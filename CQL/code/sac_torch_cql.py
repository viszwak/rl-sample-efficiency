import os                    # NEW
import torch
import torch.nn.functional as F
import numpy as np
from networks import ActorNetwork as Actor, CriticNetwork


class TwinCritic(torch.nn.Module):
    def __init__(self, state_dim, action_dim, chkpt_dir='tmp'):  # ← added chkpt_dir
        super().__init__()
        self.q1 = CriticNetwork(
            beta=3e-4,
            input_dims=(state_dim,),
            fc1_dims=256,
            fc2_dims=256,
            n_actions=action_dim,
            name='q1',
            chkpt_dir=chkpt_dir)          # use sim-specific dir

        self.q2 = CriticNetwork(
            beta=3e-4,
            input_dims=(state_dim,),
            fc1_dims=256,
            fc2_dims=256,
            n_actions=action_dim,
            name='q2',
            chkpt_dir=chkpt_dir)          # use sim-specific dir

    def Q1(self, s, a):
        return self.q1.forward(s, a)

    def Q2(self, s, a):
        return self.q2.forward(s, a)

    def forward(self, s, a):
        return self.Q1(s, a), self.Q2(s, a)


class SAC_CQL:
    def __init__(
        self,
        state_dim,
        action_dim,
        max_action,
        device,
        sim=1,                                   # ← new sim arg (default = 1)
        discount=0.99,
        tau=0.005,
        alpha=0.2,
        cql_alpha=0.1
    ):
        self.device = device
        self.sim = sim                           # store sim ID

        # -------- per-sim checkpoint directory ----------
        self.chkpt_dir = f"models/sim{self.sim}"
        os.makedirs(self.chkpt_dir, exist_ok=True)
        # -------------------------------------------------

        self.actor = Actor(
            alpha=3e-4,
            input_dims=(state_dim,),
            fc1_dims=256,
            fc2_dims=256,
            n_actions=action_dim,
            max_action=max_action,
            name='actor',
            chkpt_dir=self.chkpt_dir             # sim-specific dir
        ).to(device)

        self.critic = TwinCritic(state_dim, action_dim, chkpt_dir=self.chkpt_dir).to(device)
        self.critic_target = TwinCritic(state_dim, action_dim, chkpt_dir=self.chkpt_dir).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_optimizer = self.actor.optimizer
        self.critic_optimizer = torch.optim.Adam(list(self.critic.parameters()), lr=3e-4)

        self.max_action = max_action
        self.discount = discount
        self.tau = tau
        self.alpha = alpha
        self.cql_alpha = cql_alpha
        self.action_dim = action_dim
        self.state_dim = state_dim

    # ------------------------------------------------------------------
    #  NEW helpers to save / load all model components for this sim
    # ------------------------------------------------------------------
    def save_models(self):
        """Save actor and twin-critic checkpoints to models/sim{N}."""
        self.actor.save_checkpoint()
        self.critic.q1.save_checkpoint()
        self.critic.q2.save_checkpoint()

    def load_models(self):
        """Load actor and twin-critic checkpoints from models/sim{N}."""
        self.actor.load_checkpoint()
        self.critic.q1.load_checkpoint()
        self.critic.q2.load_checkpoint()
    # ------------------------------------------------------------------

    def select_action(self, state):
        state = torch.tensor(state.reshape(1, -1), dtype=torch.float32).to(self.device)
        action, _ = self.actor.sample_normal(state, reparameterize=False)
        return action.cpu().data.numpy().flatten()

    def train(self, replay_buffer, batch_size=256):
        state, action, reward, next_state, done = replay_buffer.sample_buffer(batch_size)

        state = torch.tensor(state, dtype=torch.float32).to(self.device)
        action = torch.tensor(action, dtype=torch.float32).to(self.device)
        reward = torch.tensor(reward, dtype=torch.float32).unsqueeze(1).to(self.device)
        next_state = torch.tensor(next_state, dtype=torch.float32).to(self.device)
        not_done = torch.tensor(1 - done, dtype=torch.float32).unsqueeze(1).to(self.device)

        with torch.no_grad():
            next_action, _ = self.actor.sample_normal(next_state, reparameterize=False)
            target_q1 = self.critic_target.Q1(next_state, next_action)
            target_q2 = self.critic_target.Q2(next_state, next_action)
            target_Q = reward + not_done * self.discount * torch.min(target_q1, target_q2)

        current_q1 = self.critic.Q1(state, action)
        current_q2 = self.critic.Q2(state, action)

        q1_loss = F.mse_loss(current_q1, target_Q)
        q2_loss = F.mse_loss(current_q2, target_Q)

        # CQL penalty
        num_samples = 2
        with torch.no_grad():
            rand_actions = torch.empty(batch_size * num_samples, self.action_dim).uniform_(-1, 1).to(self.device)

        s_repeat = state.unsqueeze(1).repeat(1, num_samples, 1).reshape(-1, self.state_dim)
        pol_actions, _ = self.actor.sample_normal(s_repeat, reparameterize=False)
        pol_actions = pol_actions.detach()

        q1_rand = self.critic.Q1(s_repeat, rand_actions).reshape(batch_size, num_samples)
        q2_rand = self.critic.Q2(s_repeat, rand_actions).reshape(batch_size, num_samples)

        cql_q1 = (torch.logsumexp(q1_rand, dim=1) - current_q1.squeeze()).mean()
        cql_q2 = (torch.logsumexp(q2_rand, dim=1) - current_q2.squeeze()).mean()

        q1_loss += self.cql_alpha * cql_q1
        q2_loss += self.cql_alpha * cql_q2

        critic_loss = q1_loss + q2_loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Actor update
        pi, _ = self.actor.sample_normal(state, reparameterize=True)
        q1_pi = self.critic.Q1(state, pi)
        q2_pi = self.critic.Q2(state, pi)
        actor_loss = (-torch.min(q1_pi, q2_pi)).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Soft target update
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)