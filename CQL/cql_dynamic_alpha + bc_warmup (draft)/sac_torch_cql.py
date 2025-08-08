import os
import numpy as np
import torch
import torch.nn.functional as F
from networks import ActorNetwork as Actor, CriticNetwork


class TwinCritic(torch.nn.Module):
    def __init__(self, state_dim, action_dim, chkpt_dir='tmp'):
        super().__init__()
        self.q1 = CriticNetwork(
            beta=3e-4, input_dims=(state_dim,), fc1_dims=256, fc2_dims=256,
            n_actions=action_dim, name='q1', chkpt_dir=chkpt_dir
        )
        self.q2 = CriticNetwork(
            beta=3e-4, input_dims=(state_dim,), fc1_dims=256, fc2_dims=256,
            n_actions=action_dim, name='q2', chkpt_dir=chkpt_dir
        )

    def Q1(self, s, a): return self.q1(s, a)
    def Q2(self, s, a): return self.q2(s, a)
    def forward(self, s, a): return self.Q1(s, a), self.Q2(s, a)


class SAC_CQL:
    def __init__(
        self,
        state_dim,
        action_dim,
        max_action,
        device,
        sim=1,
        discount=0.99,
        tau=0.005,
        alpha=0.2,                 # SAC entropy temperature (fixed here)
        cql_alpha_init=2.0,        # initial CQL Lagrange weight
        bc_warmup_steps=20_000,    # steps to do BC before SAC objective
        num_cql_random=10,         # random actions per state for CQL
        use_next_policy_actions=True
    ):
        self.device = device
        self.sim = sim
        self.chkpt_dir = f"models/sim{self.sim}"
        os.makedirs(self.chkpt_dir, exist_ok=True)

        self.actor = Actor(
            alpha=3e-4, input_dims=(state_dim,), fc1_dims=256, fc2_dims=256,
            n_actions=action_dim, max_action=max_action, name='actor', chkpt_dir=self.chkpt_dir
        ).to(device)
        self.actor_optimizer = self.actor.optimizer

        self.critic = TwinCritic(state_dim, action_dim, chkpt_dir=self.chkpt_dir).to(device)
        self.critic_target = TwinCritic(state_dim, action_dim, chkpt_dir=self.chkpt_dir).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)

        # CQL Lagrange weight (learnable): alpha_cql = exp(log_alpha)
        self.log_cql_alpha = torch.tensor(np.log(cql_alpha_init), dtype=torch.float32,
                                          requires_grad=True, device=self.device)
        self.cql_alpha_optimizer = torch.optim.Adam([self.log_cql_alpha], lr=1e-4)

        self.max_action = float(max_action)
        self.discount = discount
        self.tau = tau
        self.alpha = alpha
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.bc_warmup_steps = bc_warmup_steps
        self.num_cql_random = num_cql_random
        self.use_next_policy_actions = use_next_policy_actions

        self.total_it = 0
        self.last_cql_q1_penalty = 0.0
        self.last_cql_q2_penalty = 0.0

    @property
    def cql_alpha(self):
        return self.log_cql_alpha.exp()

    # ---------- persistence ----------
    def save_models(self):
        self.actor.save_checkpoint()
        self.critic.q1.save_checkpoint()
        self.critic.q2.save_checkpoint()

    def load_models(self):
        self.actor.load_checkpoint()
        self.critic.q1.load_checkpoint()
        self.critic.q2.load_checkpoint()

    # ---------- policy helpers ----------
    def _policy(self, s, reparameterize: bool):
        return self.actor.sample_normal(s, reparameterize=reparameterize)

    def select_action(self, state, deterministic: bool = True):
        s = torch.tensor(state, dtype=torch.float32, device=self.device).view(1, -1)
        if deterministic:
            a = self.actor.mean_action(s)
        else:
            a, _ = self._policy(s, reparameterize=False)
        return a.detach().cpu().numpy().flatten()

    # ---------- training ----------
    def train(self, replay_buffer, batch_size=256):
        self.total_it += 1

        state, action, reward, next_state, done = replay_buffer.sample_buffer(batch_size)

        state      = torch.tensor(state,      dtype=torch.float32, device=self.device)
        action     = torch.tensor(action,     dtype=torch.float32, device=self.device)
        reward     = torch.tensor(reward,     dtype=torch.float32, device=self.device).unsqueeze(1)
        next_state = torch.tensor(next_state, dtype=torch.float32, device=self.device)
        not_done   = torch.tensor(1 - done,   dtype=torch.float32, device=self.device).unsqueeze(1)

        # --- SAC target with entropy ---
        with torch.no_grad():
            next_action, log_pi_next = self._policy(next_state, reparameterize=False)
            tq1 = self.critic_target.Q1(next_state, next_action)
            tq2 = self.critic_target.Q2(next_state, next_action)
            target_v = torch.min(tq1, tq2) - self.alpha * log_pi_next
            target_Q = reward + not_done * self.discount * target_v

        # --- critic MSE ---
        current_q1 = self.critic.Q1(state, action)
        current_q2 = self.critic.Q2(state, action)
        q1_loss = F.mse_loss(current_q1, target_Q)
        q2_loss = F.mse_loss(current_q2, target_Q)

        # --- CQL(H) penalty (union of actions) ---
        B = state.size(0)
        K = self.num_cql_random

        # random actions ~ U([-max_action, max_action])
        with torch.no_grad():
            rand_actions = torch.empty(B, K, self.action_dim, device=self.device)\
                .uniform_(-self.max_action, self.max_action)

        # current-policy actions at s
        pi_s, _ = self._policy(state, reparameterize=False)
        q1_pi_s = self.critic.Q1(state, pi_s)   # [B,1]
        q2_pi_s = self.critic.Q2(state, pi_s)   # [B,1]

        # next-policy actions at s' (optional but helpful)
        if self.use_next_policy_actions:
            with torch.no_grad():
                pi_next, _ = self._policy(next_state, reparameterize=False)
            q1_pi_next = self.critic.Q1(next_state, pi_next)  # [B,1]
            q2_pi_next = self.critic.Q2(next_state, pi_next)  # [B,1]
        else:
            q1_pi_next = q2_pi_next = None


        # evaluate random actions on current-state Q
        s_rep = state.unsqueeze(1).repeat(1, K, 1).view(-1, self.state_dim)
        rand_a = rand_actions.view(-1, self.action_dim)
        q1_rand = self.critic.Q1(s_rep, rand_a).view(B, K)
        q2_rand = self.critic.Q2(s_rep, rand_a).view(B, K)

        # build union sets
        q1_union = [q1_rand, q1_pi_s]
        q2_union = [q2_rand, q2_pi_s]
        if q1_pi_next is not None:
            q1_union.append(q1_pi_next)
            q2_union.append(q2_pi_next)

        q1_union = torch.cat(q1_union, dim=1)  # [B, K + 1 (+1)]
        q2_union = torch.cat(q2_union, dim=1)

        logK = torch.log(torch.tensor(q1_union.size(1), dtype=torch.float32, device=self.device))
        cql_q1 = (torch.logsumexp(q1_union, dim=1) - logK - current_q1.squeeze(1)).mean()
        cql_q2 = (torch.logsumexp(q2_union, dim=1) - logK - current_q2.squeeze(1)).mean()

        self.last_cql_q1_penalty = float(cql_q1.detach().cpu())
        self.last_cql_q2_penalty = float(cql_q2.detach().cpu())

        alpha_cql = self.cql_alpha.detach()
        critic_loss = q1_loss + q2_loss + alpha_cql * (cql_q1 + cql_q2)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # --- Lagrange update for alpha_cql ---
        # Increase alpha_cql when gap < target; decrease when gap > target
        cql_gap = 0.5 * (cql_q1.detach() + cql_q2.detach())
        target_gap = torch.tensor(5.0, dtype=torch.float32, device=self.device)  # tune in [1,10]
        cql_alpha_loss = self.cql_alpha * (cql_gap - target_gap)

        self.cql_alpha_optimizer.zero_grad()
        cql_alpha_loss.backward()
        self.cql_alpha_optimizer.step()

        with torch.no_grad():
            self.log_cql_alpha.clamp_(min=np.log(1e-3), max=np.log(1e3))

        # --- Actor update (BC warmup -> SAC) ---
        pi, log_pi = self._policy(state, reparameterize=True)
        if self.total_it < self.bc_warmup_steps:
            actor_loss = F.mse_loss(pi, action)  # behavior cloning
        else:
            q1_pi = self.critic.Q1(state, pi)
            q2_pi = self.critic.Q2(state, pi)
            actor_loss = (self.alpha * log_pi - torch.min(q1_pi, q2_pi)).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # --- Soft target update ---
        with torch.no_grad():
            for p, tp in zip(self.critic.parameters(), self.critic_target.parameters()):
                tp.data.mul_(1.0 - self.tau).add_(self.tau * p.data)
