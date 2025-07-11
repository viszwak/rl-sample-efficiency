import os
import torch
import torch.nn.functional as F
import numpy as np
from networks import ActorNetwork as Actor, CriticNetwork


class TwinCritic(torch.nn.Module):
    """Two independent Q-networks sharing no parameters."""

    def __init__(self, state_dim: int, action_dim: int, chkpt_dir: str = "tmp"):
        super().__init__()
        self.q1 = CriticNetwork(
            beta=3e-4,
            input_dims=(state_dim,),
            fc1_dims=256,
            fc2_dims=256,
            n_actions=action_dim,
            name="q1",
            chkpt_dir=chkpt_dir,
        )
        self.q2 = CriticNetwork(
            beta=3e-4,
            input_dims=(state_dim,),
            fc1_dims=256,
            fc2_dims=256,
            n_actions=action_dim,
            name="q2",
            chkpt_dir=chkpt_dir,
        )

    # Convenience wrappers --------------------------------------------------
    def Q1(self, s, a):
        return self.q1(s, a)

    def Q2(self, s, a):
        return self.q2(s, a)

    def forward(self, s, a):
        return self.Q1(s, a), self.Q2(s, a)


class SAC_CQL:
    """Soft‑Actor‑Critic with Conservative Q‑Learning regulariser and optional VAE encoder."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        max_action: float,
        device: torch.device,
        *,
        sim: int = 1,
        discount: float = 0.99,
        tau: float = 0.005,
        alpha: float = 0.2,
        cql_alpha: float = 0.1,
        vae: torch.nn.Module | None = None,
    ) -> None:
        self.device = device
        self.vae = vae  # frozen encoder (pre‑trained); can be None

        # ------------------------------------------------------------
        #   Use latent dimension if VAE is provided
        # ------------------------------------------------------------
        if self.vae is not None:
            latent_dim = 6  # ↳ hard‑coded; change if you retrain VAE with a different size
            state_dim = latent_dim
        self.state_dim = state_dim
        self.action_dim = action_dim

        # ------------------------------------------------------------
        #   Per‑simulation checkpoint directory
        # ------------------------------------------------------------
        self.chkpt_dir = os.path.join("models", f"sim{sim}")
        os.makedirs(self.chkpt_dir, exist_ok=True)

        # ------------------------------------------------------------
        #   Actor & critics
        # ------------------------------------------------------------
        self.actor = Actor(
            alpha=3e-4,
            input_dims=(state_dim,),
            fc1_dims=256,
            fc2_dims=256,
            n_actions=action_dim,
            max_action=max_action,
            name="actor",
            chkpt_dir=self.chkpt_dir,
        ).to(device)

        self.critic = TwinCritic(state_dim, action_dim, chkpt_dir=self.chkpt_dir).to(device)
        self.critic_target = TwinCritic(state_dim, action_dim, chkpt_dir=self.chkpt_dir).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_optimizer = self.actor.optimizer
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)

        # hyper‑params --------------------------------------------------------
        self.max_action = max_action
        self.discount = discount
        self.tau = tau
        self.alpha = alpha
        self.cql_alpha = cql_alpha

    # ------------------------------------------------------------------
    #  Checkpoint helpers
    # ------------------------------------------------------------------
    def save_models(self):
        """Save actor + twin‑critic weights to sim‑specific folder."""
        self.actor.save_checkpoint()
        self.critic.q1.save_checkpoint()
        self.critic.q2.save_checkpoint()

    def load_models(self):
        self.actor.load_checkpoint()
        self.critic.q1.load_checkpoint()
        self.critic.q2.load_checkpoint()

    # ------------------------------------------------------------------
    #  Interaction helpers
    # ------------------------------------------------------------------
    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode raw states via VAE if provided (runs under no_grad)."""
        if self.vae is None:
            return x
        with torch.no_grad():
            return self.vae.encode_latent(x)

    def select_action(self, state: np.ndarray) -> np.ndarray:
        state = torch.tensor(state.reshape(1, -1), dtype=torch.float32, device=self.device)
        state = self._encode(state)
        action, _ = self.actor.sample_normal(state, reparameterize=False)
        return action.cpu().numpy().flatten()

    # ------------------------------------------------------------------
    #  Training step
    # ------------------------------------------------------------------
    def train(self, replay_buffer, batch_size: int = 256):
        # Sample batch -------------------------------------------------
        state, action, reward, next_state, done = replay_buffer.sample_buffer(batch_size)

        state = torch.tensor(state, dtype=torch.float32, device=self.device)
        action = torch.tensor(action, dtype=torch.float32, device=self.device)
        reward = torch.tensor(reward, dtype=torch.float32, device=self.device).unsqueeze(1)
        next_state = torch.tensor(next_state, dtype=torch.float32, device=self.device)
        not_done = torch.tensor(1 - done, dtype=torch.float32, device=self.device).unsqueeze(1)

        # Encode states via VAE ----------------------------------------
        state = self._encode(state)
        next_state = self._encode(next_state)

        # --------------------------------------------------------------
        # Critic target ------------------------------------------------
        with torch.no_grad():
            next_action, _ = self.actor.sample_normal(next_state, reparameterize=False)
            target_q1 = self.critic_target.Q1(next_state, next_action)
            target_q2 = self.critic_target.Q2(next_state, next_action)
            target_Q = reward + not_done * self.discount * torch.min(target_q1, target_q2)

        current_q1 = self.critic.Q1(state, action)
        current_q2 = self.critic.Q2(state, action)

        q1_loss = F.mse_loss(current_q1, target_Q)
        q2_loss = F.mse_loss(current_q2, target_Q)

        # --------------------------------------------------------------
        # Conservative Q‑Learning penalty -----------------------------
        num_samples = 2
        with torch.no_grad():
            rand_actions = torch.empty(batch_size * num_samples, self.action_dim, device=self.device).uniform_(-1, 1)
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

        # Update critic ------------------------------------------------
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # --------------------------------------------------------------
        # Actor update -------------------------------------------------
        pi, _ = self.actor.sample_normal(state, reparameterize=True)
        q1_pi = self.critic.Q1(state, pi)
        q2_pi = self.critic.Q2(state, pi)
        actor_loss = (-torch.min(q1_pi, q2_pi)).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # --------------------------------------------------------------
        # Soft target update ------------------------------------------
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)
