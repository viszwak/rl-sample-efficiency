import os
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class CriticNetwork(nn.Module):
    def __init__(self, beta, input_dims, fc1_dims, fc2_dims, n_actions,
                 name, chkpt_dir='tmp/sac'):
        super().__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name+'_sac')

        self.fc1 = nn.Linear(self.input_dims[0] + n_actions, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.q1 = nn.Linear(self.fc2_dims, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=beta)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state, action):
        x = T.cat([state, action], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        q = self.q1(x)
        return q

    def save_checkpoint(self):
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))


class ActorNetwork(nn.Module):
    """
    Tanh-Gaussian actor.
    Returns:
      - sample_normal(...): (action, log_prob) where log_prob includes tanh correction
      - mean_action(...): deterministic tanh(mu) * max_action for evaluation
    """
    def __init__(self, alpha, input_dims, fc1_dims, fc2_dims, max_action,
                 n_actions, name, chkpt_dir='tmp/sac'):
        super().__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.name = name
        self.max_action = float(max_action)
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name+'_sac')
        self.eps = 1e-6

        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.mu = nn.Linear(self.fc2_dims, self.n_actions)
        self.log_std = nn.Linear(self.fc2_dims, self.n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        mu = self.mu(x)
        # clamp log_std to a sane range to avoid numerical issues
        log_std = self.log_std(x)
        log_std = T.clamp(log_std, min=-20.0, max=2.0)
        std = T.exp(log_std)
        return mu, std, log_std

    def sample_normal(self, state, reparameterize=True):
        mu, std, _ = self.forward(state)
        dist = T.distributions.Normal(mu, std)
        if reparameterize:
            u = dist.rsample()
        else:
            u = dist.sample()
        # squashed action
        a = T.tanh(u)
        action = a * self.max_action

        # log prob with tanh correction: sum over action dims
        # log π(a) = log N(u; μ, σ) - sum log(1 - tanh(u)^2)
        log_prob = dist.log_prob(u) - T.log(1 - a.pow(2) + self.eps)
        log_prob = log_prob.sum(dim=1, keepdim=True)
        return action, log_prob

    @T.no_grad()
    def mean_action(self, state):
        mu, _, _ = self.forward(state)
        a = T.tanh(mu) * self.max_action
        return a

    def save_checkpoint(self):
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))


class ValueNetwork(nn.Module):
    def __init__(self, beta, input_dims, fc1_dims, fc2_dims,
                 name, chkpt_dir='tmp/sac'):
        super().__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name+'_sac')

        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.v = nn.Linear(self.fc2_dims, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=beta)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        v = self.v(x)
        return v

    def save_checkpoint(self):
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))