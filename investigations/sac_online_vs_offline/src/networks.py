import os
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions.normal import Normal

class CriticNetwork(nn.Module):
    def __init__(self, beta, input_dims, fc1_dims, fc2_dims,
                 n_actions, name):
        super().__init__()
        self.name = name
        self.checkpoint_file = os.path.join('models', name + '_critic.ckpt')
        os.makedirs(os.path.dirname(self.checkpoint_file), exist_ok=True)

        self.fc1 = nn.Linear(input_dims[0] + n_actions, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.q   = nn.Linear(fc2_dims, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=beta)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state, action):
        x = T.cat([state, action], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.q(x)

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))


class ValueNetwork(nn.Module):
    def __init__(self, beta, input_dims, fc1_dims, fc2_dims, name):
        super().__init__()
        self.name = name
        self.checkpoint_file = os.path.join('models', name + '_value.ckpt')
        os.makedirs(os.path.dirname(self.checkpoint_file), exist_ok=True)

        self.fc1 = nn.Linear(input_dims[0], fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.v   = nn.Linear(fc2_dims, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=beta)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.v(x)

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))


class ActorNetwork(nn.Module):
    def __init__(self, alpha, input_dims, fc1_dims, fc2_dims,
                 n_actions, name, max_action):
        super().__init__()
        self.name = name
        self.checkpoint_file = os.path.join('models', name + '_actor.ckpt')
        os.makedirs(os.path.dirname(self.checkpoint_file), exist_ok=True)

        self.fc1 = nn.Linear(input_dims[0],    fc1_dims)
        self.fc2 = nn.Linear(fc1_dims,        fc2_dims)
        self.mu_layer    = nn.Linear(fc2_dims, n_actions)
        self.log_std_layer = nn.Linear(fc2_dims, n_actions)

        self.max_action = T.tensor(max_action, dtype=T.float32).to(
            T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        )

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def sample_normal(self, state, reparameterize=True):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))

        mu        = self.mu_layer(x)
        log_sigma = self.log_std_layer(x)

        # clamp log_sigma, then sigma
        log_sigma = T.clamp(log_sigma, min=-20.0, max=2.0)
        sigma     = T.exp(log_sigma).clamp(min=1e-6, max=1.0)

        dist = Normal(mu, sigma)
        raw_actions = dist.rsample() if reparameterize else dist.sample()

        actions = T.tanh(raw_actions) * self.max_action

        log_probs = dist.log_prob(raw_actions)
        log_probs -= T.log(
            self.max_action * (1 - T.tanh(raw_actions) ** 2) + 1e-6
        )
        log_probs = log_probs.sum(axis=1, keepdim=True)

        return actions, log_probs

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))
