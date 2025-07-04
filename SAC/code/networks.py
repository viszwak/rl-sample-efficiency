import os
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions.normal import Normal
import numpy as np

class CriticNetwork(nn.Module):
    def __init__(self, beta, input_dims, fc1_dims, fc2_dims, n_actions,
                 name, chkpt_dir='tmp/sac'):
        super(CriticNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dims[0] + n_actions, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.q1 = nn.Linear(fc2_dims, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=beta)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state, action):
        q1_action_value = F.relu(self.fc1(T.cat([state, action], dim=1)))
        q1_action_value = F.relu(self.fc2(q1_action_value))
        q1 = self.q1(q1_action_value)
        return q1

    def save_checkpoint(self, filepath):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        T.save(self.state_dict(), filepath)

    def load_checkpoint(self, filepath):
        self.load_state_dict(T.load(filepath))

class ActorNetwork(nn.Module):
    def __init__(self, alpha, input_dims, fc1_dims, fc2_dims, max_action,
                 n_actions, name, chkpt_dir='tmp/sac'):
        super(ActorNetwork, self).__init__()
        self.max_action = max_action
        self.reparam_noise = 1e-6

        self.fc1 = nn.Linear(*input_dims, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.mu = nn.Linear(fc2_dims, n_actions)
        self.sigma = nn.Linear(fc2_dims, n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        prob = F.relu(self.fc1(state))
        prob = F.relu(self.fc2(prob))
        mu = self.mu(prob)
        sigma = T.clamp(self.sigma(prob), min=self.reparam_noise, max=1)
        return mu, sigma

    def sample_normal(self, state, reparameterize=True):
        mu, sigma = self.forward(state)
        probabilities = T.distributions.Normal(mu, sigma)
        actions = probabilities.rsample() if reparameterize else probabilities.sample()
        action = T.tanh(actions) * T.tensor(self.max_action).to(self.device)

        log_probs = probabilities.log_prob(actions)
        log_probs -= T.log(1 - action.pow(2) + self.reparam_noise)
        log_probs = log_probs.sum(1, keepdim=True)
        return action, log_probs

    def save_checkpoint(self, filepath):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        T.save(self.state_dict(), filepath)

    def load_checkpoint(self, filepath):
        self.load_state_dict(T.load(filepath))

class ValueNetwork(nn.Module):
    def __init__(self, beta, input_dims, fc1_dims, fc2_dims,
                 name, chkpt_dir='tmp/sac'):
        super(ValueNetwork, self).__init__()
        self.fc1 = nn.Linear(*input_dims, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.v = nn.Linear(fc2_dims, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=beta)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        state_value = F.relu(self.fc1(state))
        state_value = F.relu(self.fc2(state_value))
        v = self.v(state_value)
        return v

    def save_checkpoint(self, filepath):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        T.save(self.state_dict(), filepath)

    def load_checkpoint(self, filepath):
        self.load_state_dict(T.load(filepath))
