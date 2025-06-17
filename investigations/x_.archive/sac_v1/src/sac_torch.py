import os
import torch as T
import torch.nn.functional as F
import numpy as np
from buffer import ReplayBuffer
from networks import ActorNetwork, CriticNetwork, ValueNetwork

class Agent():
    def __init__(self, alpha, beta, input_dims, tau, env,
                 env_id, gamma=0.99, 
                 n_actions=2, max_size=1000000, layer1_size=256,
                 layer2_size=256, batch_size=100, reward_scale=2):
        self.gamma = gamma
        self.tau = tau
        self.memory = ReplayBuffer(max_size, input_dims, n_actions)
        self.batch_size = batch_size
        self.n_actions = n_actions

        self.actor = ActorNetwork(
            alpha, input_dims, layer1_size, layer2_size,
            n_actions=n_actions,
            name=env_id + '_actor',
            max_action=env.action_space.high
        )
        self.critic_1 = CriticNetwork(
            beta, input_dims, layer1_size, layer2_size,
            n_actions=n_actions,
            name=env_id + '_critic_1'
        )
        self.critic_2 = CriticNetwork(
            beta, input_dims, layer1_size, layer2_size,
            n_actions=n_actions,
            name=env_id + '_critic_2'
        )
        self.value = ValueNetwork(
            beta, input_dims, layer1_size, layer2_size,
            name=env_id + '_value'
        )
        self.target_value = ValueNetwork(
            beta, input_dims, layer1_size, layer2_size,
            name=env_id + '_target_value'
        )

        self.scale = reward_scale
        # initialize target networks to match
        self.update_network_parameters(tau=1)

    def choose_action(self, observation):
        if isinstance(observation, tuple):
            observation = observation[0]
        state = T.tensor(
            observation,
            dtype=T.float32,
            device=self.actor.device
        ).unsqueeze(0)
        actions, _ = self.actor.sample_normal(state, reparameterize=False)
        return actions.cpu().detach().numpy()[0]

    def remember(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return

        states, actions, rewards, states_, dones = \
            self.memory.sample_buffer(self.batch_size)

        device = self.critic_1.device
        states  = T.tensor(states,  dtype=T.float32, device=device)
        actions = T.tensor(actions, dtype=T.float32, device=device)
        rewards = T.tensor(rewards, dtype=T.float32, device=device)
        states_ = T.tensor(states_, dtype=T.float32, device=device)
        dones   = T.tensor(dones,   dtype=T.bool,    device=device)

        # — Value network update —
        values      = self.value(states).view(-1)
        values_next = self.target_value(states_).view(-1)
        values_next[dones] = 0.0

        new_actions, log_probs = self.actor.sample_normal(
            states, reparameterize=False)
        log_probs = log_probs.view(-1)
        q1_new = self.critic_1(states, new_actions)
        q2_new = self.critic_2(states, new_actions)
        q_min  = T.min(q1_new, q2_new).view(-1)

        value_target = q_min - log_probs
        value_loss   = 0.5 * F.mse_loss(values, value_target)

        self.value.optimizer.zero_grad()
        value_loss.backward(retain_graph=True)
        T.nn.utils.clip_grad_norm_(self.value.parameters(), max_norm=1.0)
        self.value.optimizer.step()

        # — Actor network update —
        new_actions, log_probs = self.actor.sample_normal(
            states, reparameterize=True)
        log_probs = log_probs.view(-1)
        q1_new = self.critic_1(states, new_actions)
        q2_new = self.critic_2(states, new_actions)
        q_min  = T.min(q1_new, q2_new).view(-1)

        actor_loss = T.mean(log_probs - q_min)
        self.actor.optimizer.zero_grad()
        actor_loss.backward(retain_graph=True)
        T.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
        self.actor.optimizer.step()

        # — Critic networks update —
        self.critic_1.optimizer.zero_grad()
        self.critic_2.optimizer.zero_grad()

        q_target = self.scale * rewards + self.gamma * values_next
        q1_old   = self.critic_1(states, actions).view(-1)
        q2_old   = self.critic_2(states, actions).view(-1)

        critic_1_loss = 0.5 * F.mse_loss(q1_old, q_target)
        critic_2_loss = 0.5 * F.mse_loss(q2_old, q_target)
        critic_loss   = critic_1_loss + critic_2_loss

        critic_loss.backward()
        T.nn.utils.clip_grad_norm_(self.critic_1.parameters(), max_norm=1.0)
        T.nn.utils.clip_grad_norm_(self.critic_2.parameters(), max_norm=1.0)
        self.critic_1.optimizer.step()
        self.critic_2.optimizer.step()

        # — Soft update of target value network —
        self.update_network_parameters()

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        target_params = dict(self.target_value.named_parameters())
        source_params = dict(self.value.named_parameters())

        for name in source_params:
            target_params[name] = (
                tau * source_params[name].clone()
                + (1 - tau) * target_params[name].clone()
            )

        self.target_value.load_state_dict(target_params)

    def save_models(self):
        print('.... saving models ....')
        self.actor.save_checkpoint()
        self.value.save_checkpoint()
        self.target_value.save_checkpoint()
        self.critic_1.save_checkpoint()
        self.critic_2.save_checkpoint()

    def load_models(self):
        print('.... loading models ....')
        self.actor.load_checkpoint()
        self.value.load_checkpoint()
        self.target_value.load_checkpoint()
        self.critic_1.load_checkpoint()
        self.critic_2.load_checkpoint()
