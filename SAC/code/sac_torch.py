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
                 layer2_size=256, batch_size=256, reward_scale=2):
        
        self.gamma = gamma
        self.tau = tau
        self.memory = ReplayBuffer(max_size, input_dims, n_actions)
        self.batch_size = batch_size
        self.n_actions = n_actions
        self.temperature = 0.2
        self.scale = reward_scale

        self.actor = ActorNetwork(alpha, input_dims, layer1_size,
                                  layer2_size, n_actions=n_actions,
                                  name=env_id+'_actor', 
                                  max_action=env.action_space.high)
        
        self.critic_1 = CriticNetwork(beta, input_dims, layer1_size,
                                      layer2_size, n_actions=n_actions,
                                      name=env_id+'_critic_1')
        
        self.critic_2 = CriticNetwork(beta, input_dims, layer1_size,
                                      layer2_size, n_actions=n_actions,
                                      name=env_id+'_critic_2')
       
        self.value = ValueNetwork(beta, input_dims, layer1_size,
                                  layer2_size, name=env_id+'_value')
        
        self.target_value = ValueNetwork(beta, input_dims, layer1_size,
                                         layer2_size, name=env_id+'_target_value')

        self.update_network_parameters(tau=1)

    def choose_action(self, observation, step_count=None):
        if isinstance(observation, tuple):
            observation = np.array(observation[0])
        
        state = T.tensor(observation, dtype=T.float32).unsqueeze(0).to(self.actor.device)
        actions, _ = self.actor.sample_normal(state, reparameterize=False)
        return actions.cpu().detach().numpy()[0]

    def remember(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return

        state, action, reward, new_state, done = \
                self.memory.sample_buffer(self.batch_size)

        reward = T.tensor(reward, dtype=T.float).to(self.critic_1.device)
        done = T.tensor(done).to(self.critic_1.device)
        state_ = T.tensor(new_state, dtype=T.float).to(self.critic_1.device)
        state = T.tensor(state, dtype=T.float).to(self.critic_1.device)
        action = T.tensor(action, dtype=T.float).to(self.critic_1.device)
        
        value = self.value(state).view(-1)
        value_ = self.target_value(state_).view(-1)
        value_[done] = 0.0
       
        actions, log_probs = self.actor.sample_normal(state, reparameterize=False)
        log_probs = log_probs.view(-1)
        q1_new_policy = self.critic_1.forward(state, actions)
        q2_new_policy = self.critic_2.forward(state, actions)
        critic_value = T.min(q1_new_policy, q2_new_policy).view(-1)
        
        self.value.optimizer.zero_grad()
        value_target = critic_value - self.temperature * log_probs
        value_loss = 0.5 * F.mse_loss(value, value_target)
        value_loss.backward()
        self.value.optimizer.step()

        actions, log_probs = self.actor.sample_normal(state, reparameterize=True)
        log_probs = log_probs.view(-1)
        q1_new_policy = self.critic_1.forward(state, actions)
        q2_new_policy = self.critic_2.forward(state, actions)
        critic_value = T.min(q1_new_policy, q2_new_policy).view(-1)

        actor_loss = T.mean(self.temperature * log_probs - critic_value)
        self.actor.optimizer.zero_grad()
        actor_loss.backward()
        self.actor.optimizer.step()

        self.critic_1.optimizer.zero_grad()
        self.critic_2.optimizer.zero_grad()
        q_hat = self.scale * reward + self.gamma * value_
        q1_old_policy = self.critic_1.forward(state, action).view(-1)
        q2_old_policy = self.critic_2.forward(state, action).view(-1)
        critic_1_loss = 0.5 * F.mse_loss(q1_old_policy, q_hat)
        critic_2_loss = 0.5 * F.mse_loss(q2_old_policy, q_hat)
        critic_loss = critic_1_loss + critic_2_loss
        critic_loss.backward()
        self.critic_1.optimizer.step()
        self.critic_2.optimizer.step()

        self.update_network_parameters()

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        target_params = self.target_value.named_parameters()
        value_params = self.value.named_parameters()
        target_dict = dict(target_params)
        value_dict = dict(value_params)

        for name in value_dict:
            value_dict[name] = tau * value_dict[name].clone() + \
                              (1 - tau) * target_dict[name].clone()

        self.target_value.load_state_dict(value_dict)

    def save_models(self, prefix):
        print('.... saving models ....')
        self.actor.save_checkpoint(f'{prefix}_actor.pt')
        self.value.save_checkpoint(f'{prefix}_value.pt')
        self.target_value.save_checkpoint(f'{prefix}_target_value.pt')
        self.critic_1.save_checkpoint(f'{prefix}_critic1.pt')
        self.critic_2.save_checkpoint(f'{prefix}_critic2.pt')

    def load_models(self, prefix):
        print('.... loading models ....')
        self.actor.load_checkpoint(f'{prefix}_actor.pt')
        self.value.load_checkpoint(f'{prefix}_value.pt')
        self.target_value.load_checkpoint(f'{prefix}_target_value.pt')
        self.critic_1.load_checkpoint(f'{prefix}_critic1.pt')
        self.critic_2.load_checkpoint(f'{prefix}_critic2.pt')
