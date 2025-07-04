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
                 layer2_size=256, batch_size=256, reward_scale=2,
                 sim=1):

        self.gamma = gamma
        self.tau = tau
        self.memory = ReplayBuffer(max_size, input_dims, n_actions)
        self.batch_size = batch_size
        self.n_actions = n_actions
        self.temperature = 0.2
        self.sim = sim

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

    def choose_action(self, observation):
        state = T.tensor(np.array([observation]), dtype=T.float32).to(self.actor.device)
        actions, _ = self.actor.sample_normal(state, reparameterize=False)
        return actions[0].cpu().numpy()

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return

        states, actions, rewards, next_states, dones = \
            self.memory.sample_buffer(self.batch_size)

        states = T.tensor(states, dtype=T.float32).to(self.actor.device)
        actions = T.tensor(actions, dtype=T.float32).to(self.actor.device)
        rewards = T.tensor(rewards, dtype=T.float32).to(self.actor.device)
        next_states = T.tensor(next_states, dtype=T.float32).to(self.actor.device)
        dones = T.tensor(dones, dtype=T.float32).to(self.actor.device)

        # Value target: r + γ * V(s′)
        value_next = self.value(next_states).view(-1)
        q_target = rewards + self.gamma * value_next * (1 - dones)

        q1_pred = self.critic_1(states, actions).view(-1)
        q2_pred = self.critic_2(states, actions).view(-1)

        critic_1_loss = F.mse_loss(q1_pred, q_target.detach())
        critic_2_loss = F.mse_loss(q2_pred, q_target.detach())

        self.critic_1.optimizer.zero_grad()
        critic_1_loss.backward()
        self.critic_1.optimizer.step()

        self.critic_2.optimizer.zero_grad()
        critic_2_loss.backward()
        self.critic_2.optimizer.step()

        new_actions, log_pi = self.actor.sample_normal(states, reparameterize=True)
        q1_new = self.critic_1(states, new_actions)
        q2_new = self.critic_2(states, new_actions)
        q_min = T.min(q1_new, q2_new)

        actor_loss = (self.temperature * log_pi - q_min).mean()

        self.actor.optimizer.zero_grad()
        actor_loss.backward()
        self.actor.optimizer.step()

        value = self.value(states).view(-1)
        value_target = (q_min - self.temperature * log_pi).view(-1).detach()
        value_loss = F.mse_loss(value, value_target)

        self.value.optimizer.zero_grad()
        value_loss.backward()
        self.value.optimizer.step()

        with T.no_grad():
            for param, target_param in zip(self.value.parameters(), self.value.target.parameters()):
                target_param.data.copy_(
                    self.tau * param.data + (1 - self.tau) * target_param.data
                )

    def save_models(self):
        print('... saving models ...')
        folder = f'models/offline_sim_{self.sim}'
        os.makedirs(folder, exist_ok=True)
        self.actor.save_checkpoint(f'{folder}/actor.pth')
        self.critic_1.save_checkpoint(f'{folder}/critic_1.pth')
        self.critic_2.save_checkpoint(f'{folder}/critic_2.pth')
        self.value.save_checkpoint(f'{folder}/value.pth')

    def load_models(self):
        print('... loading models ...')
        folder = f'models/offline_sim_{self.sim}'
        self.actor.load_checkpoint(f'{folder}/actor.pth')
        self.critic_1.load_checkpoint(f'{folder}/critic_1.pth')
        self.critic_2.load_checkpoint(f'{folder}/critic_2.pth')
        self.value.load_checkpoint(f'{folder}/value.pth')
