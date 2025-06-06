import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pickle
import matplotlib.pyplot as plt
from collections import deque
import random

# Q-network
class DQN(nn.Module):
    def __init__(self, obs_dim, n_actions):
        super(DQN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions)
        )

    def forward(self, x):
        return self.fc(x)

# Replay buffer
class ReplayBuffer:
    def __init__(self, max_size=100000):
        self.buffer = deque(maxlen=max_size)

    def store(self, transition):
        self.buffer.append(transition)

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        return zip(*batch)

# Epsilon-greedy policy
def select_action(state, q_net, epsilon, n_actions):
    if np.random.rand() < epsilon:
        return np.random.randint(n_actions)
    state_tensor = torch.tensor([state], dtype=torch.float32)
    with torch.no_grad():
        return q_net(state_tensor).argmax().item()

# Training loop
def train_dqn():
    env = gym.make('LunarLander-v2')
    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n

    q_net = DQN(obs_dim, n_actions)
    target_net = DQN(obs_dim, n_actions)
    target_net.load_state_dict(q_net.state_dict())

    optimizer = optim.Adam(q_net.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    buffer = ReplayBuffer()
    
    gamma = 0.99
    epsilon = 1.0
    epsilon_decay = 0.995
    epsilon_min = 0.05
    batch_size = 64
    target_update_freq = 100
    n_episodes = 1000
    rewards = []
    dataset = []  # ← for offline use

    for ep in range(n_episodes):
        state, _ = env.reset()
        done = False
        total_reward = 0

        while not done:
            action = select_action(state, q_net, epsilon, n_actions)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            transition = (state, action, reward, next_state, done)
            buffer.store(transition)
            dataset.append(transition)
            state = next_state
            total_reward += reward

            if len(buffer.buffer) >= batch_size:
                s, a, r, s2, d = buffer.sample(batch_size)
                s = torch.tensor(s, dtype=torch.float32)
                a = torch.tensor(a, dtype=torch.int64).unsqueeze(1)
                r = torch.tensor(r, dtype=torch.float32).unsqueeze(1)
                s2 = torch.tensor(s2, dtype=torch.float32)
                d = torch.tensor(d, dtype=torch.float32).unsqueeze(1)

                q_values = q_net(s).gather(1, a)
                with torch.no_grad():
                    target_q = r + gamma * target_net(s2).max(1, keepdim=True)[0] * (1 - d)
                loss = loss_fn(q_values, target_q)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        rewards.append(total_reward)
        epsilon = max(epsilon_min, epsilon * epsilon_decay)

        if ep % target_update_freq == 0:
            target_net.load_state_dict(q_net.state_dict())

        print(f"Episode {ep}, Reward: {total_reward:.2f}, Epsilon: {epsilon:.2f}")

    env.close()

    # Save results
    with open("dqn_lunarlander_scores.pkl", "wb") as f:
        pickle.dump(rewards, f)

    with open("dqn_lunarlander_dataset.pkl", "wb") as f:
        pickle.dump(dataset, f)

    plt.plot(rewards)
    plt.title("DQN on LunarLander-v2")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.grid()
    plt.savefig("dqn_lunarlander_plot.png")
    print("✅ DQN training and dataset saving complete.")

if __name__ == "__main__":
    train_dqn()
