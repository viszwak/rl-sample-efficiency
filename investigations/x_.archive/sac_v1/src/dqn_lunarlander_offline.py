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

# Replay buffer for fixed dataset
class OfflineReplayBuffer:
    def __init__(self, data):
        self.buffer = data

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        return zip(*batch)

# Training loop (offline)
def train_offline_dqn():
    # Load dataset
    with open("dqn_lunarlander_dataset.pkl", "rb") as f:
        dataset = pickle.load(f)
    print(f"✅ Loaded dataset with {len(dataset)} transitions")

    env = gym.make('LunarLander-v2')
    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n

    q_net = DQN(obs_dim, n_actions)
    target_net = DQN(obs_dim, n_actions)
    target_net.load_state_dict(q_net.state_dict())

    optimizer = optim.Adam(q_net.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    buffer = OfflineReplayBuffer(dataset)
    
    gamma = 0.99
    batch_size = 64
    target_update_freq = 100
    total_steps = 550_000
    eval_interval = 10_000
    eval_episodes = 5
    rewards = []

    for step in range(1, total_steps + 1):
        s, a, r, s2, d = buffer.sample(batch_size)
        s = torch.tensor(np.array(s), dtype=torch.float32)
        a = torch.tensor(a, dtype=torch.int64).unsqueeze(1)
        r = torch.tensor(r, dtype=torch.float32).unsqueeze(1)
        s2 = torch.tensor(np.array(s2), dtype=torch.float32)
        d = torch.tensor(d, dtype=torch.float32).unsqueeze(1)

        q_values = q_net(s).gather(1, a)
        with torch.no_grad():
            target_q = r + gamma * target_net(s2).max(1, keepdim=True)[0] * (1 - d)
        loss = loss_fn(q_values, target_q)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % target_update_freq == 0:
            target_net.load_state_dict(q_net.state_dict())

        if step % eval_interval == 0:
            total_eval_reward = 0
            for _ in range(eval_episodes):
                reset_output = env.reset()
                state = reset_output[0] if isinstance(reset_output, tuple) else reset_output
                done = False
                episode_reward = 0
                while not done:
                    state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
                    action = q_net(state_tensor).argmax().item()
                    next_state, reward, terminated, truncated, _ = env.step(action)
                    done = terminated or truncated
                    state = next_state
                    episode_reward += reward
                total_eval_reward += episode_reward
            avg_reward = total_eval_reward / eval_episodes
            rewards.append((step, avg_reward))
            print(f"[Step {step}] Eval avg reward: {avg_reward:.2f}")

    # Save scores
    with open("dqn_lunarlander_offline_scores.pkl", "wb") as f:
        pickle.dump(rewards, f)

    # Plot
    steps, scores = zip(*rewards)
    plt.plot(steps, scores, label="DQN Offline")
    plt.title("Offline DQN on LunarLander-v2")
    plt.xlabel("Training Steps")
    plt.ylabel("Average Evaluation Reward")
    plt.grid()
    plt.legend()
    plt.savefig("dqn_lunarlander_offline_plot.png")
    print("✅ Offline DQN training complete and plot saved.")

if __name__ == "__main__":
    train_offline_dqn()
