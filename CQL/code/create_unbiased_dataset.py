import gym
import numpy as np
import pickle
import os
import argparse
from buffer import ReplayBuffer
import matplotlib.pyplot as plt

class UnbiasedDatasetGenerator:

    def __init__(self, env_name='LunarLanderContinuous-v2', buffer_size=1000000):
        self.env = gym.make(env_name)
        self.buffer = ReplayBuffer(buffer_size,
                                   self.env.observation_space.shape,
                                   self.env.action_space.shape[0])

        self.policy_transitions = {}
        self.episode_returns = []
        self.success_count = 0
        self.total_episodes = 0

    def random_policy(self, state):
        return self.env.action_space.sample()

    def noisy_expert_policy(self, state, noise_level=0.3):
        x_pos, y_pos, x_vel, y_vel, angle, ang_vel, left_leg, right_leg = state

        thrust = 0.0
        rotation = 0.0

        target_y_vel = -0.2
        if y_vel < target_y_vel - 0.1:
            thrust = min(1.0, (target_y_vel - y_vel) * 2)

        if abs(x_pos) > 0.1:
            rotation = -x_pos * 2.0 - x_vel * 0.5

        if abs(angle) > 0.1:
            rotation = -angle * 3.0 - ang_vel * 0.5

        thrust += np.random.normal(0, noise_level)
        rotation += np.random.normal(0, noise_level)

        return np.clip([thrust, rotation], -1, 1)

    def failing_policy(self, state):
        if np.random.random() < 0.3:
            return [np.random.choice([-0.8, 0.8]), np.random.uniform(-0.3, 0.3)]
        else:
            return [np.random.uniform(-0.2, 0.2), np.random.choice([-0.9, 0.9])]

    def conservative_policy(self, state):
        x_pos, y_pos, x_vel, y_vel, angle, ang_vel, left_leg, right_leg = state

        thrust = 0.0
        rotation = 0.0

        if y_vel < -0.8:
            thrust = 0.5
        elif y_pos < 0.2 and y_vel < -0.3:
            thrust = 0.3

        if abs(angle) > 0.3:
            rotation = -angle * 0.5

        return [thrust, rotation]

    def exploratory_policy(self, state, prev_action=None, momentum=0.7):
        if prev_action is None:
            return self.random_policy(state)

        delta = np.random.normal(0, 0.2, size=2)
        new_action = momentum * np.array(prev_action) + (1 - momentum) * delta

        if np.random.random() < 0.1:
            new_action = self.random_policy(state)

        return np.clip(new_action, -1, 1)

    def collect_episode(self, policy_fn, policy_name, max_steps=1000):
        state = self.env.reset()
        if isinstance(state, tuple):
            state = state[0]

        episode_return = 0
        episode_length = 0
        prev_action = None

        for step in range(max_steps):
            if policy_name == "exploratory":
                action = policy_fn(state, prev_action)
                prev_action = action
            else:
                action = policy_fn(state)

            next_state, reward, done, info = self.env.step(action)[:4]
            if isinstance(next_state, tuple):
                next_state = next_state[0]

            self.buffer.store_transition(state, action, reward, next_state, done)

            if policy_name not in self.policy_transitions:
                self.policy_transitions[policy_name] = 0
            self.policy_transitions[policy_name] += 1

            episode_return += reward
            episode_length += 1
            state = next_state

            if done:
                break

        self.episode_returns.append(episode_return)
        self.total_episodes += 1
        if episode_return > 200:
            self.success_count += 1

        return episode_return, episode_length

    def generate_dataset(self, target_transitions=600000):

        print(f"Generating unbiased dataset with {target_transitions} transitions...")

        policies = [
            ("random", self.random_policy, 0.20),
            ("noisy_expert_low", lambda s: self.noisy_expert_policy(s, 0.2), 0.20),
            ("noisy_expert_high", lambda s: self.noisy_expert_policy(s, 0.5), 0.15),
            ("failing", self.failing_policy, 0.15),
            ("conservative", self.conservative_policy, 0.15),
            ("exploratory", self.exploratory_policy, 0.15),
        ]

        round_num = 0
        while self.buffer.mem_cntr < target_transitions:
            round_num += 1
            print(f"\nRound {round_num} - Buffer: {self.buffer.mem_cntr}/{target_transitions}")

            for policy_name, policy_fn, target_prop in policies:
                if self.buffer.mem_cntr >= target_transitions:
                    break

                episode_return, episode_length = self.collect_episode(policy_fn, policy_name)

                if self.total_episodes % 50 == 0:
                    success_rate = (self.success_count / self.total_episodes) * 100
                    print(f"  Episode {self.total_episodes}: {policy_name} policy, "
                          f"Return: {episode_return:.1f}, Success rate: {success_rate:.1f}%")

        total_trans = sum(self.policy_transitions.values())
        for policy, count in self.policy_transitions.items():
            print(f"  {policy}: {count} transitions ({count/total_trans*100:.1f}%)")

    def analyze_dataset(self):

        data_size = min(self.buffer.mem_cntr, self.buffer.mem_size)

        states = self.buffer.state_memory[:data_size]
        actions = self.buffer.action_memory[:data_size]
        rewards = self.buffer.reward_memory[:data_size]

        state_names = ['x_pos', 'y_pos', 'x_vel', 'y_vel', 'angle', 'ang_vel', 'left_leg', 'right_leg']
        for i, name in enumerate(state_names):
            print(f"  {name}: mean={np.mean(states[:, i]):.3f}, "
                  f"std={np.std(states[:, i]):.3f}, "
                  f"min={np.min(states[:, i]):.3f}, "
                  f"max={np.max(states[:, i]):.3f}")

        action_names = ['thrust', 'rotation']
        for i, name in enumerate(action_names):
            print(f"  {name}: mean={np.mean(actions[:, i]):.3f}, "
                  f"std={np.std(actions[:, i]):.3f}, "
                  f"min={np.min(actions[:, i]):.3f}, "
                  f"max={np.max(actions[:, i]):.3f}")

        print(f"  Mean: {np.mean(rewards):.3f}")
        print(f"  Std: {np.std(rewards):.3f}")
        print(f"  Min: {np.min(rewards):.3f}")
        print(f"  Max: {np.max(rewards):.3f}")
        print(f"  Positive rewards: {np.sum(rewards > 0)} ({np.sum(rewards > 0)/len(rewards)*100:.1f}%)")

        print(f"  Mean: {np.mean(self.episode_returns):.1f}")
        print(f"  Std: {np.std(self.episode_returns):.1f}")
        print(f"  Min: {np.min(self.episode_returns):.1f}")
        print(f"  Max: {np.max(self.episode_returns):.1f}")
        print(f"  Median: {np.median(self.episode_returns):.1f}")

        self.visualize_dataset()

    def visualize_dataset(self):
        data_size = min(self.buffer.mem_cntr, self.buffer.mem_size)

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))

        axes[0, 0].hist(self.episode_returns, bins=50, edgecolor='black', alpha=0.7)
        axes[0, 0].axvline(x=200, color='r', linestyle='--', label='Success threshold')
        axes[0, 0].set_title('Episode Return Distribution')
        axes[0, 0].set_xlabel('Episode Return')
        axes[0, 0].set_ylabel('Count')
        axes[0, 0].legend()

        rewards = self.buffer.reward_memory[:data_size]
        axes[0, 1].hist(rewards, bins=50, edgecolor='black', alpha=0.7)
        axes[0, 1].set_title('Step Reward Distribution')
        axes[0, 1].set_xlabel('Reward')
        axes[0, 1].set_ylabel('Count')

        actions = self.buffer.action_memory[:min(10000, data_size)]
        axes[0, 2].scatter(actions[:, 0], actions[:, 1], alpha=0.5, s=1)
        axes[0, 2].set_title('Action Space Coverage (first 10k)')
        axes[0, 2].set_xlabel('Thrust')
        axes[0, 2].set_ylabel('Rotation')
        axes[0, 2].set_xlim(-1.1, 1.1)
        axes[0, 2].set_ylim(-1.1, 1.1)

        states = self.buffer.state_memory[:min(10000, data_size)]
        axes[1, 0].scatter(states[:, 0], states[:, 1], alpha=0.5, s=1)
        axes[1, 0].set_title('Position Distribution (first 10k)')
        axes[1, 0].set_xlabel('X Position')
        axes[1, 0].set_ylabel('Y Position')

        axes[1, 1].scatter(states[:, 2], states[:, 3], alpha=0.5, s=1)
        axes[1, 1].set_title('Velocity Distribution (first 10k)')
        axes[1, 1].set_xlabel('X Velocity')
        axes[1, 1].set_ylabel('Y Velocity')

        window = 50
        success_rate = []
        for i in range(window, len(self.episode_returns)):
            window_returns = self.episode_returns[i - window:i]
            success_rate.append(sum(r > 200 for r in window_returns) / window * 100)

        axes[1, 2].plot(range(window, len(self.episode_returns)), success_rate)
        axes[1, 2].set_title(f'Success Rate (rolling {window} episodes)')
        axes[1, 2].set_xlabel('Episode')
        axes[1, 2].set_ylabel('Success Rate (%)')
        axes[1, 2].set_ylim(0, 100)

        plt.tight_layout()
        plt.savefig('unbiased_dataset_analysis.png', dpi=150)
        plt.show()

    def save_dataset(self, filepath):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(self.buffer, f)
        print(f"Dataset saved to {filepath}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sim', type=int, default=1, help='Simulation ID')
    parser.add_argument('--transitions', type=int, default=600000, help='Number of transitions to collect')
    args = parser.parse_args()

    generator = UnbiasedDatasetGenerator()

    generator.generate_dataset(target_transitions=args.transitions)

    generator.analyze_dataset()

    dataset_path = f'dataset/unbiased_sim_{args.sim}/replay_buffer.pkl'
    generator.save_dataset(dataset_path)

    print(f"\nDataset ready for CQL training at: {dataset_path}")
