import gym
import numpy as np
import pickle
import os
import argparse
from buffer import ReplayBuffer
import matplotlib.pyplot as plt
import random
from sklearn.cluster import KMeans
from collections import defaultdict

class UnbiasedDatasetGenerator:
     
    def __init__(self, env_name='LunarLanderContinuous-v2', buffer_size=1000000):
        self.env = gym.make(env_name)
        self.buffer = ReplayBuffer(buffer_size,
                                  self.env.observation_space.shape,
                                  self.env.action_space.shape[0])

        self.policy_memory = [] 
        
        self.policy_name_to_id = {}
        self.policy_id_to_name = {}
        self.next_policy_id = 0
        
        self.policy_transitions = {}
        self.episode_returns = []
        self.success_count = 0
        self.total_episodes = 0

        self.state_visitation_counts = {}
        self.state_discretization_level = 1  
        self.trajectory_lengths = []

        self.state_clusters = None
        self.n_clusters = 100
        
    def register_policy(self, policy_name):

        if policy_name not in self.policy_name_to_id:
            policy_id = self.next_policy_id
            self.policy_name_to_id[policy_name] = policy_id
            self.policy_id_to_name[policy_id] = policy_name
            self.next_policy_id += 1
        return self.policy_name_to_id[policy_name]
        
    def random_policy(self, state):
        return np.array(self.env.action_space.sample())

    def noisy_expert_policy(self, state, noise_level=0.1):
        s = state
        angle_targ = s[0] * 0.5 + s[2] * 1.0  
        angle_targ = np.clip(angle_targ, -0.4, 0.4)
        hover_targ = 0.55 * np.abs(s[0])  

        angle_todo = (angle_targ - s[4]) * 0.5 - (s[5]) * 1.0
        hover_todo = (hover_targ - s[1]) * 0.5 - (s[3]) * 0.5

        if s[6] or s[7]:  
            angle_todo = 0
            hover_todo = -(s[3]) * 0.5  

        a = np.array([hover_todo * 20 - 1, -angle_todo * 20])
        a = np.clip(a, -1, +1)
        
        # Add noise
        if noise_level > 0:
            a = a + np.random.normal(0, noise_level, size=2)
            a = np.clip(a, -1, +1)
        
        return a

    def failing_policy(self, state):
        if np.random.random() < 0.7:

            action = self.noisy_expert_policy(state, noise_level=0.15)

            if np.random.random() < 0.2:
                action[0] *= 0.5  
            if np.random.random() < 0.2:
                action[1] *= -1  
            return action
        else:

            return self.random_policy(state)

    
    def conservative_policy(self, state):
        x_pos, y_pos, x_vel, y_vel, angle, ang_vel, left_leg, right_leg = state
        
        thrust = 0.0
        rotation = 0.0
        
        if y_vel < -0.5:  
            thrust = 0.8  
        elif y_vel < -0.3:
            thrust = 0.5
        elif y_pos < 0.5 and y_vel < -0.2:  
            thrust = 0.6  
            
        # Better angle control
        if abs(angle) > 0.2:  
            rotation = -angle * 1.5  
        elif abs(x_pos) > 0.5:
            rotation = -x_pos * 0.3 
            
        return np.array([thrust, rotation])
    
    def exploratory_policy(self, state, prev_action=None, momentum=0.7):
        if prev_action is None:
            return self.random_policy(state)
        
        delta = np.random.normal(0, 0.2, size=2)
        new_action = momentum * np.array(prev_action) + (1 - momentum) * delta
        
        if np.random.random() < 0.1:
            new_action = self.random_policy(state)
            
        return np.clip(new_action, -1, 1)
    
    def collect_episode(self, policy_fn, policy_name, max_steps=1000):
        if self.buffer.mem_cntr >= self.buffer.mem_size:
            print(f"Buffer is full at {self.buffer.mem_cntr} transitions")
            return 0, 0 
            
        state = self.env.reset()
        if isinstance(state, tuple):
            state = state[0]
            
        episode_return = 0
        episode_length = 0
        prev_action = None
        
        policy_id = self.register_policy(policy_name)
        
        for step in range(max_steps):

            if self.buffer.mem_cntr >= self.buffer.mem_size:
                break
                
            if policy_name == "exploratory":
                action = policy_fn(state, prev_action)
                prev_action = action
            else:
                action = policy_fn(state)

            action = np.clip(action, -1, 1)
            
            step_out = self.env.step(action)
            if len(step_out) == 5:  
                next_state, reward, terminated, truncated, info = step_out
                done = terminated or truncated
            else:  
                next_state, reward, done, info = step_out

            if isinstance(next_state, tuple):
                next_state = next_state[0]

            state_key = tuple(np.round(state, self.state_discretization_level))
            if state_key not in self.state_visitation_counts:
                self.state_visitation_counts[state_key] = 0
            self.state_visitation_counts[state_key] += 1
            
            self.buffer.store_transition(state, action, reward, next_state, done)
            self.policy_memory.append(policy_id)
            
            if policy_name not in self.policy_transitions:
                self.policy_transitions[policy_name] = 0
            self.policy_transitions[policy_name] += 1
            
            episode_return += reward
            episode_length += 1
            state = next_state
            
            if done:
                break
        
        self.episode_returns.append(episode_return)
        self.trajectory_lengths.append(episode_length)
        self.total_episodes += 1
        if episode_return > 200:
            self.success_count += 1
            
        return episode_return, episode_length

    def mixed_policy(self, state):
        choice = np.random.random()
        if choice < 0.33:
            return self.noisy_expert_policy(state, 0.05)
        elif choice < 0.66:
            return self.conservative_policy(state)
        else:
            return self.random_policy(state)



    def generate_dataset(self, target_transitions=1000000):
        
        policies = [
            ("expert_perfect", lambda s: self.noisy_expert_policy(s, 0.0), 0.3),      # 10%
            ("expert_noisy", lambda s: self.noisy_expert_policy(s, 0.1), 0.2),        # 5%
            ("conservative", self.conservative_policy, 0.2),                          # 20%
            ("failing", self.failing_policy, 0.05),                                    # 20%
            ("exploratory", self.exploratory_policy, 0.2),                           # 20%
            ("random", self.random_policy, 0.05),                                     # 20%
            ("expert_high_noise", lambda s: self.noisy_expert_policy(s, 0.2), 0.025),   # 2.5%
            ("mixed", self.mixed_policy, 0.025),                                       # 2.5%
        ]
        
        policy_weights = [p[2] for p in policies]
        cumulative_weights = np.cumsum(policy_weights)
        
        target_transitions = min(target_transitions, self.buffer.mem_size)
        self.buffer = ReplayBuffer(
            self.buffer.mem_size,
            self.env.observation_space.shape,
            self.env.action_space.shape[0]
        )
        self.policy_transitions = {}
        self.episode_returns = []
        self.policy_memory = []
        self.total_episodes = 0
        self.state_visitation_counts = {}  
        self.trajectory_lengths = []  
        
        while self.buffer.mem_cntr < target_transitions:

            rand_val = np.random.random()
            policy_idx = np.searchsorted(cumulative_weights, rand_val)
            policy_name, policy_fn, _ = policies[policy_idx]
            
            state = self.env.reset()
            if isinstance(state, tuple):
                state = state[0]
            
            episode_return = 0
            episode_length = 0
            prev_action = None
            policy_id = self.register_policy(policy_name)
            
            for step in range(1000):
                if self.buffer.mem_cntr >= target_transitions:
                    break
                
                state_key = tuple(np.round(state, self.state_discretization_level))
                if state_key not in self.state_visitation_counts:
                    self.state_visitation_counts[state_key] = 0
                self.state_visitation_counts[state_key] += 1
                
                if policy_name == "exploratory":
                    action = policy_fn(state, prev_action)
                    prev_action = action
                else:
                    action = policy_fn(state)
                
                action = np.clip(action, -1, 1)
                
                step_out = self.env.step(action)
                if len(step_out) == 5:
                    next_state, reward, terminated, truncated, info = step_out
                    done = terminated or truncated
                else:
                    next_state, reward, done, info = step_out
                
                if isinstance(next_state, tuple):
                    next_state = next_state[0]
                

                self.buffer.store_transition(state, action, reward, next_state, done)
                self.policy_memory.append(policy_id)
                
                if policy_name not in self.policy_transitions:
                    self.policy_transitions[policy_name] = 0
                self.policy_transitions[policy_name] += 1
                
                episode_return += reward
                episode_length += 1
                state = next_state
                
                if done:
                    break
            
            self.episode_returns.append(episode_return)
            self.trajectory_lengths.append(episode_length)
            self.total_episodes += 1
            if episode_return > 200:
                self.success_count += 1
            
            if self.total_episodes % 50 == 0:
                success_rate = (self.success_count / self.total_episodes) * 100
                print(f"Episodes: {self.total_episodes}, Transitions: {self.buffer.mem_cntr}/{target_transitions}, Success rate: {success_rate:.1f}%")
                for policy, count in self.policy_transitions.items():
                    print(f"  {policy}: {count} ({count/self.buffer.mem_cntr*100:.1f}%)")
        
        print(f"\nDataset complete: {self.buffer.mem_cntr} transitions")
        print(f"Final success rate: {(self.success_count/self.total_episodes)*100:.1f}%")


    def perform_state_clustering(self, n_samples=10000):
        data_size = min(self.buffer.mem_cntr, self.buffer.mem_size)
        sample_size = min(n_samples, data_size)
        
        if sample_size < self.n_clusters:
            print(f"Warning: Not enough samples for {self.n_clusters} clusters. Using {sample_size//10} clusters.")
            self.n_clusters = max(10, sample_size // 10)
        
        sample_indices = np.random.choice(data_size, sample_size, replace=False)
        sampled_states = self.buffer.state_memory[sample_indices]
        
        self.state_clusters = KMeans(n_clusters=self.n_clusters, random_state=42)
        cluster_labels = self.state_clusters.fit_predict(sampled_states)
        
        cluster_counts = np.bincount(cluster_labels, minlength=self.n_clusters)
        
        return cluster_counts
    
    def analyze_dataset(self):
        data_size = min(self.buffer.mem_cntr, self.buffer.mem_size)
        
        if data_size == 0:
            print("No data to analyze!")
            return
        
        states = self.buffer.state_memory[:data_size]
        actions = self.buffer.action_memory[:data_size]
        rewards = self.buffer.reward_memory[:data_size]
        
        policy_ids = self.policy_memory[:data_size] if self.policy_memory else []

        actions = np.clip(actions, -1, 1)

        # Analyze policy contribution
        if policy_ids:
            print("\n=== Policy Contribution Analysis ===")
            policy_counts = defaultdict(int)
            policy_rewards = defaultdict(list)
            
            for pid, reward in zip(policy_ids, rewards):
                if pid in self.policy_id_to_name:
                    policy_name = self.policy_id_to_name[pid]
                    policy_counts[policy_name] += 1
                    policy_rewards[policy_name].append(reward)
            
            for policy_name in sorted(policy_counts.keys()):
                count = policy_counts[policy_name]
                avg_reward = np.mean(policy_rewards[policy_name])
                print(f"  {policy_name}: {count} transitions ({count/data_size*100:.1f}%), "
                      f"avg reward: {avg_reward:.3f}")

        print("\nState Coverage Statistics:")
        print(f"Unique discretized states visited: {len(self.state_visitation_counts)}")
        
        if self.state_visitation_counts:
            visit_values = list(self.state_visitation_counts.values())
            print(f"Max state visits: {max(visit_values)}")
            print(f"Mean state visits: {np.mean(visit_values):.2f}")
            print(f"Median state visits: {np.median(visit_values):.2f}")
        

        if self.trajectory_lengths and len(self.trajectory_lengths) > 1:
            print(f"Trajectory length mean: {np.mean(self.trajectory_lengths):.2f}")
            print(f"Trajectory length std: {np.std(self.trajectory_lengths):.2f}")
            print(f"Trajectory length min: {min(self.trajectory_lengths)}")
            print(f"Trajectory length max: {max(self.trajectory_lengths)}")
        else:
            print("Not enough trajectory data for statistics")

        if data_size >= 100:  
            cluster_counts = self.perform_state_clustering()
            coverage_ratio = np.sum(cluster_counts > 0) / self.n_clusters
            print(f"State cluster coverage: {coverage_ratio:.2%} of {self.n_clusters} clusters")
            print(f"Cluster visitation variance: {np.std(cluster_counts):.2f}")

        print("\nState Statistics:")
        state_names = ['x_pos', 'y_pos', 'x_vel', 'y_vel', 'angle', 'ang_vel', 'left_leg', 'right_leg']
        for i, name in enumerate(state_names):
            print(f"  {name}: mean={np.mean(states[:, i]):.3f}, "
                  f"std={np.std(states[:, i]):.3f}, "
                  f"min={np.min(states[:, i]):.3f}, "
                  f"max={np.max(states[:, i]):.3f}")
        
        print("\nAction Statistics:")
        action_names = ['thrust', 'rotation']
        for i, name in enumerate(action_names):
            print(f"  {name}: mean={np.mean(actions[:, i]):.3f}, "
                  f"std={np.std(actions[:, i]):.3f}, "
                  f"min={np.min(actions[:, i]):.3f}, "
                  f"max={np.max(actions[:, i]):.3f}")

        print("\nAction Space Coverage:")
        for i, name in enumerate(action_names):
            hist, _ = np.histogram(actions[:, i], bins=50)
            hist = hist / hist.sum()
            hist = hist[hist > 0]  # Remove zero bins to avoid log(0)
            action_entropy = -np.sum(hist * np.log(hist))
            print(f"  {name} entropy: {action_entropy:.3f}")
        
        hist_2d, _, _ = np.histogram2d(actions[:, 0], actions[:, 1], bins=20)
        coverage_ratio = np.sum(hist_2d > 0) / hist_2d.size
        print(f"  2D action space coverage: {coverage_ratio:.2%}")

        print("\nReward Statistics:")
        print(f"  Mean: {np.mean(rewards):.3f}")
        print(f"  Std: {np.std(rewards):.3f}")
        print(f"  Min: {np.min(rewards):.3f}")
        print(f"  Max: {np.max(rewards):.3f}")
        print(f"  Positive rewards: {np.sum(rewards > 0)} ({np.sum(rewards > 0)/len(rewards)*100:.1f}%)")
        
        if self.episode_returns:
            print("\nEpisode Return Statistics:")
            print(f"  Mean: {np.mean(self.episode_returns):.1f}")
            print(f"  Std: {np.std(self.episode_returns):.1f}")
            print(f"  Min: {np.min(self.episode_returns):.1f}")
            print(f"  Max: {np.max(self.episode_returns):.1f}")
            print(f"  Median: {np.median(self.episode_returns):.1f}")

        successful_landings = 0
        crash_landings = 0
        perfect_landings = 0
        
        for i in range(min(data_size, len(rewards))):
            if rewards[i] >= 100:  
                successful_landings += 1
                if rewards[i] == 100:  
                    perfect_landings += 1
            elif rewards[i] <= -100:  
                crash_landings += 1

        print(f"\nLanding Analysis:")
        print(f"  Successful landings (reward >= 100): {successful_landings}")
        print(f"  Perfect landings (reward = 100): {perfect_landings}")
        print(f"  Crashes (reward <= -100): {crash_landings}")
        print(f"  Success rate: {successful_landings/data_size*100:.3f}%")

        # Check diversity score
        state_diversity = np.mean(np.std(states, axis=0))
        action_diversity = np.mean(np.std(actions, axis=0))
        print(f"\nDiversity Metrics:")
        print(f"  State diversity: {state_diversity:.3f}")
        print(f"  Action diversity: {action_diversity:.3f}")

        episode_success_rate = (self.success_count / self.total_episodes) * 100
        print(f"  Transition-level success rate: {successful_landings/data_size*100:.3f}%")
        print(f"  Episode-level success rate: {episode_success_rate:.1f}%")

        
        self.visualize_dataset()
    
    def visualize_dataset(self):
        data_size = min(self.buffer.mem_cntr, self.buffer.mem_size)
        
        if data_size == 0:
            print("No data to visualize!")
            return
        
        fig, axes = plt.subplots(3, 3, figsize=(15, 15))
        
        if self.episode_returns:
            axes[0, 0].hist(self.episode_returns, bins=50, edgecolor='black', alpha=0.7)
            axes[0, 0].axvline(x=200, color='r', linestyle='--', label='Success threshold')
            axes[0, 0].set_title('Episode Return Distribution')
            axes[0, 0].set_xlabel('Episode Return')
            axes[0, 0].set_ylabel('Count')
            axes[0, 0].legend()
        else:
            axes[0, 0].text(0.5, 0.5, 'No episode data', ha='center', va='center')
        
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
        
        if len(self.episode_returns) > 50:
            window = 50
            success_rate = []
            for i in range(window, len(self.episode_returns)):
                window_returns = self.episode_returns[i-window:i]
                success_rate.append(sum(r > 200 for r in window_returns) / window * 100)
            
            axes[1, 2].plot(range(window, len(self.episode_returns)), success_rate)
            axes[1, 2].set_title(f'Success Rate (rolling {window} episodes)')
            axes[1, 2].set_xlabel('Episode')
            axes[1, 2].set_ylabel('Success Rate (%)')
            axes[1, 2].set_ylim(0, 100)
        else:
            axes[1, 2].text(0.5, 0.5, 'Not enough episodes', ha='center', va='center')
        
        policy_ids = self.policy_memory[:data_size] if self.policy_memory else []
        policy_counts = defaultdict(int)
        for pid in policy_ids:
            if pid in self.policy_id_to_name:
                policy_counts[self.policy_id_to_name[pid]] += 1
        
        if policy_counts:
            axes[2, 0].pie(policy_counts.values(), labels=policy_counts.keys(), autopct='%1.1f%%')
            axes[2, 0].set_title('Policy Contribution to Dataset')
        else:
            axes[2, 0].text(0.5, 0.5, 'No policy data', ha='center', va='center')
        
        if self.state_visitation_counts:
            visit_counts = list(self.state_visitation_counts.values())
            axes[2, 1].hist(visit_counts, bins=30, edgecolor='black', alpha=0.7)
            axes[2, 1].set_title('State Visitation Frequency Distribution')
            axes[2, 1].set_xlabel('Visit Count')
            axes[2, 1].set_ylabel('Number of States')
            axes[2, 1].set_yscale('log')
        else:
            axes[2, 1].text(0.5, 0.5, 'No visitation data', ha='center', va='center')
        
        if self.trajectory_lengths:
            axes[2, 2].hist(self.trajectory_lengths, bins=30, edgecolor='black', alpha=0.7)
            axes[2, 2].set_title('Trajectory Length Distribution')
            axes[2, 2].set_xlabel('Episode Length')
            axes[2, 2].set_ylabel('Count')
        else:
            axes[2, 2].text(0.5, 0.5, 'No trajectory data', ha='center', va='center')
        
        plt.tight_layout()
        plt.savefig('701_dataset_analysis.png', dpi=150)
        plt.close(fig)
        print("Visualization saved to '701_dataset_analysis.png'")

    def save_dataset(self, filepath):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        with open(filepath, 'wb') as f:
            pickle.dump(self.buffer, f)
        print(f"Dataset saved to {filepath}")
        
        metadata_filepath = filepath.replace('.pkl', '_metadata.pkl')
        metadata = {
            'policy_memory': self.policy_memory,
            'policy_mapping': self.policy_id_to_name,
            'episode_returns': self.episode_returns,
            'state_clusters': self.state_clusters,
            'state_visitation_counts': self.state_visitation_counts
        }
        with open(metadata_filepath, 'wb') as f:
            pickle.dump(metadata, f)
        print(f"Metadata saved to {metadata_filepath}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sim', type=int, default=1, help='Simulation ID')
    parser.add_argument('--transitions', type=int, default=1000000, help='Number of transitions to collect')
    args = parser.parse_args()
    
    generator = UnbiasedDatasetGenerator()

    generator.generate_dataset(target_transitions=args.transitions)
    
    generator.analyze_dataset()
    
    dataset_path = f'dataset/unbiased_sim_{args.sim}/replay_buffer.pkl'
    generator.save_dataset(dataset_path)
    
    print(f"\nDataset ready for CQL training at: {dataset_path}")