import numpy as np
import gym
import torch
import time
import pickle
import os
import argparse
from buffer import ReplayBuffer

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sim', type=int, default=999, help='Simulation run number')
    args = parser.parse_args()
    sim = args.sim

    env_id = 'LunarLanderContinuous-v2'
    env = gym.make(env_id)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    total_episodes = 1500
    score_history = []
    steps = 0

    # Save paths
    model_dir = f'models/sim_{sim}'
    dataset_dir = f'dataset/sim_{sim}'
    results_dir = f'results/sim_{sim}'
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(dataset_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    print("Using GPU:", torch.cuda.is_available())
    print(f"Collecting random transitions for {total_episodes} episodes...")

    buffer = ReplayBuffer(max_size=2_000_000, input_shape=(obs_dim,), n_actions=act_dim)
    start_time = time.time()

    for episode in range(total_episodes):
        obs = env.reset()
        done = False
        truncated = False
        score = 0

        while not (done or truncated):
            action = env.action_space.sample()
            new_obs, reward, done, truncated, _ = env.step(action)
            buffer.store_transition(obs, action, reward, new_obs, int(done))

            obs = new_obs
            score += reward
            steps += 1

        score_history.append(score)
        print(f"Episode {episode+1}/{total_episodes} | Score: {score:.1f} | Total Steps: {steps}")

    total_time = time.time() - start_time
    print(f"✅ Collected {steps} steps from {total_episodes} episodes in {total_time / 60:.2f} minutes.")

    # Save buffer and scores
    with open(f'{dataset_dir}/replay_buffer.pkl', 'wb') as f:
        pickle.dump(buffer, f)
    np.save(f'{results_dir}/scores.npy', np.array(score_history))

    print(f"Replay buffer saved to: {dataset_dir}/replay_buffer.pkl")
    print(f"Scores saved to: {results_dir}/scores.npy")
