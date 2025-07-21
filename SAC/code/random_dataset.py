import numpy as np
import gym
import torch
import time
import pickle
import os
import argparse
from sac_torch import Agent  # Your existing SAC Agent class

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sim', type=int, default=999, help='Simulation run number')
    args = parser.parse_args()
    sim = args.sim

    # -------------------------------------------------------------
    # Environment and Agent Setup
    # -------------------------------------------------------------
    env_id = 'LunarLanderContinuous-v2'
    env = gym.make(env_id)

    agent = Agent(
        alpha=0.0003,
        beta=0.0003,
        reward_scale=2,
        env_id=env_id,
        input_dims=env.observation_space.shape,
        tau=0.005,
        env=env,
        batch_size=256,
        layer1_size=256,
        layer2_size=256,
        n_actions=env.action_space.shape[0]
    )

    total_episodes = 1500
    score_history = []
    steps = 0

    # -------------------------------------------------------------
    # Create folders for saving models, buffer, and scores
    # -------------------------------------------------------------
    model_dir = f'models/sim_{sim}'
    dataset_dir = f'dataset/sim_{sim}'
    results_dir = f'results/sim_{sim}'

    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(dataset_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    # -------------------------------------------------------------
    # GPU Check
    # -------------------------------------------------------------
    print("Using GPU:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU name:", torch.cuda.get_device_name(0))
    start_time = time.time()

    print("Collecting structured random dataset (untrained actor)...")

    # -------------------------------------------------------------
    # Data Collection Loop (No training)
    # -------------------------------------------------------------
    for i in range(total_episodes):
        ep_start = time.time()
        observation = env.reset()
        done = False
        truncated = False
        score = 0

        while not (done or truncated):
            action = agent.choose_action(observation)  # uses untrained actor
            observation_, reward, done, truncated, info = env.step(action)

            steps += 1
            agent.remember(observation, action, reward, observation_, done)  # collect into buffer

            # 🚫 No learning
            score += reward
            observation = observation_

        score_history.append(score)
        avg_score = np.mean(score_history[-100:])

        print(f"Episode {i+1} | Score: {score:.1f} | Avg (last 100): {avg_score:.1f} | Steps: {steps}")
        print(f"Episode time: {time.time() - ep_start:.2f} seconds")

    # -------------------------------------------------------------
    # Save Replay Buffer & Scores
    # -------------------------------------------------------------
    total_time = time.time() - start_time
    print(f"✅ Dataset collection complete in {total_time / 60:.2f} minutes.")

    np.save(f'{results_dir}/scores.npy', np.array(score_history))
    with open(f'{dataset_dir}/replay_buffer.pkl', 'wb') as f:
        pickle.dump(agent.memory, f)

    print(f"Scores saved to: {results_dir}/scores.npy")
    print(f"Replay buffer saved to: {dataset_dir}/replay_buffer.pkl")
