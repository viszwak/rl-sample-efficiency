import numpy as np
import gym
import torch
import time
import pickle
import os
import argparse
from sac_torch import Agent

if __name__ == '__main__':
    # -------------------------------------------------------------
    # Parse the simulation number from CLI to separate run outputs
    # Example: python sac_online.py --sim 3
    # -------------------------------------------------------------
    parser = argparse.ArgumentParser()
    parser.add_argument('--sim', type=int, default=1, help='Simulation run number')
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

    n_games = 1500  # Number of training episodes
    best_score = env.reward_range[0]  # Initialize best score to lowest possible
    score_history = []  # List to store score per episode
    load_checkpoint = False  # Flag to skip training if loading model
    steps = 0  # Total environment steps taken

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

    # -------------------------------------------------------------
    # Warm-Up Replay Buffer (before SAC can start learning)
    # -------------------------------------------------------------
    print("Filling Replay Buffer with transitions...")
    while agent.memory.mem_cntr < agent.batch_size:
        obs = env.reset()
        done = False
        while not done:
            action = env.action_space.sample()  # Random action
            new_obs, reward, done, _, _ = env.step(action)
            agent.remember(obs, action, reward, new_obs, done)
            obs = new_obs
    print("Replay Buffer filled. Starting training...")

    # -------------------------------------------------------------
    # Training Loop
    # -------------------------------------------------------------
    for i in range(n_games):
        ep_start = time.time()
        observation = env.reset()
        done = False
        truncated = False
        score = 0

        # Run one episode
        while not (done or truncated):
            action = agent.choose_action(observation)  # Use SAC policy
            observation_, reward, done, truncated, info = env.step(action)

            steps += 1
            agent.remember(observation, action, reward, observation_, done)
            if not load_checkpoint:
                agent.learn()  # Update networks using sampled batch

            score += reward
            observation = observation_

        # Logging and tracking
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])  # Rolling average over last 100

        if avg_score > best_score:
            best_score = avg_score

        # Save model every 100 episodes
        if not load_checkpoint and i % 100 == 0:
            agent.save_models(f'{model_dir}/sac')

        print(f'Episode {i} | Score: {score:.1f} | Avg (last 100): {avg_score:.1f} | Steps: {steps}')
        print(f"Episode time: {time.time() - ep_start:.2f} seconds")

    # -------------------------------------------------------------
    # Save Replay Buffer & Scores for Offline use / Analysis
    # -------------------------------------------------------------
    total_time = time.time() - start_time
    print(f"Total training time: {total_time / 60:.2f} minutes")

    if not load_checkpoint:
        # Save per-episode scores
        np.save(f'{results_dir}/scores.npy', np.array(score_history))

        # Save replay buffer to pickle
        with open(f'{dataset_dir}/replay_buffer.pkl', 'wb') as f:
            pickle.dump(agent.memory, f)

        print(f"Scores saved to {results_dir}/scores.npy")
        print(f"Replay buffer saved to {dataset_dir}/replay_buffer.pkl")
