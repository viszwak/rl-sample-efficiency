import argparse
from vae_torch import VAE
import os
import gym
import torch as T
import numpy as np
import pickle

from sac_torch_cql import SAC_CQL
from buffer import ReplayBuffer
from utils import plot_learning_curve


def evaluate_policy(env, agent, mean, std, episodes=5):
    scores = []
    for _ in range(episodes):
        obs, _ = env.reset()
        done = False
        score = 0
        while not done:
            norm_obs = (obs - mean) / std
            action = agent.select_action(norm_obs)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            score += reward
        scores.append(score)
    return np.mean(scores)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sim', type=int, default=1, help='Simulation ID')
    args = parser.parse_args()
    sim = args.sim

    # ---- Load buffer from correct path ----
    buffer_path = f'dataset/sim_{sim}/replay_buffer.pkl'
    with open(buffer_path, 'rb') as f:
        replay_buffer: ReplayBuffer = pickle.load(f)

    # ---- Load normalization stats ----
    with open("models/vae_norm_stats_4st.pkl", "rb") as f:
        norm_stats = pickle.load(f)
    mean = T.tensor(norm_stats['mean'], dtype=T.float32)
    std = T.tensor(norm_stats['std'], dtype=T.float32)

    env_id = 'LunarLanderContinuous-v2'
    env = gym.make(env_id)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])
    device = T.device('cuda' if T.cuda.is_available() else 'cpu')

    vae = VAE(input_dim=8, latent_dim=4)
    vae.load_state_dict(T.load("models/vae_state_encoder_4st.pt"))
    vae.to(device)
    vae.eval()

    # Update SAC_CQL to take mean/std tensors for normalization
    agent = SAC_CQL(
        state_dim=state_dim,
        action_dim=action_dim,
        max_action=max_action,
        device=device,
        sim=sim,
        cql_alpha=0.1,
        vae=vae,
        norm_mean=mean.to(device),
        norm_std=std.to(device),
    )

    results_dir = f'results/sim{sim}'
    os.makedirs(results_dir, exist_ok=True)

    scores, steps = [], []
    eval_interval = 10_000
    max_steps = 350_000
    batch_size = 256

    for step in range(1, max_steps + 1):
        agent.train(replay_buffer, batch_size=batch_size)

        if step % eval_interval == 0:
            avg_score = evaluate_policy(env, agent, mean.numpy(), std.numpy(), episodes=5)
            print(f"[Sim {sim} | Step {step}] Avg Eval Score: {avg_score:.2f}")
            scores.append(avg_score)
            steps.append(step)

    agent.save_models()

    np.save(os.path.join(results_dir, 'scores_4st.npy'), np.array(scores))
    np.save(os.path.join(results_dir, 'steps_4st.npy'),  np.array(steps))
