# cql_sac_vae.py
import argparse
import os
import gym
import torch as T
import numpy as np
import pickle

from sac_torch_cql import SAC_CQL
from model.vae_vector import VAE
from buffer import ReplayBuffer
from misc_utils import plot_learning_curve

def evaluate_policy(env, agent, episodes=5):
    scores = []
    for _ in range(episodes):
        obs, _ = env.reset()
        done = False
        score = 0
        while not done:
            vae_obs = encode_state(obs)
            action = agent.select_action(vae_obs)
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

    # ---- Load VAE-encoded buffer ----
    buffer_path = f'dataset/sim_{sim}/vae_encoded_buffer.pkl'
    with open(buffer_path, 'rb') as f:
        replay_buffer: ReplayBuffer = pickle.load(f)

    print("✅ Loaded buffer from:", buffer_path)
    print("State shape in buffer:", replay_buffer.state_memory.shape)
    assert replay_buffer.state_memory.shape[1] == 7, "Expected 7D latent states"

    env_id = 'LunarLanderContinuous-v2'
    env = gym.make(env_id)

    state_dim = 7  # Using 7D VAE-encoded states
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])
    device = T.device('cuda' if T.cuda.is_available() else 'cpu')

    # --- Load VAE model and normalization stats ---
    vae_model_path = "results/vae_lunar/vae_epoch2.pt"
    norm_stats_path = "results/vae_lunar/norm_stats.pkl"

    vae = VAE(input_dim=8, latent_dim=7).to(device)
    vae.load_state_dict(T.load(vae_model_path, map_location=device))
    vae.eval()

    with open(norm_stats_path, "rb") as f:
        stats = pickle.load(f)
    mean = T.tensor(stats["mean"], dtype=T.float32).to(device)
    std = T.tensor(stats["std"], dtype=T.float32).to(device)

    def encode_state(state):
        state = T.tensor(state, dtype=T.float32).to(device).unsqueeze(0)  # Add batch dimension [1, 8]
        norm_state = (state - mean) / std
        with T.no_grad():
            latent = vae.encode_latent(norm_state)
        return latent.squeeze(0).cpu().numpy()  # Remove batch dimension and return to CPU

    agent = SAC_CQL(
        state_dim=state_dim,
        action_dim=action_dim,
        max_action=max_action,
        device=device,
        sim=sim,
        cql_alpha=1.0
    )

    results_dir = f'results/vae_sim{sim}'
    os.makedirs(results_dir, exist_ok=True)

    scores, steps = [], []
    eval_interval = 10_000
    max_steps = 350_000
    batch_size = 256

    for step in range(1, max_steps + 1):
        agent.train(replay_buffer, batch_size=batch_size)

        if step % eval_interval == 0:
            avg_score = evaluate_policy(env, agent, episodes=5)
            print(f"[Sim {sim} | Step {step}] Avg Eval Score: {avg_score:.2f}")
            scores.append(avg_score)
            steps.append(step)

    agent.save_models()

    np.save(os.path.join(results_dir, 'scores.npy'), np.array(scores))
    np.save(os.path.join(results_dir, 'steps.npy'), np.array(steps))

def evaluate_policy(env, agent, episodes=5):
    scores = []
    for _ in range(episodes):
        obs, _ = env.reset()
        done = False
        score = 0
        while not done:
            vae_obs = encode_state(obs)
            vae_obs = torch.tensor(vae_obs, dtype=torch.float32).to(agent.device)
            action = agent.select_action(vae_obs)
            action = action * agent.max_action  # Scale action if needed
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            score += reward
        scores.append(score)
    return np.mean(scores)
