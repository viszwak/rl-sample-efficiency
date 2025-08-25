import argparse
import os
import gym
import torch as T
import numpy as np
import pickle

from sac_torch_cql import SAC_CQL
from buffer import ReplayBuffer
#from utils import plot_learning_curve

# --- Only needed for eval encoding ---
from model.vae_vector import VAE

def make_encoder(device, state_dim_from_buffer):
    """
    Load the trained VAE + scaler and return a function that maps
    an 8-D Lunar obs -> 6-D latent (z + 2 flags).
    """
    # paths where you saved the VAE (adjust if you used different dirs)
    vae_ckpt_path = "results/vae_lunar/vae_best.pt"
    stats_path    = "results/vae_lunar/norm_stats.pkl"

    ckpt = T.load(vae_ckpt_path, map_location=device)
    # infer latent dim from checkpoint (e.g., 4). Fallback: state_dim-2
    if isinstance(ckpt, dict) and "fc_mu.weight" in ckpt:
        z_dim = ckpt["fc_mu.weight"].shape[0]
    else:
        z_dim = state_dim_from_buffer - 2

    vae = VAE(input_dim=6, latent_dim=z_dim).to(device)
    vae.load_state_dict(ckpt)
    vae.eval()

    with open(stats_path, "rb") as f:
        stats = pickle.load(f)
    mean = T.tensor(stats["mean_cont"], dtype=T.float32, device=device)  # [6]
    std  = T.tensor(stats["std_cont"],  dtype=T.float32, device=device)  # [6]

    @T.no_grad()
    def encode_obs(obs8_np):
        x = T.tensor(obs8_np, dtype=T.float32, device=device).unsqueeze(0)   # [1,8]
        x_cont = (x[:, :6] - mean) / (std + 1e-6)
        mu, _ = vae.encode(x_cont)                                          # [1,z]
        return T.cat([mu, x[:, 6:]], dim=-1).squeeze(0).cpu().numpy()       # [6]
    return encode_obs


def evaluate_policy(env, agent, encode_obs, episodes=5):
    import torch as T
    scores = []
    for _ in range(episodes):
        obs, _ = env.reset()
        done = False
        score = 0.0
        while not done:
            # 8D -> 6D latent
            s_lat = encode_obs(obs)
            # deterministic action = tanh(mu)
            with T.no_grad():
                s = T.tensor(s_lat, dtype=T.float32, device=agent.device).unsqueeze(0)
                mu, _ = agent.actor(s)  # use the mean; ignore std
                a = T.tanh(mu) * T.as_tensor(agent.max_action, device=agent.device, dtype=mu.dtype)
                action = a.squeeze(0).cpu().numpy()
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            score += reward
        scores.append(score)
    return float(np.mean(scores))



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sim', type=int, default=1, help='Simulation ID')
    args = parser.parse_args()
    sim = args.sim

    # ---- Load the VAE-encoded buffer ----
    buffer_path = 'dataset/unbiased_sim_1/replay_buffer_vae.pkl'   # <— VAE file
    with open(buffer_path, 'rb') as f:
        replay_buffer: ReplayBuffer = pickle.load(f)

    # ---- Get dims from buffer (REQUIRED for VAE) ----
    state_dim = replay_buffer.state_memory.shape[1]   # should be 6 (z + 2 flags)
    action_dim = replay_buffer.action_memory.shape[1]

    env_id = 'LunarLanderContinuous-v2'
    env = gym.make(env_id)
    max_action = float(env.action_space.high[0])      # [-1,1] for Lunar
    device = T.device('cuda' if T.cuda.is_available() else 'cpu')

    agent = SAC_CQL(
        state_dim=state_dim,
        action_dim=action_dim,
        max_action=max_action,
        device=device,
        sim=sim,
        cql_alpha=0.05   # keep your original hparam choices
    )

    # --- Encoder for eval (only used in evaluate_policy) ---
    encode_obs = make_encoder(device, state_dim_from_buffer=state_dim)

    results_dir = 'results/unbiased_sim_4_vae(0.05)'   # tag it as vae-run
    os.makedirs(results_dir, exist_ok=True)

    scores, steps = [], []
    eval_interval = 10_000
    max_steps = 1_000_000
    batch_size = 128

    for step in range(1, max_steps + 1):
        agent.train(replay_buffer, batch_size=batch_size)

        if step % eval_interval == 0:
            avg_score = evaluate_policy(env, agent, encode_obs, episodes=10)
            print(f"[Sim {sim} | Step {step}] Avg Eval Score: {avg_score:.2f}")
            scores.append(avg_score)
            steps.append(step)

    agent.save_models()

    np.save(os.path.join(results_dir, 'scores.npy'), np.array(scores))
    np.save(os.path.join(results_dir, 'steps.npy'),  np.array(steps))
