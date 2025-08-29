import argparse, os, pickle, gym, numpy as np
import torch as T
import torch  

from sac_torch_cql import SAC_CQL
from model.vae_vector import VAE



def _load_norm_stats(vae_dir, device):
    with open(os.path.join(vae_dir, "norm_stats.pkl"), "rb") as f:
        stats = pickle.load(f)
    center = T.tensor(stats["mean_cont"], dtype=T.float32, device=device)
    scale  = T.tensor(stats["std_cont"],  dtype=T.float32, device=device).clamp_min(1e-6)
    return center, scale

def _load_latent_stats(vae_dir, device):
    with open(os.path.join(vae_dir, "latent_z_stats.pkl"), "rb") as f:
        zs = pickle.load(f)
    z_mean = T.tensor(zs["z_mean"], dtype=T.float32, device=device).unsqueeze(0)
    z_std  = T.tensor(zs["z_std"],  dtype=T.float32, device=device).unsqueeze(0).clamp_min(1e-6)
    return z_mean, z_std

def make_encoder(device, state_dim_from_buffer=None, latent_dim=None, vae_dir="results/vae_lunar_28"):
    if latent_dim is None:
        assert state_dim_from_buffer is not None, "Provide state_dim_from_buffer or latent_dim"
        latent_dim = int(state_dim_from_buffer) - 2  # subtract 2 flag dims

    vae = VAE(input_dim=6, latent_dim=latent_dim).to(device)
    ckpt = T.load(os.path.join(vae_dir, "vae_best.pt"), map_location=device)
    vae.load_state_dict(ckpt)
    vae.eval()

    center, scale = _load_norm_stats(vae_dir, device)
    z_mean, z_std = _load_latent_stats(vae_dir, device)

    @T.no_grad()
    def encode_obs(obs_np):
        x6 = T.tensor(obs_np[:6], dtype=T.float32, device=device).unsqueeze(0)
        flags = T.tensor(obs_np[6:], dtype=T.float32, device=device).unsqueeze(0)
        x6n = T.clamp((x6 - center) / scale, -5.0, 5.0)
        if hasattr(vae, "encode_latent"):
            z = vae.encode_latent(x6n, deterministic=True)
        else:
            mu, _ = vae.encode(x6n); z = mu
        zn = (z - z_mean) / z_std
        lat = T.cat([zn, flags], dim=1)
        return lat.squeeze(0).cpu().numpy()

    return encode_obs


# ---------- deterministic eval (uses policy mean action) ----------
def evaluate_policy(env, agent, encode_obs, episodes=5):
    scores = []
    for _ in range(episodes):
        obs, _ = env.reset()
        done, score = False, 0.0
        with T.no_grad():
            while not done:
                lat_obs = encode_obs(obs)
                s = T.tensor(lat_obs, dtype=T.float32, device=agent.device).unsqueeze(0)
                mu, _ = agent.actor(s)  # deterministic mean
                action = T.tanh(mu) * agent.max_action
                action = action.cpu().numpy().flatten()
                obs, r, term, trunc, _ = env.step(action)
                done = term or trunc
                score += r
        scores.append(score)
    return float(np.mean(scores))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--sim', type=int, default=1)
    ap.add_argument('--buf', type=str,
                    default='results/vae_lunar_28/replay_buffer_latent_z4.pkl',
                    help='Path to VAE-encoded ReplayBuffer')
    ap.add_argument('--vae_dir', type=str,
                    default='results/vae_lunar_28',
                    help='Dir with vae_best.pt, norm_stats.pkl, latent_z_stats.pkl')
    ap.add_argument('--env_id', type=str, default='LunarLanderContinuous-v2')
    ap.add_argument('--eval_every', type=int, default=5000)
    ap.add_argument('--max_steps', type=int, default=1_000_000)
    ap.add_argument('--batch_size', type=int, default=256)
    ap.add_argument('--alpha', type=float, default=0.2)
    ap.add_argument('--cql_alpha', type=float, default=0.01)
    ap.add_argument('--discount', type=float, default=0.99)
    ap.add_argument('--tau', type=float, default=0.005)
    args = ap.parse_args()
    sim = args.sim

    # ---- load buffer ----
    with open(args.buf, 'rb') as f:
        replay_buffer = pickle.load(f)

    state_dim  = int(replay_buffer.state_memory.shape[1])
    action_dim = int(replay_buffer.action_memory.shape[1])
    print(f"Buffer state dim: {state_dim}")
    print(f"Buffer size: {replay_buffer.mem_cntr}")
    rw = replay_buffer.reward_memory[:min(1000, replay_buffer.mem_cntr)]
    print(f"Sample reward range: [{rw.min():.2f}, {rw.max():.2f}]")

    # ---- env / device ----
    env = gym.make(args.env_id)
    max_action = float(env.action_space.high[0])
    device = T.device('cuda' if T.cuda.is_available() else 'cpu')

    # ---- agent ----
    agent = SAC_CQL(
        state_dim=state_dim,
        action_dim=action_dim,
        max_action=max_action,
        device=device,
        sim=sim,
        cql_alpha=args.cql_alpha,
        alpha=args.alpha,
        discount=args.discount,
        tau=args.tau
    )

    # ---- encoder for eval ----
    encode_obs = make_encoder(device, state_dim_from_buffer=state_dim, vae_dir=args.vae_dir)

    # ---- training loop with monitors ----
    results_dir = f'results/sim_{sim}_vae_cql'
    os.makedirs(results_dir, exist_ok=True)

    scores, steps = [], []
    val_size = min(10_000, replay_buffer.mem_cntr)
    val_states  = replay_buffer.state_memory[-val_size:]
    val_actions = replay_buffer.action_memory[-val_size:]
    val_rewards = replay_buffer.reward_memory[-val_size:]

    for step in range(1, args.max_steps + 1):
        agent.train(replay_buffer, batch_size=args.batch_size)

        if step % args.eval_every == 0:
            avg_score = evaluate_policy(env, agent, encode_obs, episodes=5)

            sample_states  = replay_buffer.state_memory[:100]
            sample_actions = replay_buffer.action_memory[:100]
            val_idx = np.random.choice(val_size, 100, replace=False)

            with torch.no_grad():
                s = torch.tensor(sample_states, dtype=torch.float32, device=device)
                a = torch.tensor(sample_actions, dtype=torch.float32, device=device)
                q1 = agent.critic.Q1(s, a).mean().item()

                s_val = torch.tensor(val_states[val_idx], dtype=torch.float32, device=device)
                a_val = torch.tensor(val_actions[val_idx], dtype=torch.float32, device=device)
                q1_val = agent.critic.Q1(s_val, a_val).mean().item()
                val_returns = float(val_rewards[val_idx].mean())

                mu_policy, _ = agent.actor(s)
                actions_policy = torch.tanh(mu_policy) * agent.max_action
                action_diff = float(((actions_policy - a) ** 2).mean().item())

                print(f"[Sim {sim} | Step {step}] "
                      f"Score: {avg_score:.2f} | "
                      f"Train Q: {q1:.2f} | Val Q: {q1_val:.2f} | "
                      f"Val r(mean): {val_returns:.2f} | Action MSE: {action_diff:.4f}")

            scores.append(avg_score)
            steps.append(step)

    agent.save_models()
    np.save(os.path.join(results_dir, 'scores.npy'), np.array(scores))
    np.save(os.path.join(results_dir, 'steps.npy'), np.array(steps))
