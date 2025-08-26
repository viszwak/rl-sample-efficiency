# encode_dataset.py
import os
import json
import pickle
import logging
import numpy as np
from types import SimpleNamespace

import torch
from model.vae_vector import VAE   # same VAE class as used in training

logging.basicConfig(level=logging.INFO)

def load_stats(save_dir, device):
    with open(os.path.join(save_dir, "norm_stats.pkl"), "rb") as f:
        stats = pickle.load(f)
    # renamed fields from the updated trainer
    median = torch.tensor(stats["median"], dtype=torch.float32, device=device)
    scale  = torch.tensor(stats["mad_scale"], dtype=torch.float32, device=device)
    return median, scale

@torch.no_grad()
def encode_all(vae, x6, batch_size=4096):
    zs = []
    for i in range(0, x6.shape[0], batch_size):
        xb = x6[i:i+batch_size]
        zb = vae.encode_latent(xb)  # μ only (deterministic), VAE must be in eval()
        zs.append(zb)
    return torch.cat(zs, dim=0)

def main():
    # -------------------
    # Load config & setup
    # -------------------
    with open("config/vae_lunar.json") as f:
        config = json.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    z_dim  = int(config["model"]["latent_dim"])
    save_dir = config["paths"]["save_dir"]
    os.makedirs(save_dir, exist_ok=True)

    # -------------------
    # Load original buffer
    # -------------------
    with open(config["paths"]["replay_buffer"], "rb") as f:
        buf = pickle.load(f)

    # Support both styles: attribute mem_cntr/mem_size, or just array length
    N_total = getattr(buf, "mem_cntr", None)
    if N_total is None or N_total == 0:
        N_total = len(buf.state_memory)
    N_total = min(N_total, len(buf.state_memory))

    # Pull arrays and trim to N_total
    S  = torch.tensor(np.asarray(buf.state_memory[:N_total]),     dtype=torch.float32, device=device)  # [N,8]
    S2 = torch.tensor(np.asarray(buf.new_state_memory[:N_total]), dtype=torch.float32, device=device)  # [N,8]
    A  = np.asarray(buf.action_memory[:N_total])                                                       # [N,act_dim]
    R  = np.asarray(buf.reward_memory[:N_total])                                                       # [N]
    D  = np.asarray(buf.terminal_memory[:N_total]).astype(np.float32)                                  # [N]

    # -------------------
    # Load VAE + robust stats
    # -------------------
    vae = VAE(input_dim=6, latent_dim=z_dim).to(device)
    vae_ckpt = os.path.join(save_dir, "vae_best.pt")
    vae.load_state_dict(torch.load(vae_ckpt, map_location=device))
    vae.eval()

    median, scale = load_stats(save_dir, device)

    # -------------------
    # Robust normalize + clamp [-5,5] exactly like training
    # -------------------
    X6  = S[:,  :6]
    X62 = S2[:, :6]
    F   = S[:,  6:]   # flags (2 dims)
    F2  = S2[:, 6:]

    X6n  = torch.clamp((X6  - median) / scale, -5.0, 5.0)
    X62n = torch.clamp((X62 - median) / scale, -5.0, 5.0)

    # -------------------
    # Encode μ only (no sampling)
    # -------------------
    z   = encode_all(vae, X6n)
    z2  = encode_all(vae, X62n)

    # -------------------
    # Standardize latent (stabilizes CQL)
    # -------------------
    z_mean = z.mean(dim=0, keepdim=True)
    z_std  = z.std(dim=0, keepdim=True).clamp_min(1e-6)

    z_norm  = (z  - z_mean) / z_std
    z2_norm = (z2 - z_mean) / z_std      # IMPORTANT: use same stats for next states

    # Save the latent standardization stats for downstream use (e.g., online eval)
    with open(os.path.join(save_dir, "latent_z_stats.pkl"), "wb") as f:
        pickle.dump({
            "z_mean": z_mean.squeeze(0).cpu().numpy(),
            "z_std":  z_std.squeeze(0).cpu().numpy()
        }, f)

    # -------------------
    # Build latent states with flags concatenated
    # -------------------
    state_latent      = torch.cat([z_norm,  F],  dim=1).cpu().numpy()   # [N, z_dim+2]
    next_state_latent = torch.cat([z2_norm, F2], dim=1).cpu().numpy()   # [N, z_dim+2]

    # -------------------
    # Pack into a simple buffer-like object and save
    # -------------------
    latent_buf = SimpleNamespace(
        state_memory=state_latent,
        new_state_memory=next_state_latent,
        action_memory=A,
        reward_memory=R,
        terminal_memory=D,
        mem_size=state_latent.shape[0],
        mem_cntr=state_latent.shape[0]
    )

    out_path = os.path.join(save_dir, f"replay_buffer_latent_z{z_dim}.pkl")
    with open(out_path, "wb") as f:
        pickle.dump(latent_buf, f)

    logging.info(f"Encoded latent dataset saved: {out_path}")
    logging.info(f"Obs dim for CQL now = {state_latent.shape[1]} (z_dim {z_dim} + 2 flags)")

if __name__ == "__main__":
    main()
