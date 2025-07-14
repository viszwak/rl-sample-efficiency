import torch
import numpy as np
import pickle
import matplotlib.pyplot as plt
import os

# === Load Files ===
vae_path = "results/vae_lunar/vae_epoch100.pt"
norm_stats_path = "results/vae_lunar/norm_stats.pkl"
buffer_path = "dataset/sim_1/replay_buffer.pkl"
loss_file = "results/vae_lunar/losses.npy"

# === Load VAE Model ===
from model.vae_vector import VAE
vae = VAE(input_dim=8, latent_dim=4)
vae.load_state_dict(torch.load(vae_path, map_location='cpu'))
vae.eval()

# === Load Normalization Stats ===
with open(norm_stats_path, "rb") as f:
    stats = pickle.load(f)
mean = torch.tensor(stats['mean'], dtype=torch.float32)
std = torch.tensor(stats['std'], dtype=torch.float32)

# === Load and Normalize States ===
with open(buffer_path, "rb") as f:
    replay_buffer = pickle.load(f)
states = torch.tensor(replay_buffer.state_memory, dtype=torch.float32)
norm_states = (states - mean) / std

# === Reconstruct and Denormalize ===
with torch.no_grad():
    recon, _, _ = vae(norm_states)
recon_denorm = recon * std + mean

# === Reconstruction Error ===
recon_error = ((states - recon_denorm)**2).mean(dim=1).numpy()

# === Plot Directory ===
os.makedirs("vae_plots", exist_ok=True)

# === 1. Loss Curve ===
if os.path.exists(loss_file):
    losses = np.load(loss_file)
    plt.figure()
    plt.plot(losses)
    plt.title("VAE Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid()
    plt.savefig("vae_plots/loss_curve.png")

# === 2. Reconstruction Error Histogram ===
plt.figure()
plt.hist(recon_error, bins=50, alpha=0.7)
plt.title("Histogram of Reconstruction Error (MSE)")
plt.xlabel("MSE")
plt.ylabel("Frequency")
plt.grid()
plt.savefig("vae_plots/reconstruction_error_hist.png")

# === 3. Original vs Reconstructed States ===
sample_indices = np.random.choice(len(states), 5, replace=False)
for i in sample_indices:
    plt.figure()
    plt.plot(states[i].numpy(), label="Original")
    plt.plot(recon_denorm[i].numpy(), label="Reconstructed", linestyle="--")
    plt.title(f"State {i} Reconstruction")
    plt.legend()
    plt.grid()
    plt.savefig(f"vae_plots/state_reconstruction_{i}.png")
