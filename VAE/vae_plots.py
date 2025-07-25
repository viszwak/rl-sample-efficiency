import torch
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# === Load Files ===
vae_path = "results/vae_lunar/vae_epoch2.pt"
norm_stats_path = "results/vae_lunar/norm_stats.pkl"
buffer_path = "dataset/sim_1/replay_buffer.pkl"
loss_file = "results/vae_lunar/losses.npy"

# === Load VAE Model ===
from model.vae_vector import VAE
vae = VAE(input_dim=8, latent_dim=7)
vae.load_state_dict(torch.load(vae_path, map_location='cpu'))
vae.eval()

# === Load Normalization Stats ===
with open(norm_stats_path, "rb") as f:
    stats = pickle.load(f)
mean = torch.tensor(stats['mean'], dtype=torch.float32)
std = torch.tensor(stats['std'], dtype=torch.float32)

# === Load and Downsample States ===
with open(buffer_path, "rb") as f:
    replay_buffer = pickle.load(f)
states = torch.tensor(replay_buffer.state_memory, dtype=torch.float32)
# Downsample to 100,000 states to reduce memory usage
subset_size = 100000
indices = np.random.choice(len(states), subset_size, replace=False)
norm_states = (states[indices] - mean) / std

# === Reconstruct and Denormalize ===
with torch.no_grad():
    recon, _, _ = vae(norm_states)
recon_denorm = recon * std + mean
mse_per_dim = ((norm_states - recon)**2).mean(dim=0).numpy()  # MSE on normalized data
print("Per-dimension MSE:", np.round(mse_per_dim, 4))
print("Average reconstruction MSE:", np.round(mse_per_dim.mean(), 4))

# === Reconstruction Error ===
recon_error = ((norm_states - recon)**2).mean(dim=1).numpy()

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
    plt.close()

# === 2. Reconstruction Error Histogram ===
plt.figure()
plt.hist(recon_error, bins=50, alpha=0.7)
plt.title("Histogram of Reconstruction Error (MSE)")
plt.xlabel("MSE")
plt.ylabel("Frequency")
plt.grid()
plt.savefig("vae_plots/reconstruction_error_hist.png")
plt.close()

# === 3. Original vs Reconstructed States ===
sample_indices = np.random.choice(subset_size, 5, replace=False)
for i in sample_indices:
    plt.figure()
    plt.plot(norm_states[i].numpy(), label="Original (Normalized)")
    plt.plot(recon[i].numpy(), label="Reconstructed (Normalized)", linestyle="--")
    plt.title(f"State {indices[i]} Reconstruction")
    plt.legend()
    plt.grid()
    plt.savefig(f"vae_plots/state_reconstruction_{indices[i]}.png")
    plt.close()

# === 4. t-SNE Visualization of Latent Space ===
with torch.no_grad():
    latent = vae.encode_latent(norm_states).cpu().numpy()

rewards = np.array(replay_buffer.reward_memory)[indices]
tsne = TSNE(n_components=2, perplexity=30, learning_rate=100, random_state=42)  # Reduced learning_rate
reduced = tsne.fit_transform(latent[:5000])  # Further downsample to 5,000 points
rewards_subset = rewards[:5000]

plt.figure(figsize=(8, 6))
scatter = plt.scatter(reduced[:, 0], reduced[:, 1], c=rewards_subset, cmap='coolwarm', s=0.6)
plt.colorbar(scatter, label="Reward")
plt.title("t-SNE of VAE Latent Space (Colored by Reward)")
plt.grid(True)
plt.tight_layout()
plt.savefig("vae_plots/tsne_reward.png")
plt.close()
print(" Saved t-SNE plot to vae_plots/tsne_reward.png")