import torch
import pickle
import numpy as np
import os
from model.vae_vector import VAE

# ---- Config ----
vae_model_path = "results/vae_lunar/vae_epoch2.pt"
norm_stats_path = "results/vae_lunar/norm_stats.pkl"
buffer_path = "dataset/sim_1/replay_buffer.pkl"
save_path = "dataset/sim_1/vae_encoded_buffer.pkl"
batch_size = 1024  # Adjust based on GPU memory (e.g., 512 if 4GB GPU)

# ---- Load VAE model ----
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vae = VAE(input_dim=8, latent_dim=7).to(device)
vae.load_state_dict(torch.load(vae_model_path, map_location=device))
vae.eval()

# ---- Load normalization stats ----
with open(norm_stats_path, "rb") as f:
    stats = pickle.load(f)
mean = torch.tensor(stats["mean"], dtype=torch.float32).to(device)
std = torch.tensor(stats["std"], dtype=torch.float32).to(device)

# ---- Load original replay buffer ----
with open(buffer_path, "rb") as f:
    buffer = pickle.load(f)
print(f"Buffer size: {len(buffer.state_memory)} samples")

# ---- Encode in batches ----
n_samples = len(buffer.state_memory)
latent_states = np.zeros((n_samples, 7))
latent_next_states = np.zeros((n_samples, 7))

for i in range(0, n_samples, batch_size):
    start_idx = i
    end_idx = min(i + batch_size, n_samples)
    states = torch.tensor(buffer.state_memory[start_idx:end_idx], dtype=torch.float32).to(device)
    next_states = torch.tensor(buffer.new_state_memory[start_idx:end_idx], dtype=torch.float32).to(device)

    norm_states = (states - mean) / std
    norm_next_states = (next_states - mean) / std

    with torch.no_grad():
        latent_states[start_idx:end_idx] = vae.encode_latent(norm_states).cpu().numpy()
        latent_next_states[start_idx:end_idx] = vae.encode_latent(norm_next_states).cpu().numpy()

buffer.state_memory = latent_states
buffer.new_state_memory = latent_next_states

# ---- Save encoded buffer ----
os.makedirs(os.path.dirname(save_path), exist_ok=True)
with open(save_path, "wb") as f:
    pickle.dump(buffer, f)

print(f"VAE-encoded buffer saved to: {save_path}")