# encode_dataset.py
import torch
import pickle
import numpy as np
import os
from model.vae_vector import VAE

# ---- Config ----
vae_model_path = "results/vae_lunar/vae_epoch100.pt"
norm_stats_path = "results/vae_lunar/norm_stats.pkl"
buffer_path = "dataset/sim_1/replay_buffer.pkl"
save_path = "dataset/sim_1/vae_encoded_buffer.pkl"

# ---- Load VAE model ----
vae = VAE(input_dim=8, latent_dim=4)
vae.load_state_dict(torch.load(vae_model_path, map_location="cpu"))
vae.eval()

# ---- Load normalization stats ----
with open(norm_stats_path, "rb") as f:
    stats = pickle.load(f)
mean = torch.tensor(stats["mean"], dtype=torch.float32)
std = torch.tensor(stats["std"], dtype=torch.float32)

# ---- Load original replay buffer ----
with open(buffer_path, "rb") as f:
    buffer = pickle.load(f)

# ---- Encode both states and next_states ----
states = torch.tensor(buffer.state_memory, dtype=torch.float32)
next_states = torch.tensor(buffer.new_state_memory, dtype=torch.float32)  # <-- FIXED

norm_states = (states - mean) / std
norm_next_states = (next_states - mean) / std

with torch.no_grad():
    latent_states = vae.encode_latent(norm_states).numpy()
    latent_next_states = vae.encode_latent(norm_next_states).numpy()

buffer.state_memory = latent_states
buffer.new_state_memory = latent_next_states


with open(save_path, "wb") as f:
    pickle.dump(buffer, f)

print(f" VAE-encoded buffer saved to: {save_path}")
