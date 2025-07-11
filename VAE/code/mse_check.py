import torch
import pickle
import numpy as np
import matplotlib.pyplot as plt
from vae_torch import VAE
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load dataset
with open("dataset/sim_1/replay_buffer.pkl", "rb") as f:
    dataset = pickle.load(f)
states = dataset.state_memory[:1000]
states_tensor = torch.tensor(states, dtype=torch.float32).to(device)

# Load VAE models
vae_4 = VAE(input_dim=8, latent_dim=4).to(device)
vae_4.load_state_dict(torch.load("models/vae_state_encoder_4st.pt"))
vae_4.eval()

vae_6 = VAE(input_dim=8, latent_dim=6).to(device)
vae_6.load_state_dict(torch.load("models/vae_state_encoder_6st.pt"))
vae_6.eval()

# Reconstruct
with torch.no_grad():
    recon_4, _, _ = vae_4(states_tensor)
    recon_6, _, _ = vae_6(states_tensor)

# Move to CPU for NumPy
recon_4 = recon_4.cpu().numpy()
recon_6 = recon_6.cpu().numpy()
states_np = states_tensor.cpu().numpy()

# Compute MSE
mse_4 = np.mean((states_np - recon_4) ** 2)
mse_6 = np.mean((states_np - recon_6) ** 2)

# Print results
print(f"🔍 VAE Reconstruction MSE (latent=4): {mse_4:.6f}")
print(f"🔍 VAE Reconstruction MSE (latent=6): {mse_6:.6f}")

# Plot bar chart
plt.figure(figsize=(6, 5))
labels = ['Latent 4D', 'Latent 6D']
mses = [mse_4, mse_6]
colors = ['#1f77b4', '#ff7f0e']

plt.bar(labels, mses, color=colors)
plt.title("VAE Reconstruction MSE by Latent Dimension")
plt.ylabel("Mean Squared Error")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()

# Save
os.makedirs("plots", exist_ok=True)
plt.savefig("plots/vae_mse_comparison.png")
plt.show()
