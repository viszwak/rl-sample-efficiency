import torch
import pickle
import matplotlib.pyplot as plt
import numpy as np
from vae_torch import VAE
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load state dataset
with open("dataset/sim_1/replay_buffer.pkl", "rb") as f:
    dataset = pickle.load(f)
states = dataset.state_memory[:5]  # (5, 8)
states_tensor = torch.tensor(states, dtype=torch.float32).to(device)

# Load all VAE models
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


recon_4 = recon_4.cpu().numpy()
recon_6 = recon_6.cpu().numpy()

# Save plots
output_dir = "plots"
os.makedirs(output_dir, exist_ok=True)

feature_names = [
    "x-pos", "y-pos", "x-vel", "y-vel",
    "angle", "angular vel", "left leg", "right leg"
]

for i in range(5):
    plt.figure(figsize=(10, 4))
    plt.plot(states[i], label="Original", marker='o')
    plt.plot(recon_6[i], label="VAE-6D", linestyle=':', marker='s')
    plt.plot(recon_4[i], label="VAE-4D", linestyle='-.', marker='d')
    plt.title(f"State Reconstruction Comparison – Sample {i+1}")
    plt.xlabel("State Feature")
    plt.ylabel("Value")
    plt.xticks(ticks=np.arange(8), labels=feature_names, rotation=45)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/recon_{i+1}.png")
    plt.close()

print("✅ Reconstruction plots (4D/6D/) saved in: plots/")
