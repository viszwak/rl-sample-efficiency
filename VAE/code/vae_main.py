import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pickle
from torch.utils.data import DataLoader, TensorDataset
from vae_torch import VAE
import os
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load state dataset
with open('dataset/sim_1/replay_buffer.pkl', 'rb') as f:
    dataset = pickle.load(f)
states = dataset.state_memory  # shape: (N, 8)

states_tensor = torch.tensor(states, dtype=torch.float32)

# Compute normalization stats
mean = states_tensor.mean(dim=0)
std = states_tensor.std(dim=0) + 1e-6  # avoid div-by-zero

# Normalize states
normalized_states = (states_tensor - mean) / std

loader = DataLoader(TensorDataset(normalized_states), batch_size=1024, shuffle=True)

# Initialize VAE
vae = VAE(input_dim=8, latent_dim=4).to(device)
optimizer = optim.Adam(vae.parameters(), lr=1e-3)

# Create output directory
save_dir = "results/vae_4d"
os.makedirs(save_dir, exist_ok=True)

# Define VAE loss function
def vae_loss(recon_x, x, mu, logvar):
    recon_loss = F.mse_loss(recon_x, x, reduction='mean')
    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kl

# Train VAE
losses = []
for epoch in range(100):
    total_loss = 0
    for batch in loader:
        x = batch[0].to(device)
        recon_x, mu, logvar = vae(x)
        loss = vae_loss(recon_x, x, mu, logvar)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(loader)
    losses.append(avg_loss)
    print(f"Epoch {epoch+1} | Loss: {avg_loss:.4f}")

# Save VAE model
vae_path = os.path.join(save_dir, "vae_state_encoder_4st.pt")
torch.save(vae.state_dict(), vae_path)
print(f"✅ VAE model saved to: {vae_path}")

# Save normalization stats
norm_stats = {
    'mean': mean.numpy(),
    'std': std.numpy()
}
with open(os.path.join(save_dir, "vae_norm_stats_4st.pkl"), 'wb') as f:
    pickle.dump(norm_stats, f)
print(f"📊 Normalization stats saved to: {save_dir}/vae_norm_stats_4st.pkl")

# Save training loss for plotting
np.save(os.path.join(save_dir, "losses_4d.npy"), np.array(losses))
print(f"📉 Losses saved to: {save_dir}/losses_4d.npy")
