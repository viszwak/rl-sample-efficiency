import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pickle
from torch.utils.data import DataLoader, TensorDataset
from vae_torch import VAE
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load state dataset
with open('dataset/sim_1/replay_buffer.pkl', 'rb') as f:
    dataset = pickle.load(f)
states = dataset.state_memory  # shape: (N, 8)

states_tensor = torch.tensor(states, dtype=torch.float32)
loader = DataLoader(TensorDataset(states_tensor), batch_size=1024, shuffle=True)

vae = VAE(input_dim=8, latent_dim=6).to(device)
optimizer = optim.Adam(vae.parameters(), lr=1e-3)

def vae_loss(recon_x, x, mu, logvar):
    recon_loss = F.mse_loss(recon_x, x, reduction='mean')
    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kl

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

    print(f"Epoch {epoch+1} | Loss: {total_loss/len(loader):.4f}")

vae_dir = "models"
os.makedirs(vae_dir, exist_ok=True)
vae_path = os.path.join(vae_dir, "vae_state_encoder_6st.pt")

torch.save(vae.state_dict(), vae_path)
print(f" VAE model saved to: {vae_path}")
