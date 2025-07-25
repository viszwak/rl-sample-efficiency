import torch
import torch.nn as nn
import torch.optim as optim
import pickle
import numpy as np
import os
import json
import logging
from torch.utils.data import DataLoader, TensorDataset, random_split
from model.vae_vector import VAE
from utils.noise import apply_noise

best_val_loss = float('inf')  # Track the best validation loss
patience = 10                # Number of epochs to wait for improvement
trigger_times = 0            # Counter for epochs without improvement

logging.basicConfig(level=logging.INFO)

# Load config
with open("config/vae_lunar.json") as f:
    config = json.load(f)

# Load dataset
with open(config["paths"]["replay_buffer"], "rb") as f:
    buffer = pickle.load(f)
states = torch.tensor(buffer.state_memory, dtype=torch.float32)

# Normalize
mean = states.mean(dim=0)
std = states.std(dim=0) + 1e-6
states = (states - mean) / std

# Dataset splits
N = len(states)
val_size = int(0.1 * N)
test_size = int(0.1 * N)
train_size = N - val_size - test_size
train_set, val_set, test_set = random_split(states, [train_size, val_size, test_size])

train_loader = DataLoader(train_set, batch_size=config["model"]["batch_size"], shuffle=True)
val_loader = DataLoader(val_set, batch_size=256)
test_loader = DataLoader(test_set, batch_size=256)

# Model
vae = VAE(input_dim=8, latent_dim=config["model"]["latent_dim"]).to("cuda")
optimizer = optim.AdamW(vae.parameters(), lr=config["training"]["lr"], weight_decay=config["training"]["weight_decay"])
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
criterion = nn.MSELoss(reduction="sum")

# Training
os.makedirs(config["paths"]["save_dir"], exist_ok=True)
train_losses, val_losses = [], []

kl_weight = config["training"].get("kl_weight", 0.1)

for epoch in range(config["model"]["epochs"]):
    vae.train()
    total_loss = 0
    for batch in train_loader:
        x = batch.to("cuda")
        x_noisy = apply_noise(x, std_dev=0.05).to("cuda")
        recon_x, mu, logvar = vae(x_noisy)

        recon_loss = criterion(recon_x, x)
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        # Linearly increase KL weight from 0 → target over first 50 epochs
        anneal_kl = min(1.0, epoch / 100)
        loss = recon_loss + (kl_weight * anneal_kl) * kl_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / train_size
    train_losses.append(avg_loss)

    # Log KL and recon loss for debugging
    if (epoch + 1) % 10 == 0:
        with torch.no_grad():
            x_debug = next(iter(train_loader)).to("cuda")
            recon_x, mu, logvar = vae(x_debug)
            recon = criterion(recon_x, x_debug) / len(x_debug)
            kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / len(x_debug)
        logging.info(f"Epoch {epoch+1} | Recon: {recon:.4f} | KL: {kl:.4f}")

    # Validation
    vae.eval()
    val_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            x = batch.to("cuda")
            recon_x, _, _ = vae(x)
            val_loss += criterion(recon_x, x).item()
    avg_val_loss = val_loss / val_size
    val_losses.append(avg_val_loss)

    scheduler.step(avg_val_loss)

    # Early stopping logic
    if avg_val_loss > best_val_loss:
        trigger_times += 1
        logging.info(f"Validation loss did not improve. Trigger times: {trigger_times}/{patience}")
        if trigger_times >= patience:
            logging.info(f"Early stopping at epoch {epoch + 1}")
            break
    else:
        best_val_loss = avg_val_loss
        trigger_times = 0
        logging.info(f"Validation loss improved to {best_val_loss:.4f}. Saving model...")
        torch.save(vae.state_dict(), os.path.join(config["paths"]["save_dir"], f"vae_epoch{epoch+1}.pt"))

    logging.info(f"Epoch {epoch+1} | Train: {avg_loss:.4f} | Val: {avg_val_loss:.4f}")

    # Save checkpoint
    if (epoch + 1) % 10 == 0:
        torch.save(vae.state_dict(), os.path.join(config["paths"]["save_dir"], f"vae_epoch{epoch+1}.pt"))

# Save stats
with open(os.path.join(config["paths"]["save_dir"], "norm_stats.pkl"), "wb") as f:
    pickle.dump({"mean": mean.numpy(), "std": std.numpy()}, f)

np.save(os.path.join(config["paths"]["save_dir"], "losses.npy"), np.array(train_losses))
logging.info("Training completed.")