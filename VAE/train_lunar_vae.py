import os
import json
import pickle
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split

from model.vae_vector import VAE
from utils.noise import apply_noise

logging.basicConfig(level=logging.INFO)

# -------------------
# Load config & setup
# -------------------
with open("config/vae_lunar.json") as f:
    config = json.load(f)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(config["training"].get("seed", 42))
np.random.seed(config["training"].get("seed", 42))

# -------------------
# Load dataset
# -------------------
with open(config["paths"]["replay_buffer"], "rb") as f:
    buffer = pickle.load(f)

states = torch.tensor(buffer.state_memory, dtype=torch.float32)      # [N, 8]
x_cont = states[:, :6]                                               # first 6 continuous dims
# flags = states[:, 6:] passthrough during RL

# Standardize continuous dims
mean = x_cont.mean(dim=0)
std = x_cont.std(dim=0) + 1e-6
x_cont = (x_cont - mean) / std

# Dataset & splits
dataset = TensorDataset(x_cont)
N = len(dataset)
val_size = int(0.1 * N)
test_size = int(0.1 * N)
train_size = N - val_size - test_size

g = torch.Generator().manual_seed(config["training"].get("seed", 42))
train_set, val_set, test_set = random_split(dataset, [train_size, val_size, test_size], generator=g)

train_loader = DataLoader(train_set, batch_size=config["model"]["batch_size"], shuffle=True)
val_loader   = DataLoader(val_set,   batch_size=256)
# test_loader  = DataLoader(test_set,  batch_size=256)  # optional


# Model, opt, sched

vae = VAE(input_dim=6, latent_dim=config["model"]["latent_dim"]).to(device)
optimizer = optim.AdamW(
    vae.parameters(),
    lr=config["training"]["lr"],
    weight_decay=config["training"]["weight_decay"]
)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
criterion = nn.MSELoss(reduction="sum")

save_dir = config["paths"]["save_dir"]
os.makedirs(save_dir, exist_ok=True)

kl_weight     = config["training"].get("kl_weight", 1.0)
denoise_std   = config["training"].get("denoise_std", 0.05)
anneal_epochs = max(1, int(config["model"].get("anneal_epochs", 50)))

best_val_loss = float("inf")
patience = 10
trigger_times = 0

train_losses, val_losses = [], []


# Training loop

for epoch in range(config["model"]["epochs"]):
    vae.train()
    total_loss = 0.0
    anneal_kl_train = min(1.0, epoch / anneal_epochs)  # used in TRAIN
    anneal_kl_val   = 1.0                              # fixed at target for VALIDATION


    for (batch_x,) in train_loader:
        x = batch_x.to(device)
        x_noisy = apply_noise(x, std_dev=denoise_std)  # denoising AE: noisy-in, clean target

        recon_x, mu, logvar = vae(x_noisy, sample=True)
        recon_loss = criterion(recon_x, x)
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        loss = recon_loss + (kl_weight * anneal_kl_train) * kl_loss


        optimizer.zero_grad()
        loss.backward()
        # Optional: gradient clipping for stability
        # torch.nn.utils.clip_grad_norm_(vae.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()

    avg_train = total_loss / train_size
    train_losses.append(avg_train)

    # Validation with the same objective
    vae.eval()
    val_total = 0.0
    with torch.no_grad():
        for (batch_x,) in val_loader:
            x = batch_x.to(device)
            x_noisy = apply_noise(x, std_dev=denoise_std)
            recon_x, mu, logvar = vae(x_noisy, sample=True)
            recon_loss = criterion(recon_x, x)
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            val_total += (recon_loss + (kl_weight * anneal_kl_val) * kl_loss).item()


    avg_val = val_total / val_size
    val_losses.append(avg_val)
    scheduler.step(avg_val)

    # Periodic debug (per-sample recon/KL)
    if (epoch + 1) % 10 == 0:
        with torch.no_grad():
            (x_dbg,) = next(iter(train_loader))
            x_dbg = x_dbg.to(device)
            recon_dbg, mu_dbg, logvar_dbg = vae(x_dbg, sample=True)
            recon_ps = (criterion(recon_dbg, x_dbg) / len(x_dbg)).item()
            kl_ps = (-0.5 * torch.sum(1 + logvar_dbg - mu_dbg.pow(2) - logvar_dbg.exp()) / len(x_dbg)).item()
        logging.info(f"Epoch {epoch+1} | Recon/sample: {recon_ps:.4f} | KL/sample: {kl_ps:.4f}")

    # Early stopping & best checkpoint
    if avg_val > best_val_loss:
        trigger_times += 1
        logging.info(f"Val not improved. Trigger {trigger_times}/{patience}")
        if trigger_times >= patience:
            logging.info(f"Early stopping at epoch {epoch + 1}")
            break
    else:
        best_val_loss = avg_val
        trigger_times = 0
        logging.info(f"Val improved to {best_val_loss:.6f}. Saving best...")
        torch.save(vae.state_dict(), os.path.join(save_dir, "vae_best.pt"))

    # Periodic snapshot
    if (epoch + 1) % 10 == 0:
        torch.save(vae.state_dict(), os.path.join(save_dir, f"vae_epoch{epoch+1}.pt"))

    logging.info(f"Epoch {epoch+1:03d} | Train: {avg_train:.6f} | Val: {avg_val:.6f}")

# Save last
torch.save(vae.state_dict(), os.path.join(save_dir, "vae_last.pt"))

# Save normalization stats (continuous only)
with open(os.path.join(save_dir, "norm_stats.pkl"), "wb") as f:
    pickle.dump({"mean_cont": mean.cpu().numpy(), "std_cont": std.cpu().numpy()}, f)

# Save losses
np.save(os.path.join(save_dir, "train_losses.npy"), np.array(train_losses))
np.save(os.path.join(save_dir, "val_losses.npy"),   np.array(val_losses))

logging.info("Training completed.")
