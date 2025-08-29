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

data_size = min(buffer.mem_cntr, buffer.mem_size)
good_indices = []
for i in range(data_size):
    # Skip terminal states with very negative rewards (crashes)
    if buffer.reward_memory[i] > -100:  # Not a crash
        good_indices.append(i)

print(f"Using {len(good_indices)}/{data_size} transitions (filtered crashes)")

# Use filtered data
states = torch.tensor(buffer.state_memory[good_indices], dtype=torch.float32)  # [N, 8]
x_cont = states[:, :6]  # first 6 continuous dims

median = x_cont.median(dim=0)[0]
mad = (x_cont - median).abs().median(dim=0)[0] + 1e-6
# Use robust scaling
x_cont = (x_cont - median) / (mad * 1.4826)  # 1.4826 converts MAD to std equivalent

# Clip extreme values to prevent outliers from breaking training
x_cont = torch.clamp(x_cont, -5, 5)

def augment_batch(batch, noise_scale=0.01):
    if np.random.random() < 0.5:  
        noise = torch.randn_like(batch) * noise_scale
        return batch + noise
    return batch

# Dataset & splits
dataset = TensorDataset(x_cont)
N = len(dataset)
val_size = int(0.1 * N)
test_size = int(0.1 * N)
train_size = N - val_size - test_size

g = torch.Generator().manual_seed(config["training"].get("seed", 42))
train_set, val_set, test_set = random_split(dataset, [train_size, val_size, test_size], generator=g)

batch_size = min(512, config["model"]["batch_size"] * 2)  # Double the batch size
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True)
val_loader = DataLoader(val_set, batch_size=256)

# Model, opt, sched
vae = VAE(input_dim=6, latent_dim=config["model"]["latent_dim"]).to(device)

optimizer = optim.AdamW(
    vae.parameters(),
    lr=config["training"]["lr"] * 0.5, 
    weight_decay=config["training"]["weight_decay"] * 10,  
    betas=(0.9, 0.999)
)

scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer, 
    T_0=20,  
    T_mult=2,  
    eta_min=1e-6
)

criterion = nn.MSELoss(reduction="sum")

save_dir = config["paths"]["save_dir"]
os.makedirs(save_dir, exist_ok=True)

base_kl_weight = config["training"].get("kl_weight", 1.0) * 0.0001  
denoise_std = config["training"].get("denoise_std", 0.05) * 0.5  
anneal_epochs = max(1, int(config["model"].get("anneal_epochs", 50)))

best_val_loss = float("inf")
patience = 20  
trigger_times = 0

train_losses, val_losses, recon_losses, kl_losses = [], [], [], []

warmup_epochs = 10

for epoch in range(config["model"]["epochs"]):
    vae.train()
    total_loss = 0.0
    total_recon = 0.0
    total_kl = 0.0
    
    if epoch < warmup_epochs:
        # Warmup: gradually increase KL weight
        kl_weight = base_kl_weight * (epoch / warmup_epochs)
    else:
        # Cyclical annealing
        cycle = (epoch - warmup_epochs) % 30
        if cycle < 15:
            kl_weight = base_kl_weight * (cycle / 15)
        else:
            kl_weight = base_kl_weight
    
    for (batch_x,) in train_loader:
        x = batch_x.to(device)
        
        if epoch < 30:
            current_denoise_std = denoise_std * (1 - epoch / 30)  
        else:
            current_denoise_std = 0.01 
            
        x_noisy = apply_noise(x, std_dev=current_denoise_std)
        
        x_noisy = augment_batch(x_noisy, noise_scale=0.005)
        
        recon_x, mu, logvar = vae(x_noisy, sample=True)
        
        dim_weights = torch.tensor([1.0, 2.0, 1.0, 2.0, 1.5, 1.0], device=device)
        weighted_diff = (recon_x - x) ** 2 * dim_weights.unsqueeze(0)
        recon_loss = weighted_diff.sum()
        
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        kl_threshold = 0.1 * batch_size * config["model"]["latent_dim"]
        kl_loss = torch.max(kl_loss, torch.tensor(kl_threshold, device=device))
        
        loss = recon_loss + kl_weight * kl_loss
        
        optimizer.zero_grad()
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(vae.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        total_loss += loss.item()
        total_recon += recon_loss.item()
        total_kl += kl_loss.item()
    
    scheduler.step()
    
    avg_train = total_loss / len(train_set)
    avg_recon = total_recon / len(train_set)
    avg_kl = total_kl / len(train_set)
    
    train_losses.append(avg_train)
    recon_losses.append(avg_recon)
    kl_losses.append(avg_kl)
    
    vae.eval()
    val_total = 0.0
    val_recon = 0.0
    
    with torch.no_grad():
        for (batch_x,) in val_loader:
            x = batch_x.to(device)
            # No noise during validation
            recon_x, mu, logvar = vae(x, sample=False)  # Use mean during validation
            
            # Same weighted reconstruction
            weighted_diff = (recon_x - x) ** 2 * dim_weights.unsqueeze(0)
            recon_loss = weighted_diff.sum()
            
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            val_total += (recon_loss + kl_weight * kl_loss).item()
            val_recon += recon_loss.item()
    
    avg_val = val_total / len(val_set)
    avg_val_recon = val_recon / len(val_set)
    val_losses.append(avg_val)
    
    if avg_val_recon < best_val_loss:
        best_val_loss = avg_val_recon
        trigger_times = 0
        logging.info(f"Val recon improved to {best_val_loss:.6f}. Saving best...")
        torch.save(vae.state_dict(), os.path.join(save_dir, "vae_best.pt"))
    else:
        trigger_times += 1
        if trigger_times >= patience:
            logging.info(f"Early stopping at epoch {epoch + 1}")
            break
    
    if (epoch + 1) % 10 == 0:
        with torch.no_grad():
            (x_dbg,) = next(iter(val_loader))
            x_dbg = x_dbg.to(device)
            recon_dbg, mu_dbg, logvar_dbg = vae(x_dbg, sample=False)
            
            # Check reconstruction quality per dimension
            per_dim_error = ((recon_dbg - x_dbg) ** 2).mean(dim=0)
            logging.info(f"Per-dim MSE: {per_dim_error.cpu().numpy()}")
    
    if (epoch + 1) % 20 == 0:
        torch.save(vae.state_dict(), os.path.join(save_dir, f"vae_epoch{epoch+1}.pt"))
    
    logging.info(f"Epoch {epoch+1:03d} | Train: {avg_train:.6f} | Val: {avg_val:.6f} | "
                f"Recon: {avg_recon:.6f} | KL: {avg_kl:.6f} | KL_w: {kl_weight:.6f} | "
                f"LR: {optimizer.param_groups[0]['lr']:.6f}")

torch.save(vae.state_dict(), os.path.join(save_dir, "vae_last.pt"))


with open(os.path.join(save_dir, "norm_stats.pkl"), "wb") as f:
    pickle.dump({
        "mean_cont": median.cpu().numpy(),  # Use median instead of mean
        "std_cont": mad.cpu().numpy() * 1.4826,  # Use MAD-based std
        "robust": True
    }, f)

# Save losses
np.save(os.path.join(save_dir, "train_losses.npy"), np.array(train_losses))
np.save(os.path.join(save_dir, "val_losses.npy"), np.array(val_losses))
np.save(os.path.join(save_dir, "recon_losses.npy"), np.array(recon_losses))
np.save(os.path.join(save_dir, "kl_losses.npy"), np.array(kl_losses))

import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

axes[0, 0].plot(train_losses, label='Train Total')
axes[0, 0].plot(val_losses, label='Val Total')
axes[0, 0].set_title('Total Loss')
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].legend()
axes[0, 0].grid(True)

axes[0, 1].plot(recon_losses)
axes[0, 1].set_title('Reconstruction Loss')
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].grid(True)

axes[1, 0].plot(kl_losses)
axes[1, 0].set_title('KL Loss')
axes[1, 0].set_xlabel('Epoch')
axes[1, 0].grid(True)

lrs = []
test_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optim.Adam(vae.parameters(), lr=config["training"]["lr"] * 0.5),
    T_0=20, T_mult=2, eta_min=1e-6
)
for _ in range(len(train_losses)):
    lrs.append(test_scheduler.get_last_lr()[0])
    test_scheduler.step()

axes[1, 1].plot(lrs)
axes[1, 1].set_title('Learning Rate Schedule')
axes[1, 1].set_xlabel('Epoch')
axes[1, 1].grid(True)

plt.tight_layout()
plt.savefig(os.path.join(save_dir, "training_curves.png"))
plt.close()

logging.info("Training completed successfully!")
logging.info(f"Best validation reconstruction loss: {best_val_loss:.6f}")