# train_lunar_vae_dyn.py (updated)
import os, json, pickle, logging, numpy as np, torch
import torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from model.vae_vector import VAE
from utils.noise import apply_noise

logging.basicConfig(level=logging.INFO)

with open("config/vae_lunar.json") as f:
    config = json.load(f)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seed = config["training"].get("seed", 42)
torch.manual_seed(seed); np.random.seed(seed)

# ---------------- Load buffer ----------------
with open(config["paths"]["replay_buffer"], "rb") as f:
    buf = pickle.load(f)

N = min(getattr(buf, "mem_cntr", len(buf.state_memory)), buf.state_memory.shape[0])

# Filter out hard crashes (same as v2)
good_idx = [i for i in range(N) if buf.reward_memory[i] > -100]
logging.info(f"Using {len(good_idx)}/{N} transitions (filtered crashes)")

S  = torch.tensor(buf.state_memory[good_idx],     dtype=torch.float32)  # [M,8]
S2 = torch.tensor(buf.new_state_memory[good_idx], dtype=torch.float32)  # [M,8]
A  = torch.tensor(buf.action_memory[good_idx],    dtype=torch.float32)  # [M,act_dim]
R  = torch.tensor(buf.reward_memory[good_idx],    dtype=torch.float32).unsqueeze(1) # [M,1]
D  = torch.tensor(buf.terminal_memory[good_idx],  dtype=torch.float32).unsqueeze(1) # [M,1]

Xc   = S[:,  :6]; F  = S[:,  6:]   # cont + flags (flags are passthrough for RL later)
Xc2  = S2[:, :6]; F2 = S2[:, 6:]   # next cont + flags

# Robust scaling (median/MAD) on continuous 6 dims
median = Xc.median(dim=0)[0]
mad    = (Xc - median).abs().median(dim=0)[0] + 1e-6
scale  = mad * 1.4826
Xc_std  = torch.clamp((Xc  - median) / scale,  -5, 5)
Xc2_std = torch.clamp((Xc2 - median) / scale,  -5, 5)

# Dataset: (s_t, a_t, r_t, d_t, s_{t+1})
dataset = TensorDataset(Xc_std, A, R, D, Xc2_std)

M = len(dataset)
val_sz = int(0.1*M); test_sz = int(0.1*M); train_sz = M - val_sz - test_sz
g = torch.Generator().manual_seed(seed)
train_set, val_set, test_set = random_split(dataset, [train_sz, val_sz, test_sz], generator=g)

batch_size = min(512, config["model"]["batch_size"]*2)
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True)
val_loader   = DataLoader(val_set,   batch_size=256)

# ---------------- Model + heads ----------------
z_dim   = config["model"]["latent_dim"]   
act_dim = A.shape[1]

vae = VAE(input_dim=6, latent_dim=z_dim).to(device)

class TransitionHead(nn.Module):
    def __init__(self, z_dim, a_dim, hid=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim + a_dim, hid), nn.ReLU(),
            nn.Linear(hid, hid), nn.ReLU(),
            nn.Linear(hid, z_dim)
        )
    def forward(self, z, a): return self.net(torch.cat([z, a], dim=-1))

class RewardHead(nn.Module):
    def __init__(self, z_dim, a_dim, hid=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim + a_dim, hid), nn.ReLU(),
            nn.Linear(hid, 1)
        )
    def forward(self, z, a): return self.net(torch.cat([z, a], dim=-1))

class DoneHead(nn.Module):
    def __init__(self, z_dim, a_dim, hid=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim + a_dim, hid), nn.ReLU(),
            nn.Linear(hid, 1)  # logits
        )
    def forward(self, z, a): return self.net(torch.cat([z, a], dim=-1))

T_head = TransitionHead(z_dim, act_dim).to(device)
R_head = RewardHead(z_dim, act_dim).to(device)
D_head = DoneHead(z_dim, act_dim).to(device)

# ---------------- Opt + sched ----------------
lr = config["training"]["lr"] * 0.5
wd = config["training"]["weight_decay"] * 10
params = list(vae.parameters()) + list(T_head.parameters()) + list(R_head.parameters()) + list(D_head.parameters())
opt = optim.AdamW(params, lr=lr, weight_decay=wd, betas=(0.9,0.999))

sched = optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=20, T_mult=2, eta_min=1e-6)

huber   = nn.SmoothL1Loss(reduction="mean")
bce_logits = nn.BCEWithLogitsLoss(reduction="mean")

save_dir = config["paths"]["save_dir"]
os.makedirs(save_dir, exist_ok=True)

# KL / weights
base_beta   = config["training"].get("kl_weight", 0.5) * 1e-4   
anneal_ep   = max(1, int(config["model"].get("anneal_epochs", 50)))
denoise_std = config["training"].get("denoise_std", 0.0)

lambda_dyn  = 1.0
lambda_rew  = 0.1
lambda_done = 0.05
free_nats   = 1.0  # per-latent-dim

best_val = float("inf"); patience=20; triggers=0

# ---- hoist dim weights once ----
dim_w = torch.tensor([0.5, 3.0, 0.5, 3.0, 2.5, 2.0], device=device)

# (Assumes VAE exposes encode_latent(x) -> mu)
for epoch in range(config["model"]["epochs"]):
    vae.train(); T_head.train(); R_head.train(); D_head.train()
    tot=tot_rec=tot_kl=tot_dyn=tot_rew=tot_done = 0.0

    # anneal beta (train); fixed beta for val
    beta = min(1.0, epoch/anneal_ep) * base_beta

    for xb, ab, rb, db, x2b in train_loader:
        x  = xb.to(device); a = ab.to(device)
        r  = rb.to(device); d = db.to(device)
        x2 = x2b.to(device)

        # input noise (optional)
        x_noisy = apply_noise(x, std_dev=denoise_std) if denoise_std>0 else x

        recon_x, mu, logvar = vae(x_noisy, sample=True)

        # ---- mean-normalized reconstruction ----
        rec = ((recon_x - x)**2 * dim_w.unsqueeze(0)).mean()

        # ---- free-nats KL: mean per-sample * z_dim ----
        kl_per_dim = -0.5*(1 + logvar - mu.pow(2) - logvar.exp())
        kl = torch.clamp(kl_per_dim - free_nats, min=0.0).mean() * z_dim

        # ----- Dynamics-aware targets -----
        with torch.no_grad():
            z_next_tgt = vae.encode_latent(x2)  # uses mu

        z_t = vae.encode_latent(x_noisy)       # current latent (mu)
        z_next_pred = T_head(z_t, a)
        dyn_loss = nn.functional.mse_loss(z_next_pred, z_next_tgt, reduction="mean") * z_dim

        # Reward / Done auxiliaries (light weights)
        r_pred = R_head(z_t, a)
        d_logit = D_head(z_t, a)
        rew_loss  = huber(r_pred, r)
        done_loss = bce_logits(d_logit, d)

        loss = rec + beta*kl + lambda_dyn*dyn_loss + lambda_rew*rew_loss + lambda_done*done_loss

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()

        # accumulate per-batch means
        tot      += loss.item()
        tot_rec  += rec.item()
        tot_kl   += kl.item()
        tot_dyn  += dyn_loss.item()
        tot_rew  += rew_loss.item()
        tot_done += done_loss.item()

    sched.step()

    # ---- Validation (fixed beta=base_beta), use mean-normalized losses ----
    vae.eval(); T_head.eval(); R_head.eval(); D_head.eval()
    val_tot = val_rec = val_dyn = 0.0
    with torch.no_grad():
        for xb, ab, rb, db, x2b in val_loader:
            x  = xb.to(device); a = ab.to(device); x2 = x2b.to(device)
            recon_x, mu, logvar = vae(x, sample=False)

            rec = ((recon_x - x)**2 * dim_w.unsqueeze(0)).mean()
            kl_per_dim = -0.5*(1 + logvar - mu.pow(2) - logvar.exp())
            kl = torch.clamp(kl_per_dim - free_nats, min=0.0).mean() * z_dim

            z_next_tgt = vae.encode_latent(x2)
            z_t = vae.encode_latent(x)
            z_next_pred = T_head(z_t, a)
            dyn_loss = nn.functional.mse_loss(z_next_pred, z_next_tgt, reduction="mean") * z_dim

            val_tot += (rec + base_beta*kl + lambda_dyn*dyn_loss).item()
            val_rec += rec.item(); val_dyn += dyn_loss.item()

    # average by number of batches (consistent with per-batch mean accumulation)
    val_obj = val_tot / max(1, len(val_loader))

    if val_obj < best_val:
        best_val = val_obj; triggers = 0
        logging.info(f"Val improved to {best_val:.6f}. Saving best...")
        torch.save(vae.state_dict(), os.path.join(save_dir, "vae_best.pt"))
        torch.save({"T":T_head.state_dict(), "R":R_head.state_dict(), "D":D_head.state_dict()},
                   os.path.join(save_dir, "aux_heads.pt"))
    else:
        triggers += 1
        if triggers >= patience:
            logging.info(f"Early stopping at epoch {epoch+1}")
            break

    # epoch logging averaged by number of batches
    nb = max(1, len(train_loader))
    logging.info(
        f"Epoch {epoch+1:03d} | "
        f"train_tot:{tot/nb:.4f} | val_tot:{val_obj:.4f} | "
        f"rec:{tot_rec/nb:.4f} | kl:{tot_kl/nb:.4f} | dyn:{tot_dyn/nb:.4f} | "
        f"rew:{tot_rew/nb:.4f} | done:{tot_done/nb:.4f} | beta:{beta:.6f}"
    )

# Save robust stats for encoder (renamed fields to reflect median/MAD)
with open(os.path.join(save_dir, "norm_stats.pkl"), "wb") as f:
    pickle.dump({
        "median": median.cpu().numpy(),
        "mad_scale": scale.cpu().numpy(),
        "robust": True
    }, f)

logging.info("Training completed.")
