# encode_dataset.py  (mem_cntr-aware + preserve original mem_size)
import os, pickle, numpy as np, torch
from model.vae_vector import VAE

vae_model_path = "results/vae_lunar/vae_best.pt"
norm_stats_path = "results/vae_lunar/norm_stats.pkl"
buffer_path = "dataset/unbiased_sim_1/replay_buffer.pkl"
save_path = "dataset/unbiased_sim_1/replay_buffer_vae.pkl"
batch_size = 2048

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load VAE (input_dim=6; latent inferred)
ckpt = torch.load(vae_model_path, map_location=device)
z_dim = ckpt["fc_mu.weight"].shape[0] if isinstance(ckpt, dict) and "fc_mu.weight" in ckpt else 4
vae = VAE(input_dim=6, latent_dim=z_dim).to(device)
vae.load_state_dict(ckpt)
vae.eval()
for p in vae.parameters(): p.requires_grad = False

# Load scaler (continuous 6 dims)
with open(norm_stats_path, "rb") as f:
    stats = pickle.load(f)
if "mean_cont" in stats:
    mean_cont = torch.tensor(stats["mean_cont"], dtype=torch.float32, device=device)
    std_cont  = torch.tensor(stats["std_cont"],  dtype=torch.float32, device=device)
else:
    mean_full = torch.tensor(stats["mean"], dtype=torch.float32, device=device)
    std_full  = torch.tensor(stats["std"],  dtype=torch.float32, device=device)
    mean_cont, std_cont = mean_full[:6], std_full[:6]

# Load buffer
with open(buffer_path, "rb") as f:
    buffer = pickle.load(f)

mem_size  = buffer.state_memory.shape[0]
filled    = getattr(buffer, "mem_cntr", mem_size)  # number of valid rows
new_dim   = z_dim + 2

# Preallocate to ORIGINAL mem_size so shapes stay compatible
latent_states      = np.zeros((mem_size, new_dim), dtype=np.float32)
latent_next_states = np.zeros((mem_size, new_dim), dtype=np.float32)

# Encode ONLY the filled rows
for start in range(0, filled, batch_size):
    end = min(start + batch_size, filled)

    states      = torch.tensor(buffer.state_memory[start:end],     dtype=torch.float32, device=device)
    next_states = torch.tensor(buffer.new_state_memory[start:end], dtype=torch.float32, device=device)

    x_cont      = states[:, :6]
    flags       = states[:, 6:]
    x_cont_next = next_states[:, :6]
    flags_next  = next_states[:, 6:]

    x_cont_std      = (x_cont      - mean_cont) / (std_cont + 1e-6)
    x_cont_next_std = (x_cont_next - mean_cont) / (std_cont + 1e-6)

    with torch.no_grad():
        mu, _      = vae.encode(x_cont_std)       # deterministic latents
        mu_next, _ = vae.encode(x_cont_next_std)

        s_latent      = torch.cat([mu, flags], dim=-1)
        s_next_latent = torch.cat([mu_next, flags_next], dim=-1)

    latent_states[start:end]      = s_latent.cpu().numpy()
    latent_next_states[start:end] = s_next_latent.cpu().numpy()

# Replace arrays in buffer; keep mem_size & mem_cntr as-is
buffer.state_memory     = latent_states
buffer.new_state_memory = latent_next_states
# Optionally expose new input shape for downstream code that reads it
buffer.input_shape = (new_dim,)  # harmless addition

# Save
os.makedirs(os.path.dirname(save_path), exist_ok=True)
with open(save_path, "wb") as f:
    pickle.dump(buffer, f, protocol=pickle.HIGHEST_PROTOCOL)

print(f"Filled transitions encoded: {filled}/{mem_size}")
print(f"Encoded state dim: {new_dim} (z={z_dim} + 2 flags)")
print(f"Saved to: {save_path}")
