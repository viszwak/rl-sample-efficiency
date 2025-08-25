import torch
import torch.nn as nn
import torch.nn.functional as F

class VAE(nn.Module):
    """
    VAE for LunarLander continuous state:
    - Train on the first 6 continuous dims (standardized).
    - Use deterministic μ for RL inference (encode_latent(..., deterministic=True)).
    - LayerNorm for encoder (robust to batch size 1). No norm in decoder.
    """
    def __init__(self, input_dim=6, latent_dim=4, hidden_dim=128, use_layernorm=True):
        super().__init__()
        self.logvar_min, self.logvar_max = -10.0, 10.0

        # Encoder
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.n1  = nn.LayerNorm(hidden_dim) if use_layernorm else nn.Identity()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.n2  = nn.LayerNorm(hidden_dim // 2) if use_layernorm else nn.Identity()

        self.fc_mu     = nn.Linear(hidden_dim // 2, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim // 2, latent_dim)

        # Decoder (no BN/LayerNorm)
        self.fc_dec1 = nn.Linear(latent_dim, hidden_dim // 2)
        self.fc_dec2 = nn.Linear(hidden_dim // 2, hidden_dim)
        self.fc_out  = nn.Linear(hidden_dim, input_dim)

        nn.init.zeros_(self.fc_out.bias)

    def encode(self, x):
        h = F.relu(self.n1(self.fc1(x)))
        h = F.relu(self.n2(self.fc2(h)))
        mu = self.fc_mu(h)
        logvar = torch.clamp(self.fc_logvar(h), self.logvar_min, self.logvar_max)
        return mu, logvar

    def reparameterize(self, mu, logvar, sample=True):
        if not sample:
            return mu
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = F.relu(self.fc_dec1(z))
        h = F.relu(self.fc_dec2(h))
        return self.fc_out(h)  # unbounded; inputs are z-scored

    def forward(self, x, sample=True):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar, sample=sample)
        recon_x = self.decode(z)
        return recon_x, mu, logvar

    def encode_latent(self, x, deterministic=True):
        mu, logvar = self.encode(x)
        return self.reparameterize(mu, logvar, sample=not deterministic)
