import torch
import torch.nn as nn
import torch.nn.functional as F

class VAE(nn.Module):
    def __init__(self, input_dim=8, latent_dim=7, hidden_dim=2048):  # Increase to 2048
        super(VAE, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.bn2 = nn.BatchNorm1d(hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, hidden_dim // 4)
        self.bn3 = nn.BatchNorm1d(hidden_dim // 4)
        self.fc_mu = nn.Linear(hidden_dim // 4, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim // 4, latent_dim)

        self.fc_dec1 = nn.Linear(latent_dim, hidden_dim // 4)
        self.bn_dec1 = nn.BatchNorm1d(hidden_dim // 4)
        self.fc_dec2 = nn.Linear(hidden_dim // 4, hidden_dim // 2)
        self.bn_dec2 = nn.BatchNorm1d(hidden_dim // 2)
        self.fc_dec3 = nn.Linear(hidden_dim // 2, hidden_dim)
        self.bn_dec3 = nn.BatchNorm1d(hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, input_dim)

    def encode(self, x):
        h = F.relu(self.bn1(self.fc1(x)))
        h = F.relu(self.bn2(self.fc2(h)))
        h = F.relu(self.bn3(self.fc3(h)))
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = F.relu(self.bn_dec1(self.fc_dec1(z)))
        h = F.relu(self.bn_dec2(self.fc_dec2(h)))
        h = F.relu(self.bn_dec3(self.fc_dec3(h)))
        return self.fc_out(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar

    def encode_latent(self, x):
        mu, logvar = self.encode(x)
        return self.reparameterize(mu, logvar)