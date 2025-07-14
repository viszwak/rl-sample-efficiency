# noise.py
import torch

def apply_noise(x, std_dev):
    noise = torch.randn_like(x) * std_dev
    return torch.clamp(x + noise, -5, 5)