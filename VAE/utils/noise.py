import torch

def apply_noise(x, std_dev=0.05, clip=None, p=1.0, per_feature_std=None):
    """
    Add Gaussian noise for denoising autoencoding.

    Args:
        x: Tensor [..., D] (already standardized if applicable).
        std_dev: float, global noise std if per_feature_std is None.
        clip: float or None. If set, clamp to [-clip, clip].
        p: float in [0,1]. Probability to apply noise per element.
        per_feature_std: optional tensor of shape [D] for per-dim std.

    Returns:
        Noisy tensor on same device/dtype/shape as x.
    """
    if per_feature_std is not None:
        std = per_feature_std.view(*(1,) * (x.dim() - 1), -1).to(x)
    else:
        std = torch.as_tensor(std_dev, device=x.device, dtype=x.dtype)

    noise = torch.randn_like(x) * std

    if p < 1.0:
        mask = (torch.rand_like(x) < p).to(x.dtype)
        noise = noise * mask

    y = x + noise
    if clip is not None:
        y = torch.clamp(y, -clip, clip)
    return y
