import numpy as np
import matplotlib.pyplot as plt

# Load loss histories
loss_4d = np.load("results/vae_4d/losses_4d.npy")
loss_6d = np.load("results/vae_6d/losses_6d.npy")

# Plot
plt.figure(figsize=(8, 5))
plt.plot(loss_4d, label="VAE-4D", linestyle='-.', marker='o', markersize=4)
plt.plot(loss_6d, label="VAE-6D", linestyle='--', marker='s', markersize=4)

plt.title("VAE Training Loss per Epoch")
plt.xlabel("Epoch")
plt.ylabel("Total Loss (Reconstruction + KL)")
plt.grid(True)
plt.legend()
plt.tight_layout()

# Save figure
plt.savefig("plots/vae_loss_comparison.png")
print("✅ Saved: plots/vae_loss_comparison.png")
