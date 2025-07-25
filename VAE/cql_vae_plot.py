import numpy as np
import matplotlib.pyplot as plt
import os

# Load paths
scores_path = "results/vae_sim1/scores.npy"
steps_path = "results/vae_sim1/steps.npy"

# Load data
scores = np.load(scores_path)
steps = np.load(steps_path)

# Plot
plt.figure(figsize=(10, 5))
plt.plot(steps, scores, marker='o', linestyle='-', label="CQL-VAE Evaluation Score")
plt.title("CQL with VAE-Encoded States: Evaluation Over Time")
plt.xlabel("Training Steps")
plt.ylabel("Average Evaluation Score")
plt.grid(True)
plt.legend()

# Save and show
os.makedirs("vae_plots", exist_ok=True)
plt.savefig("vae_plots/cql_vae_score_curve.png")
plt.show()
