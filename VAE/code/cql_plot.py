import numpy as np
import matplotlib.pyplot as plt
import os

# Define paths to the VAE-encoded CQL score arrays
file_paths = {
    '4D VAE': 'results/sim1/scores_4st.npy',
    '6D VAE': 'results/sim1/scores_6st.npy',
}

colors = {
    '4D VAE': 'blue',
    '6D VAE': 'orange',
}

eval_interval = 10000
window = 3  # Rolling average window

plt.figure(figsize=(10, 6))

for label, path in file_paths.items():
    if not os.path.exists(path):
        print(f"Missing: {path}")
        continue

    scores = np.load(path)
    x = np.arange(1, len(scores) + 1) * eval_interval

    # Rolling average and std
    rolling_avg = np.convolve(scores, np.ones(window)/window, mode='valid')
    rolling_std = [np.std(scores[max(0, j - window):j]) for j in range(window, len(scores) + 1)]
    x_plot = x[window - 1:]

    plt.plot(x_plot, rolling_avg, label=label, color=colors[label])
    plt.fill_between(x_plot, rolling_avg - rolling_std, rolling_avg + rolling_std,
                     color=colors[label], alpha=0.2)

plt.title("CQL Performance with VAE Encoded States (4D vs 6D)")
plt.xlabel("Environment Steps")
plt.ylabel("Average Evaluation Score")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("plots/cql_vae_comparison.png")
plt.show()
