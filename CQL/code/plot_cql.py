import numpy as np
import matplotlib.pyplot as plt
import os

# Paths to the saved score arrays for CQL simulations
file_paths = [
    'results/sim1/scores.npy',
    'results/sim2/scores.npy',
    'results/sim3/scores.npy',
    'results/sim4/scores.npy',
    'results/sim5/scores.npy'
]

# Plotting settings
colors = ['blue', 'green', 'orange', 'red', 'purple']
labels = ['Sim 1', 'Sim 2', 'Sim 3', 'Sim 4', 'Sim 5']
eval_interval = 10000
window = 3  # Rolling average window

plt.figure(figsize=(12, 7))

for i, path in enumerate(file_paths):
    if not os.path.exists(path):
        print(f"File not found: {path}")
        continue

    scores = np.load(path)
    x = np.arange(1, len(scores) + 1) * eval_interval

    # Compute rolling average and rolling standard deviation
    rolling_avg = np.convolve(scores, np.ones(window)/window, mode='valid')
    rolling_std = [np.std(scores[max(0, j - window):j]) for j in range(window, len(scores) + 1)]
    x_plot = x[window - 1:]

    # Plot mean with standard deviation shading
    plt.plot(x_plot, rolling_avg, label=labels[i], color=colors[i])
    plt.fill_between(x_plot,
                     rolling_avg - rolling_std,
                     rolling_avg + rolling_std,
                     color=colors[i], alpha=0.2)

plt.title("CQL Performance - 5 Simulations")
plt.xlabel("Environment Steps")
plt.ylabel("Average Score")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("plots/cql_plot.png")
plt.show()
