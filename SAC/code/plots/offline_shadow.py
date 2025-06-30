import numpy as np
import matplotlib.pyplot as plt

# Offline SAC score files
file_paths = [
    'plots/offline_sac_scores.npy',
    'plots/offline_sac_scores2.npy',
    'plots/offline_sac_scores3.npy',
    'plots/offline_sac_scores4.npy',
    'plots/offline_sac_scores5.npy'
]

# Plot settings
colors = ['blue', 'green', 'orange', 'red', 'purple']
labels = ['Sim 1', 'Sim 2', 'Sim 3', 'Sim 4', 'Sim 5']
eval_interval = 10000
window = 3  # smaller because lower number of points

plt.figure(figsize=(12, 7))

for i, path in enumerate(file_paths):
    scores = np.load(path)
    x = np.arange(1, len(scores) + 1) * eval_interval

    # Rolling average and std
    rolling_avg = np.convolve(scores, np.ones(window)/window, mode='valid')
    rolling_std = [np.std(scores[max(0, j-window):j]) for j in range(window, len(scores)+1)]
    x_plot = x[window-1:]

    # Plot each simulation
    plt.plot(x_plot, rolling_avg, label=labels[i], color=colors[i])
    plt.fill_between(x_plot,
                     rolling_avg - rolling_std,
                     rolling_avg + rolling_std,
                     color=colors[i], alpha=0.2)

plt.title("Offline SAC Performance - 5 Simulations")
plt.xlabel("Environment Steps")
plt.ylabel("Average Score")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("plots/offline_sac_all_shadow_plot.png")
plt.show()
