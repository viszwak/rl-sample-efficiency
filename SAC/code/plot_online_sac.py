import numpy as np
import matplotlib.pyplot as plt

# File paths (in order)
file_paths = [
    'results/sim_1/scores.npy',
    'results/sim_2/scores.npy',
    'results/sim_3/scores.npy',
    'results/sim_4/scores.npy',
    'results/sim_5/scores.npy'
]

# Colors and labels
colors = ['blue', 'green', 'orange', 'red', 'purple']
labels = ['Sim 1', 'Sim 2', 'Sim 3', 'Sim 4', 'Sim 5']

# Rolling window
window = 50

plt.figure(figsize=(12, 7))

for i, path in enumerate(file_paths):
    scores = np.load(path)
    x = np.arange(1, len(scores) + 1)
    
    # Compute rolling average and std
    rolling_avg = np.convolve(scores, np.ones(window)/window, mode='valid')
    rolling_std = [np.std(scores[max(0, j-window):j]) for j in range(window, len(scores)+1)]
    x_plot = x[window-1:]

    # Plot
    plt.plot(x_plot, rolling_avg, label=f'{labels[i]}', color=colors[i])
    plt.fill_between(x_plot,
                     rolling_avg - rolling_std,
                     rolling_avg + rolling_std,
                     color=colors[i], alpha=0.2)

plt.title("Online SAC Performance - 5 Simulations")
plt.xlabel("Episode")
plt.ylabel("Score")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("plots/online_sac_plot.png")
plt.show()
