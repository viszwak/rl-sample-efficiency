import numpy as np
import matplotlib.pyplot as plt

# File paths
offline_paths = [
    'results/offline_sim_1/scores.npy',
    'results/offline_sim_2/scores.npy',
    'results/offline_sim_3/scores.npy',
    'results/offline_sim_4/scores.npy',
    'results/offline_sim_5/scores.npy'
]

online_paths = [
    'results/sim_1/scores.npy',
    'results/sim_2/scores.npy',
    'results/sim_3/scores.npy',
    'results/sim_4/scores.npy',
    'results/sim_5/scores.npy'
]

# Load and trim to min length
offline_runs = [np.load(path) for path in offline_paths]
online_runs = [np.load(path) for path in online_paths]

min_len_offline = min(len(run) for run in offline_runs)
min_len_online = min(len(run) for run in online_runs)

offline_trimmed = [r[:min_len_offline] for r in offline_runs]
online_trimmed = [r[:min_len_online] for r in online_runs]

offline_avg = np.mean(offline_trimmed, axis=0)
online_avg = np.mean(online_trimmed, axis=0)

# Step intervals
offline_steps = np.arange(1, len(offline_avg)+1) * 10000
online_steps = np.arange(1, len(online_avg)+1) * 1000

# Plot colors
offline_color = 'blue'
online_color = 'orange'
window = 3

plt.figure(figsize=(12, 7))

# Plot offline individual runs
for run in offline_trimmed:
    steps = np.arange(1, len(run)+1) * 10000
    smooth = np.convolve(run, np.ones(window)/window, mode='valid')
    plt.plot(steps[window-1:], smooth, color=offline_color, alpha=0.3)

# Plot online individual runs
for run in online_trimmed:
    steps = np.arange(1, len(run)+1) * 1000
    smooth = np.convolve(run, np.ones(window)/window, mode='valid')
    plt.plot(steps[window-1:], smooth, color=online_color, alpha=0.3)

# Plot average curves
offline_avg_smooth = np.convolve(offline_avg, np.ones(window)/window, mode='valid')
online_avg_smooth = np.convolve(online_avg, np.ones(window)/window, mode='valid')

plt.plot(offline_steps[window-1:], offline_avg_smooth, color=offline_color, linewidth=2.5, label="Offline SAC (avg)")
plt.plot(online_steps[window-1:], online_avg_smooth, color=online_color, linewidth=2.5, label="Online SAC (avg)")

# Format
plt.title("Online vs Offline SAC Performance (5 Simulations Each)")
plt.xlabel("Environment Steps")
plt.ylabel("Score")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("plots/sac_comparison(online_vs_offline).png")
plt.show()
