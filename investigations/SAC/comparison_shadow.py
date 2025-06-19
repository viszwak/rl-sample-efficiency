import numpy as np
import matplotlib.pyplot as plt

# File paths
offline_paths = [
    'plots/offline_sac_scores.npy',
    'plots/offline_sac_scores2.npy',
    'plots/offline_sac_scores3.npy',
    'plots/offline_sac_scores4.npy',
    'plots/offline_sac_scores5.npy'
]

online_paths = [
    'plots/online_sac_scores.npy',
    'plots/online_sac_score_2.npy',
    'plots/online_sac_score_3.npy',
    'plots/online_sac_score_4.npy',
    'plots/online_sac_score_5.npy'
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
plt.savefig("plots/online_vs_offline_sac_shadow.png")
plt.show()
