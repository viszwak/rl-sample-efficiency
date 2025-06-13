import numpy as np
import matplotlib.pyplot as plt

# === Load Online SAC scores ===
online_scores = np.load('plots/online_sac_scores.npy')
online_total_steps = 1285160  # from your training logs
x_online = np.linspace(0, online_total_steps, len(online_scores))

window_online = 50
avg_online = np.convolve(online_scores, np.ones(window_online)/window_online, mode='valid')
std_online = [np.std(online_scores[max(0, i-window_online):i]) for i in range(window_online, len(online_scores)+1)]
x_online_plot = x_online[window_online-1:]

# === Load Offline SAC scores ===
offline_scores = np.load('plots/offline_sac_scores.npy')  # <-- your filename
eval_interval = 10000  # evaluation frequency during training
x_offline = np.arange(1, len(offline_scores) + 1) * eval_interval

window_offline = 5
avg_offline = np.convolve(offline_scores, np.ones(window_offline)/window_offline, mode='valid')
std_offline = [np.std(offline_scores[max(0, i-window_offline):i]) for i in range(window_offline, len(offline_scores)+1)]
x_offline_plot = x_offline[window_offline-1:]

# === Plot ===
plt.figure(figsize=(12, 6))
plt.plot(x_online_plot, avg_online, label='Online SAC', color='blue')
plt.fill_between(x_online_plot, avg_online - std_online, avg_online + std_online, color='blue', alpha=0.3)

plt.plot(x_offline_plot, avg_offline, label='Offline SAC', color='green')
plt.fill_between(x_offline_plot, avg_offline - std_offline, avg_offline + std_offline, color='green', alpha=0.3)

plt.title("Online vs Offline SAC Performance (Aligned by Training Steps)")
plt.xlabel("Training Steps")
plt.ylabel("Score")
plt.legend()
plt.grid(True)
plt.savefig("plots/sac_online_vs_offline_rescaled.png")
plt.show()
