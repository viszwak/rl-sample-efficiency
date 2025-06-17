import numpy as np
import matplotlib.pyplot as plt

# Load saved scores
scores = np.load('plots/online_sac_scores.npy')
x = np.arange(1, len(scores) + 1)

# Compute rolling average and std deviation
window = 50  # You can adjust this
rolling_avg = np.convolve(scores, np.ones(window)/window, mode='valid')
rolling_std = [np.std(scores[max(0, i-window):i]) for i in range(window, len(scores)+1)]

# Shorten x accordingly
x_plot = x[window-1:]

# Plot with shadow
plt.figure(figsize=(10, 6))
plt.plot(x_plot, rolling_avg, label='Running Avg (window=50)', color='blue')
plt.fill_between(x_plot,
                 rolling_avg - rolling_std,
                 rolling_avg + rolling_std,
                 color='blue', alpha=0.3, label='±1 std dev')
plt.title("SAC Performance with Shadow Plot")
plt.xlabel("Episode")
plt.ylabel("Score")
plt.legend()
plt.grid(True)
plt.savefig("plots/sac_shadow_plot.png")
plt.show()

##Offline shadow
import numpy as np
import matplotlib.pyplot as plt

scores = np.load('plots/offline_sac_scores.npy')
x = np.arange(1, len(scores) + 1) * 2500  # Each score every 2500 steps

window = 5  # Smaller window due to lower resolution
rolling_avg = np.convolve(scores, np.ones(window)/window, mode='valid')
rolling_std = [np.std(scores[max(0, i-window):i]) for i in range(window, len(scores)+1)]
x_plot = x[window-1:]

plt.figure(figsize=(10, 6))
plt.plot(x_plot, rolling_avg, label='Offline SAC Avg', color='green')
plt.fill_between(x_plot, rolling_avg - rolling_std, rolling_avg + rolling_std, 
                 color='green', alpha=0.3, label='±1 std dev')
plt.title("Offline SAC Shadow Plot")
plt.xlabel("Training Step")
plt.ylabel("Score")
plt.legend()
plt.grid(True)
plt.savefig("plots/offline_sac_shadow_plot.png")
plt.show()
