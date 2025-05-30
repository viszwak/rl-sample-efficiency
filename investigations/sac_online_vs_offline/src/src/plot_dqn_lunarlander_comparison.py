import matplotlib.pyplot as plt
import pickle
import numpy as np

def moving_average(data, window_size=25):
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

# Load score files
with open("dqn_lunarlander_scores.pkl", "rb") as f:
    online_scores = pickle.load(f)

with open("dqn_lunarlander_offline_scores.pkl", "rb") as f:
    offline_scores = pickle.load(f)

# Apply smoothing to online
smoothed_online_scores = moving_average(online_scores, window_size=25)
smoothed_online_steps = [ep * 300 for ep in range(1, len(smoothed_online_scores) + 1)]

# Offline scores
offline_steps, offline_vals = zip(*offline_scores)

# Plot
plt.figure(figsize=(10, 6))
plt.plot(smoothed_online_steps, smoothed_online_scores, label="DQN Online")
plt.plot(offline_steps, offline_vals, label="DQN Offline")

plt.title("DQN Online vs Offline – LunarLander-v2")
plt.xlabel("Training Steps")
plt.ylabel("Average Evaluation Reward")
plt.grid(True)
plt.legend()

plt.savefig("dqn_lunarlander_online_vs_offline_smoothed.png")
print(" Saved: dqn_lunarlander_online_vs_offline_smoothed.png")
