import pickle
import numpy as np
import matplotlib.pyplot as plt

# --- 1. Load Online SAC Data ---
# Load the list of episode rewards for online training
with open('lunarlander_online_scores.pkl', 'rb') as f:
    ll_online_scores = pickle.load(f)

# Try to load per-episode lengths (environment steps) so we can build a precise x-axis.
# If you logged the length of each episode (e.g. number of timesteps until terminal),
# save them in 'lunarlander_online_lens.pkl' as a list of ints.
try:
    with open('lunarlander_online_lens.pkl', 'rb') as f:
        ll_online_lens = pickle.load(f)  # e.g. [200, 180, 220, …]
    # Build a cumulative-sum array: at the end of episode i, how many env steps have been used.
    ll_cum_steps_online = np.cumsum(ll_online_lens)
except FileNotFoundError:
    # If you did NOT log actual episode lengths, fall back to evenly spacing total steps:
    # Adjust total_steps_ll to the true number of env steps you ran online.
    total_steps_ll = 280_246
    ll_cum_steps_online = np.array([
        int(i * (total_steps_ll / len(ll_online_scores)))
        for i in range(1, len(ll_online_scores) + 1)
    ])

# --- 2. Load Offline SAC Data ---
# The offline pickle should contain a list of (gradient_step, avg_reward) tuples, e.g.:
#     [(5000, 15.2), (10000, 18.5), ...]
with open('lunarlander_offline_scores.pkl', 'rb') as f:
    ll_offline_data = pickle.load(f)

# Unzip into two sequences:
offline_steps, offline_scores = zip(*ll_offline_data)
offline_steps   = np.array(offline_steps)   # e.g. [5000, 10000, 15000, …]
offline_scores  = np.array(offline_scores)  # e.g. [15.2, 18.5, …]

# --- 3. Plotting: “Steps” on X-Axis for Sample-Efficiency Comparison ---
plt.figure(figsize=(10, 6))

# Plot Online: x = cumulative environment steps, y = episode reward
plt.plot(
    ll_cum_steps_online,
    ll_online_scores,
    label='Online SAC',
    linewidth=2
)

# Plot Offline: x = gradient-update steps, y = evaluation reward
plt.plot(
    offline_steps,
    offline_scores,
    label='Offline SAC',
    linewidth=2
)

plt.title('LunarLanderContinuous-v2: Online vs. Offline SAC (by Steps)')
plt.xlabel('Training Steps')
plt.ylabel('Average Reward')
plt.legend()
plt.grid(True)
plt.tight_layout()

# Optionally save the figure to disk
plt.savefig('plots/lunarlander_sac_step_comparison.png', dpi=300)
print("Saved: plots/lunarlander_sac_step_comparison.png")

plt.show()
