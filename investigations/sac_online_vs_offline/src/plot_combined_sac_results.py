import pickle
import matplotlib.pyplot as plt
import numpy as np
import os

# Ensure "plots/" directory exists
os.makedirs("plots", exist_ok=True)

# ------------------------------
# 1) LunarLander (Raw, by Steps)
# ------------------------------

# Load LunarLander online scores (one reward per episode)
with open('lunarlander_online_scores.pkl', 'rb') as f:
    ll_online_scores = pickle.load(f)

# Load LunarLander offline scores (list of (step, reward) tuples)
with open('lunarlander_offline_scores.pkl', 'rb') as f:
    ll_offline_data = pickle.load(f)
ll_offline_steps, ll_offline_scores = zip(*ll_offline_data)

# Use actual total training steps for online (280,246)
total_steps_ll = 280246
n_ll_episodes = len(ll_online_scores)
ll_online_steps = [
    int(i * (total_steps_ll / n_ll_episodes))
    for i in range(1, n_ll_episodes + 1)
]

# Plot raw (unsmoothed) LunarLander comparison
plt.figure(figsize=(10, 5))
plt.plot(
    ll_online_steps,
    ll_online_scores,
    label='LunarLander Online',
    linewidth=2
)
plt.plot(
    ll_offline_steps,
    ll_offline_scores,
    label='LunarLander Offline',
    linewidth=2
)

plt.title('LunarLanderContinuous-v2: SAC Performance (Raw Scores)')
plt.xlabel('Training Steps')
plt.ylabel('Reward')
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig('plots/lunarlander_sac_raw_comparison.png', dpi=300)
print("Saved: plots/lunarlander_sac_raw_comparison.png")
plt.close()

# ------------------------------
# 2) Pendulum (Raw, by Steps)
# ------------------------------

# Load Pendulum online scores (one reward per episode)
with open('pendulum_online_scores.pkl', 'rb') as f:
    pend_online_scores = pickle.load(f)

# Load Pendulum offline scores (list of (step, reward) tuples)
with open('pendulum_offline_scores.pkl', 'rb') as f:
    pend_offline_data = pickle.load(f)
pend_offline_steps, pend_offline_scores = zip(*pend_offline_data)

# Use actual total training steps for Pendulum online (200,000)
total_steps_pend = 200000
n_pend_episodes = len(pend_online_scores)
pend_online_steps = [
    int(i * (total_steps_pend / n_pend_episodes))
    for i in range(1, n_pend_episodes + 1)
]

# Plot raw (unsmoothed) Pendulum comparison
plt.figure(figsize=(10, 5))
plt.plot(
    pend_online_steps,
    pend_online_scores,
    label='Pendulum Online',
    linewidth=2
)
plt.plot(
    pend_offline_steps,
    pend_offline_scores,
    label='Pendulum Offline',
    linewidth=2
)

plt.title('Pendulum-v1: SAC Performance (Raw Scores)')
plt.xlabel('Training Steps')
plt.ylabel('Reward')
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig('plots/pendulum_sac_raw_comparison.png', dpi=300)
print("Saved: plots/pendulum_sac_raw_comparison.png")
plt.close()
