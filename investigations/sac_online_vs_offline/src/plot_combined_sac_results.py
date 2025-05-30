import pickle
import matplotlib.pyplot as plt
import numpy as np

# - Rolling average smoothing 
def smooth_curve(y, window_size=20):
    return np.convolve(y, np.ones(window_size)/window_size, mode='valid')

#  Load LunarLander scores
with open('lunarlander_online_scores.pkl', 'rb') as f:
    ll_online_scores = pickle.load(f)
with open('lunarlander_offline_scores.pkl', 'rb') as f:
    ll_offline_data = pickle.load(f)
ll_offline_steps, ll_offline_scores = zip(*ll_offline_data)

# Use actual total training steps
total_steps_ll = 772104
ll_online_steps = [int(i * (total_steps_ll / len(ll_online_scores))) for i in range(1, len(ll_online_scores) + 1)]

# Smooth the online curve
ll_online_smooth = smooth_curve(ll_online_scores, window_size=20)
ll_online_smooth_steps = ll_online_steps[:len(ll_online_smooth)]

# - Load Pendulum scores ---
with open('pendulum_online_scores.pkl', 'rb') as f:
    pend_online_scores = pickle.load(f)
with open('pendulum_offline_scores.pkl', 'rb') as f:
    pend_offline_data = pickle.load(f)
pend_offline_steps, pend_offline_scores = zip(*pend_offline_data)

total_steps_pend = 200000
pend_online_steps = [int(i * (total_steps_pend / len(pend_online_scores))) for i in range(1, len(pend_online_scores) + 1)]
pend_online_smooth = smooth_curve(pend_online_scores, window_size=20)
pend_online_smooth_steps = pend_online_steps[:len(pend_online_smooth)]

# --- Plot 1: LunarLander -
plt.figure(figsize=(10, 5))
plt.plot(ll_online_smooth_steps, ll_online_smooth, label='LunarLander Online', linewidth=2)
plt.plot(ll_offline_steps, ll_offline_scores, label='LunarLander Offline', linewidth=2)
plt.title('LunarLanderContinuous-v2: SAC Performance')
plt.xlabel('Training Steps')
plt.ylabel('Average Reward')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('plots/lunarlander_sac_comparison.png', dpi=300)
print(" Saved: plots/lunarlander_sac_comparison.png")

# --- Plot 2: Pendulum ---
plt.figure(figsize=(10, 5))
plt.plot(pend_online_smooth_steps, pend_online_smooth, label='Pendulum Online', linewidth=2)
plt.plot(pend_offline_steps, pend_offline_scores, label='Pendulum Offline', linewidth=2)
plt.title('Pendulum-v1: SAC Performance')
plt.xlabel('Training Steps')
plt.ylabel('Average Reward')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('plots/pendulum_sac_comparison.png', dpi=300)
print(" Saved: plots/pendulum_sac_comparison.png")

plt.show()
