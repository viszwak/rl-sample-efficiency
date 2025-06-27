#!/usr/bin/env python3
import os
import numpy as np
import matplotlib.pyplot as plt

# ----------------------------
# 1. File paths
# ----------------------------
offline_sac_paths = [
    'plots/offline_sac_scores.npy',
    'plots/offline_sac_scores2.npy',
    'plots/offline_sac_scores3.npy',
    'plots/offline_sac_scores4.npy',
    'plots/offline_sac_scores5.npy'
]

online_sac_paths = [
    'plots/online_sac_scores.npy',
    'plots/online_sac_score_2.npy',
    'plots/online_sac_score_3.npy',
    'plots/online_sac_score_4.npy',
    'plots/online_sac_score_5.npy'
]

offline_cql_paths = [
    'plots/cql_scores.npy',
    'plots/cql_scores_2.npy',
    'plots/cql_scores_3.npy',
    'plots/cql_scores_4.npy',
    'plots/cql_scores_5.npy'
]

# ----------------------------
# 2. Helpers
# ----------------------------
def load_runs(paths):
    return [np.load(p) for p in paths if os.path.isfile(p)]

def trim_to_min(runs):
    min_len = min(len(r) for r in runs)
    return [r[:min_len] for r in runs], min_len

def smooth(y, window=10):
    return np.convolve(y, np.ones(window) / window, mode='valid')

# ----------------------------
# 3. Load, trim, and cutoff
# ----------------------------
offline_sac_runs = load_runs(offline_sac_paths)
online_sac_runs  = load_runs(online_sac_paths)
offline_cql_runs = load_runs(offline_cql_paths)

if not offline_sac_runs or not online_sac_runs or not offline_cql_runs:
    raise ValueError("❌ One or more groups are empty.")

offline_sac_trim, len_off_sac = trim_to_min(offline_sac_runs)
online_sac_trim, len_on_sac   = trim_to_min(online_sac_runs)
offline_cql_trim, len_off_cql = trim_to_min(offline_cql_runs)

MAX_STEPS = 1_000_000
pts_offline = min(MAX_STEPS // 10_000, len_off_sac)
pts_online  = min(MAX_STEPS // 1_000, len_on_sac)

offline_sac_trim = [r[:pts_offline] for r in offline_sac_trim]
offline_cql_trim = [r[:pts_offline] for r in offline_cql_trim]
online_sac_trim  = [r[:pts_online]  for r in online_sac_trim]

offline_sac_avg = np.mean(offline_sac_trim, axis=0)
online_sac_avg  = np.mean(online_sac_trim,  axis=0)
offline_cql_avg = np.mean(offline_cql_trim, axis=0)

offline_steps = np.arange(1, pts_offline + 1) * 10_000
online_steps  = np.arange(1, pts_online  + 1) * 1_000
cql_steps     = offline_steps.copy()

# ----------------------------
# 4. Plotting (avg only)
# ----------------------------
window = 10
colors = dict(offline_sac='blue', online_sac='orange', offline_cql='green')

plt.figure(figsize=(12, 7))

# Offline SAC (avg only)
if len(offline_sac_avg) >= window:
    smoothed = smooth(offline_sac_avg, window)
    steps = offline_steps[window-1:window-1+len(smoothed)]
    plt.plot(steps, smoothed, color=colors['offline_sac'], linewidth=3, label='Offline SAC')

# Online SAC (avg only)
if len(online_sac_avg) >= window:
    smoothed = smooth(online_sac_avg, window)
    steps = online_steps[window-1:window-1+len(smoothed)]
    plt.plot(steps, smoothed, color=colors['online_sac'], linewidth=3, label='Online SAC')

# Offline CQL (avg only)
if len(offline_cql_avg) >= window:
    smoothed = smooth(offline_cql_avg, window)
    steps = cql_steps[window-1:window-1+len(smoothed)]
    plt.plot(steps, smoothed, color=colors['offline_cql'], linewidth=3, label='Offline CQL')

# ----------------------------
# 5. Final formatting
# ----------------------------
plt.title("SAC vs CQL Performance (Smoothed Average Only, ≤ 1M steps)")
plt.xlabel("Environment Steps")
plt.ylabel("Score")
plt.legend()
plt.grid(True, linestyle='--', linewidth=0.5)
plt.tight_layout()

out_file = "plots/sac_vs_cql_smooth_1M.png"
plt.savefig(out_file, dpi=300)
plt.show()

print(f"✅ Plot saved to: {out_file}")
