#!/usr/bin/env python3
import os
import numpy as np
import pandas as pd

# Config
threshold = 200
window = 10
obs_dim = 8  # LunarLanderContinuous-v2

# Paths
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

# -----------------------
# Utilities
# -----------------------
def load_runs(paths):
    return [np.load(p) for p in paths if os.path.isfile(p)]

def trim_to_min(runs):
    min_len = min(len(r) for r in runs)
    return [r[:min_len] for r in runs]

def smooth(y, window):
    return np.convolve(y, np.ones(window) / window, mode='valid')

def compute_metrics(runs, step_interval):
    steps_to_thresh = []
    data_to_thresh = []
    total_steps = []
    total_data = []
    successes = []

    for r in runs:
        smoothed = smooth(r, window) if len(r) >= window else r
        reached = False
        for i, val in enumerate(smoothed):
            if val >= threshold:
                step = (i + window - 1) * step_interval
                steps_to_thresh.append(step)
                data_to_thresh.append(step * obs_dim)
                reached = True
                break
        if not reached:
            steps_to_thresh.append(np.nan)
            data_to_thresh.append(np.nan)
        total = len(r) * step_interval
        total_steps.append(total)
        total_data.append(total * obs_dim)
        successes.append(reached)

    return {
        "Steps to Threshold": np.nanmean(steps_to_thresh),
        "Data to Threshold": np.nanmean(data_to_thresh),
        "Total Steps": np.nanmean(total_steps),
        "Total Data Used": np.nanmean(total_data),
        "Success %": np.mean(successes) * 100
    }

# -----------------------
# Load & Compute
# -----------------------
online_sac = trim_to_min(load_runs(online_sac_paths))
offline_sac = trim_to_min(load_runs(offline_sac_paths))
offline_cql = trim_to_min(load_runs(offline_cql_paths))

if not online_sac or not offline_sac or not offline_cql:
    raise ValueError("❌ Missing or empty .npy score files!")

online_metrics  = compute_metrics(online_sac, 1_000)
offline_metrics = compute_metrics(offline_sac, 10_000)
cql_metrics     = compute_metrics(offline_cql, 10_000)

# -----------------------
# Assemble Table
# -----------------------
df = pd.DataFrame([
    {
        "Solved Task (≥200)": "✅ Yes" if online_metrics["Success %"] > 0 else "❌ No",
        "Success %": f"{online_metrics['Success %']:.1f}%",
        "Total Steps": int(online_metrics["Total Steps"]),
        "Total Data Used": int(online_metrics["Total Data Used"]),
        "Steps to Reach 200": (f"{int(online_metrics['Steps to Threshold'])}"
                               if not np.isnan(online_metrics["Steps to Threshold"]) else "—"),
        "Data to Reach 200": (f"{int(online_metrics['Data to Threshold'])}"
                              if not np.isnan(online_metrics["Data to Threshold"]) else "—")
    },
    {
        "Solved Task (≥200)": "✅ Yes" if offline_metrics["Success %"] > 0 else "❌ No",
        "Success %": f"{offline_metrics['Success %']:.1f}%",
        "Total Steps": int(offline_metrics["Total Steps"]),
        "Total Data Used": int(offline_metrics["Total Data Used"]),
        "Steps to Reach 200": (f"{int(offline_metrics['Steps to Threshold'])}"
                               if not np.isnan(offline_metrics["Steps to Threshold"]) else "—"),
        "Data to Reach 200": (f"{int(offline_metrics['Data to Threshold'])}"
                              if not np.isnan(offline_metrics["Data to Threshold"]) else "—")
    },
    {
        "Solved Task (≥200)": "✅ Yes" if cql_metrics["Success %"] > 0 else "❌ No",
        "Success %": f"{cql_metrics['Success %']:.1f}%",
        "Total Steps": int(cql_metrics["Total Steps"]),
        "Total Data Used": int(cql_metrics["Total Data Used"]),
        "Steps to Reach 200": (f"{int(cql_metrics['Steps to Threshold'])}"
                               if not np.isnan(cql_metrics["Steps to Threshold"]) else "—"),
        "Data to Reach 200": (f"{int(cql_metrics['Data to Threshold'])}"
                              if not np.isnan(cql_metrics["Data to Threshold"]) else "—")
    }
], index=["Online SAC", "Offline SAC", "Offline CQL"])

# -----------------------
# Display
# -----------------------
print("\nSample Efficiency Summary (Score ≥ 200):\n")
print(df)

# Optional display in notebook or ChatGPT
try:
    import ace_tools as tools
    tools.display_dataframe_to_user(name="Sample Efficiency Table", dataframe=df)
except ImportError:
    pass
