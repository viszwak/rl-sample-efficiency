
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def main():
    # 1) Ensure “plots” folder exists
    os.makedirs("plots", exist_ok=True)

    # 2) Load your online returns from the NumPy file (might be shape (N,1), so we squeeze)
    online_returns = np.load("online_returns.npy").squeeze()

    # 3) Convert to pandas Series for rolling-window calculations
    returns_series = pd.Series(online_returns)

    # 4) Define rolling window size (e.g. last 50 episodes for smoothing)
    window_size = 50

    # 5) Compute rolling mean and rolling standard deviation
    rolling_mean = returns_series.rolling(window=window_size, min_periods=1).mean()
    rolling_std = returns_series.rolling(window=window_size, min_periods=1).std()

    # 6) Prepare x-axis (episode indices)
    episodes = np.arange(1, len(online_returns) + 1)

    # 7) Plot everything
    plt.figure(figsize=(10, 5))

    # Raw per-episode returns (light grey, semi-transparent)
    plt.plot(episodes, online_returns, color="lightgrey", alpha=0.5, label="Episode Returns")

    # Rolling mean (blue)
    plt.plot(episodes, rolling_mean, color="blue", label=f"Rolling Mean (window={window_size})")

    # Shaded region for ±1 rolling std
    plt.fill_between(
        episodes,
        rolling_mean - rolling_std,
        rolling_mean + rolling_std,
        color="blue",
        alpha=0.3,
        label="Rolling Std Dev"
    )

    # Horizontal line at return = 200
    plt.axhline(200, color="r", linestyle="--", label="Score = 200")

    plt.xlabel("Episode")
    plt.ylabel("Return")
    plt.title("Online SAC Learning Curve on LunarLanderContinuous-v2")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # 8) Save the figure into the “plots” folder
    save_path = os.path.join("plots", "online_sac_learning_curve.png")
    plt.savefig(save_path)

    # 9) Display it
    plt.show()

    print(f"✔ Plot saved to: {save_path}")

if __name__ == "__main__":
    main()
