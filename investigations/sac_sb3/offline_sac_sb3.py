# offline_sac_sb3.py

import numpy as np
import os
import gymnasium as gym
from gymnasium.spaces import Box
from math import ceil

from stable_baselines3 import SAC
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback


class OfflineEnv(gym.Env):
    """
    A dummy Gymnasium environment whose only job is to expose observation_space and action_space.
    SB3 will never actually call `step()` or `reset()` during offline training.
    """
    metadata = {"render_modes": []}

    def __init__(self, obs_space: Box, act_space: Box):
        super().__init__()
        self.observation_space = obs_space
        self.action_space = act_space

    def reset(self, seed=None, options=None):
        # Never actually used during offline training
        return np.zeros(self.observation_space.shape, dtype=np.float32), {}

    def step(self, action):
        # SB3 should not call this during offline training
        raise RuntimeError("OfflineEnv.step() should not be called during offline training.")

    def render(self):
        pass

    def close(self):
        pass


def train_offline_sac_sb3(
    dataset_path="lunar_lander_dataset.npz",
    model_save_path="sac_lunar_offline_sb3.zip",
    total_gradient_steps=200_000,
    batch_size=256,
    eval_freq_steps=10_000,
    n_eval_episodes=5,
    tensorboard_log="./sac_offline_sb3_tb/"
):
    """
    1) Load transitions from `lunar_lander_dataset.npz`.
    2) Flatten obs to shape (N, 8) if necessary.
    3) Create a dummy OfflineEnv and fill an SB3 ReplayBuffer with offline data.
    4) Train SAC purely offline (no env interactions) for `total_gradient_steps` gradient updates.
    5) Use EvalCallback on the real LunarLanderContinuous-v2 to log performance every `eval_freq_steps`.
    6) Save final policy to `sac_lunar_offline_sb3.zip`.
    """

    # --- 1) Load the saved transitions ---
    data = np.load(dataset_path)
    obs = data["obs"]  # shape (N, 8) or (N, 1, 8)
    if obs.ndim == 3 and obs.shape[1] == 1:
        obs = obs.reshape(obs.shape[0], obs.shape[2])
    else:
        obs = obs.reshape(obs.shape[0], -1)

    actions = data["actions"]  # (N, 2)
    rewards = data["rewards"]  # (N,)
    next_obs = data["next_obs"]
    if next_obs.ndim == 3 and next_obs.shape[1] == 1:
        next_obs = next_obs.reshape(next_obs.shape[0], next_obs.shape[2])
    else:
        next_obs = next_obs.reshape(next_obs.shape[0], -1)

    dones = data["dones"].astype(bool)  # (N,)

    N = obs.shape[0]
    print(f"Loaded dataset with {N} transitions; each obs is shape {obs.shape[1:]}.")

    # --- 2) Define observation & action spaces using Gymnasium Box ---
    obs_dim = obs.shape[1]      # should be 8
    act_dim = actions.shape[1]  # should be 2
    obs_space = Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
    act_space = Box(low=-1.0, high=1.0, shape=(act_dim,), dtype=np.float32)

    # --- 3) Create a DummyVecEnv for SB3 SAC to build its networks ---
    dummy_env = DummyVecEnv([lambda: OfflineEnv(obs_space, act_space)])

    # --- 4) Initialize SAC with a buffer sized to fit the offline dataset ---
    model = SAC(
        "MlpPolicy",
        dummy_env,
        verbose=1,
        buffer_size=N,        # exactly holds all transitions
        batch_size=batch_size,
        learning_rate=3e-4,
        train_freq=1,
        gradient_steps=1,
        tensorboard_log=tensorboard_log,
        device="cpu"          # or "cuda" if you have a GPU
    )

    # Replace SB3’s replay buffer with our offline data
    model.replay_buffer = ReplayBuffer(
        buffer_size=N,
        observation_space=obs_space,
        action_space=act_space,
        device=model.device,
    )

    # Fill the buffer: wrap each entry in a batch dimension (1, ...)
    for i in range(N):
        obs_i = obs[i : i + 1]             # shape (1, 8)
        next_obs_i = next_obs[i : i + 1]   # shape (1, 8)
        action_i = actions[i : i + 1]      # shape (1, 2)
        reward_i = np.array([rewards[i]])  # shape (1,)
        done_i = np.array([dones[i]], dtype=bool)  # shape (1,)

        model.replay_buffer.add(obs_i, next_obs_i, action_i, reward_i, done_i, {})

    # --- 5) Set up the real evaluation environment & EvalCallback ---
    eval_env = gym.make("LunarLanderContinuous-v2")
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=None,
        log_path="./sac_offline_sb3_eval_logs/",
        eval_freq=eval_freq_steps,
        n_eval_episodes=n_eval_episodes,
        deterministic=True,
        render=False
    )

    # --- 6) Train purely offline ---
    #   - total_timesteps=0 ensures no env.step() during training
    #   - gradient_steps=total_gradient_steps requests that many gradient updates
    model.learn(
        total_timesteps=0,
        gradient_steps=total_gradient_steps,
        callback=eval_callback,
        log_interval=10,
        reset_num_timesteps=False
    )

    # --- 7) Save final offline policy ---
    model.save(model_save_path)
    print(f"✔ Offline SAC policy saved to: {model_save_path}")

    eval_env.close()
    dummy_env.close()


if __name__ == "__main__":
    # Create logging directories if they don’t exist
    os.makedirs("./sac_offline_sb3_eval_logs/", exist_ok=True)
    os.makedirs("./sac_offline_sb3_tb/", exist_ok=True)

    train_offline_sac_sb3(
        dataset_path="lunar_lander_dataset.npz",
        model_save_path="sac_lunar_offline_sb3.zip",
        total_gradient_steps=200_000,   # change if you want more/fewer updates
        batch_size=256,
        eval_freq_steps=10_000,         # evaluate every 10k gradient steps
        n_eval_episodes=5,
        tensorboard_log="./sac_offline_sb3_tb/"
    )
