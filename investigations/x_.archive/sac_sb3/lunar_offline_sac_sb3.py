# lunar_offline_sac_sb3.py

import numpy as np
import os
import gymnasium as gym
from gymnasium import spaces # Import spaces explicitly for creating Box objects
from stable_baselines3 import SAC
from stable_baselines3.common.buffers import ReplayBuffer
import torch

def train_offline_sb3(
    dataset_path="lunar_lander_dataset.npz",
    model_save_path="sac_lunar_offline_sb3.zip",
    total_gradient_steps=234400, # Corresponds to 100 epochs * 2344 updates/epoch
    batch_size=256,
    learning_starts=0, # Start training immediately
    tensorboard_log_dir="./sac_offline_sb3_tb/"
):
    """
    1) Loads `lunar_lander_dataset.npz`.
    2) Initializes an SB3 SAC model.
    3) Populates SB3's ReplayBuffer with the offline data.
    4) Implements a custom training loop to perform gradient updates by sampling from the ReplayBuffer.
    5) Saves the final offline SAC policy.
    """

    print("--- Starting Offline SAC Training with Stable-Baselines3 ---")

    # 1) Load the saved transitions
    print(f"Loading dataset from: {dataset_path}")
    data = np.load(dataset_path)
    obs = data["obs"]
    act = data["actions"]
    rew = data["rewards"]
    dones = data["dones"].astype(bool)
    next_obs = data["next_obs"]

    n_transitions = obs.shape[0]
    print(f"Dataset size: {n_transitions} transitions")

    # Manually define observation and action spaces for SB3
    # This bypasses the problematic environment patching in SB3
    observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(8,), dtype=np.float32)
    action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

    # 2) Initialize an SB3 SAC model
    model = SAC(
        "MlpPolicy",
        observation_space, # Pass the manually created observation_space
        action_space,      # Pass the manually created action_space
        buffer_size=n_transitions,
        learning_starts=learning_starts,
        batch_size=batch_size,
        tau=0.005,
        gamma=0.99,
        train_freq=(1, "step"),
        gradient_steps=1,
        verbose=1,
        tensorboard_log=tensorboard_log_dir,
        policy_kwargs=dict(net_arch=[256, 256]), # Match D3RLpy's typical 256, 256 layers
        seed=42,
        device="cpu" # Use CPU for broader compatibility
    )
    print("SB3 SAC model initialized.")

    # 3) Populate SB3's ReplayBuffer with the offline data
    replay_buffer = ReplayBuffer(
        buffer_size=n_transitions,
        observation_space=observation_space,
        action_space=action_space,
        device="cpu",
        n_envs=1,
        handle_timeout_termination=False # Our dataset has `dones`, not separate timeouts
    )

    print("Populating ReplayBuffer...")
    for i in range(n_transitions):
        replay_buffer.add(
            obs=obs[i],
            next_obs=next_obs[i],
            action=act[i],
            reward=rew[i],
            done=dones[i],
            infos=[{}] # SB3 expects infos list for n_envs
        )
    print("ReplayBuffer populated.")
    print(f"ReplayBuffer size: {replay_buffer.size()}")


    # 4) Implement a custom training loop
    print(f"Starting custom training loop for {total_gradient_steps} gradient steps...")
    model.n_updates = 0 # Initialize update counter for logging

    for step in range(total_gradient_steps):
        # Sample a batch from the replay buffer
        data_sample = replay_buffer.sample(batch_size, env=None) # env=None for offline

        # Perform one gradient update
        model.train(gradient_steps=1, batch_size=batch_size, replay_buffer=data_sample)

        # Log progress if needed (SB3 SAC already logs to tensorboard periodically from train)
        if total_gradient_steps > 0 and step % (total_gradient_steps // 10) == 0: # Log 10 times throughout training
            print(f"Step {step}/{total_gradient_steps} completed.")

    print("Offline training loop completed.")

    # 5) Save the final offline SAC policy
    model.save(model_save_path)
    print(f"✔ Offline SB3 SAC policy saved to: {model_save_path}")
    print(f"TensorBoard logs available at: {tensorboard_log_dir}")


if __name__ == "__main__":
    # Create the directory for TensorBoard logs if it doesn't exist
    os.makedirs("./sac_offline_sb3_tb/", exist_ok=True)
    
    train_offline_sb3()