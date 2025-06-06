# lunar_offline_sac.py

import numpy as np
import os
from math import ceil

from d3rlpy.algos import SACConfig
from d3rlpy.datasets import MDPDataset
from d3rlpy.logging import TensorboardAdapterFactory
from d3rlpy.metrics.evaluators import EnvironmentEvaluator
import gymnasium as gym

def train_offline(
    dataset_path="lunar_lander_dataset.npz",
    model_save_path="sac_lunar_offline.zip",
    epochs=100,
    batch_size=256,
    tensorboard_dir="./sac_offline_tb/",
    logdir="./sac_offline_logs/"
):
    """
    1) Loads `lunar_lander_dataset.npz` (collected during your online run).
    2) Trains offline SAC (with D3RLpy 2.0.4) for `epochs` passes over the data.
    3) Logs metrics to TensorBoard under `tensorboard_dir` and performs evaluation.
    4) Saves the final offline SAC policy to `sac_lunar_offline.zip`.
    """

    # 1) Load the saved transitions
    data = np.load(dataset_path)
    obs          = data["obs"]
    act          = data["actions"]
    rew          = data["rewards"]
    dones        = data["dones"].astype(bool)
    next_obs     = data["next_obs"]

    # 2) Build a D3RLpy MDPDataset (rewards must be shape (N, 1))
    dataset = MDPDataset(
        observations=obs,
        actions=act,
        rewards=rew.reshape(-1, 1), # Reshape rewards to (N, 1) as required
        terminals=dones,
    )

    # 3) Compute how many gradient‐update steps per “epoch” and total steps
    N = obs.shape[0]
    updates_per_epoch = ceil(N / batch_size)
    total_steps = updates_per_epoch * epochs

    print(f"Dataset size: {N} transitions")
    print(f"Batch size: {batch_size}")
    print(f"Updates per epoch: {updates_per_epoch}")
    print(f"Training for {epochs} epochs → total_steps = {total_steps}")

    # 4) Create SACConfig
    # For d3rlpy==2.0.4, these parameters should be accepted.
    sac_config = SACConfig(
        batch_size=batch_size,
        action_size=2,         # LunarLanderContinuous has a 2D action space
        observation_shape=(8,), # LunarLanderContinuous has an 8D observation space
    )
    offline_sac = sac_config.create(device="cpu")

    # 5) Ensure logging folders exist
    os.makedirs(tensorboard_dir, exist_ok=True)
    os.makedirs(logdir, exist_ok=True)

    # 6) Create a TensorboardAdapterFactory instance for logging
    tensorboard_logger_factory = TensorboardAdapterFactory(
        root_dir=tensorboard_dir,
    )

    # 7) Prepare the evaluation environment
    eval_env = gym.make("LunarLanderContinuous-v3")

    # 8) Create an EnvironmentEvaluator for policy evaluation
    evaluator = EnvironmentEvaluator(
        env=eval_env,
    )

    # 9) Train (“fit”) offline SAC for total_steps gradient updates
    print("\nStarting offline SAC training...")
    offline_sac.fit(
        dataset=dataset,
        n_steps=total_steps,
        n_steps_per_epoch=updates_per_epoch,
        logger_adapter=tensorboard_logger_factory,
        evaluators={"environment": evaluator},
        save_interval=updates_per_epoch * 5,
    )
    print("Offline SAC training complete.")

    # 10) Close the evaluation environment
    eval_env.close()

    # 11) Save the final offline policy
    offline_sac.save_policy(model_save_path)
    print(f"✔ Offline SAC policy saved to: {model_save_path}")

if __name__ == "__main__":
    train_offline()