# lunar_online_sac.py

import gym
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import EvalCallback

def collect_and_train(
    env_id="LunarLanderContinuous-v2",
    total_timesteps=600_000,
    dataset_path="lunar_lander_dataset.npz",
    model_path="sac_lunar_online.zip",
    eval_freq=50_000,
    n_eval_episodes=5
):
    """
    1) Trains SAC online on LunarLanderContinuous-v2.
    2) Collects all transitions into Python lists.
    3) Saves the dataset (obs, actions, rewards, next_obs, dones) to a .npz.
    4) Logs per-episode returns to an .npy file.
    5) Clears env/logger before saving the final model to avoid pickling errors.
    """

    # --- 1) Create environment and eval_env ---
    env = gym.make(env_id)
    eval_env = gym.make(env_id)

    # --- 2) Prepare storage buffers for offline dataset ---
    obs_buf = []
    act_buf = []
    rew_buf = []
    next_obs_buf = []
    done_buf = []

    # Also keep track of per-episode returns:
    episode_rewards = []
    current_ep_reward = 0.0

    # --- 3) Instantiate SAC for online training ---
    model = SAC(
        "MlpPolicy",
        env,
        verbose=1,
        batch_size=256,
        buffer_size=int(1e6),
        learning_rate=3e-4,
        train_freq=1,
        gradient_steps=1,
        ent_coef="auto",
        tau=0.005,
        gamma=0.99,
        seed=0,
        tensorboard_log="./sac_online_tb/"
    )

    # --- 4) Set up an EvalCallback that only logs eval results (no best_model_save_path) ---
    eval_callback = EvalCallback(
        eval_env,
        # We omit best_model_save_path to avoid any mid-training save.
        log_path="./sac_online_eval_logs/",
        eval_freq=eval_freq,
        n_eval_episodes=n_eval_episodes,
        deterministic=True,
        render=False
    )

    # --- 5) Monkey-patch SAC’s internal _on_step() so we can copy the latest transition ---
    original_on_step = model._on_step  # keep a reference to the original

    def patched_on_step() -> bool:
        """
        1. Call the original _on_step() (which handles replay_buffer.add & training).
        2. Immediately grab the most-recent transition from model.replay_buffer and append to our lists.
        3. Keep track of per-episode reward and detect end of episode.
        """
        # 1) Run the original (this inserts into replay buffer and updates networks)
        ret = original_on_step()

        # 2) Grab the replay buffer directly from the model
        replay_buffer = model.replay_buffer

        # The most recent transition sits at index (pos - 1)
        idx = (replay_buffer.pos - 1) % replay_buffer.buffer_size

        # Copy out the data
        obs_buf.append(replay_buffer.observations[idx].copy())
        act_buf.append(replay_buffer.actions[idx].copy())
        rew = replay_buffer.rewards[idx]
        rew_buf.append(rew)
        next_obs_buf.append(replay_buffer.next_observations[idx].copy())
        done_flag = replay_buffer.dones[idx]
        done_buf.append(done_flag)

        # 3) Update per-episode reward
        nonlocal current_ep_reward
        current_ep_reward += rew

        # If this transition ended an episode, record it and reset
        if done_flag:
            episode_rewards.append(current_ep_reward)
            current_ep_reward = 0.0

        return ret

    # Overwrite model._on_step with our patched version
    model._on_step = patched_on_step  # type: ignore

    # --- 6) Train online SAC (this also triggers patched_on_step each step) ---
    model.learn(
        total_timesteps=total_timesteps,
        callback=eval_callback,
        tb_log_name="sac_online_run"
    )

    # --- 7) Before saving, clear out env/logger to avoid pickling errors ---
    model.env      = None
    model.eval_env = None
    model.set_logger(None)

    # --- 8) Save the final online SAC model (clean) ---
    model.save(model_path)
    print(f"✔ Online SAC model saved to: {model_path}")

    # --- 9) Convert buffers to NumPy arrays and save to .npz ---
    obs_arr = np.array(obs_buf)
    act_arr = np.array(act_buf)
    rew_arr = np.array(rew_buf)
    next_obs_arr = np.array(next_obs_buf)
    done_arr = np.array(done_buf)

    np.savez(
        dataset_path,
        obs=obs_arr,
        actions=act_arr,
        rewards=rew_arr,
        next_obs=next_obs_arr,
        dones=done_arr
    )
    print(f"✔ Dataset saved to: {dataset_path}")

    # --- 10) Save per-episode returns so we can plot later ---
    episode_rewards = np.array(episode_rewards)
    np.save("online_returns.npy", episode_rewards)
    print("✔ Per-episode returns saved to: online_returns.npy")

    # --- Cleanup ---
    env.close()
    eval_env.close()


if __name__ == "__main__":
    collect_and_train()
