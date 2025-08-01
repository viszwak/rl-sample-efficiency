import os
import gym
import torch as T
import numpy as np
import pickle

from sac_torch_cql import SAC_CQL
from buffer import ReplayBuffer


def evaluate_policy(env, agent, episodes=5):
    scores = []
    for _ in range(episodes):
        obs, _ = env.reset()
        done = False
        score = 0
        while not done:
            action = agent.select_action(obs)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            score += reward
        scores.append(score)
    return np.mean(scores)


def train_cql(
    sim=0,
    cql_alpha=0.5,
    batch_size=256,
    num_samples=10,
    max_steps=350_000,
    eval_interval=10_000
):
    buffer_path = f'dataset/unbiased_sim_1/replay_buffer.pkl'
    with open(buffer_path, 'rb') as f:
        replay_buffer: ReplayBuffer = pickle.load(f)

    env_id = 'LunarLanderContinuous-v2'
    env = gym.make(env_id)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])
    device = T.device('cuda' if T.cuda.is_available() else 'cpu')

    agent = SAC_CQL(
        state_dim=state_dim,
        action_dim=action_dim,
        max_action=max_action,
        device=device,
        sim=sim,
        cql_alpha=cql_alpha
    )

    # ✅ Set the CQL sampling parameter here
    agent.num_samples = num_samples

    results_dir = f'results/unbiased_sim_optuna_{sim}'
    os.makedirs(results_dir, exist_ok=True)

    scores, steps = [], []
    best_score = -np.inf

    for step in range(1, max_steps + 1):
        agent.train(replay_buffer, batch_size=batch_size)

        if step % eval_interval == 0:
            avg_score = evaluate_policy(env, agent, episodes=5)
            print(f"[Trial {sim} | Step {step}] Avg Eval Score: {avg_score:.2f}")
            scores.append(avg_score)
            steps.append(step)
            if avg_score > best_score:
                best_score = avg_score

    agent.save_models()
    np.save(os.path.join(results_dir, 'scores.npy'), np.array(scores))
    np.save(os.path.join(results_dir, 'steps.npy'), np.array(steps))

    return best_score
