import gym
import torch as T
import numpy as np
import pickle
import os

from sac_torch_cql import SAC_CQL
from buffer import ReplayBuffer
from utils import plot_learning_curve

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

if __name__ == '__main__':
    env_id = 'LunarLanderContinuous-v2'
    env = gym.make(env_id)

    with open('tmp/offline_sac_dataset_4.pkl', 'rb') as f:
        replay_buffer: ReplayBuffer = pickle.load(f)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])
    device = T.device('cuda' if T.cuda.is_available() else 'cpu')

    agent = SAC_CQL(
        state_dim=state_dim,
        action_dim=action_dim,
        max_action=max_action,
        device=device,
        cql_alpha=0.1  # conservative penalty strength
    )

    scores, steps = [], []
    eval_interval = 10000
    max_steps = 300_000
    batch_size = 256

    for step in range(1, max_steps + 1):
        agent.train(replay_buffer, batch_size=batch_size)

        if step % eval_interval == 0:
            avg_score = evaluate_policy(env, agent, episodes=5)
            print(f"[Step {step}] Avg Eval Score: {avg_score:.2f}")
            scores.append(avg_score)
            steps.append(step)

    os.makedirs("plots", exist_ok=True)
    np.save('plots/cql_scores_4.npy', np.array(scores))
    plot_learning_curve(steps, scores, 'plots/cql_eval_4.png')
