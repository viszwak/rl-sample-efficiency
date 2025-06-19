import gym
import torch as T
import numpy as np
import pickle
from sac_torch_offline import Agent
from buffer import ReplayBuffer
from utils import plot_learning_curve
import os

def evaluate_policy(env, agent, episodes=5):
    scores = []
    for _ in range(episodes):
        obs, _ = env.reset()
        done = False
        score = 0
        while not done:
            action = agent.choose_action(obs)  # no evaluate flag
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            score += reward
        scores.append(score)
    return np.mean(scores)

if __name__ == '__main__':
    env = gym.make('LunarLanderContinuous-v2')  # Only for evaluation

    # Load saved buffer
    with open('tmp/offline_sac_dataset.pkl', 'rb') as f:
        replay_buffer: ReplayBuffer = pickle.load(f)

    # Setup agent with your actual constructor
    agent = Agent(alpha=0.001,
                  beta=3e-4,
                  input_dims=env.observation_space.shape,
                  tau=0.005,
                  env=env,
                  env_id='LunarLanderContinuous-v2',
                  n_actions=env.action_space.shape[0])

    eval_interval = 10000
    max_steps = 1_250_000
    batch_size = 256  # may not be used if agent uses internal batch_size

    scores, steps = [], []

    for step in range(1, max_steps + 1):
        agent.learn()  # no arguments

        if step % eval_interval == 0:
            avg_score = evaluate_policy(env, agent, episodes=5)
            print(f"[Step {step}] Avg Eval Score: {avg_score:.2f}")
            scores.append(avg_score)
            steps.append(step)

    # Save scores
    np.save('plots/offline_sac_scores5.npy', np.array(scores))
    plot_learning_curve(steps, scores, 'plots/offline_sac.png')
