import gym
import torch as T
import numpy as np
import pickle
import os
from sac_torch_offline import Agent
from buffer import ReplayBuffer
#from utils import plot_learning_curve  # not used

def evaluate_policy(env, agent, episodes=5):
    scores = []
    for _ in range(episodes):
        obs, _ = env.reset()
        done, score = False, 0
        while not done:
            action = agent.choose_action(obs)
            obs, reward, term, trunc, _ = env.step(action)
            done = term or trunc
            score += reward
        scores.append(score)
    return np.mean(scores)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--sim', type=int, required=True)
    args = parser.parse_args()
    sim = args.sim

    env = gym.make('LunarLanderContinuous-v2')

    with open(f'dataset/sim_{sim}/replay_buffer.pkl', 'rb') as f:
        replay_buffer: ReplayBuffer = pickle.load(f)

    agent = Agent(alpha=0.001,
                  beta=3e-4,
                  input_dims=env.observation_space.shape,
                  tau=0.005,
                  env=env,
                  env_id='LunarLanderContinuous-v2',
                  n_actions=env.action_space.shape[0],
                  sim=sim)

    eval_interval = 10_000
    max_steps = 1_250_000
    scores, steps = [], []

    for step in range(1, max_steps + 1):
        agent.learn()

        if step % eval_interval == 0:
            avg_score = evaluate_policy(env, agent)
            print(f"[Step {step}] Avg Eval Score: {avg_score:.2f}")
            scores.append(avg_score)
            steps.append(step)

    os.makedirs(f"results/offline_sim_{sim}", exist_ok=True)
    np.save(f"results/offline_sim_{sim}/scores.npy", np.array(scores))
    np.save(f"results/offline_sim_{sim}/steps.npy",  np.array(steps))
    agent.save_models()

