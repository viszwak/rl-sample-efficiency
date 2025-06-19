import gym
import numpy as np
import pickle
import torch
from sac_torch import Agent
from utils import plot_learning_curve
import time

if __name__ == '__main__':
    env_id = 'LunarLanderContinuous-v2'
    env = gym.make(env_id)

    agent = Agent(alpha=0.0003, beta=0.0003, reward_scale=2, env_id=env_id,
                  input_dims=env.observation_space.shape, tau=0.005,
                  env=env, batch_size=4096, layer1_size=256, layer2_size=256,
                  n_actions=env.action_space.shape[0])
    
    print("Actor device:", agent.actor.device)
    print("Critic device:", agent.critic_1.device)
    print("CUDA available:", torch.cuda.is_available())


    print("Loading offline dataset...")
    with open("tmp/offline_sac_dataset.pkl", "rb") as f:
        agent.memory = pickle.load(f)
    print(f"Loaded {agent.memory.mem_cntr} transitions.")

    n_steps = 800_000  # adjust 
    eval_interval = 20000  # evaluate every X learning steps
    score_history = []
    eval_games = 5  # episodes per evaluation

    start_time = time.time()
    print("Starting offline SAC training...")

    for i in range(n_steps):
        agent.learn()
        
        if i % 10000 == 0:
            print(f"[Step {i}] Time elapsed: {(time.time() - start_time)/60:.2f} min")

        if i % eval_interval == 0 or i == n_steps - 1:
            total_score = 0
            for _ in range(eval_games):
                obs = env.reset()
                done = False
                truncated = False
                score = 0
                while not (done or truncated):
                    action = agent.choose_action(obs)
                    obs, reward, done, truncated, _ = env.step(action)
                    score += reward
                total_score += score
            avg_score = total_score / eval_games
            score_history.append(avg_score)
            print(f"[Step {i}] Avg Eval Score: {avg_score:.2f}")

    total_time = time.time() - start_time
    print(f"Offline training completed in {total_time/60:.2f} minutes")

    agent.save_models()
    print("Offline SAC models saved.")

    # Save scores + plot
    x = [i * eval_interval for i in range(len(score_history))]
    np.save("plots/offline_sac_scores.npy", np.array(score_history))
    plot_learning_curve(x, score_history, "plots/LunarLander_offline_sac_plot.png")
