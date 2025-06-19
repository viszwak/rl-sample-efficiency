#import pybullet_envs
import numpy as np
from sac_torch import Agent
from utils import plot_learning_curve
import numpy as np
import gym
import torch
import time
import pickle

if __name__ == '__main__':

    env_id = 'LunarLanderContinuous-v2'
    env = gym.make(env_id) 
    agent = Agent(alpha=0.0003, beta=0.0003, reward_scale=2, env_id=env_id, 
                input_dims=env.observation_space.shape, tau=0.005,
                env=env, batch_size=256, layer1_size=256, layer2_size=256,
                n_actions=env.action_space.shape[0])
    n_games = 1500
    filename = env_id + '_'+ str(n_games) + 'games_scale' + str(agent.scale) + \
                    '_clamp_on_sigma.png'
    figure_file = 'plots/' + filename

    best_score = env.reward_range[0]
    score_history = []
    load_checkpoint = False
    steps = 0
    print("Using GPU:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU name:", torch.cuda.get_device_name(0))
    start_time = time.time()  # start training timer


    print("Filling Replay Buffer with transitions...") # REPLAY BUFFER WARM-UP
    while agent.memory.mem_cntr < agent.batch_size:
        obs = env.reset()
        done = False
        while not done:
            action = env.action_space.sample()
            new_obs, reward, done, _, _ = env.step(action)
            agent.remember(obs, action, reward, new_obs, done)
            obs = new_obs
    print("Replay Buffer filled. Starting training...")
    
    for i in range(n_games):
        ep_start = time.time()  # start episode timer
        observation = env.reset()
        done = False
        truncated = False
        score = 0
        while not (done or truncated):
            action = agent.choose_action(observation)
            observation_, reward, done, truncated, info = env.step(action)
            steps += 1
            agent.remember(observation, action, reward, observation_, done)
            if not load_checkpoint:
                agent.learn()
            score += reward
            observation = observation_
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])

        if avg_score > best_score:
            best_score = avg_score
        if not load_checkpoint and i % 100 == 0:
            agent.save_models()


        print('episode ', i, 'score %.1f' % score,
                'training 100 games avg %.1f' % avg_score, 
                'steps %d' % steps)
        print(f"Episode time: {time.time() - ep_start:.2f} seconds")
    
    total_time = time.time() - start_time
    print(f"Total training time: {total_time / 60:.2f} minutes")

    if not load_checkpoint:
        x = [i+1 for i in range(n_games)]
        plot_learning_curve(x, score_history, figure_file)
        np.save('plots/online_sac_score_5.npy', np.array(score_history))

        with open("tmp/offline_sac_dataset_5.pkl", "wb") as f:
            pickle.dump(agent.memory, f)
        print("Replay buffer saved to tmp/offline_sac_dataset_5.pkl")

