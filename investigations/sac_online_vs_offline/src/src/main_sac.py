import matplotlib
matplotlib.use('Agg')
import numpy as np
from sac_torch import Agent
from utils import plot_learning_curve
import numpy as np
import gym
import pickle
data = []


if __name__ == '__main__':
    env_id = 'LunarLanderContinuous-v2'
    env = gym.make(env_id )
    agent = Agent(alpha=0.0003, beta=0.0003, reward_scale=2, env_id=env_id, 
                input_dims=env.observation_space.shape, tau=0.005,
                env=env, batch_size=256, layer1_size=256, layer2_size=256,
                n_actions=env.action_space.shape[0])
    n_games = 1000
    filename = env_id + '_'+ str(n_games) + 'games_scale' + str(agent.scale) + \
                    '_clamp_on_sigma.png'
    figure_file = 'plots/' + filename

    best_score = env.reward_range[0]
    score_history = []
    load_checkpoint = False
    steps = 0
    for i in range(n_games):
        observation = env.reset()
        done = False
        truncated = False
        score = 0
        while not (done or truncated):
            action = agent.choose_action(observation)
            observation_, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            steps += 1
            agent.remember(observation, action, reward, observation_, done)
            data.append((observation, action, reward, observation_, done))
            if not load_checkpoint:
                agent.learn()
            score += reward
            observation = observation_
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])

        if i % 100 == 0 and not load_checkpoint:
            agent.save_models()
            print(f" Saved model at episode {i}")

        print('episode ', i, 'score %.1f' % score,
                'trailing 100 games avg %.1f' % avg_score, 
                'steps %d' % steps, env_id)
    if not load_checkpoint:
        x = [i+1 for i in range(n_games)]
        plot_learning_curve(x, score_history, figure_file)
    with open('./lunarlander_online_dataset.pkl', 'wb') as f:
        pickle.dump(data, f)
    print(" Dataset saved as 'lunarlander_online_dataset.pkl'")
    with open('lunarlander_online_scores.pkl', 'wb') as f:
        pickle.dump(score_history, f)
    print(" Online scores saved as 'lunarlander_online_scores.pkl'")


