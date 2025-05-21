import matplotlib
matplotlib.use('Agg')       # for headless plotting
import numpy as np
from sac_torch import Agent
from utils import plot_learning_curve
import gym
import pickle

if __name__ == '__main__':
    env_id = 'Pendulum-v1'
    env = gym.make(env_id)
    agent = Agent(
        alpha=3e-5, beta=3e-5, reward_scale=1,
        env_id=env_id, input_dims=env.observation_space.shape,
        tau=0.005, env=env, batch_size=256,
        layer1_size=256, layer2_size=256,
        n_actions=env.action_space.shape[0]
    )

    n_games = 1000
    figure_file = f"plots/{env_id}_{n_games}games_scale{agent.scale}.png"
    dataset_file = 'pendulum_online_dataset.pkl'

    score_history = []
    data = []
    total_steps = 0
    load_checkpoint = False

    RANDOM_STEPS = 10_000
    print(f"🔄 Warming up buffer with {RANDOM_STEPS} random steps...")

    for episode in range(n_games):
        observation, _ = env.reset()
        done = False
        score = 0

        while not done:
            if total_steps < RANDOM_STEPS:
                action = env.action_space.sample()
            else:
                action = agent.choose_action(observation)

            obs_new, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_steps += 1

            agent.remember(observation, action, reward, obs_new, done)
            data.append((observation, action, reward, obs_new, done))

            if not load_checkpoint and total_steps >= RANDOM_STEPS:
                agent.learn()

            score += reward
            observation = obs_new

        score_history.append(score)
        avg_score = np.mean(score_history[-100:])

        if episode % 100 == 0 and not load_checkpoint:
            agent.save_models()
            print(f"💾 Saved model at episode {episode}")

        print(
            f"episode {episode:4d}   "
            f"score {score:8.2f}   "
            f"avg_last_100 {avg_score:8.2f}   "
            f"steps {total_steps}"
        )

    if not load_checkpoint:
        x = list(range(1, n_games + 1))
        plot_learning_curve(x, score_history, figure_file)

    with open(dataset_file, 'wb') as f:
        pickle.dump(data, f)

    print(f"✅ Pendulum dataset saved as '{dataset_file}'")
