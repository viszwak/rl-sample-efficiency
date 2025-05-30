import pickle
import numpy as np
import gym

from sac_torch import Agent
from utils import plot_learning_curve

def evaluate_policy(agent, env, n_episodes=10):
    """Run the current policy without learning and return mean episode return."""
    scores = []
    for _ in range(n_episodes):
        state = env.reset()
        done = False
        total_reward = 0.0
        while not done:
            # no 'evaluate' flag—just ask for the action
            action = agent.choose_action(state)
            # clip to action space bounds
            action = np.clip(action, env.action_space.low, env.action_space.high)
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
        scores.append(total_reward)
    return np.mean(scores)

if __name__ == "__main__":
    #  Load  recorded transitions
    with open('lunarlander_online_dataset.pkl', 'rb') as f:
        dataset = pickle.load(f)

    #  Create env & SAC agent
    env_id = 'LunarLanderContinuous-v2'
    env = gym.make(env_id)
    agent = Agent(
        alpha=0.0003, beta=0.0003, reward_scale=2, env_id=env_id,
        input_dims=env.observation_space.shape, tau=0.005,
        env=env, batch_size=256, layer1_size=256, layer2_size=256,
        n_actions=env.action_space.shape[0]
    )

    #  Pre-fill the replay buffer
    for obs, act, rew, obs_next, done in dataset:
        agent.remember(obs, act, rew, obs_next, done)

    #  Offline update loop
    n_train_steps = 750000
    eval_interval = 5000
    eval_scores = []
    for step in range(1, n_train_steps + 1):
        agent.learn()
        if step % eval_interval == 0:
            avg_score = evaluate_policy(agent, env, n_episodes=5)
            eval_scores.append((step, avg_score))
            print(f"[Step {step:6d}] Eval avg reward: {avg_score:.2f}")
            agent.save_models()

    #  Plot evaluation curve
    steps, scores = zip(*eval_scores)
    figure_file = 'plots/lunarlander_offline_eval.png'
    plot_learning_curve(steps, scores, figure_file)
    print(f" Saved learning curve to {figure_file}")

    # Save evaluation scores for plotting later
    with open('lunarlander_offline_scores.pkl', 'wb') as f:
        pickle.dump(eval_scores, f)
    print(" Saved offline evaluation scores to 'lunarlander_offline_scores.pkl'")


