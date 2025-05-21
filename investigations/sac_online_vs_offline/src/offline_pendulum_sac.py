import matplotlib
matplotlib.use('Agg')            # headless plotting
import pickle
import gym
import numpy as np
from sac_torch import Agent
from utils import plot_learning_curve

# ——— Configuration ———
ENV_ID        = 'Pendulum-v1'
DATASET_FILE  = 'pendulum_online_dataset.pkl'
N_TRAIN_STEPS = 200_000        # total gradient steps
EVAL_INTERVAL = 10_000         # steps between evaluations
N_EVAL_EPIS   = 5              # episodes per evaluation

# hyper-params (you can tweak these)
ALPHA         = 3e-5
BETA          = 3e-5
REWARD_SCALE  = 1
TAU           = 0.005
BATCH_SIZE    = 256
LAYER1_SIZE   = 256
LAYER2_SIZE   = 256

if __name__ == '__main__':
    # 1) Load your offline dataset
    with open(DATASET_FILE, 'rb') as f:
        data = pickle.load(f)   # list of (obs, action, reward, obs_, done)

    # 2) Make env & Agent
    env = gym.make(ENV_ID)
    agent = Agent(
        alpha=ALPHA, beta=BETA,
        input_dims=env.observation_space.shape,
        tau=TAU,
        env=env,
        env_id=ENV_ID,
        reward_scale=REWARD_SCALE,
        batch_size=BATCH_SIZE,
        layer1_size=LAYER1_SIZE,
        layer2_size=LAYER2_SIZE,
        n_actions=env.action_space.shape[0]
    )

    # 3) Fill agent.memory with your dataset
    for (s, a, r, s2, done) in data:
        agent.memory.store_transition(s, a, r, s2, done)
    print(f"⚡ Loaded {len(data)} transitions into replay buffer.")

    # 4) Offline training loop
    eval_steps  = []
    eval_scores = []
    for step in range(1, N_TRAIN_STEPS + 1):
        agent.learn()

        # periodic evaluation
        if step % EVAL_INTERVAL == 0:
            scores = []
            for _ in range(N_EVAL_EPIS):
                obs, _ = env.reset()
                done = False
                total_r = 0
                while not done:
                    a = agent.choose_action(obs)
                    obs, r, terminated, truncated, _ = env.step(a)
                    done = terminated or truncated
                    total_r += r
                scores.append(total_r)
            mean_score = np.mean(scores)
            eval_steps.append(step)
            eval_scores.append(mean_score)
            print(f"[Step {step:6d}] eval avg reward → {mean_score:7.2f}")

    # 5) Plot and save the offline learning curve
    figure_file = f"plots/{ENV_ID}_offline_sac.png"
    plot_learning_curve(eval_steps, eval_scores, figure_file)
    print(f"✅ Offline SAC learning curve saved to '{figure_file}'")
