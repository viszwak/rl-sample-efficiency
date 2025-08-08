import argparse
import os
import gym
import torch as T
import numpy as np
import pickle
from contextlib import nullcontext

from sac_torch_cql import SAC_CQL
from buffer import ReplayBuffer
from utils import plot_learning_curve


def evaluate_policy(env, agent, episodes=5):
    scores = []
    # no gradients + deterministic actions during eval
    with T.no_grad():
        for _ in range(episodes):
            reset_out = env.reset()
            obs = reset_out[0] if isinstance(reset_out, tuple) else reset_out
            done = False
            score = 0.0
            while not done:
                action = agent.select_action(obs, deterministic=True)
                step_out = env.step(action)
                if len(step_out) == 5:
                    obs, reward, terminated, truncated, _ = step_out
                    done = terminated or truncated
                else:
                    obs, reward, done, _ = step_out
                score += float(reward)
            scores.append(score)
    return float(np.mean(scores))


def set_seed(seed, env=None):
    np.random.seed(seed)
    T.manual_seed(seed)
    if T.cuda.is_available():
        T.cuda.manual_seed_all(seed)
    try:
        if env is not None:
            env.reset(seed=seed)
            env.action_space.seed(seed)
            env.observation_space.seed(seed)
    except Exception:
        pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sim', type=int, default=1, help='Simulation ID')
    parser.add_argument('--buffer', type=str, default='dataset/medium_sim_1/replay_buffer.pkl')
    parser.add_argument('--results', type=str, default=None)
    parser.add_argument('--eval_every', type=int, default=10_000)
    parser.add_argument('--max_steps', type=int, default=700_000)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    sim = args.sim
    results_dir = args.results or f'results/medium_sim_{sim}'
    os.makedirs(results_dir, exist_ok=True)

    # ---- Load offline buffer (read-only) ----
    with open(args.buffer, 'rb') as f:
        replay_buffer: ReplayBuffer = pickle.load(f)

    env_id = 'LunarLanderContinuous-v2'
    env = gym.make(env_id)
    set_seed(args.seed, env)

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
        # you can tweak these via kwargs if you want:
        # alpha=0.2, cql_alpha_init=5.0, num_cql_random=10, bc_warmup_steps=20_000
    )

    scores, steps = [], []
    eval_interval = args.eval_every
    max_steps = args.max_steps
    batch_size = args.batch_size

    best_score = -np.inf
    best_path = os.path.join(results_dir, 'best.pt')

    # slightly faster printing without flushing every step
    print_ctx = nullcontext()

    for step in range(1, max_steps + 1):
        agent.train(replay_buffer, batch_size=batch_size)

        if step % eval_interval == 0:
            avg_score = evaluate_policy(env, agent, episodes=5)
            cql_q1 = agent.last_cql_q1_penalty
            cql_q2 = agent.last_cql_q2_penalty
            cql_alpha = agent.cql_alpha.item()

            with print_ctx:
                print(f"[Sim {sim} | Step {step}] Avg Eval Score: {avg_score:.2f}")
                print(f"   CQL Q1 penalty: {cql_q1:.2f} | CQL Q2 penalty: {cql_q2:.2f} | CQL alpha: {cql_alpha:.4f}")

            scores.append(avg_score)
            steps.append(step)

            # save best checkpoint
            if avg_score > best_score:
                best_score = avg_score
                # lightweight checkpoint: just actor/critics + alpha param
                agent.save_models()
                T.save(
                    {
                        'step': step,
                        'best_score': best_score,
                        'log_cql_alpha': agent.log_cql_alpha.detach().cpu(),
                    },
                    best_path
                )

    # final save
    agent.save_models()
    np.save(os.path.join(results_dir, 'scores.npy'), np.array(scores))
    np.save(os.path.join(results_dir, 'steps.npy'),  np.array(steps))

    # optional plot if your utils has it
    try:
        plot_learning_curve(steps, scores, os.path.join(results_dir, 'learning_curve.png'))
    except Exception as e:
        print(f"Plotting failed (ignored): {e}")
