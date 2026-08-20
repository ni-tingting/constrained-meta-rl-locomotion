import argparse
import pickle
import random
from pathlib import Path

import numpy as np
import torch

from utils.tools import compute_task_reward_cost, create_sigle_envs
from utils.torch import tensor


def sample_truncated_gaussian(count: int, mean: float, variance: float, low: float, high: float, seed: int):
    rng = np.random.default_rng(seed)
    std = float(np.sqrt(variance))
    values = []
    while len(values) < count:
        batch = rng.normal(loc=mean, scale=std, size=count)
        accepted = batch[(batch >= low) & (batch <= high)]
        values.extend(accepted.tolist())
    return values[:count]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate meta policy on 100 random environments sampled from truncated Gaussian."
    )
    parser.add_argument("--model-path", required=True, help="Directory containing model.p or model_last.p")
    parser.add_argument("--env-name", default="Hopper")
    parser.add_argument("--num-envs", type=int, default=100)
    parser.add_argument("--num-trajectories", type=int, default=100)
    parser.add_argument("--time-horizon", type=int, default=200)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--mean", type=float, default=0.5)
    parser.add_argument("--variance", type=float, default=0.1)
    parser.add_argument("--low", type=float, default=0.0)
    parser.add_argument("--high", type=float, default=1.0)
    parser.add_argument("--mean-action", action="store_true", default=True)
    return parser.parse_args()


def load_policy(model_dir: Path):
    model_last = model_dir / "model_last.p"
    model_file = model_last if model_last.exists() else model_dir / "model.p"
    if not model_file.exists():
        raise FileNotFoundError(f"No model.p or model_last.p in {model_dir}")
    with model_file.open("rb") as file:
        policy_net, _, _, running_state, _ = pickle.load(file)
    policy_net.eval()
    policy_net.to(torch.device("cpu"))
    return policy_net, running_state


def evaluate_policy_on_env(
    env,
    policy,
    running_state,
    env_name,
    env_parameter,
    gamma,
    horizon,
    num_trajectories,
    seed,
    mean_action,
):
    episode_rewards = []
    episode_costs = []

    previous_fix = None
    if running_state is not None and hasattr(running_state, "fix"):
        previous_fix = running_state.fix
        running_state.fix = True

    try:
        for i in range(num_trajectories):
            state, _ = env.reset(seed=seed + i)
            if running_state is not None:
                state = running_state(state)

            reward_sum = 0.0
            discounted_cost_sum = 0.0

            for t in range(horizon):
                state_var = tensor(state).unsqueeze(0)
                with torch.no_grad():
                    if mean_action:
                        action = policy(state_var)[0][0].numpy()
                    else:
                        action = policy.select_action(state_var)[0].numpy()

                action = int(action) if policy.is_disc_action else action.astype(np.float64)
                step_result = env.step(action)
                if len(step_result) == 6:
                    next_state, reward_raw, _, done, truncated, info = step_result
                else:
                    next_state, reward_raw, done, truncated, info = step_result

                reward, cost = compute_task_reward_cost(env_name, env_parameter, reward_raw, info)
                reward_sum += float(reward)
                discounted_cost_sum += float(cost) * (gamma ** t)

                if done or truncated:
                    break

                if running_state is not None:
                    next_state = running_state(next_state)
                state = next_state

            episode_rewards.append(reward_sum)
            episode_costs.append(discounted_cost_sum)
    finally:
        if previous_fix is not None:
            running_state.fix = previous_fix

    return {
        "avg_reward": float(np.mean(episode_rewards)),
        "std_reward": float(np.std(episode_rewards)),
        "avg_discounted_cost": float(np.mean(episode_costs)),
        "std_discounted_cost": float(np.std(episode_costs)),
    }


def main() -> None:
    args = parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    model_dir = Path(args.model_path).expanduser().resolve()
    policy, running_state = load_policy(model_dir)

    env_args = argparse.Namespace(env_name=args.env_name, env_num=1, seed=args.seed)
    env, _ = create_sigle_envs(env_args)

    env_parameters = sample_truncated_gaussian(
        count=args.num_envs,
        mean=args.mean,
        variance=args.variance,
        low=args.low,
        high=args.high,
        seed=args.seed,
    )

    reward_means = []
    reward_stds = []
    cost_means = []
    cost_stds = []
    count_cost_above = 0

    for index, env_parameter in enumerate(env_parameters):
        stats = evaluate_policy_on_env(
            env,
            policy,
            running_state,
            args.env_name,
            env_parameter,
            args.gamma,
            args.time_horizon,
            args.num_trajectories,
            args.seed + index * 1000,
            args.mean_action,
        )
        reward_means.append(stats["avg_reward"])
        reward_stds.append(stats["std_reward"])
        cost_means.append(stats["avg_discounted_cost"])
        cost_stds.append(stats["std_discounted_cost"])
        if stats["avg_discounted_cost"] > 5.0:
            count_cost_above += 1
        print(
            f"env {index:03d} velocity={env_parameter:.6f} "
            f"reward_mean={stats['avg_reward']:.6f} reward_std={stats['std_reward']:.6f} "
            f"cost_mean={stats['avg_discounted_cost']:.6f} cost_std={stats['std_discounted_cost']:.6f}"
        )

    print("\nSummary over environments")
    print(
        f"reward_mean_avg={float(np.mean(reward_means)):.6f} "
        f"reward_std_avg={float(np.mean(reward_stds)):.6f} "
        f"cost_mean_avg={float(np.mean(cost_means)):.6f} "
        f"cost_std_avg={float(np.mean(cost_stds)):.6f}"
    )
    print(f"envs_with_cost_mean_above_5={count_cost_above}/{args.num_envs}")


if __name__ == "__main__":
    main()
