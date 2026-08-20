import argparse
import json
import pickle
import random
from pathlib import Path
from types import SimpleNamespace
from typing import List, Tuple

import numpy as np
import torch

from utils.tools import compute_task_reward_cost, create_env_parameter_list, create_sigle_envs
from utils.torch import tensor


def estimate_constraint_value_np(costs: List[float], masks: List[int], gamma: float) -> float:
    discounted = 0.0
    step_index = 1
    traj_num = 1

    for cost, mask in zip(costs, masks):
        discounted += float(cost) * (gamma ** (step_index - 1))
        if mask == 0:
            step_index = 1
            traj_num += 1
        else:
            step_index += 1

    return discounted / max(traj_num, 1)


def collect_policy_rollout(
    env,
    policy,
    running_state,
    env_name: str,
    env_parameter: float,
    min_batch_size: int,
    horizon: int,
    gamma: float,
    seed: int,
) -> Tuple[float, float]:
    num_steps = 0
    num_episodes = 0
    env_total_reward = 0.0
    costs_all: List[float] = []
    masks_all: List[int] = []

    while num_steps < min_batch_size:
        state, _ = env.reset(seed=seed)
        if running_state is not None:
            state = running_state(state)

        episode_reward = 0.0

        for t in range(horizon):
            state_var = tensor(state).unsqueeze(0)
            with torch.no_grad():
                action = policy.select_action(state_var)[0].numpy()
            action = int(action) if policy.is_disc_action else action.astype(np.float64)

            next_state, reward_raw, done, truncated, info = env.step(action)
            reward, cost = compute_task_reward_cost(env_name, env_parameter, reward_raw, info)

            if running_state is not None:
                next_state = running_state(next_state)

            mask = 0 if done else 1
            costs_all.append(float(cost))
            masks_all.append(mask)
            episode_reward += float(reward)

            if done or truncated:
                break

            state = next_state

        num_steps += (t + 1)
        num_episodes += 1
        env_total_reward += episode_reward

    avg_reward = env_total_reward / max(num_episodes, 1)
    avg_cost = estimate_constraint_value_np(costs_all, masks_all, gamma=gamma)
    return avg_reward, avg_cost


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute shared pre-adaptation baseline from initial checkpoint.")
    parser.add_argument("--model-path", required=True, help="Directory containing model.p")
    parser.add_argument("--env-name", default="Hopper")
    parser.add_argument("--env-num", type=int, default=1)
    parser.add_argument("--use-cover-set-tasks", default=False)
    parser.add_argument("--cover-set-path", default="assets/cover_set.json")
    parser.add_argument("--min-batch-size", type=int, default=200)
    parser.add_argument("--time-horizon", type=int, default=200)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", required=True, help="Output JSON file path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    model_file = Path(args.model_path) / "model_last.p"
    if not model_file.exists():
        model_file = Path(args.model_path) / "model.p"
    if not model_file.exists():
        raise FileNotFoundError(f"Model file not found: {model_file}")

    with model_file.open("rb") as f:
        policy, _, _, running_state, _ = pickle.load(f)

    policy.eval()
    policy.to(torch.device("cpu"))

    if running_state is not None:
        running_state.fix = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    env_args = SimpleNamespace(
        env_name=args.env_name,
        env_num=args.env_num,
        use_cover_set_tasks=str(args.use_cover_set_tasks).lower() in {"true", "1", "yes", "y"},
        cover_set_path=args.cover_set_path,
    )
    env, _ = create_sigle_envs(env_args)
    task_list = create_env_parameter_list(env_args) + create_env_parameter_list(env_args)

    reward_values = []
    cost_values = []

    for task_index, env_parameter in enumerate(task_list):
        rollout_seed = args.seed + task_index
        avg_reward, avg_cost = collect_policy_rollout(
            env,
            policy,
            running_state,
            args.env_name,
            env_parameter,
            args.min_batch_size,
            args.time_horizon,
            args.gamma,
            rollout_seed,
        )
        reward_values.append(avg_reward)
        cost_values.append(avg_cost)

    baseline = {
        "baseline_reward": float(np.mean(reward_values)) if reward_values else 0.0,
        "baseline_cost": float(np.mean(cost_values)) if cost_values else 0.0,
        "num_tasks": len(task_list),
        "seed": args.seed,
        "env_name": args.env_name,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(baseline, indent=2))
    print(f"Saved baseline: {output_path}")
    print(baseline)


if __name__ == "__main__":
    main()
