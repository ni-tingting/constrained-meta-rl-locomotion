"""
Evaluation routines shared by the ``main.py`` subcommands.

Three things live here, all of which produce numbers rather than policies:

- ``evaluate_cover_set_policies``: scores the CPO experts and the safe
  meta-policy on every cover-set task. Its output JSON is the ``U_hat`` that
  ``algos/our_algorithm.py`` consumes.
- ``compute_shared_baseline``: the common pre-adaptation reference point that
  every algorithm's test curve is measured against.
- ``evaluate_on_random_envs``: held-out generalisation check on tasks freshly
  drawn from the task distribution.

Also home to ``find_latest_model_dir``, the checkpoint-discovery helper that was
previously duplicated in four scripts.
"""

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from utils.tools import (
    compute_task_reward_cost,
    create_env_parameter_list,
    create_sigle_envs,
    evaluate_policy_on_task,
)
from utils.torch import tensor


def load_cover_set(path: Path) -> List[float]:
    """Read the cover-set JSON, validating that it is a non-empty list."""
    if not path.exists():
        raise FileNotFoundError(f"Cover set file not found: {path}")
    values = json.loads(path.read_text())
    if not isinstance(values, list) or len(values) == 0:
        raise ValueError(f"Cover set must be a non-empty JSON list: {path}")
    return [float(value) for value in values]


def find_latest_model_dir(repo_root: Path, algo_name: str, env_name: str) -> Optional[Path]:
    """
    Most recently modified run directory for ``algo_name`` on ``env_name``.

    Matches folders whose name ends in ``-<env_name>`` and that hold a usable
    checkpoint. Returns None when nothing has been trained yet.
    """
    base = repo_root / "assets" / "learned_models" / algo_name
    if not base.exists():
        return None

    candidates = []
    for run_dir in base.iterdir():
        if not run_dir.is_dir():
            continue
        if not run_dir.name.endswith(f"-{env_name}"):
            continue
        model_path = run_dir / "model.p"
        model_last = run_dir / "model_last.p"
        if model_path.exists() or model_last.exists():
            candidates.append(run_dir)

    if not candidates:
        return None

    def newest(run_dir: Path) -> float:
        stamps = [
            (run_dir / name).stat().st_mtime
            for name in ("model.p", "model_last.p")
            if (run_dir / name).exists()
        ]
        return max(stamps)

    return max(candidates, key=newest)


# --------------------------------------------------------------------------- #
# Cover-set evaluation -> U_hat
# --------------------------------------------------------------------------- #

def evaluate_cover_set_policies(
    env,
    env_name: str,
    cover_values: List[float],
    cpo_export_dir: Path,
    safemeta_policy,
    safemeta_state,
    num_trajectories: int,
    gamma: float,
    horizon: int,
    seed: int,
) -> Dict:
    """
    Score the CPO expert and the safe meta-policy on each cover-set task.

    The two policies are evaluated with offset seeds (``seed`` and
    ``seed + 1000``) so their rollouts are independent.
    """
    from algos.our_algorithm import cpo_policy_dir, load_policy_from_dir

    results = []
    for velocity in cover_values:
        cpo_policy, cpo_state = load_policy_from_dir(cpo_policy_dir(cpo_export_dir, velocity))

        cpo_reward, cpo_cost = evaluate_policy_on_task(
            env, cpo_policy, cpo_state, env_name, velocity, gamma, horizon,
            num_trajectories=num_trajectories, base_seed=seed,
        )
        meta_reward, meta_cost = evaluate_policy_on_task(
            env, safemeta_policy, safemeta_state, env_name, velocity, gamma, horizon,
            num_trajectories=num_trajectories, base_seed=seed + 1000,
        )

        results.append(
            {
                "velocity": velocity,
                "cpo": {"avg_reward": float(cpo_reward), "avg_cost": float(cpo_cost)},
                "safemeta": {"avg_reward": float(meta_reward), "avg_cost": float(meta_cost)},
            }
        )

    return {"results": results}


# --------------------------------------------------------------------------- #
# Shared pre-adaptation baseline
# --------------------------------------------------------------------------- #

def estimate_constraint_value_np(costs: List[float], masks: List[int], gamma: float) -> float:
    """
    NumPy twin of ``core.common.estimate_constraint_value``.

    Kept separate because the baseline is computed outside the torch training
    loop; ``masks[i] == 0`` marks a trajectory boundary.
    """
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
    """Sample until ``min_batch_size`` steps, returning ``(avg_reward, avg_cost)``."""
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


def compute_shared_baseline(
    policy,
    running_state,
    env_name: str,
    env_num: int,
    use_cover_set_tasks: bool,
    cover_set_path: str,
    min_batch_size: int,
    horizon: int,
    gamma: float,
    seed: int,
) -> Dict:
    """
    Average reward / cost of the un-adapted policy over the task list.

    Uses twice the task list (as the meta-test task list does) so the baseline
    is drawn from the same task count the algorithms are scored on.
    """
    if running_state is not None:
        running_state.fix = True

    env_args = SimpleNamespace(
        env_name=env_name,
        env_num=env_num,
        use_cover_set_tasks=use_cover_set_tasks,
        cover_set_path=cover_set_path,
    )
    env, _ = create_sigle_envs(env_args)
    task_list = create_env_parameter_list(env_args) + create_env_parameter_list(env_args)

    reward_values = []
    cost_values = []

    for task_index, env_parameter in enumerate(task_list):
        avg_reward, avg_cost = collect_policy_rollout(
            env, policy, running_state, env_name, env_parameter,
            min_batch_size, horizon, gamma, seed + task_index,
        )
        reward_values.append(avg_reward)
        cost_values.append(avg_cost)

    return {
        "baseline_reward": float(np.mean(reward_values)) if reward_values else 0.0,
        "baseline_cost": float(np.mean(cost_values)) if cost_values else 0.0,
        "num_tasks": len(task_list),
        "seed": seed,
        "env_name": env_name,
    }


# --------------------------------------------------------------------------- #
# Held-out random-task evaluation
# --------------------------------------------------------------------------- #

def sample_random_tasks(count: int, mean: float, variance: float, low: float, high: float, seed: int):
    """
    Draw ``count`` task parameters from a truncated Gaussian by rejection.

    Note this takes a *variance* while ``utils.tools.sample_truncated_gaussian``
    takes a standard deviation.
    """
    rng = np.random.default_rng(seed)
    std = float(np.sqrt(variance))
    values = []
    while len(values) < count:
        batch = rng.normal(loc=mean, scale=std, size=count)
        accepted = batch[(batch >= low) & (batch <= high)]
        values.extend(accepted.tolist())
    return values[:count]


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
    """Per-task reward/cost mean and std over ``num_trajectories`` rollouts."""
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


def evaluate_on_random_envs(
    policy,
    running_state,
    env_name: str,
    num_envs: int,
    num_trajectories: int,
    horizon: int,
    gamma: float,
    seed: int,
    mean: float,
    variance: float,
    low: float,
    high: float,
    mean_action: bool,
    constraint_threshold: float = 5.0,
    verbose: bool = True,
) -> Dict:
    """
    Evaluate one policy on ``num_envs`` freshly sampled tasks.

    Reports the averaged reward/cost plus how many tasks exceeded
    ``constraint_threshold`` -- the constraint-violation rate on unseen tasks.
    """
    env_args = SimpleNamespace(env_name=env_name, env_num=1, seed=seed)
    env, _ = create_sigle_envs(env_args)

    env_parameters = sample_random_tasks(num_envs, mean, variance, low, high, seed)

    reward_means, reward_stds, cost_means, cost_stds = [], [], [], []
    count_cost_above = 0

    for index, env_parameter in enumerate(env_parameters):
        stats = evaluate_policy_on_env(
            env, policy, running_state, env_name, env_parameter, gamma, horizon,
            num_trajectories, seed + index * 1000, mean_action,
        )
        reward_means.append(stats["avg_reward"])
        reward_stds.append(stats["std_reward"])
        cost_means.append(stats["avg_discounted_cost"])
        cost_stds.append(stats["std_discounted_cost"])
        if stats["avg_discounted_cost"] > constraint_threshold:
            count_cost_above += 1
        if verbose:
            print(
                f"env {index:03d} velocity={env_parameter:.6f} "
                f"reward_mean={stats['avg_reward']:.6f} reward_std={stats['std_reward']:.6f} "
                f"cost_mean={stats['avg_discounted_cost']:.6f} cost_std={stats['std_discounted_cost']:.6f}"
            )

    return {
        "reward_mean_avg": float(np.mean(reward_means)),
        "reward_std_avg": float(np.mean(reward_stds)),
        "cost_mean_avg": float(np.mean(cost_means)),
        "cost_std_avg": float(np.mean(cost_stds)),
        "envs_above_threshold": count_cost_above,
        "num_envs": num_envs,
        "constraint_threshold": constraint_threshold,
    }
