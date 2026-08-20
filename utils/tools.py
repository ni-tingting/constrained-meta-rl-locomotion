"""
Environment construction, task sampling, and the reward / cost definitions.

Key pieces:
- ``create_sigle_envs`` / ``create_env_parameter_list``: build a MuJoCo env and
  the list of task parameters (goal velocities for Hopper / HalfCheetah /
  Swimmer, goal directions for Humanoid).
- ``sample_truncated_gaussian``: the Hopper task distribution
  (N(0.5, 0.01) truncated to [0, 1]).
- ``load_fixed_cover_set_tasks``: use a fixed task set from
  ``assets/cover_set.json`` instead of sampling.
- ``compute_task_reward_cost``: turns a raw MuJoCo step into the
  task-conditioned reward and the safety cost. This is the single definition of
  the constrained MDP -- change it here and every algorithm sees the change.
"""

from os import path
import json
import random
import numpy as np
import torch

try:
    import gymnasium as gym
except ImportError:
    import gym


HOPPER_TASK_MEAN = 0.5
HOPPER_TASK_VARIANCE = 0.01
HOPPER_TASK_STD = np.sqrt(HOPPER_TASK_VARIANCE)
HOPPER_TASK_MIN = 0.0
HOPPER_TASK_MAX = 1.0


def sample_truncated_gaussian(
    mean=HOPPER_TASK_MEAN,
    std=HOPPER_TASK_STD,
    lower=HOPPER_TASK_MIN,
    upper=HOPPER_TASK_MAX,
    rng=None,
    max_attempts=10000,
):
    generator = rng if rng is not None else random
    for _ in range(max_attempts):
        value = generator.gauss(mean, std)
        if lower <= value <= upper:
            return value
    return min(max(value, lower), upper)

def hline():
    print('==============================================')


def load_fixed_cover_set_tasks(args):
    if getattr(args, "env_name", None) != "Hopper":
        return None
    if not getattr(args, "use_cover_set_tasks", False):
        return None

    cover_set_path = getattr(args, "cover_set_path", "assets/cover_set.json")
    if not path.isabs(cover_set_path):
        cover_set_path = path.abspath(path.join(path.dirname(path.abspath(__file__)), "..", cover_set_path))

    if not path.exists(cover_set_path):
        raise FileNotFoundError(f"Cover set file not found: {cover_set_path}")

    with open(cover_set_path, "r") as file:
        raw_values = json.load(file)

    if not isinstance(raw_values, list) or len(raw_values) == 0:
        raise ValueError(f"Cover set file must contain a non-empty JSON list: {cover_set_path}")

    return [float(value) for value in raw_values]

def create_sigle_envs(args):
    env = None
    last_error = None
    for env_id in (f"{args.env_name}-v5", f"{args.env_name}-v4"):
        try:
            env = gym.make(env_id)
            break
        except Exception as error:
            last_error = error
    if env is None:
        raise RuntimeError(
            f"Failed to create MuJoCo env for {args.env_name}. Tried -v5 and -v4. Last error: {last_error}"
        )
    env_parameter_list=[]
    if args.env_name == "HalfCheetah":
        for i in range(0, args.env_num):
            env_parameter_list.append(random.uniform(0.0, 2.0))
    elif args.env_name == "Swimmer":
        for i in range(0, args.env_num):
            env_parameter_list.append(random.uniform(0.0, 1.0))
    elif args.env_name == "Humanoid":
        for i in range(0, args.env_num):
            aaaa=random.uniform(0.0, 0.9)
            env_parameter_list.append(np.cos(aaaa))
    elif args.env_name == "Hopper":
        fixed_tasks = load_fixed_cover_set_tasks(args)
        if fixed_tasks is not None:
            env_parameter_list.extend(fixed_tasks)
            print(f"Hopper tasks from cover set: {env_parameter_list}")
        else:
            for i in range(0, args.env_num):
                env_parameter_list.append(sample_truncated_gaussian())
            print(f"Hopper tasks sampled from truncated Gaussian: {env_parameter_list}")

    return env, env_parameter_list

def create_env_parameter_list(args):
    env_parameter_list=[]
    if args.env_name == "HalfCheetah":
        for i in range(0, args.env_num):
            env_parameter_list.append(random.uniform(0.0, 2.0))
    elif args.env_name == "Swimmer":
        for i in range(0, args.env_num):
            env_parameter_list.append(random.uniform(0.0, 1.0))
    elif args.env_name == "Humanoid":
        for i in range(0, args.env_num):
            aaaa=random.uniform(0.0, 0.9)
            env_parameter_list.append(np.cos(aaaa))
    elif args.env_name == "Hopper":
        fixed_tasks = load_fixed_cover_set_tasks(args)
        if fixed_tasks is not None:
            env_parameter_list.extend(fixed_tasks)
        else:
            for i in range(0, args.env_num):
                env_parameter_list.append(sample_truncated_gaussian())
    
    return env_parameter_list


def create_env_parameter_list_deterministic(args, seed_offset=0):
    rng = random.Random(int(args.seed) + int(seed_offset))
    env_parameter_list = []

    if args.env_name == "HalfCheetah":
        for _ in range(args.env_num):
            env_parameter_list.append(rng.uniform(0.0, 2.0))
    elif args.env_name == "Swimmer":
        for _ in range(args.env_num):
            env_parameter_list.append(rng.uniform(0.0, 1.0))
    elif args.env_name == "Humanoid":
        for _ in range(args.env_num):
            angle = rng.uniform(0.0, 0.9)
            env_parameter_list.append(np.cos(angle))
    elif args.env_name == "Hopper":
        for _ in range(args.env_num):
            env_parameter_list.append(sample_truncated_gaussian(rng=rng))

    return env_parameter_list


def create_meta_test_task_list(args):
    first = create_env_parameter_list_deterministic(args, seed_offset=0)
    second = create_env_parameter_list_deterministic(args, seed_offset=1)
    return first + second


def deterministic_rollout_seed(base_seed, env_index, meta_iter, offset=0):
    """
    Reproducible rollout seed for a given (task, meta-iteration).

    ``offset`` distinguishes the different rollouts taken within one
    meta-iteration (adaptation batch, second batch, evaluation batch) so they do
    not share a seed. Keeping this deterministic makes runs comparable across
    algorithms at a fixed ``--seed``.
    """
    seed = int(base_seed)
    seed = (seed * 1000003 + int(env_index) * 9176 + int(meta_iter) * 6361 + int(offset) * 811) % (2**31 - 1)
    if seed <= 0:
        seed += 12345
    return seed


def evaluate_policy_on_task(
    env,
    policy,
    running_state,
    env_name,
    env_parameter,
    gamma,
    horizon,
    num_trajectories=500,
    base_seed=0,
):
    total_reward = 0.0
    total_discounted_cost = 0.0

    previous_fix = None
    if running_state is not None and hasattr(running_state, "fix"):
        previous_fix = running_state.fix
        running_state.fix = True

    try:
        for episode_index in range(num_trajectories):
            episode_seed = int(base_seed) + episode_index
            state, _ = env.reset(seed=episode_seed)
            if running_state is not None:
                state = running_state(state)

            episode_reward = 0.0
            episode_discounted_cost = 0.0

            for t in range(horizon):
                state_var = torch.tensor(state).unsqueeze(0)
                with torch.no_grad():
                    action = policy(state_var)[0][0].numpy()

                action = int(action) if policy.is_disc_action else action.astype(np.float64)
                step_result = env.step(action)
                if len(step_result) == 6:
                    next_state, reward_raw, _, done, truncated, info = step_result
                else:
                    next_state, reward_raw, done, truncated, info = step_result

                reward, cost = compute_task_reward_cost(env_name, env_parameter, reward_raw, info)

                episode_reward += float(reward)
                episode_discounted_cost += float(cost) * (gamma ** t)

                if done or truncated:
                    break

                if running_state is not None:
                    next_state = running_state(next_state)
                state = next_state

            total_reward += episode_reward
            total_discounted_cost += episode_discounted_cost
    finally:
        if previous_fix is not None:
            running_state.fix = previous_fix

    avg_reward = total_reward / max(num_trajectories, 1)
    avg_discounted_cost = total_discounted_cost / max(num_trajectories, 1)
    return avg_reward, avg_discounted_cost


def compute_task_reward_cost(env_name, env_parameter, reward_raw, info):
    x_velocity = float(info.get('x_velocity', 0.0))
    reward_ctrl = float(info.get('reward_ctrl', 0.0))
    reward_alive = float(info.get('reward_alive', 0.0))
    reward_quadctrl = float(info.get('reward_quadctrl', 0.0))
    y_velocity = float(info.get('y_velocity', 0.0))

    if env_name == "HalfCheetah":
        reward = -abs(x_velocity - env_parameter)
        cost = -reward_ctrl
    elif env_name == "Swimmer":
        reward = -abs(x_velocity - env_parameter)
        cost = -reward_ctrl * 100.0
    elif env_name == "Humanoid":
        reward = (x_velocity * env_parameter + y_velocity * np.sqrt(1 - env_parameter**2)) * 1.25 + reward_alive + reward_quadctrl
        cost = -reward_quadctrl
    elif env_name == "Hopper":
        reward = -abs(x_velocity - env_parameter) + reward_raw - x_velocity
        cost = -(reward_raw - x_velocity - 1.0) * 100.0
    else:
        reward = float(reward_raw)
        cost = -reward_ctrl

    return reward, cost

def assets_dir():
    return path.abspath(path.join(path.dirname(path.abspath(__file__)), '../assets'))

