import argparse
import math
import random
import pickle
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

try:
    from utils.tools import create_sigle_envs, sample_truncated_gaussian, compute_task_reward_cost
except Exception:  # pragma: no cover - optional dependency for standalone use
    create_sigle_envs = None
    sample_truncated_gaussian = None
    compute_task_reward_cost = None


@dataclass
class SafeHopperEnvArgs:
    env_name: str = "Hopper"
    env_num: int = 1
    use_cover_set_tasks: bool = False
    cover_set_path: str = "assets/cover_set.json"
    seed: int = 0


def create_safe_hopper_env(args: Optional[SafeHopperEnvArgs] = None) -> Tuple[Any, float]:
    """
    Create a Safe Hopper environment and sample a single goal velocity.

    The goal velocity is sampled from the truncated Gaussian defined in `utils.tools`.
    Returns the env and the sampled task parameter so callers can attach it to the env
    if their environment needs an explicit setter.
    """
    if args is None:
        args = SafeHopperEnvArgs()

    if create_sigle_envs is not None:
        env, env_params = create_sigle_envs(args)
        env_parameter = env_params[0] if env_params else 0.0
        return env, float(env_parameter)

    if sample_truncated_gaussian is None:
        raise RuntimeError("utils.tools is unavailable; cannot sample Hopper task parameter.")

    env_parameter = float(sample_truncated_gaussian())
    return None, env_parameter


def _load_policy_from_dir(model_dir: Path) -> Tuple[torch.nn.Module, object]:
    model_file = model_dir / "model_last.p"
    if not model_file.exists():
        model_file = model_dir / "model.p"
    if not model_file.exists():
        raise FileNotFoundError(f"No model.p or model_last.p found in {model_dir}")

    with model_file.open("rb") as handle:
        policy_net, _, _, running_state, _ = pickle.load(handle)

    policy_net.eval()
    policy_net.to(torch.device("cpu"))
    return policy_net, running_state


def load_safemeta_policy(model_dir: str) -> Tuple[torch.nn.Module, object]:
    """Load the SafeMeta policy (pi_s) and running state from a checkpoint directory."""
    policy_net, running_state = _load_policy_from_dir(Path(model_dir))
    return policy_net, running_state


def _cpo_policy_dir(export_dir: Path, velocity: float) -> Path:
    safe_label = str(velocity).replace(".", "p")
    return export_dir / f"velocity_{safe_label}"


def build_u_hat_from_cover_set(
    eval_json_path: str,
    cpo_export_dir: str,
) -> List[Dict[str, Any]]:
    """
    Build U_hat from the JSON produced by evaluate_cover_set_policies.py.

    Each entry contains:
    - policy: CPO policy for the corresponding cover-set velocity
    - u_l, v_l: reward/cost for the CPO policy
    - u_l_s, v_l_s: reward/cost for the SafeMeta policy
    """
    eval_path = Path(eval_json_path)
    export_dir = Path(cpo_export_dir)
    results = eval_path.read_text()
    payload = json.loads(results)
    if "results" not in payload:
        raise ValueError("Invalid evaluation JSON: missing 'results' key")

    u_hat: List[Dict[str, Any]] = []
    for item in payload["results"]:
        velocity = float(item["velocity"])
        cpo_policy, _ = _load_policy_from_dir(_cpo_policy_dir(export_dir, velocity))
        u_hat.append(
            {
                "policy": cpo_policy,
                "u_l": float(item["cpo"]["avg_reward"]),
                "v_l": float(item["cpo"]["avg_cost"]),
                "u_l_s": float(item["safemeta"]["avg_reward"]),
                "v_l_s": float(item["safemeta"]["avg_cost"]),
            }
        )
    return u_hat


def update_mixture_weight(m: int, v_l_s: float, epsilon: float, alpha: float, constraint_threshold: float) -> float:
    """
    Calculates the mixture weight $\alpha_{l,m}$ based on Equation 6 of the paper.

    Formulas:
    $L = some-tunable variable$
    $C_l = \frac{2v_{l,s} + (4L+9)\epsilon}{3v_{l,s}}$,

    For $m=0$:
    $\alpha_{l,1} = \frac{v_{l,s} - 2\epsilon(L+2)}{v_{l,s} - 2\epsilon(L+2) + 2/(1-\gamma)}$

    For $m \ge 1$:
    $\alpha_{l,m+1} = \frac{(v_{l,s} - (4L+9)\epsilon)\alpha_{l,1}}{v_{l,s}\alpha_{l,1} + (v_{l,s} - (4L+9)\epsilon - v_{l,s}\alpha_{l,1})C_l^m}$
    """
    L = 3
    v_l_s = max(abs(constraint_threshold - v_l_s), 1e-6)

    alpha_l_1_num = v_l_s
    alpha_l_1_den = v_l_s + 2
    if alpha_l_1_den == 0:
        alpha_l_1 = 0.0
    else:
        alpha_l_1 = alpha_l_1_num / alpha_l_1_den

    alpha_l_1 = float(np.clip(alpha_l_1, 0.0, 1.0))
    if m == 0:
        return alpha_l_1

    numerator = 3 * v_l_s * alpha
    denominator = v_l_s * ( 2 + alpha) 
    if denominator == 0:
        return 0.0
    return float(np.clip(numerator / denominator, 0.0, 1.0))


def _policy_action(policy: Any, state: np.ndarray) -> np.ndarray:
    if hasattr(policy, "select_action"):
        target_dtype = None
        if isinstance(policy, torch.nn.Module):
            try:
                target_dtype = next(policy.parameters()).dtype
            except StopIteration:
                target_dtype = None
        if target_dtype is None:
            target_dtype = torch.float32

        if isinstance(state, np.ndarray):
            state_tensor = torch.from_numpy(state).to(dtype=target_dtype).unsqueeze(0)
        elif isinstance(state, torch.Tensor):
            state_tensor = state.to(dtype=target_dtype)
            if state_tensor.dim() == 1:
                state_tensor = state_tensor.unsqueeze(0)
        else:
            state_tensor = torch.tensor(state, dtype=target_dtype).unsqueeze(0)

        action = policy.select_action(state_tensor)[0]
        if isinstance(action, torch.Tensor):
            action = action.detach().cpu().numpy()
        return np.asarray(action)
    if hasattr(policy, "act"):
        return np.asarray(policy.act(state))
    if callable(policy):
        return np.asarray(policy(state))
    raise ValueError("Policy does not implement select_action, act, or __call__.")


def _policy_forward_action(policy: Any, state: np.ndarray) -> np.ndarray:
    if not hasattr(policy, "__call__"):
        return _policy_action(policy, state)

    state_var = torch.tensor(state).unsqueeze(0)
    with torch.no_grad():
        action = policy(state_var)[0][0].numpy()

    if hasattr(policy, "is_disc_action"):
        return int(action) if policy.is_disc_action else action.astype(np.float64)
    return np.asarray(action)


def evaluate_policy(
    env,
    policy,
    eval_trajectories: int,
    H: int,
    gamma: float,
    base_seed: int,
    running_state: Optional[object],
    env_name: Optional[str],
    env_parameter: Optional[float],
) -> Tuple[float, float, float, float]:
    eval_rewards = []
    eval_costs = []

    previous_fix = None
    if running_state is not None and hasattr(running_state, "fix"):
        previous_fix = running_state.fix
        running_state.fix = True
    try:
        for episode_index in range(eval_trajectories):
            episode_seed = int(base_seed) + episode_index
            state, _ = env.reset(seed=episode_seed)
            if running_state is not None:
                state = running_state(state)
            total_reward = 0.0
            total_cost = 0.0

            for t in range(H):
                action = _policy_forward_action(policy, state)
                step_result = env.step(action)
                if len(step_result) == 6:
                    next_state, reward_raw, step_cost, terminated, truncated, info = step_result
                else:
                    next_state, reward_raw, terminated, truncated, info = step_result
                    step_cost = None
                if compute_task_reward_cost is not None and env_name is not None and env_parameter is not None:
                    reward, cost = compute_task_reward_cost(env_name, env_parameter, reward_raw, info)
                else:
                    reward = reward_raw
                    cost = step_cost if step_cost is not None else 0.0
                total_reward += float(reward)
                total_cost += float(cost) * (gamma ** t)
                state = next_state
                if running_state is not None:
                    state = running_state(state)
                if terminated or truncated:
                    break
            eval_rewards.append(total_reward)
            eval_costs.append(total_cost)
    finally:
        if previous_fix is not None:
            running_state.fix = previous_fix

    reward_mean = float(np.mean(eval_rewards))
    reward_std = float(np.std(eval_rewards))
    cost_mean = float(np.mean(eval_costs))
    cost_std = float(np.std(eval_costs))
    return reward_mean, reward_std, cost_mean, cost_std



def test_time_adaptation(
    env,
    pi_s,
    U_hat: List[Dict[str, Any]],
    K: int,
    H: int,
    running_state: Optional[object] = None,
    base_seed: int = 0,
    env_name: Optional[str] = None,
    env_parameter: Optional[float] = None,
    delta: float = 0.1,
    epsilon: float = 0.1,
    gamma: float = 0.99,
    eval_every: int = 500,
    eval_trajectories: int = 500,
    constraint_threshold: float = 5.0,
    save_path: Optional[str] = None,
):
    """
    Implements Algorithm 2: Testing phase with safe exploration.

    Given a safe meta-policy $\pi_s$ and a policy-value set $\hat{\mathcal{U}}$, it adaptively mixes 
    a candidate policy $\pi_l$ with $\pi_s$ to maximize reward while ensuring safe exploration.

    The mixture policy is defined as:
    $\pi_{l,m} = \alpha_{l,m}\pi_l + (1-\alpha_{l,m})\pi_s$

    The predicted values are:
    $u_{l,m} = \alpha_{l,m}u_l + (1-\alpha_{l,m})u_{l,s}$
    $v_{l,m} = \alpha_{l,m}v_l + (1-\alpha_{l,m})v_{l,s}$

    Confidence Bound Threshold:
    $\text{threshold} = \sqrt{\frac{2\ln(4K/\delta)}{(k-k_0+1)(1-\gamma)^2}} + \epsilon(L+1)$
    """
    if not U_hat:
        return pi_s

    L = 200
    U_hat_sorted = sorted(U_hat, key=lambda item: item["u_l"], reverse=True)

    k_0 = 1
    m = 0
    alpha = 0.0

    eval_records: List[Dict[str, float]] = []

    R_history: List[float] = []
    C_history: List[float] = []

    previous_fix = None
    if running_state is not None and hasattr(running_state, "fix"):
        previous_fix = running_state.fix
        running_state.fix = True
    
    reward_mean_safe, reward_std_safe, cost_mean_safe, cost_std_safe = evaluate_policy(
        env=env,
        policy=pi_s,
        eval_trajectories=eval_trajectories,
        H=H,
        gamma=gamma,
        base_seed=base_seed,
        running_state=running_state,
        env_name=env_name,
        env_parameter=env_parameter,
    )
    print(f"SafeMeta policy evaluation: reward_mean={reward_mean_safe:.2f}, cost_mean={cost_mean_safe:.2f}")
    current_candidate_key = None
    reward_mean_candidate = 0.0
    reward_std_candidate = 0.0
    cost_mean_candidate = 0.0
    cost_std_candidate = 0.0
    for k in range(1, K + 1):
        if not U_hat_sorted:
            if k % eval_every == 0:
                eval_records.append(
                    {
                        "k": float(k),
                        "reward_mean": reward_mean_safe,
                        "reward_std": reward_std_safe,
                        "cost_mean": cost_mean_safe,
                        "cost_std": cost_std_safe,
                    }
                )
            continue

        candidate = U_hat_sorted[0]
        candidate_key = (candidate.get("u_l"), candidate.get("v_l"))
        if candidate_key != current_candidate_key:
            reward_mean_candidate, reward_std_candidate, cost_mean_candidate, cost_std_candidate = evaluate_policy(
                env=env,
                policy=candidate["policy"],
                eval_trajectories=eval_trajectories,
                H=H,
                gamma=gamma,
                base_seed=base_seed,
                running_state=running_state,
                env_name=env_name,
                env_parameter=env_parameter,
            )
            print(f"Evaluating candidate policy with u_l={candidate['u_l']:.2f}, v_l={candidate['v_l']:.2f}: reward_mean={reward_mean_candidate:.2f}, cost_mean={cost_mean_candidate:.2f}")
            current_candidate_key = candidate_key

        u_l = float(candidate["u_l"])
        v_l = float(candidate["v_l"])
        u_l_s = float(candidate["u_l_s"])
        v_l_s = float(candidate["v_l_s"])

        u_l_m = alpha * u_l + (1 - alpha) * u_l_s
        v_l_m = alpha * v_l + (1 - alpha) * v_l_s

        episode_seed = int(base_seed) + (k - 1)
        state, _ = env.reset(seed=episode_seed)
        if running_state is not None:
            state = running_state(state)
        R_k = 0.0
        C_k = 0.0
        if random.random() < alpha:
            flag_policy = 1
        else:
            flag_policy = 0

        for t in range(H):
            if flag_policy == 1:
                action = _policy_forward_action(candidate["policy"], state)
            else:
                action = _policy_forward_action(pi_s, state)

            step_result = env.step(action)
            if len(step_result) == 6:
                next_state, reward_raw, step_cost, terminated, truncated, info = step_result
            else:
                next_state, reward_raw, terminated, truncated, info = step_result
                step_cost = None
            if compute_task_reward_cost is not None and env_name is not None and env_parameter is not None:
                reward, cost = compute_task_reward_cost(env_name, env_parameter, reward_raw, info)
            else:
                reward = reward_raw
                cost = step_cost if step_cost is not None else 0.0
            R_k += float(reward)
            C_k += float(cost) * (gamma ** t)
            state = next_state
            if running_state is not None:
                state = running_state(state)
            if terminated or truncated:
                break

        R_history.append(R_k)
        C_history.append(C_k)

        if k % eval_every == 0:
            eval_records.append(
                {
                    "k": float(k),
                    "reward_mean": alpha * reward_mean_candidate + (1 - alpha) * reward_mean_safe,
                    "reward_std": alpha ** 2 * reward_std_candidate + (1 - alpha) ** 2 * reward_std_safe,
                    "cost_mean": alpha * cost_mean_candidate + (1 - alpha) * cost_mean_safe,
                    "cost_std": alpha ** 2 * cost_std_candidate + (1 - alpha) ** 2 * cost_std_safe,
                }
            )
            print(f"Evaluation at k={k}: reward_mean={alpha * reward_mean_candidate + (1 - alpha) * reward_mean_safe:.2f}, cost_mean={alpha * cost_mean_candidate + (1 - alpha) * cost_mean_safe:.2f}")

        n = k - k_0 + 1
        mean_R = float(np.mean(R_history))
        mean_C = float(np.mean(C_history))

        thresh =  np.sqrt(60000/n) + epsilon * 280 
        thresh_c = np.sqrt(600/n)+ epsilon * 10
        condition_1 = abs(mean_R - u_l_m) > thresh or abs(mean_C - v_l_m) > thresh_c
        condition_2 = n > 300

        if condition_1 and condition_2:
            U_hat_sorted.pop(0)
            print(f"Removing candidate with u_l={u_l:.2f}, v_l={v_l:.2f} at k={k} due to failed confidence bound test. mean_R={mean_R - u_l_m:.2f}, mean_C={mean_C - v_l_m:.2f}, threshold_R={thresh:.2f}, threshold_C={thresh_c:.2f}")
            k_0 = k + 1
            m = 0
            alpha = 0.0
            R_history.clear()
            C_history.clear()
            continue

        denom = ((v_l_m - constraint_threshold) ** 2) if v_l_m != 0 else None
        if denom is not None:
            min_samples = min(max(100 / denom, 650), 350)
            if n >= min_samples and m < 15:
                alpha = update_mixture_weight(m, v_l_s, epsilon, alpha, constraint_threshold)
                print(f"Updating mixture weight at k={k}: m={m}, alpha={alpha:.4f}, v_l={v_l:.4f}, v_l_m={v_l_m:.4f}, threshold={constraint_threshold}")
                k_0 = k + 1
                m += 1
                R_history.clear()
                C_history.clear()

    if save_path is not None and eval_records:
        import json

        with open(save_path, "w") as handle:
            json.dump(eval_records, handle, indent=2)

    if previous_fix is not None:
        running_state.fix = previous_fix

    if U_hat_sorted:
        return U_hat_sorted[0]["policy"]
    return pi_s


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run safe PCE adaptation.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save-path", default="assets/plots/safe_pce_eval.json")
    parser.add_argument("--safemeta-model-path", default="assets/learned_models/SafeMeta/2026-03-27-exp-SafeMeta-Hopper")
    parser.add_argument("--cover-set-eval", default="assets/plots/cover_set_eval.json")
    parser.add_argument("--cpo-export-dir", default="assets/learned_models/CPO_cover_set")
    parser.add_argument("--k", type=int, default= 20 * 300)
    parser.add_argument("--h", type=int, default=200)
    parser.add_argument("--delta", type=float, default=0.1)
    parser.add_argument("--epsilon", type=float, default=0.1)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--eval-every", type=int, default=300)
    parser.add_argument("--eval-trajectories", type=int, default=500)
    parser.add_argument("--constraint-threshold", type=float, default=5.0)
    parser.add_argument("--env-name", default="Hopper")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    pi_s, running_state = load_safemeta_policy(args.safemeta_model_path)
    U_hat = build_u_hat_from_cover_set(
        args.cover_set_eval,
        args.cpo_export_dir,
    )
    env, env_parameter = create_safe_hopper_env()
    # print(f"Sampled task velocity: {env_parameter:.6f}")
    test_time_adaptation(
        env,
        pi_s,
        U_hat,
        K = args.k,
        H = args.h,
        running_state = running_state,
        base_seed = args.seed,
        env_name = args.env_name,
        env_parameter = env_parameter,
        delta = args.delta,
        epsilon = args.epsilon,
        gamma = args.gamma,
        eval_every = args.eval_every,
        eval_trajectories = args.eval_trajectories,
        constraint_threshold = args.constraint_threshold,
        save_path=args.save_path,
    )
