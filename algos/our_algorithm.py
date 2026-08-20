"""
Our method: safe meta-RL via dual-method-based policy adaptation.

This module holds the two phases that are specific to our algorithm. The safe
meta-policy itself is trained by ``algos/SafeMeta.py``; what follows is the
pre-training construction that decides *which* tasks need an expert, and the
test-time loop that adapts to a new task without ever violating the constraint.

Phase 1 -- task cover set (``build_cover_set``)
    Greedy set cover over sampled task parameters. Two tasks within ``epsilon``
    of each other are treated as solvable by the same policy, so a handful of
    experts covers the whole task distribution. Written to
    ``assets/cover_set.json`` and consumed by every downstream stage.

Phase 2 -- safe test-time adaptation (``test_time_adaptation``, Algorithm 2)
    Given the safe meta-policy ``pi_s`` and a set ``U_hat`` of candidate
    (policy, reward, cost) tuples, mix a candidate into ``pi_s``

        pi_{l,m} = alpha * pi_l + (1 - alpha) * pi_s

    and raise ``alpha`` only when the observed reward and cost stay inside their
    confidence bounds. ``alpha`` starts at 0 -- pure ``pi_s``, known safe -- so
    the mixture is safe at *every* step, not just at convergence. A candidate
    whose predicted values fail the bound test is discarded and the next-best
    one is tried.
"""

import json
import math
import pickle
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from utils.tools import compute_task_reward_cost, create_sigle_envs


# --------------------------------------------------------------------------- #
# Phase 1: task cover set
# --------------------------------------------------------------------------- #

COVER_SEED = 1
COVER_EPSILON = 0.1
COVER_MEAN = 0.5
COVER_STD = 0.1
COVER_LOW = 0.0
COVER_HIGH = 1.0
COVER_MAX_PHASES = 20


def covers(noise_i: float, noise_j: float, epsilon: float = COVER_EPSILON) -> int:
    """1 if task ``i`` is within ``epsilon`` of task ``j``, i.e. covered by it."""
    return int(abs(noise_i - noise_j) < epsilon)


def sample_task_noises(
    size: int,
    mean: float = COVER_MEAN,
    std: float = COVER_STD,
    low: float = COVER_LOW,
    high: float = COVER_HIGH,
    rng: "np.random.Generator | None" = None,
):
    """Draw task parameters from a Gaussian clipped to ``[low, high]``."""
    generator = rng if rng is not None else np.random.default_rng(COVER_SEED)
    samples = generator.normal(mean, std, size=size)
    return np.clip(samples, low, high)


def policy_cover_subroutine(noises: List[float], delta: float, epsilon: float) -> Tuple[List[float], int]:
    """
    Greedy set cover: repeatedly take the task covering the most uncovered tasks.

    Stops once ``(1 - 3*delta)`` of the sampled tasks are covered. Returns the
    chosen task parameters and how many were needed.
    """
    N = len(noises)

    U = []
    T = set(range(N))
    A = np.zeros((N, N), dtype=int)

    for i in range(N):
        for j in range(N):
            A[i, j] = covers(noises[i], noises[j], epsilon)

    cover_amounts = []

    for _ in range(N):
        if len(T) == 0:
            break

        best_j = None
        best_cover = -1

        for j in range(N):
            if j in U:
                continue
            cover_size = sum(A[i, j] for i in T)
            if cover_size > best_cover:
                best_cover = cover_size
                best_j = j

        if best_j is None:
            break

        U.append(best_j)

        newly_covered = {i for i in T if A[i, best_j] == 1}
        T = T - newly_covered

        cover_amounts.append(best_cover)

        if sum(cover_amounts) >= (1 - 3 * delta) * N:
            break

    U_cmdps = [noises[j] for j in U]
    return U_cmdps, len(U)


def build_cover_set(delta: float, epsilon: float = COVER_EPSILON):
    """
    Sample tasks and cover them, doubling the sample size until the cover is
    statistically adequate.

    The stopping test is ``sqrt(|U| * log(2N/delta) / (N - |U|)) <= delta``: the
    cover is accepted once its generalisation slack falls below ``delta``.
    """
    if not (0.0 < delta < 1.0):
        raise ValueError("delta must be in (0, 1)")

    rng = np.random.default_rng(COVER_SEED)

    N_f = int(math.log(1 / delta) / (delta ** 2))
    N_f = max(N_f, 1)

    phase = 1
    noises = list(sample_task_noises(size=N_f, rng=rng))
    N = N_f

    while phase <= COVER_MAX_PHASES:
        U, hat_Pi_size = policy_cover_subroutine(noises, delta, epsilon)

        denominator = max(N - hat_Pi_size, 1)
        lhs = math.sqrt((hat_Pi_size * math.log(2 * N / delta)) / denominator)
        if lhs <= delta:
            return U, hat_Pi_size

        N = 2 * N
        new_noises = list(sample_task_noises(size=N_f, rng=rng))
        noises.extend(new_noises)
        phase += 1

    U, hat_Pi_size = policy_cover_subroutine(noises, delta, epsilon)
    return U, hat_Pi_size


def ensure_cover_set_exists(cover_set_path: Path, cover_delta: float) -> bool:
    """
    Build the cover set at ``cover_set_path`` if it is not already there.

    Returns True if a new cover set was written, False if one already existed.
    """
    if cover_set_path.exists():
        return False

    cover_set, _ = build_cover_set(delta=cover_delta, epsilon=COVER_EPSILON)
    cover_set = [float(x) for x in cover_set]
    cover_set_path.parent.mkdir(parents=True, exist_ok=True)
    cover_set_path.write_text(json.dumps(cover_set))
    return True


# --------------------------------------------------------------------------- #
# Environment and checkpoint helpers
# --------------------------------------------------------------------------- #

@dataclass
class SafeHopperEnvArgs:
    env_name: str = "Hopper"
    env_num: int = 1
    use_cover_set_tasks: bool = False
    cover_set_path: str = "assets/cover_set.json"
    seed: int = 0


def create_safe_hopper_env(args: Optional[SafeHopperEnvArgs] = None) -> Tuple[Any, float]:
    """
    Create the environment and sample a single goal velocity for it.

    Returns the env and the sampled task parameter, so the caller can pass the
    parameter on to the reward/cost function (the MuJoCo env itself is
    task-agnostic; the task enters only through
    ``utils.tools.compute_task_reward_cost``).
    """
    if args is None:
        args = SafeHopperEnvArgs()

    env, env_params = create_sigle_envs(args)
    env_parameter = env_params[0] if env_params else 0.0
    return env, float(env_parameter)


def load_policy_from_dir(model_dir: Path) -> Tuple[torch.nn.Module, object]:
    """Load ``(policy, running_state)``, preferring ``model_last.p`` over ``model.p``."""
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
    """Load the safe meta-policy ``pi_s`` and its state normaliser."""
    return load_policy_from_dir(Path(model_dir))


def cpo_policy_dir(export_dir: Path, velocity: float) -> Path:
    """Directory holding the CPO expert trained for ``velocity``."""
    safe_label = str(velocity).replace(".", "p")
    return export_dir / f"velocity_{safe_label}"


def build_u_hat_from_cover_set(
    eval_json_path: str,
    cpo_export_dir: str,
) -> List[Dict[str, Any]]:
    """
    Build the candidate set ``U_hat`` from the JSON written by the
    ``eval-cover-set`` stage.

    Each entry carries the expert policy for one cover-set task plus the four
    numbers the confidence-bound test needs:

    - ``u_l``, ``v_l``: reward / cost of the CPO expert on that task
    - ``u_l_s``, ``v_l_s``: reward / cost of the safe meta-policy on that task
    """
    eval_path = Path(eval_json_path)
    export_dir = Path(cpo_export_dir)
    payload = json.loads(eval_path.read_text())
    if "results" not in payload:
        raise ValueError("Invalid evaluation JSON: missing 'results' key")

    u_hat: List[Dict[str, Any]] = []
    for item in payload["results"]:
        velocity = float(item["velocity"])
        cpo_policy, _ = load_policy_from_dir(cpo_policy_dir(export_dir, velocity))
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


# --------------------------------------------------------------------------- #
# Phase 2: safe test-time adaptation
# --------------------------------------------------------------------------- #

def update_mixture_weight(m: int, v_l_s: float, epsilon: float, alpha: float, constraint_threshold: float) -> float:
    r"""
    Mixture weight :math:`\alpha_{l,m}` (Equation 6 of the paper).

    ``v_l_s`` is first turned into the *slack* against the constraint
    threshold, so a policy sitting far below the threshold is allowed a larger
    step towards the candidate than one already close to it.

    For :math:`m = 0`:
        :math:`\alpha_{l,1} = \frac{v_{l,s}}{v_{l,s} + 2}`

    For :math:`m \ge 1`, the weight is ratcheted up geometrically:
        :math:`\alpha_{l,m+1} = \frac{3 v_{l,s}\alpha}{v_{l,s}(2 + \alpha)}`

    The result is clipped to ``[0, 1]``.
    """
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
    denominator = v_l_s * (2 + alpha)
    if denominator == 0:
        return 0.0
    return float(np.clip(numerator / denominator, 0.0, 1.0))


def _policy_action(policy: Any, state: np.ndarray) -> np.ndarray:
    """Sample an action, tolerating policies that expose ``select_action``, ``act`` or ``__call__``."""
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
    """Deterministic (mean) action from a policy network."""
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
    """
    Roll out ``policy`` deterministically and return
    ``(reward_mean, reward_std, cost_mean, cost_std)``.

    Costs are discounted by ``gamma``. The state normaliser is frozen for the
    duration so evaluation does not shift the running statistics.
    """
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
                if env_name is not None and env_parameter is not None:
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
    r"""
    Algorithm 2: testing phase with safe exploration.

    Candidates in ``U_hat`` are tried in decreasing order of predicted reward.
    For the current candidate, the predicted mixture values are

        :math:`u_{l,m} = \alpha u_l + (1-\alpha) u_{l,s}`
        :math:`v_{l,m} = \alpha v_l + (1-\alpha) v_{l,s}`

    Each episode is run under either the candidate or ``pi_s``, chosen by a coin
    flip with bias ``alpha``. Two things can then happen:

    - **Candidate rejected.** If the empirical reward or cost has drifted
      outside its confidence bound (and enough episodes have accumulated for
      that to be meaningful), the candidate's predicted values were wrong: drop
      it, reset ``alpha`` to 0, and move to the next candidate.
    - **Weight raised.** Otherwise, once enough episodes support the current
      estimate, ``alpha`` is increased by ``update_mixture_weight``, shifting
      probability mass towards the higher-reward candidate.

    Either way the counters and history reset, so each ``alpha`` is judged only
    on episodes actually collected under it.

    Returns the final policy, and writes the periodic evaluation records to
    ``save_path`` if given.
    """
    if not U_hat:
        return pi_s

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
            # Every candidate was rejected; nothing left but the safe policy.
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
            print(
                f"Evaluating candidate policy with u_l={candidate['u_l']:.2f}, v_l={candidate['v_l']:.2f}: "
                f"reward_mean={reward_mean_candidate:.2f}, cost_mean={cost_mean_candidate:.2f}"
            )
            current_candidate_key = candidate_key

        u_l = float(candidate["u_l"])
        v_l = float(candidate["v_l"])
        u_l_s = float(candidate["u_l_s"])
        v_l_s = float(candidate["v_l_s"])

        u_l_m = alpha * u_l + (1 - alpha) * u_l_s
        v_l_m = alpha * v_l + (1 - alpha) * v_l_s

        # One episode under the mixture: pick which arm to follow with prob alpha.
        episode_seed = int(base_seed) + (k - 1)
        state, _ = env.reset(seed=episode_seed)
        if running_state is not None:
            state = running_state(state)
        R_k = 0.0
        C_k = 0.0
        follow_candidate = random.random() < alpha

        for t in range(H):
            if follow_candidate:
                action = _policy_forward_action(candidate["policy"], state)
            else:
                action = _policy_forward_action(pi_s, state)

            step_result = env.step(action)
            if len(step_result) == 6:
                next_state, reward_raw, step_cost, terminated, truncated, info = step_result
            else:
                next_state, reward_raw, terminated, truncated, info = step_result
                step_cost = None
            if env_name is not None and env_parameter is not None:
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
            print(
                f"Evaluation at k={k}: "
                f"reward_mean={alpha * reward_mean_candidate + (1 - alpha) * reward_mean_safe:.2f}, "
                f"cost_mean={alpha * cost_mean_candidate + (1 - alpha) * cost_mean_safe:.2f}"
            )

        n = k - k_0 + 1
        mean_R = float(np.mean(R_history))
        mean_C = float(np.mean(C_history))

        # Confidence bounds on the reward and cost estimates. NOTE: the
        # constants below are tuned for Hopper rather than derived from delta;
        # see the README for the caveat.
        thresh = np.sqrt(60000 / n) + epsilon * 280
        thresh_c = np.sqrt(600 / n) + epsilon * 10
        outside_bounds = abs(mean_R - u_l_m) > thresh or abs(mean_C - v_l_m) > thresh_c
        enough_episodes = n > 300

        if outside_bounds and enough_episodes:
            U_hat_sorted.pop(0)
            print(
                f"Removing candidate with u_l={u_l:.2f}, v_l={v_l:.2f} at k={k} due to failed "
                f"confidence bound test. mean_R={mean_R - u_l_m:.2f}, mean_C={mean_C - v_l_m:.2f}, "
                f"threshold_R={thresh:.2f}, threshold_C={thresh_c:.2f}"
            )
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
                print(
                    f"Updating mixture weight at k={k}: m={m}, alpha={alpha:.4f}, "
                    f"v_l={v_l:.4f}, v_l_m={v_l_m:.4f}, threshold={constraint_threshold}"
                )
                k_0 = k + 1
                m += 1
                R_history.clear()
                C_history.clear()

    if save_path is not None and eval_records:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w") as handle:
            json.dump(eval_records, handle, indent=2)

    if previous_fix is not None:
        running_state.fix = previous_fix

    if U_hat_sorted:
        return U_hat_sorted[0]["policy"]
    return pi_s
