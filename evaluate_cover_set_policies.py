import argparse
import json
import pickle
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

from utils.tools import evaluate_policy_on_task, create_sigle_envs


def load_cover_set(path: Path) -> List[float]:
    if not path.exists():
        raise FileNotFoundError(f"Cover set file not found: {path}")
    values = json.loads(path.read_text())
    if not isinstance(values, list) or len(values) == 0:
        raise ValueError(f"Cover set must be a non-empty JSON list: {path}")
    return [float(value) for value in values]


def load_policy(model_dir: Path) -> Tuple[torch.nn.Module, object]:
    model_file = model_dir / "model_last.p"
    if not model_file.exists():
        model_file = model_dir / "model.p"
    if not model_file.exists():
        raise FileNotFoundError(f"No model.p or model_last.p found in {model_dir}")

    with model_file.open("rb") as file:
        policy_net, _, _, running_state, _ = pickle.load(file)

    policy_net.eval()
    policy_net.to(torch.device("cpu"))
    return policy_net, running_state


def find_latest_safemeta_dir(repo_root: Path, env_name: str) -> Path | None:
    base = repo_root / "assets" / "learned_models" / "SafeMeta"
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
        if model_last.exists() or model_path.exists():
            candidates.append(run_dir)

    if not candidates:
        return None
    return max(candidates, key=lambda p: max((p / "model_last.p").stat().st_mtime if (p / "model_last.p").exists() else 0, (p / "model.p").stat().st_mtime if (p / "model.p").exists() else 0))


def get_cpo_policy_dir(export_dir: Path, velocity: float) -> Path:
    safe_label = str(velocity).replace(".", "p")
    return export_dir / f"velocity_{safe_label}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate CPO policies (per cover-set velocity) and SafeMeta policy on Hopper tasks."
    )
    parser.add_argument("--cover-set-path", default="assets/cover_set.json")
    parser.add_argument("--cpo-export-dir", default="assets/learned_models/CPO_cover_set")
    parser.add_argument("--safemeta-model-path", default=None)
    parser.add_argument("--env-name", default="Hopper")
    parser.add_argument("--num-trajectories", type=int, default=100)
    parser.add_argument("--time-horizon", type=int, default=200)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", default="assets/plots/cover_set_eval.json")
    return parser.parse_args()


def evaluate_on_velocity(
    env,
    env_name: str,
    velocity: float,
    cpo_policy,
    cpo_state,
    safemeta_policy,
    safemeta_state,
    num_trajectories: int,
    gamma: float,
    horizon: int,
    seed: int,
) -> Dict:
    cpo_reward, cpo_cost = evaluate_policy_on_task(
        env,
        cpo_policy,
        cpo_state,
        env_name,
        velocity,
        gamma,
        horizon,
        num_trajectories=num_trajectories,
        base_seed=seed,
    )

    meta_reward, meta_cost = evaluate_policy_on_task(
        env,
        safemeta_policy,
        safemeta_state,
        env_name,
        velocity,
        gamma,
        horizon,
        num_trajectories=num_trajectories,
        base_seed=seed + 1000,
    )

    return {
        "velocity": velocity,
        "cpo": {"avg_reward": float(cpo_reward), "avg_cost": float(cpo_cost)},
        "safemeta": {"avg_reward": float(meta_reward), "avg_cost": float(meta_cost)},
    }


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parent

    cover_set_path = Path(args.cover_set_path)
    if not cover_set_path.is_absolute():
        cover_set_path = repo_root / cover_set_path

    cpo_export_dir = Path(args.cpo_export_dir)
    if not cpo_export_dir.is_absolute():
        cpo_export_dir = repo_root / cpo_export_dir

    safemeta_dir = Path(args.safemeta_model_path) if args.safemeta_model_path else find_latest_safemeta_dir(repo_root, args.env_name)
    if safemeta_dir is None:
        raise RuntimeError("SafeMeta model not found. Provide --safemeta-model-path explicitly.")
    if not safemeta_dir.is_absolute():
        safemeta_dir = repo_root / safemeta_dir

    safemeta_policy, safemeta_state = load_policy(safemeta_dir)

    env_args = argparse.Namespace(env_name=args.env_name, env_num=1)
    env, _ = create_sigle_envs(env_args)

    cover_values = load_cover_set(cover_set_path)

    results = []
    for velocity in cover_values:
        cpo_policy_dir = get_cpo_policy_dir(cpo_export_dir, velocity)
        cpo_policy, cpo_state = load_policy(cpo_policy_dir)
        metrics = evaluate_on_velocity(
            env,
            args.env_name,
            velocity,
            cpo_policy,
            cpo_state,
            safemeta_policy,
            safemeta_state,
            args.num_trajectories,
            args.gamma,
            args.time_horizon,
            args.seed,
        )
        results.append(metrics)

    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = repo_root / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps({"results": results}, indent=2))

    print(json.dumps({"results": results}, indent=2))


if __name__ == "__main__":
    main()
