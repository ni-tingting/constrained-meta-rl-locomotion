"""
Single entry point for the whole project.

Everything is a subcommand of this file:

    python main.py train           # meta-train / meta-test one of the four algorithms
    python main.py cover-set       # build the task cover set, then meta-train on it
    python main.py cpo-experts     # train one CPO expert per cover-set task
    python main.py eval-cover-set  # score those experts -> the U_hat for `adapt`
    python main.py adapt           # our algorithm: safe test-time adaptation
    python main.py baseline        # shared pre-adaptation reference point
    python main.py eval-random     # held-out evaluation on freshly sampled tasks
    python main.py compare         # baseline -> all algorithms -> comparison plot

Run ``python main.py <subcommand> --help`` for the flags of any one stage, or
``python main.py`` for the list above.

The full Safe PCE pipeline, in order:

    cover-set -> cpo-experts -> eval-cover-set -> adapt

Heavy imports (torch, the algorithms) are deliberately done inside each
subcommand so that ``--help`` and the cheap stages stay fast.
"""

import argparse
import json
import os
import pickle
import random
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

REPO_ROOT = Path(__file__).resolve().parent


# --------------------------------------------------------------------------- #
# Shared helpers
# --------------------------------------------------------------------------- #

def _resolve(path_like) -> Path:
    """Interpret relative paths against the repository root, not the CWD."""
    path = Path(path_like)
    return path if path.is_absolute() else REPO_ROOT / path


def _run_subcommand(subcommand: str, arguments: List[str], python: str = sys.executable,
                    dry_run: bool = False) -> None:
    """Invoke another subcommand of this file as a subprocess."""
    command = [python, str(REPO_ROOT / "main.py"), subcommand, *arguments]
    print("\n>>>", " ".join(command))
    if dry_run:
        return
    completed = subprocess.run(command, cwd=REPO_ROOT)
    if completed.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {completed.returncode}: {' '.join(command)}")


def _seed_everything(seed: int) -> None:
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _sanitize(value: float) -> str:
    """``0.5041`` -> ``0p5041``, for use in directory names."""
    return str(value).replace(".", "p")


def _training_flags(args, extra: Optional[List[str]] = None) -> List[str]:
    """Translate the shared experiment settings into ``train`` subcommand flags."""
    flags = [
        "--env-name", args.env_name,
        "--max-iter-num", str(args.max_iter_num),
        "--meta-iter-num", str(args.meta_iter_num),
        "--min-batch-size", str(args.min_batch_size),
        "--max-batch-size", str(args.max_batch_size),
        "--time-horizon", str(args.time_horizon),
        "--seed", str(args.seed),
        "--gamma", str(args.gamma),
        "--max-constraint", str(args.max_constraint),
    ]
    if extra:
        flags.extend(extra)
    return flags


def _add_experiment_flags(parser: argparse.ArgumentParser, *, env_num: int = 10,
                          max_iter_num: int = 300, batch_size: int = 500,
                          max_constraint: float = 5.0) -> None:
    """Experiment settings common to the stages that launch training."""
    parser.add_argument("--env-name", default="Hopper")
    parser.add_argument("--env-num", type=int, default=env_num)
    parser.add_argument("--max-iter-num", type=int, default=max_iter_num)
    parser.add_argument("--meta-iter-num", type=int, default=20)
    parser.add_argument("--min-batch-size", type=int, default=batch_size)
    parser.add_argument("--max-batch-size", type=int, default=batch_size)
    parser.add_argument("--time-horizon", type=int, default=200)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--max-constraint", type=float, default=max_constraint)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--python", default=sys.executable,
                        help="Python executable used for nested training runs.")


# --------------------------------------------------------------------------- #
# train
# --------------------------------------------------------------------------- #

def cmd_train(argv: List[str]) -> None:
    """Meta-train (or meta-test) one of SafeMeta / MAML_constraint / CPOMeta / CPO."""
    from datetime import date

    from torch.utils.tensorboard import SummaryWriter

    from utils import (ZFilter, assets_dir, create_sigle_envs, np,
                       parse_all_arguments, save_info, torch)
    from models.continuous_policy import Policy
    from models.critic import Value
    from models.discrete_policy import DiscretePolicy
    from algos.CPO import CPO
    from algos.CPOMeta import CPOMeta
    from algos.SafeMeta import SafeMeta
    from algos.MAML_constraint import MAML_constraint

    print("Today date is: ", date.today())

    args = parse_all_arguments(argv)
    print("Arguments: ", args)

    """Data type and compute device"""
    dtype = torch.float64
    torch.set_default_dtype(dtype)
    device = torch.device('cuda', index=args.gpu_index) if torch.cuda.is_available() else torch.device('cpu')
    if torch.cuda.is_available():
        print('using gpu')
        torch.cuda.set_device(args.gpu_index)

    """environment"""
    env, env_parameter_list = create_sigle_envs(args)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    print('\nstate dim: ', state_dim)
    print('action dim: ', action_dim, '\n')

    is_disc_action = len(env.action_space.shape) == 0
    running_state = ZFilter((state_dim,), clip=5)

    """seeding"""
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    """create all the paths to save learned models/data"""
    save_info_obj = save_info(assets_dir(), args.algo_name, args.algo_name, args.env_name)
    save_info_obj.create_all_paths()
    writer = SummaryWriter(os.path.join(assets_dir(), save_info_obj.saving_path, 'runs/'))
    print('Saving path: ', save_info_obj.saving_path)

    """define actor and critic"""
    model_file = None
    if args.model_path is not None:
        model_file = os.path.join(args.model_path, 'model.p')
        if not os.path.exists(model_file):
            model_last = os.path.join(args.model_path, 'model_last.p')
            if os.path.exists(model_last):
                model_file = model_last

    if model_file is None or not os.path.exists(model_file):
        if model_file is not None:
            print(f"Model file not found at {model_file}. Starting from scratch.")
        if is_disc_action:
            policy_net = DiscretePolicy(state_dim, action_dim)
        else:
            policy_net = Policy(state_dim, action_dim, log_std=args.log_std)
        value_net = Value(state_dim)
        cost_net = Value(state_dim)
    else:
        print('TRAINING FROM PREVIOUS PARAMETERS. . .', args)
        cli_args = args
        policy_net, value_net, cost_net, running_state, prev_args = pickle.load(open(model_file, "rb"))
        for key, value in vars(cli_args).items():
            setattr(prev_args, key, value)
        args = prev_args

    policy_net.to(device)
    value_net.to(device)
    cost_net.to(device)

    # CPO is single-task, so it takes a list of envs; the meta algorithms take one.
    algorithms = {
        'CPO': (CPO, 'train_CPO', True),
        'CPOMeta': (CPOMeta, 'train_CPOMeta', False),
        'SafeMeta': (SafeMeta, 'train_SafeMeta', False),
        'MAML_constraint': (MAML_constraint, 'train_MAML_constraint', False),
    }
    if args.algo_name not in algorithms:
        raise ValueError(
            f"Unknown --algo-name {args.algo_name!r}. Choose from {sorted(algorithms)}."
        )

    if args.is_meta_test:
        print('meta testing')

    algo_class, train_method, wants_env_list = algorithms[args.algo_name]
    algo = algo_class(
        [env] if wants_env_list else env,
        policy_net, value_net, cost_net, args, dtype, device,
        running_state=running_state, num_threads=args.num_threads,
    )
    getattr(algo, train_method)(writer, save_info_obj)


# --------------------------------------------------------------------------- #
# cover-set
# --------------------------------------------------------------------------- #

def cmd_cover_set(argv: List[str]) -> None:
    """Build the task cover set, then meta-train the safe policy on those tasks."""
    from algos.our_algorithm import ensure_cover_set_exists
    from utils.evaluation import find_latest_model_dir

    parser = argparse.ArgumentParser(
        prog="main.py cover-set",
        description="Build assets/cover_set.json if missing, then meta-train SafeMeta on it.",
    )
    parser.add_argument("--cover-delta", type=float, default=0.05,
                        help="Delta used only when creating a missing cover set.")
    parser.add_argument("--cover-set-path", default="assets/cover_set.json")
    parser.add_argument("--algo-name", default="SafeMeta",
                        help="Algorithm to train on the cover-set tasks.")
    parser.add_argument("--skip-training", action="store_true",
                        help="Only build the cover set; do not train.")
    _add_experiment_flags(parser, max_iter_num=1000, batch_size=300)
    args = parser.parse_args(argv)

    cover_set_path = _resolve(args.cover_set_path)
    if ensure_cover_set_exists(cover_set_path, args.cover_delta):
        print(f"Built cover set: {cover_set_path}")
    else:
        print(f"Cover set already exists, reusing: {cover_set_path}")
    print(json.dumps(json.loads(cover_set_path.read_text()), indent=2))

    if args.skip_training:
        return

    extra = [
        "--algo-name", args.algo_name,
        "--is-meta-test", "False",
        "--use-cover-set-tasks", "True",
        "--cover-set-path", str(cover_set_path),
        "--env-num", str(args.env_num),
    ]
    latest = find_latest_model_dir(REPO_ROOT, args.algo_name, args.env_name)
    if latest is not None:
        extra.extend(["--model-path", str(latest)])
    _run_subcommand("train", _training_flags(args, extra), python=args.python)


# --------------------------------------------------------------------------- #
# cpo-experts
# --------------------------------------------------------------------------- #

def cmd_cpo_experts(argv: List[str]) -> None:
    """Train one CPO expert per cover-set task and export it by velocity."""
    from utils.evaluation import find_latest_model_dir, load_cover_set

    parser = argparse.ArgumentParser(
        prog="main.py cpo-experts",
        description="Run CPO on each cover-set task and save the resulting policies.",
    )
    parser.add_argument("--cover-set-path", default="assets/cover_set.json")
    parser.add_argument("--export-dir", default="assets/learned_models/CPO_cover_set")
    parser.add_argument("--exp-name-prefix", default="CPO-cover")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the training commands without running them.")
    _add_experiment_flags(parser, env_num=1)
    args = parser.parse_args(argv)

    cover_set_path = _resolve(args.cover_set_path)
    export_dir = _resolve(args.export_dir)

    cover_values = load_cover_set(cover_set_path)
    tmp_dir = export_dir / "tmp_cover_sets"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    for index, velocity in enumerate(cover_values):
        # CPO trains on a single task, so hand it a one-element cover set.
        tmp_cover_file = tmp_dir / f"cover_{index}.json"
        tmp_cover_file.write_text(json.dumps([float(velocity)]))

        velocity_dir = export_dir / f"velocity_{_sanitize(velocity)}"

        extra = [
            "--algo-name", "CPO",
            "--is-meta-test", "False",
            "--env-num", "1",
            "--use-cover-set-tasks", "True",
            "--cover-set-path", str(tmp_cover_file),
            "--exp-name", f"{args.exp_name_prefix}-idx{index}-v{_sanitize(velocity)}",
            "--exp-num", str(index + 1),
        ]
        # Resume from a previously exported expert for this velocity, if any.
        if (velocity_dir / "model.p").exists():
            extra.extend(["--model-path", str(velocity_dir)])

        _run_subcommand("train", _training_flags(args, extra),
                        python=args.python, dry_run=args.dry_run)

        if args.dry_run:
            continue

        latest_dir = find_latest_model_dir(REPO_ROOT, "CPO", args.env_name)
        if latest_dir is None:
            raise RuntimeError("No CPO checkpoint found after training.")

        model_path = latest_dir / "model.p"
        if not model_path.exists():
            raise FileNotFoundError(f"model.p not found in {latest_dir}")
        velocity_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(model_path, velocity_dir / "model.p")
        (velocity_dir / "metadata.json").write_text(
            json.dumps({"velocity": velocity, "source": str(latest_dir)}, indent=2)
        )
        print(f"Exported expert for velocity {velocity} -> {velocity_dir}")

    print("Done.")


# --------------------------------------------------------------------------- #
# eval-cover-set
# --------------------------------------------------------------------------- #

def cmd_eval_cover_set(argv: List[str]) -> None:
    """Score the CPO experts and the safe meta-policy on each cover-set task."""
    from algos.our_algorithm import SafeHopperEnvArgs, create_safe_hopper_env, load_policy_from_dir
    from utils.evaluation import evaluate_cover_set_policies, find_latest_model_dir, load_cover_set

    parser = argparse.ArgumentParser(
        prog="main.py eval-cover-set",
        description="Produce the U_hat JSON consumed by `main.py adapt`.",
    )
    parser.add_argument("--cover-set-path", default="assets/cover_set.json")
    parser.add_argument("--cpo-export-dir", default="assets/learned_models/CPO_cover_set")
    parser.add_argument("--safemeta-model-path", default=None,
                        help="Defaults to the most recent SafeMeta run for this env.")
    parser.add_argument("--env-name", default="Hopper")
    parser.add_argument("--num-trajectories", type=int, default=100)
    parser.add_argument("--time-horizon", type=int, default=200)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", default="assets/plots/cover_set_eval.json")
    args = parser.parse_args(argv)

    safemeta_dir = (_resolve(args.safemeta_model_path) if args.safemeta_model_path
                    else find_latest_model_dir(REPO_ROOT, "SafeMeta", args.env_name))
    if safemeta_dir is None:
        raise RuntimeError("SafeMeta model not found. Provide --safemeta-model-path explicitly.")
    print(f"Using safe meta-policy from: {safemeta_dir}")

    safemeta_policy, safemeta_state = load_policy_from_dir(safemeta_dir)
    env, _ = create_safe_hopper_env(SafeHopperEnvArgs(env_name=args.env_name, seed=args.seed))

    payload = evaluate_cover_set_policies(
        env=env,
        env_name=args.env_name,
        cover_values=load_cover_set(_resolve(args.cover_set_path)),
        cpo_export_dir=_resolve(args.cpo_export_dir),
        safemeta_policy=safemeta_policy,
        safemeta_state=safemeta_state,
        num_trajectories=args.num_trajectories,
        gamma=args.gamma,
        horizon=args.time_horizon,
        seed=args.seed,
    )

    output_path = _resolve(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))
    print(f"Saved: {output_path}")


# --------------------------------------------------------------------------- #
# adapt  (our algorithm)
# --------------------------------------------------------------------------- #

def cmd_adapt(argv: List[str]) -> None:
    """Our algorithm: safe test-time adaptation by policy mixing."""
    from algos.our_algorithm import (SafeHopperEnvArgs, build_u_hat_from_cover_set,
                                     create_safe_hopper_env, load_safemeta_policy,
                                     test_time_adaptation)
    from utils.evaluation import find_latest_model_dir

    parser = argparse.ArgumentParser(
        prog="main.py adapt",
        description="Adapt the safe meta-policy to a new task without violating the constraint.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save-path", default="assets/plots/safe_pce_eval.json")
    parser.add_argument("--safemeta-model-path", default=None,
                        help="Defaults to the most recent SafeMeta run for this env.")
    parser.add_argument("--cover-set-eval", default="assets/plots/cover_set_eval.json")
    parser.add_argument("--cpo-export-dir", default="assets/learned_models/CPO_cover_set")
    parser.add_argument("--k", type=int, default=20 * 300, help="Total adaptation episodes K.")
    parser.add_argument("--h", type=int, default=200, help="Episode horizon H.")
    parser.add_argument("--delta", type=float, default=0.1)
    parser.add_argument("--epsilon", type=float, default=0.1)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--eval-every", type=int, default=300)
    parser.add_argument("--eval-trajectories", type=int, default=500)
    parser.add_argument("--constraint-threshold", type=float, default=5.0)
    parser.add_argument("--env-name", default="Hopper")
    args = parser.parse_args(argv)

    _seed_everything(args.seed)

    safemeta_dir = (_resolve(args.safemeta_model_path) if args.safemeta_model_path
                    else find_latest_model_dir(REPO_ROOT, "SafeMeta", args.env_name))
    if safemeta_dir is None:
        raise RuntimeError("SafeMeta model not found. Provide --safemeta-model-path explicitly.")
    print(f"Using safe meta-policy from: {safemeta_dir}")

    pi_s, running_state = load_safemeta_policy(str(safemeta_dir))
    U_hat = build_u_hat_from_cover_set(
        str(_resolve(args.cover_set_eval)),
        str(_resolve(args.cpo_export_dir)),
    )
    env, env_parameter = create_safe_hopper_env(
        SafeHopperEnvArgs(env_name=args.env_name, seed=args.seed)
    )
    print(f"Sampled task parameter: {env_parameter:.6f}")

    test_time_adaptation(
        env,
        pi_s,
        U_hat,
        K=args.k,
        H=args.h,
        running_state=running_state,
        base_seed=args.seed,
        env_name=args.env_name,
        env_parameter=env_parameter,
        delta=args.delta,
        epsilon=args.epsilon,
        gamma=args.gamma,
        eval_every=args.eval_every,
        eval_trajectories=args.eval_trajectories,
        constraint_threshold=args.constraint_threshold,
        save_path=str(_resolve(args.save_path)),
    )
    print(f"Saved: {_resolve(args.save_path)}")


# --------------------------------------------------------------------------- #
# baseline
# --------------------------------------------------------------------------- #

def cmd_baseline(argv: List[str]) -> None:
    """Pre-adaptation reward/cost of a checkpoint, shared by every test curve."""
    from algos.our_algorithm import load_policy_from_dir
    from utils.argument_parsing import str2bool
    from utils.evaluation import compute_shared_baseline

    parser = argparse.ArgumentParser(
        prog="main.py baseline",
        description="Compute the shared pre-adaptation baseline from a checkpoint.",
    )
    parser.add_argument("--model-path", required=True, help="Directory containing model.p")
    parser.add_argument("--env-name", default="Hopper")
    parser.add_argument("--env-num", type=int, default=1)
    parser.add_argument("--use-cover-set-tasks", type=str2bool, nargs="?", const=True, default=False)
    parser.add_argument("--cover-set-path", default="assets/cover_set.json")
    parser.add_argument("--min-batch-size", type=int, default=200)
    parser.add_argument("--time-horizon", type=int, default=200)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", required=True, help="Output JSON file path")
    args = parser.parse_args(argv)

    _seed_everything(args.seed)

    policy, running_state = load_policy_from_dir(_resolve(args.model_path))

    baseline = compute_shared_baseline(
        policy=policy,
        running_state=running_state,
        env_name=args.env_name,
        env_num=args.env_num,
        use_cover_set_tasks=args.use_cover_set_tasks,
        cover_set_path=args.cover_set_path,
        min_batch_size=args.min_batch_size,
        horizon=args.time_horizon,
        gamma=args.gamma,
        seed=args.seed,
    )

    output_path = _resolve(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(baseline, indent=2))
    print(f"Saved baseline: {output_path}")
    print(baseline)


# --------------------------------------------------------------------------- #
# eval-random
# --------------------------------------------------------------------------- #

def cmd_eval_random(argv: List[str]) -> None:
    """Evaluate a policy on freshly sampled, unseen tasks."""
    from algos.our_algorithm import load_policy_from_dir
    from utils.evaluation import evaluate_on_random_envs

    parser = argparse.ArgumentParser(
        prog="main.py eval-random",
        description="Evaluate a meta-policy on tasks drawn from the task distribution.",
    )
    parser.add_argument("--model-path", required=True,
                        help="Directory containing model.p or model_last.p")
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
    parser.add_argument("--constraint-threshold", type=float, default=5.0)
    parser.add_argument("--stochastic-actions", action="store_true",
                        help="Sample actions instead of using the policy mean.")
    parser.add_argument("--output", default=None, help="Optional JSON output path.")
    args = parser.parse_args(argv)

    _seed_everything(args.seed)

    policy, running_state = load_policy_from_dir(_resolve(args.model_path))

    summary = evaluate_on_random_envs(
        policy=policy,
        running_state=running_state,
        env_name=args.env_name,
        num_envs=args.num_envs,
        num_trajectories=args.num_trajectories,
        horizon=args.time_horizon,
        gamma=args.gamma,
        seed=args.seed,
        mean=args.mean,
        variance=args.variance,
        low=args.low,
        high=args.high,
        mean_action=not args.stochastic_actions,
        constraint_threshold=args.constraint_threshold,
    )

    print("\nSummary over environments")
    print(
        f"reward_mean_avg={summary['reward_mean_avg']:.6f} "
        f"reward_std_avg={summary['reward_std_avg']:.6f} "
        f"cost_mean_avg={summary['cost_mean_avg']:.6f} "
        f"cost_std_avg={summary['cost_std_avg']:.6f}"
    )
    print(
        f"envs_with_cost_mean_above_{summary['constraint_threshold']}="
        f"{summary['envs_above_threshold']}/{summary['num_envs']}"
    )

    if args.output:
        output_path = _resolve(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(summary, indent=2))
        print(f"Saved: {output_path}")


# --------------------------------------------------------------------------- #
# compare
# --------------------------------------------------------------------------- #

def cmd_compare(argv: List[str]) -> None:
    """Adapt one shared checkpoint with every algorithm and plot them together."""
    from utils.evaluation import find_latest_model_dir

    parser = argparse.ArgumentParser(
        prog="main.py compare",
        description="Baseline, then meta-test each algorithm from a shared checkpoint, then plot.",
    )
    parser.add_argument("--algos", nargs="+",
                        default=["SafeMeta", "MAML_constraint", "CPOMeta", "CPO"])
    parser.add_argument("--model-path", default=None,
                        help="Shared initial policy. Defaults to the latest SafeMeta run.")
    parser.add_argument("--skip-baseline", action="store_true")
    parser.add_argument("--skip-plot", action="store_true")
    parser.add_argument("--save-baseline", default="assets/plots/shared_baseline.json")
    parser.add_argument("--save-plot", default="assets/plots/test_metrics.png")
    _add_experiment_flags(parser, env_num=1, max_iter_num=20, batch_size=1200, max_constraint=2.0)
    parser.set_defaults(meta_iter_num=6)
    args = parser.parse_args(argv)

    print("Running algorithms:", ", ".join(args.algos))
    print(f"Environment: {args.env_name}")

    shared_model_dir = (_resolve(args.model_path) if args.model_path
                        else find_latest_model_dir(REPO_ROOT, "SafeMeta", args.env_name))
    if shared_model_dir is None:
        raise RuntimeError(
            "No initial meta-policy found. Provide --model-path, or train one with "
            "`python main.py cover-set`."
        )
    if not (shared_model_dir / "model.p").exists():
        raise FileNotFoundError(f"model.p not found under initial policy path: {shared_model_dir}")

    print(f"Using initial policy from: {shared_model_dir}")

    if not args.skip_baseline:
        _run_subcommand("baseline", [
            "--model-path", str(shared_model_dir),
            "--env-name", args.env_name,
            "--env-num", str(args.env_num),
            "--use-cover-set-tasks", "False",
            "--min-batch-size", str(args.min_batch_size),
            "--time-horizon", str(args.time_horizon),
            "--gamma", str(args.gamma),
            "--seed", str(args.seed),
            "--output", args.save_baseline,
        ], python=args.python)

    for algo in args.algos:
        extra = [
            "--algo-name", algo,
            "--model-path", str(shared_model_dir),
            "--is-meta-test", "True",
            "--env-num", str(args.env_num),
            "--use-cover-set-tasks", "False",
        ]
        _run_subcommand("train", _training_flags(args, extra), python=args.python)

    if not args.skip_plot:
        plot_command = [
            args.python, str(REPO_ROOT / "scripts" / "plot_test_metrics.py"),
            "--env-name", args.env_name,
            "--algo", *args.algos,
            "--baseline-file", args.save_baseline,
            "--save", args.save_plot,
        ]
        print("\n>>>", " ".join(plot_command))
        completed = subprocess.run(plot_command, cwd=REPO_ROOT)
        if completed.returncode != 0:
            raise RuntimeError(f"Plotting failed with exit code {completed.returncode}")
        print(f"\nDone.\nComparison plot: {args.save_plot}")


# --------------------------------------------------------------------------- #
# Dispatch
# --------------------------------------------------------------------------- #

COMMANDS = {
    "train": cmd_train,
    "cover-set": cmd_cover_set,
    "cpo-experts": cmd_cpo_experts,
    "eval-cover-set": cmd_eval_cover_set,
    "adapt": cmd_adapt,
    "baseline": cmd_baseline,
    "eval-random": cmd_eval_random,
    "compare": cmd_compare,
}

_SUMMARIES = {
    "train": "meta-train / meta-test one of the four algorithms",
    "cover-set": "build the task cover set, then meta-train on it",
    "cpo-experts": "train one CPO expert per cover-set task",
    "eval-cover-set": "score those experts -> the U_hat for `adapt`",
    "adapt": "our algorithm: safe test-time adaptation",
    "baseline": "shared pre-adaptation reference point",
    "eval-random": "held-out evaluation on freshly sampled tasks",
    "compare": "baseline -> all algorithms -> comparison plot",
}


def _usage() -> str:
    width = max(len(name) for name in COMMANDS)
    lines = [
        "usage: python main.py <subcommand> [options]",
        "",
        "subcommands:",
    ]
    lines.extend(f"  {name:<{width}}  {_SUMMARIES[name]}" for name in COMMANDS)
    lines.extend([
        "",
        "Safe PCE pipeline order:",
        "  cover-set -> cpo-experts -> eval-cover-set -> adapt",
        "",
        "Run `python main.py <subcommand> --help` for a stage's options.",
    ])
    return "\n".join(lines)


def main() -> None:
    argv = sys.argv[1:]

    if not argv or argv[0] in {"-h", "--help", "help"}:
        print(_usage())
        return

    subcommand, rest = argv[0], argv[1:]
    if subcommand not in COMMANDS:
        print(f"error: unknown subcommand {subcommand!r}\n", file=sys.stderr)
        print(_usage(), file=sys.stderr)
        raise SystemExit(2)

    COMMANDS[subcommand](rest)


if __name__ == '__main__':
    main()
