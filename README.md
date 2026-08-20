# Safe Meta-Reinforcement Learning via Dual-Method-Based Policy Adaptation

Reference implementation of **safe meta-RL with near-optimality and anytime safety
guarantees**, together with the baselines it is compared against.

The setting is a family of **constrained MDPs** (CMDPs) that share dynamics but differ
in their task parameter. On MuJoCo Hopper the task is a goal velocity drawn from a
truncated Gaussian; the agent must maximise reward while keeping the discounted safety
cost below a threshold — **at every point during adaptation**, not just at convergence.

The method has two phases:

| Phase | What happens | Code |
|---|---|---|
| **Meta-training** | Learn a *safe* meta-policy `π_s` that satisfies the constraint on the whole task distribution. | `algos/SafeMeta.py`, launched by `main.py` |
| **Test-time adaptation** | On a new task, mix a high-reward candidate policy `π_l` into `π_s` — `π_{l,m} = α·π_l + (1−α)·π_s` — raising `α` only as fast as confidence bounds allow the constraint to stay satisfied. | `safe_pce.py` (`test_time_adaptation`) |

Because `α` starts at 0 (pure `π_s`, known safe) and only grows when the observed
reward/cost stay inside their confidence intervals, the mixture is safe *anytime*.
Candidates that fail the test are discarded and the next-best one is tried.

---

## Repository structure

```
.
├── main.py                            # ENTRY POINT: meta-train / meta-test any of the 4 algorithms
├── safe_pce.py                        # ENTRY POINT: test-time adaptation (Algorithm 2 of the paper)
│
├── algos/                             # The learning algorithms
│   ├── SafeMeta.py                    #   ← the proposed method (dual-method inner step)
│   ├── MAML_constraint.py             #   baseline: MAML + constraint on the inner step
│   ├── CPOMeta.py                     #   baseline: per-task CPO step + projected meta step
│   ├── CPO.py                         #   baseline: single-task Constrained Policy Optimisation
│   └── trpo.py                        #   shared trust-region machinery (CG, line search)
│
├── core/
│   ├── agent.py                       # rollout collection + per-worker log merging
│   └── common.py                      # GAE advantages, discounted constraint value J_C(π)
│
├── models/
│   ├── continuous_policy.py           # Gaussian policy, learnable state-independent log-std
│   ├── discrete_policy.py             # categorical policy
│   └── critic.py                      # value network (used for BOTH reward and cost critics)
│
├── utils/
│   ├── tools.py                       # envs, task sampling, and the reward/cost definition
│   ├── argument_parsing.py            # all CLI flags for main.py
│   ├── model_saving.py                # checkpoint + metric-dump directory layout
│   ├── zfilter.py                     # running mean/std state normaliser
│   ├── replay_memory.py               # trajectory buffer
│   ├── torch.py                       # flatten/restore params & grads
│   └── math.py                        # Gaussian log-density / entropy
│
├── evaluate_meta_policy.py            # build the task cover set, then meta-train on it
├── run_cpo_cover_set.py               # train one CPO expert per cover-set task
├── evaluate_cover_set_policies.py     # evaluate those experts → the Û set for safe_pce.py
├── compute_shared_baseline.py         # pre-adaptation reward/cost reference point
├── evaluate_meta_policy_random_envs.py# evaluate a meta-policy on 100 held-out random tasks
├── run_all_and_plot.py                # convenience: baseline → all algorithms → comparison plot
├── plot_test_metrics.py               # test reward/cost curves per algorithm
│
├── scripts/                           # multi-seed launchers + aggregate plotting (see scripts/README.md)
└── assets/
    ├── cover_set.json                 # the fixed task set (Hopper goal velocities)
    ├── learned_models/<algo>/<run>/   # checkpoints, training_log.csv, test_log2.csv, tensorboard
    └── plots/                         # figures and evaluation JSONs
```

> **Where the problem is actually defined.** `utils/tools.py::compute_task_reward_cost`
> is the single place the reward and the safety cost of each environment are specified.
> Change it there and all four algorithms see the change. `utils/tools.py::sample_truncated_gaussian`
> is the Hopper task distribution — `N(0.5, 0.01)` truncated to `[0, 1]`.

---

## Installation

Requires Python 3.11 and a working MuJoCo install.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## Usage

### 1. Meta-training

```bash
python main.py --algo-name SafeMeta --env-name Hopper --is-meta-test False
```

`--algo-name` accepts `SafeMeta`, `MAML_constraint`, `CPOMeta`, `CPO`.
`--env-name` accepts `Hopper`, `HalfCheetah`, `Swimmer`, `Humanoid`.

Results land in `assets/learned_models/<algo>/<date>-exp-<algo>-<env>/` containing
`model.p` (best), `model_last.p`, `training_log.csv`, `test_log2.csv`, and TensorBoard
events under `runs/`.

Flags worth knowing (full list in `utils/argument_parsing.py`):

| Flag | Meaning |
|---|---|
| `--is-meta-test` | `False` = meta-train from scratch, `True` = adapt the checkpoint at `--model-path` |
| `--max-constraint` | cost threshold (Hopper/Swimmer `5`, HalfCheetah `10`, Humanoid `20`) |
| `--env-num` | tasks sampled per meta-iteration |
| `--meta-iter-num` | inner adaptation steps per task |
| `--max-iter-num` | outer (meta) iterations |
| `--meta-lambda` | Lagrange weight on the cost — `SafeMeta` / `MAML_constraint` only |
| `--max-kl`, `--damping` | trust-region size — algorithm-specific defaults |
| `--use-cover-set-tasks` | train on the fixed `assets/cover_set.json` tasks instead of sampling |

Note: some flags are only registered for certain algorithms (the parser peeks at
`--algo-name` first), so `--meta-lambda` with `--algo-name CPO` is an error, not a no-op.

### 2. Meta-testing / baseline comparison

Trains nothing new — adapts one shared initial policy with each algorithm and plots them
against a common pre-adaptation baseline:

```bash
python run_all_and_plot.py --env-name Hopper
```

### 3. Test-time adaptation (Safe PCE)

This needs the cover set and its per-task experts, in order:

```bash
# a. build the task cover set → assets/cover_set.json, then meta-train on it
python evaluate_meta_policy.py

# b. train one CPO expert per cover-set task → assets/learned_models/CPO_cover_set/velocity_*/
python run_cpo_cover_set.py

# c. evaluate experts + meta-policy on each task → assets/plots/cover_set_eval.json
python evaluate_cover_set_policies.py

# d. run the safe mixture adaptation
python safe_pce.py --seed 0 --save-path assets/plots/safe_pce_eval.json
```

Step (c) produces the `Û` set consumed by `safe_pce.py` — for each cover-set task it
records the reward/cost of the CPO expert (`u_l`, `v_l`) and of the meta-policy
(`u_l_s`, `v_l_s`), which is exactly what the confidence-bound test needs.

The **cover set** (step a) is a greedy set cover over sampled task parameters: tasks
within `epsilon` of each other are treated as covered by the same policy, so a handful of
experts suffices for the whole distribution. The shipped `assets/cover_set.json` has two
Hopper velocities, `≈0.504` and `≈0.304`.

### 4. Multi-seed runs and figures

```bash
./scripts/run_10_seeds.sh                    # meta-train 4 algorithms × 10 seeds
./scripts/run_safe_pce_20_seeds.sh           # safe_pce.py × 20 seeds
python scripts/plot_safe_pce_seeds.py        # mean / p10-p90 reward & cost bands
python scripts/plot_seeded_rewards.py        # mean ± std training curves
```

The launchers are configured with environment variables (`SEEDS`, `ALGORITHMS`,
`EXTRA_ARGS_STR`, ...) and rename each run folder to `...-seed<N>` so runs never
overwrite each other. See [`scripts/README.md`](scripts/README.md).

---

## Reproducing the figures

`assets/plots/` holds the committed results:

- `safe_pce_reward_cost_20_seeds_p10_p90.png` — reward and cost during test-time
  adaptation over 20 seeds. Cost stays under the threshold throughout, which is the
  anytime-safety claim.
- `Hopper.png` — meta-training comparison of the four algorithms.
- `safe_pce_eval_seed*.json`, `cover_set_eval.json`, `shared_baseline.json` — the raw
  numbers behind those plots.

Trained checkpoints (`*.p`) are **not** committed; rerun the commands above to
regenerate them. The metric logs and figures are committed, so the plots can be
reproduced without retraining.

---

## Notes for readers of the code

- **Sampling is seeded deterministically.** `utils/tools.py::deterministic_rollout_seed`
  derives a per-(task, meta-iteration) seed so that evaluation rollouts are comparable
  across algorithms and seeds.
- **The state normaliser is frozen at evaluation time** (`ZFilter.fix = True`), so test
  rollouts do not shift the running statistics.
- **Reward and cost use separate critics** — two instances of `models/critic.py::Value`.
- **Cost is scaled ×100 for Hopper and Swimmer** inside `compute_task_reward_cost`, so
  the cost values you see in the logs are already on the same scale as
  `--max-constraint`.
- `evaluate_meta_policy.py` is indented with tabs while the rest of the tree uses
  spaces — cosmetic, but it will look odd in a diff.

## Licence and attribution

MIT — see [`LICENSE`](LICENSE). The CPO implementation and the surrounding
`core/` / `models/` / `utils/` scaffolding derive from Sapana Chaudhary's safe meta-RL
codebase, whose copyright notice is retained in `LICENSE`.
