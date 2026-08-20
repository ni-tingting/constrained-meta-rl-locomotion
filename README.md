# Safe Meta-Reinforcement Learning via Dual-Method-Based Policy Adaptation

Reference implementation of **safe meta-RL with near-optimality and anytime safety
guarantees**, together with the baselines it is compared against.

The setting is a family of **constrained MDPs** (CMDPs) that share dynamics but differ
in their task parameter. On MuJoCo Hopper the task is a goal velocity drawn from a
truncated Gaussian; the agent must maximise reward while keeping the discounted safety
cost below a threshold — **at every point during adaptation**, not just at convergence.

Our method has two phases, both in [`algos/our_algorithm.py`](algos/our_algorithm.py):

| Phase | What happens |
|---|---|
| **Task cover set** | Greedy set cover over sampled tasks: tasks within `epsilon` of each other are treated as solvable by the same policy, so a handful of experts covers the whole distribution. |
| **Safe test-time adaptation** | On a new task, mix a high-reward candidate `π_l` into the safe meta-policy `π_s` — `π_{l,m} = α·π_l + (1−α)·π_s` — raising `α` only as fast as confidence bounds allow the constraint to stay satisfied. |

Because `α` starts at 0 (pure `π_s`, known safe) and only grows when the observed
reward and cost stay inside their confidence intervals, the mixture is safe *anytime*.
Candidates whose predicted values fail the test are discarded and the next-best one is
tried. The safe meta-policy `π_s` itself is trained by
[`algos/SafeMeta.py`](algos/SafeMeta.py).

---

## Everything runs through `main.py`

```
$ python main.py
usage: python main.py <subcommand> [options]

subcommands:
  train           meta-train / meta-test one of the four algorithms
  cover-set       build the task cover set, then meta-train on it
  cpo-experts     train one CPO expert per cover-set task
  eval-cover-set  score those experts -> the U_hat for `adapt`
  adapt           our algorithm: safe test-time adaptation
  baseline        shared pre-adaptation reference point
  eval-random     held-out evaluation on freshly sampled tasks
  compare         baseline -> all algorithms -> comparison plot
```

`python main.py <subcommand> --help` lists that stage's flags.

---

## Repository structure

```
.
├── main.py                       # THE entry point -- every stage is a subcommand
│
├── algos/                        # The learning algorithms
│   ├── our_algorithm.py          #   ← OUR METHOD: cover set + safe test-time adaptation
│   ├── SafeMeta.py               #   trains the safe meta-policy pi_s (dual-method inner step)
│   ├── MAML_constraint.py        #   baseline: MAML + constraint on the inner step
│   ├── CPOMeta.py                #   baseline: per-task CPO step + projected meta step
│   ├── CPO.py                    #   baseline: single-task Constrained Policy Optimisation
│   └── trpo.py                   #   shared trust-region machinery (CG, line search)
│
├── core/
│   ├── agent.py                  # rollout collection + per-worker log merging
│   └── common.py                 # GAE advantages, discounted constraint value J_C(π)
│
├── models/
│   ├── continuous_policy.py      # Gaussian policy, learnable state-independent log-std
│   ├── discrete_policy.py        # categorical policy
│   └── critic.py                 # value network (used for BOTH reward and cost critics)
│
├── utils/
│   ├── tools.py                  # envs, task sampling, and the reward/cost definition
│   ├── evaluation.py             # evaluation routines behind the eval-* subcommands
│   ├── argument_parsing.py       # all flags for `main.py train`
│   ├── model_saving.py           # checkpoint + metric-dump directory layout
│   ├── zfilter.py                # running mean/std state normaliser
│   ├── replay_memory.py          # trajectory buffer
│   ├── torch.py                  # flatten/restore params & grads
│   └── math.py                   # Gaussian log-density / entropy
│
├── scripts/                      # multi-seed launchers + plotting (see scripts/README.md)
└── assets/
    ├── cover_set.json            # the task cover set (Hopper goal velocities)
    ├── learned_models/<algo>/<run>/  # checkpoints, training_log.csv, test_log2.csv, tensorboard
    └── plots/                    # figures and evaluation JSONs
```

> **Where the problem is actually defined.** `utils/tools.py::compute_task_reward_cost`
> is the single place the reward and the safety cost of each environment are specified.
> Change it there and all five algorithms see the change.
> `utils/tools.py::sample_truncated_gaussian` is the Hopper task distribution —
> `N(0.5, 0.01)` truncated to `[0, 1]`.

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

### Meta-training a single algorithm

```bash
python main.py train --algo-name SafeMeta --env-name Hopper --is-meta-test False
```

`--algo-name` accepts `SafeMeta`, `MAML_constraint`, `CPOMeta`, `CPO`.
`--env-name` accepts `Hopper`, `HalfCheetah`, `Swimmer`, `Humanoid`.

Results land in `assets/learned_models/<algo>/<date>-exp-<algo>-<env>/` containing
`model.p` (best), `model_last.p`, `training_log.csv`, `test_log2.csv`, and TensorBoard
events under `runs/`.

Flags worth knowing (`python main.py train --help` for the rest):

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

Some flags exist only for certain algorithms, so `--help` shows a different set
depending on `--algo-name`, and passing another algorithm's flag is an error rather
than a silent no-op.

### The full pipeline for our method

Run these in order — each stage consumes the previous stage's output:

```bash
# 1. build the task cover set -> assets/cover_set.json, then meta-train on it
python main.py cover-set

# 2. train one CPO expert per cover-set task
#    -> assets/learned_models/CPO_cover_set/velocity_*/
python main.py cpo-experts

# 3. score experts + meta-policy on each task -> assets/plots/cover_set_eval.json
python main.py eval-cover-set

# 4. our algorithm: safe test-time adaptation
python main.py adapt --seed 0 --save-path assets/plots/safe_pce_eval.json
```

Step 3 produces the `Û` set consumed by step 4: for each cover-set task it records the
reward/cost of the CPO expert (`u_l`, `v_l`) and of the safe meta-policy (`u_l_s`,
`v_l_s`), which is exactly what the confidence-bound test needs. Stages 1–3 discover
the most recent relevant checkpoint automatically; pass `--safemeta-model-path` to pin
a specific one.

The shipped `assets/cover_set.json` holds two Hopper velocities, `≈0.504` and `≈0.304`.
`python main.py cover-set --skip-training` rebuilds it without training, and reproduces
those exact values (the construction is seeded).

### Baseline comparison

Adapts one shared checkpoint with each algorithm and plots them against a common
pre-adaptation baseline:

```bash
python main.py compare --env-name Hopper
```

### Held-out evaluation

```bash
python main.py eval-random --model-path assets/learned_models/SafeMeta/<run>
```

Reports averaged reward/cost over freshly sampled tasks plus how many exceeded the
threshold — the constraint-violation rate on unseen tasks.

### Multi-seed runs and figures

```bash
./scripts/run_10_seeds.sh                    # meta-train 4 algorithms × 10 seeds
./scripts/run_safe_pce_20_seeds.sh           # `main.py adapt` × 20 seeds
python scripts/plot_safe_pce_seeds.py        # mean / p10-p90 reward & cost bands
python scripts/plot_seeded_rewards.py        # mean ± std training curves
```

See [`scripts/README.md`](scripts/README.md).

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
  logged cost values are already on the same scale as `--max-constraint`.
- **Stages shell out to `main.py train`** rather than importing it, so each training run
  gets a clean process and its own checkpoint directory.

### Known rough edges

These are in the numerics and are deliberately left as-is rather than silently
"fixed" — flagging them so a reader is not misled:

- In `test_time_adaptation`, the confidence-bound constants are tuned for Hopper rather
  than derived from `delta`: `thresh = sqrt(60000/n) + epsilon*280` and
  `thresh_c = sqrt(600/n) + epsilon*10`. The `delta` argument is consequently unused.
- Also in `test_time_adaptation`: `min_samples = min(max(100/denom, 650), 350)` always
  evaluates to `350`, since `max(..., 650) >= 650 > 350`. It was probably meant to read
  `min(max(100/denom, 350), 650)`.

## Licence and attribution

MIT — see [`LICENSE`](LICENSE). The CPO implementation and the surrounding
`core/` / `models/` / `utils/` scaffolding derive from Sapana Chaudhary's safe meta-RL
codebase, whose copyright notice is retained in `LICENSE`.
