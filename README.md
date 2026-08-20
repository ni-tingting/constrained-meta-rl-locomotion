# Constrained Meta Reinforcement Learning with Provable Test-Time Safety

Code for the paper **"Constrained Meta Reinforcement Learning with Provable Test-Time
Safety"** — Tingting Ni, Maryam Kamgarpour (Sycamore Lab, EPFL), ICML 2026.
A copy of the paper is included: [`Constrained Meta Reinforcement Learning with Provable Test-Time Safety.pdf`](Constrained%20Meta%20Reinforcement%20Learning%20with%20Provable%20Test-Time%20Safety.pdf).

> **Scope of this repository.** This code implements the **Gym locomotion experiments of
> Appendix H.2** (Hopper and Half-Cheetah, paper Figure 4). The gridworld experiments of
> Section 6 (Figures 1 and 2) are **not** in this repository — see
> [Coverage](#coverage-what-is-and-is-not-here) below.

## The problem

We are given a family of constrained MDPs `M_all = {M_i}`, sharing state and action
spaces but differing in their task parameter. Learning has two phases:

- a **training phase** in simulation, where constraint violations are allowed and tasks
  can be sampled freely from the task distribution `D`;
- a **testing phase** in the real world, on an unknown test task `M_test ~ D`, where
  every deployed policy must be feasible — **safe exploration** (Definition 2.2).

The objective is to minimise the test-time reward regret (Eq. 1) while never violating
the constraint during adaptation.

## The algorithm

| Paper | Implementation |
|---|---|
| **Algorithm 1** — training phase: build the CMDP set `U`, learn a near-optimal policy per element, and one policy `π_s` feasible for all of them; return the policy–value set `Û` (Eq. 2) | `algos/our_algorithm.py::build_cover_set` (+ Subroutine 3), then `main.py cover-set`, `cpo-experts`, `eval-cover-set` |
| **Subroutine 3 / Algorithm 3** — greedy CMDP cover | `algos/our_algorithm.py::policy_cover_subroutine` |
| **Algorithm 2** — testing phase with safe exploration | `algos/our_algorithm.py::test_time_adaptation` |
| **Eq. 6** — mixture-weight update `α_{l,m}` | `algos/our_algorithm.py::update_mixture_weight` |

At test time the algorithm deploys the **mixture policy**

```
π_{l,m} = α_{l,m} · π_l  +  (1 − α_{l,m}) · π_s
```

where `π_l` is the current candidate from `Û` (chosen optimistically, highest predicted
reward `u_l`) and `π_s` is the feasible policy from training. Per Definition 2.3, the
mixture is realised by sampling *one* index per episode and running that policy for the
whole episode — not by mixing actions within an episode.

`α_{l,0} = 0`, i.e. deployment starts at the known-feasible `π_s`, and `α` is raised only
when enough samples have accumulated (Alg. 2 line 11). If the empirical reward or cost
leaves the confidence interval of Inequality (5), the candidate's stored values were
wrong: it is discarded from `Û`, `α` resets to 0, and the next-best candidate is tried.
This is what makes exploration safe at *every* iteration rather than only in the limit.

---

## Coverage: what is and is not here

| Paper experiment | Figure | In this repo |
|---|---|---|
| Gym Hopper, vs MAML+constraint / meta-CPO / SafeMeta / CPO | Fig. 4(b) | **Yes** — `assets/plots/Hopper.png` |
| Gym Half-Cheetah, same four baselines | Fig. 4(a) | Partly — env supported, figure not committed |

The four Appendix H.2 baselines map onto the code as:

| Paper (Appendix H.2) | `--algo-name` |
|---|---|
| (a) MAML (Finn et al., 2017) with a constraint penalty | `MAML_constraint` |
| (b) meta-CPO (Cho & Sun, 2024) | `CPOMeta` |
| (c) SafeMeta (Xu & Zhu, 2026) | `SafeMeta` |
| (d) CPO (Achiam et al., 2017) | `CPO` |
| **Our algorithm** | `main.py adapt` |

`SafeMeta` doubles as both a baseline and the source of the feasible policy `π_s`
that Algorithm 2 starts from.

---

## Repository structure

```
.
├── main.py                       # THE entry point -- every stage is a subcommand
│
├── algos/
│   ├── our_algorithm.py          #   ← OURS: Alg. 1 cover set + Alg. 2 test-time adaptation
│   ├── SafeMeta.py               #   baseline (c); also trains the feasible policy pi_s
│   ├── MAML_constraint.py        #   baseline (a)
│   ├── CPOMeta.py                #   baseline (b)
│   ├── CPO.py                    #   baseline (d)
│   └── trpo.py                   #   shared trust-region machinery (CG, line search)
│
├── core/
│   ├── agent.py                  # rollout collection + per-worker log merging
│   └── common.py                 # GAE advantages, constraint value V_c(π)
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
├── scripts/                      # plotting only -- four figure scripts
└── assets/
    ├── learned_models/<algo>/<run>/   # checkpoints, training_log.csv, test_log2.csv
    └── plots/                    # figures (PNG tracked; result JSONs are generated)
```

> **Where the CMDP is defined.** `utils/tools.py::compute_task_reward_cost` is the single
> place each environment's reward `r` and constraint cost `c` are specified. It is the
> implementation of the paper's task definitions in Appendix H.2 — see
> [Known deviations](#known-deviations-from-the-paper).

---

## Environments (Appendix H.2)

| | Hopper | Half-Cheetah |
|---|---|---|
| State / action dim | 12 / 3 | 17 / 6 |
| Reward | −\|velocity − target\| | −\|velocity − target\| |
| Task distribution | truncated on [0, 1], mean 0.5 | truncated on [0, 2], mean 1 |
| Cost | control effort | head height `h − h_0 ≤ d_τ` |
| Constraint threshold | 5 | 10 |
| Test tasks | 20 | 20 |

---

## Installation

Requires Python 3.11 and a working MuJoCo install.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

Everything is a subcommand of `main.py`; run it bare for the list.

```
$ python main.py
subcommands:
  train           meta-train / meta-test one of the four baselines
  cover-set       build the CMDP cover set U, then train the feasible policy pi_s
  cpo-experts     learn a near-optimal policy per cover-set task (oracle O_l)
  eval-cover-set  assemble the policy-value set U_hat of Eq. 2
  adapt           our algorithm: Algorithm 2, test-time adaptation
  baseline        shared pre-adaptation reference point
  eval-random     held-out evaluation on freshly sampled test tasks
  compare         baseline -> all baselines -> comparison plot
```

### Reproducing Figure 4(b) (Hopper)

**Training phase — Algorithm 1:**

```bash
# Build the CMDP cover set U (Subroutine 3), then train the feasible policy pi_s
python main.py cover-set

# Oracle O_l: a near-optimal policy for each M in U
python main.py cpo-experts

# Assemble U_hat = {(pi, V_r(pi), V_c(pi), V_r(pi_s), V_c(pi_s))} of Eq. 2
python main.py eval-cover-set
```

**Testing phase — Algorithm 2:**

```bash
python main.py adapt --seed 0 --save-path assets/plots/safe_pce_eval.json
```

**The four baselines, and the figure:**

```bash
python main.py train --algo-name SafeMeta --is-meta-test False   # and the other three
python main.py compare                                            # adapt each, then plot
```

`assets/plots/Hopper.png` is the committed result. `--max-constraint 5` for Hopper,
`10` for Half-Cheetah (`--env-name HalfCheetah`), matching Figure 4.

### Flags worth knowing

`python main.py train --help` for the rest. The flag set varies with `--algo-name`, so
passing another algorithm's flag is an error rather than a silent no-op.

| Flag | Paper symbol / meaning |
|---|---|
| `--max-constraint` | constraint threshold on `V_c` |
| `--env-num` | tasks sampled per meta-iteration (test task list is 2×) |
| `--meta-iter-num` | inner adaptation steps per task |
| `--max-iter-num` | outer (meta) iterations |
| `--gamma` | discount `γ` |
| `--meta-lambda` | Lagrange weight on the cost — `SafeMeta` / `MAML_constraint` only |
| `--max-kl`, `--damping` | trust-region size |
| `--cover-delta` | confidence `δ` for the cover-set construction |
| `--epsilon`, `--delta`, `--k`, `--h` | `ε`, `δ`, `K`, `H` of Algorithm 2 (`main.py adapt`) |

### Multi-seed runs

Test-time adaptation over seeds — the output filename carries the seed, so nothing
collides:

```bash
for seed in $(seq 0 19); do
  python main.py adapt --seed "$seed" --save-path assets/plots/safe_pce_eval_seed$seed.json
done
```

Meta-training over seeds needs one extra step. Run directories are named by *date*
(`assets/learned_models/SafeMeta/<date>-exp-SafeMeta-Hopper`), so successive seeds on the
same day land in the same folder and overwrite each other. Tag each run with its seed:

```bash
for algo in SafeMeta MAML_constraint CPOMeta CPO; do
  for seed in $(seq 0 9); do
    python main.py train --algo-name "$algo" --seed "$seed" --is-meta-test False
    latest=$(ls -td assets/learned_models/$algo/* | head -1)
    [ "${latest%-seed$seed}" = "$latest" ] && mv "$latest" "$latest-seed$seed"
  done
done
```

The `-seed<N>` suffix is not cosmetic: `plot_seeded_rewards.py` and
`plot_safe_pce_with_baselines.py` default to `--require-seed-tag` and ignore
directories not named this way.

### Plotting

Run from the repository root — these resolve `assets/` relative to the working directory.

| Script | Reads | Produces |
|---|---|---|
| `scripts/plot_test_metrics.py` | `test_log2.csv`, `shared_baseline.json` | per-algorithm test reward/cost (this is what `main.py compare` calls) |
| `scripts/plot_seeded_rewards.py` | `training_log.csv` per run | mean ± std meta-training curves |
| `scripts/plot_safe_pce_seeds.py` | `safe_pce_eval_seed*.json` | reward/cost with mean and p10–p90 bands |
| `scripts/plot_safe_pce_with_baselines.py` | the above + `test_log2.csv` | adaptation curves against the baselines |

```bash
python scripts/plot_safe_pce_seeds.py --input-dir assets/plots
python scripts/plot_seeded_rewards.py --algorithms SafeMeta MAML_constraint CPOMeta CPO
```

---

## Generated data

`assets/plots/*.png` and the `*.csv` metric logs are tracked. **All result JSONs are
generated, not committed** — `cover_set.json`, `cover_set_eval.json`,
`safe_pce_eval*.json`, `shared_baseline.json` are produced by the stages above and are
gitignored. Model checkpoints (`*.p`) are likewise not committed.

The cover-set construction is seeded (`COVER_SEED = 1`), so `python main.py cover-set
--skip-training` regenerates `assets/cover_set.json` deterministically.

---

## Known deviations from the paper

The implementation is faithful to Algorithm 1 and to the structure of Algorithm 2, but
several constants in Algorithm 2 are hand-tuned rather than computed from the paper's
formulas. These are recorded here rather than silently changed, because changing them
would alter the committed results.

**Faithful:**

- Subroutine 3 / Algorithm 3 — greedy cover, selection rule and the `|T_t| ≤ 3δN`
  stopping condition match exactly.
- Algorithm 1's doubling test `sqrt(|U| ln(2N/δ) / (N − |U|)) ≤ δ` matches exactly.
- `Û` has exactly the five fields of Eq. 2.
- Optimistic candidate selection by `argmax u_l` (Alg. 2 line 3).
- The mixture is sampled once per episode, per Definition 2.3.
- Predicted mixture values `u_{l,m}`, `v_{l,m}` follow Eq. 3.
- Constraint thresholds (Hopper 5, Half-Cheetah 10) and 20 test tasks match Figure 4.

**Deviating:**

| # | Paper | Code | Where |
|---|---|---|---|
| 1 | `R_k = Σ_t γ^t r(s_t,a_t)` (Eq. 4) — reward **discounted** | reward accumulated **undiscounted**; cost is discounted | `test_time_adaptation` |
| 2 | Inq. 5: one bound `sqrt(2ln(4K/δ) / ((k−k_0+1)(1−γ)²)) + ε(L+1)` for reward and cost | two hand-tuned bounds, `sqrt(60000/n) + 280ε` and `sqrt(600/n) + 10ε`; `δ` therefore unused | `test_time_adaptation` |
| 3 | Alg. 2 line 7: eliminate as soon as Inq. 5 is violated | additionally requires `n > 300` | `test_time_adaptation` |
| 4 | Alg. 2 line 11: `k − k_0 − 1 ≥ 32ln(4K/δ) / ((1−γ)² v_{l,s}²)`, and `m ≤ m(l) = log_{C_l} ε` | `min(max(100/denom, 650), 350)`, which is **always 350**; `m < 15` hardcoded; uses `(v_{l,m} − threshold)²` instead of `v_{l,s}²` | `test_time_adaptation` |
| 5 | Eq. 6, `m=0`: `(v_{l,s} − 2ε(L+2)) / (v_{l,s} − 2ε(L+2) + 2/(1−γ))` | `v / (v + 2)` — drops `2ε(L+2)`, uses `2` for `2/(1−γ)` | `update_mixture_weight` |
| 6 | Eq. 6, `m≥1`: geometric ratchet in `C_l^m`, `C_l = 2/3 + (4L+9)/(3v_{l,s})` | `3α / (2 + α)` | `update_mixture_weight` |
| 7 | `L = (1−γ)^{-1} + 2γ(1−γ)^{-2}` | never computed | `algos/our_algorithm.py` |
| 8 | Alg. 2 line 16: return `π_out = (1/K) Σ_k π_k` | returns the surviving candidate, or `π_s` | `test_time_adaptation` |
| 9 | Alg. 1 line 1: `N = ln²(δ)/δ²` | `ln(1/δ)/δ²` — smaller by a factor of `ln(1/δ)` | `build_cover_set` |
| 10 | Hopper reward `−\|v − target\|` | adds MuJoCo's alive bonus and control cost | `compute_task_reward_cost` |
| 11 | Half-Cheetah cost is the head-height constraint `h − h_0 ≤ d_τ` | uses control effort (`−reward_ctrl`); head height is never read | `compute_task_reward_cost` |
| 12 | Half-Cheetah tasks truncated Gaussian, mean 1, on [0, 2] | sampled **uniformly** on [0, 2] | `create_sigle_envs` |
| 13 | Hopper task distribution has **variance 0.1** | `HOPPER_TASK_VARIANCE = 0.01` (i.e. std 0.1); `main.py eval-random` defaults to variance 0.1, so the two disagree | `utils/tools.py` |

Items 10–13 mean the Hopper/Half-Cheetah CMDPs realised in code differ from their
Appendix H.2 definitions; items 1–9 mean Algorithm 2's constants are empirical rather
than the theoretically derived ones. Neither affects the qualitative claim in Figure 4
(our algorithm is the only method that both improves reward and stays under the
threshold), but both matter for anyone checking the theory against the code.

---

## Notes for readers of the code

- **Sampling is seeded deterministically.** `utils/tools.py::deterministic_rollout_seed`
  derives a per-(task, meta-iteration) seed so runs are comparable across algorithms.
- **The state normaliser is frozen at evaluation time** (`ZFilter.fix = True`).
- **Reward and cost use separate critics** — two instances of `models/critic.py::Value`.
- **Cost is scaled ×100 for Hopper** inside `compute_task_reward_cost`, so logged values
  are on the same scale as `--max-constraint`.
- **Stages shell out to `main.py train`** rather than importing it, so each training run
  gets a clean process and its own checkpoint directory.

## Citation

```bibtex
@inproceedings{ni2026constrained,
  title     = {Constrained Meta Reinforcement Learning with Provable Test-Time Safety},
  author    = {Ni, Tingting and Kamgarpour, Maryam},
  booktitle = {Proceedings of the 43rd International Conference on Machine Learning},
  series    = {PMLR},
  volume    = {306},
  year      = {2026}
}
```

Supported by the Swiss National Science Foundation under Grant 207984.

## Licence and attribution

MIT — see [`LICENSE`](LICENSE). Per Appendix H.2 the Gym experiments build on the
implementation of **Xu & Zhu (2026)**, *Efficient safe meta-reinforcement learning:
provable near-optimality and anytime safety* (NeurIPS 2026) — the `SafeMeta` baseline.
The CPO implementation and the `core/` / `models/` / `utils/` scaffolding derive from
Sapana Chaudhary's safe meta-RL codebase, whose copyright notice is retained in
`LICENSE`.
