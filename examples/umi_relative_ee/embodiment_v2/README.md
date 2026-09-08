# Embodiment experiment audit and development pilots

This directory adds monitoring and isolated controller/data ablations to the running
[Piper](../task_independent_embodiment/README.md) and
[SO-101](../so101_task_independent_embodiment/README.md) experiments. Original source
files, checkpoints, collection labels, ACT-R18 weights and validation queries are preserved.

Artifacts: `/mnt/data1/projects/lerobot-embodiment-v2`.
The original 1,459-episode corpus supplies separate embodiment training and development
motions. These pilots use **64 fixed embodiment-development rollouts**, sampled with
seed 90210 from the existing 1,000. They do not load the 100 task-validation episodes.
Earlier v1 validation results have already been inspected, so subsequent protocol
changes are exploratory; a confirmatory claim needs a new, untouched evaluation set.

## What the audit found

- Original Piper hindsight C0 has about **1.24° one-step command MAE** on sampled
  development windows, yet **115.22 mm closed-loop position RMSE** on the matched
  64-rollout development set. Good inverse-label fit does not establish stable control.
- On those same 64 rollouts, holding the previous command gives **42.02 mm** and
  the original IK-plus-residual model gives **16.58 mm**. A new direct model must
  beat the hold baseline before its movement is useful. Holding measured joint
  position is a different baseline: it drifts under gravity in this simulator.
- Hindsight's first trajectory token is exactly the current measured pose during
  training. At deployment, the requested current pose can differ. Removing that
  token eliminates this particular train/deployment invariant; it is a hypothesis
  about the failure mechanism, not a demonstrated complete solution.
- On all 1,000 original Piper development rollouts, **26.15%** of commands are
  rate-limited and **7.48%** request a goal farther away than the remaining recorded
  horizon permits. SO-101 values are **21.02%** and **0.38%**. Multiple goal commands
  can produce the same observed motion under the rate limiter.
- Replaying bounded commands on 64 Piper development rollouts changes the largest
  issued goal by **22.37° on average**, while reproducing joint positions within
  **2.9e-14 rad** and pose-matrix entries within **3.4e-14**. This is a verified
  actuator ambiguity, not evidence yet that a trained policy improves.

The teacher-forcing and saturation audits are recorded as
`inverse-teacher-forcing-audit.json` and `saturation-audit.json` in the artifact root.
Live results are in `PILOTS.md`; bounded-command results are in
`bounded_commands/PILOTS.md`. The paired comparison is in
`comparison/piper_10000steps/REPORT.md`.

`stratified/REPORT.md` and `stratified/summary.csv` separate feasible and
infeasible outcomes for both original robot sweeps and the development pilots.
The watchdog refreshes these when new evaluations complete. Subsets use separate
denominators; ACT predictions, recorded targets and development motions stay separate.
The JSON additionally records overlapping screen-failure reasons and mutually
exclusive velocity-only, outside-workspace and other-kinematic breakdowns.

## Controller pilots

`train.py` keeps the 30 Hz EE / 50 Hz command convention, eleven state observations,
ten strictly past issued commands, and five output commands. It uses:

- EE targets expressed relative to current FK, centered joint/command histories,
  and a separate absolute-current-joint token.
- A physical skip connection from measured joints (`q_delta`) or the last issued
  command (`command_delta`). Offset heads start at exactly zero in physical units.
- A masked current-time trajectory token; no unsupported positive-future windows.
- Closed-loop development checkpoint selection, saved optimizer/RNG state, and
  manifests that reject changed inputs or training source.

The direct controller uses FK to express coordinates but never solves runtime IK.
The `absolute` option provides a world-frame/absolute-output control with the same
new token/masking architecture; it is not identical to the original v1 network.

The initial 2,000-update q/command-offset pilots both failed to beat holding.
At 10,000 updates their best development errors were 75.66 and 64.47 mm,
respectively, still worse than holding. The matched command-offset model trained
with bounded commands reached 32.50 mm. Its paired improvement over the same
model trained on raw commands was 31.97 mm; a descriptive source-episode bootstrap
gave a 95% interval of 16.70–50.33 mm. The interval versus holding crosses zero,
so that smaller comparison remains uncertain on this development subset.

For the bounded-command model, the 18 screen-feasible trials average 1.52 mm,
0.26° and zero tracking failures; the 46 screen-infeasible trials average
44.63 mm, 6.93° and 43.5% failures. IK plus residual reaches 1.42 and 22.51 mm
on those respective subsets. Among the rejected trials, 15 velocity-only
screen failures average 3.91 mm, while 15 outside-workspace cases average
85.19 mm. Failure of this particular nominal IK screen does not prove physical
infeasibility.
Do not promote a model just because it improves over a drifting v1 controller.
Compare position, orientation, failures, clamp rate and the feasible subset, then
replicate promising changes on all embodiment-development rollouts and seeds.

## Bounded-command collection ablation

`canonical.py` caps successive bootstrap goals at the configured servo rate times
the command interval. It **executes every new command in simulation** and records
fresh measured states and the newly issued commands. It checks motion equivalence
against the source rollout and stops if it exceeds tolerance. It never rewrites
an old rollout or silently calls a relabeled command an executed command.

This gives a cleaner command target for the same UMI-shaped achieved trajectory.
It uses a known bootstrap actuator rate during collection; deployed model inputs
remain the requested EE trajectory and causal proprioceptive/command history.
Unit tests establish identical internal rate-limiter targets at every 500 Hz tick,
including a two-command delay. All 10,000 native Piper training replays passed:
maximum joint deviation was 7.2e-14 rad. SO-101's normalized gripper units are
handled separately, but its replay-equivalence check rejected development rollout
63 (maximum joint difference 0.000617 rad); unmodified commands replayed exactly.
The bounded SO-101 ablation was therefore not launched. Its original sweep continues.
Other dynamics must pass their own replay check before use.

The exact initial Piper generator is archived by SHA256 under `source_archive/`
in the artifact root. A narrow reader compatibility check permits reuse of its
completed collections with identical inputs after adding SO-101 gripper support;
their original manifests remain intact. Incomplete collections cannot mix generators.

`canonical_pipeline.py` replays all 10,000 D0 training rollouts, prepares hindsight
windows, and trains the same `command_delta` pilot for comparison with original
commands at the same update budget. An explicit dataset view links the new training
collection and the unchanged development data/baseline checkpoints. No write is made
through links into the original experiment. Replay timing is collection timing and
must not be reported as deployment-controller latency.

## Running and monitoring

From the repository root, use the isolated simulator environment:

```bash
export UV_CACHE_DIR=/tmp/umi-emb-uv-cache
EMB_PYTHON=/mnt/data1/projects/lerobot-embodiment-env/bin/python

# A persistent watcher is already launched; inspect before starting another.
cat /mnt/data1/projects/lerobot-embodiment-v2/WATCH.md
uv run --no-project --python "$EMB_PYTHON" python -m examples.umi_relative_ee.embodiment_v2.watch once

# Resume the bounded controller pilot comparison; completed runs are skipped.
uv run --no-project --python "$EMB_PYTHON" python -m examples.umi_relative_ee.embodiment_v2.pilots --steps 10000

# Replay and train the bounded-command data ablation; also resumable.
uv run --no-project --python "$EMB_PYTHON" python -m examples.umi_relative_ee.embodiment_v2.canonical_pipeline all

# Refresh the raw-command pilot report.
uv run --no-project --python "$EMB_PYTHON" python -m examples.umi_relative_ee.embodiment_v2.report

# Refresh separate feasible / infeasible analysis for both robot sweeps and pilots.
uv run --no-project --python "$EMB_PYTHON" python -m examples.umi_relative_ee.embodiment_v2.strata

PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run --no-project --python "$EMB_PYTHON" python -m pytest \
  tests/scripts/test_embodiment_v2.py \
  tests/scripts/test_task_independent_embodiment.py \
  tests/scripts/test_so101_task_independent_embodiment.py -q
```

The watcher updates `WATCH.md` and `watch-status.json` every 60 seconds, checks both
original source hashes, supervisor liveness, training/evaluation completion counts,
disk space and pilot failures. PID files and logs are stored beside those reports.
It flags failures for inspection rather than repeatedly restarting faulty code.

Pilot runners take an exclusive GPU lease, temporarily SIGSTOP only the two original
experiment Python processes, and SIGCONT them in cleanup. A separate watchdog can
recover a dead lease owner after its worker exits; it checks process creation times
before signaling to avoid PID reuse. CPU data preparation happens outside the lease.
The earlier sweeps share a GPU when not paused, and CPU collection can overlap pilots,
so existing timing numbers are not isolated hardware latency benchmarks.

## Research interpretation

The current study tests task-policy composition and trajectory execution on one
manipulation corpus in simulation. It cannot yet establish the proposed cross-task
O(N+M) benefit, real actuator compensation, task success, or the value of external
tracking. Keep infeasible queries in the all-query result and report feasibility
strata separately. Do not infer transfer from validation trajectory tracking alone.

The next controlled comparisons are all-development evaluation and seed replication,
then the same change on SO-101 and delay40 after replay checks pass,
matched query-vs-hindsight supervision, and C1 versus
an equal-data bootstrap control. If it does not, inspect feedback stability and
recovery-state coverage before adding more identical demonstrations or scaling models.

## Final illustrated report and scheduled cleanup

A separate completion monitor runs from the canonical parent
`/mnt/data1/projects/lerobot-embodiment`. Its plan, process ID, log and current
state are in `completion/`. It waits for both successful full-sweep exit statuses,
all 60 training completions, all 64 evaluation completions with 3,000 trials each,
complete matrix reports, completed pilots and no active experiment workers.
It also acquires the run locks before finalization.

An illustrated preview is available now in `report/preview/REPORT.md` under that
parent. Missing evaluations are explicitly marked pending. The report includes
separate feasible/infeasible comparisons, episode-bootstrap intervals and individual
training seeds, failure-screen coverage, paired iteration controls, learning curves,
the bounded-command ablation and representative trajectory figures. Markdown embeds the plots; PNG/SVG figures, JSON statistics and a PDF figure atlas
are exported alongside it. Representative trials are selected by median IK-baseline
error within each subset, independently of learned-controller performance.

After verified completion, the monitor builds `report/final/` first, then repairs
internal operational references and removes only the five old compatibility
symlinks. It preserves every rollout, checkpoint, immutable manifest and original
log. Operational files changed during cleanup are backed up under `cleanup/`.
Historical paths in immutable provenance remain documented by `relocation.json`.
The original Python experiment sources remain unchanged.

The finalization logic and report generator are `completion.py` and
`research_report.py`. Regenerate a preview using the canonical environment:

```bash
UV_CACHE_DIR=/tmp/umi-emb-uv-cache uv run --no-project \
  --python /mnt/data1/projects/lerobot-embodiment/env/bin/python \
  python -m examples.umi_relative_ee.embodiment_v2.completion preview
```
