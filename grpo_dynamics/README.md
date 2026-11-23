# Qwen3 GRPO Dynamics

Project code lives in this separate checkout and branch, based on Direct-OPD
e6cde97 (vendored verl upstream snapshot 5205aba17cd97c98398fd18af1246544e9239b00).
The prior Direct-OPD runs and source weights are read-only inputs to this experiment.
Use the versions in machine/runtime.freeze.txt. Copy machine/local.env.example
to machine/local.env and set local model/runtime/data directories and eight GPU IDs.
Never run across hosts.

## Locked data and controls

grpo_dynamics/config/protocol.json defines the shared protocol and official model
and dataset revisions. grpo_dynamics/manifests/deepmath-v1.json pins the raw and
split hashes. Fetch ten parquet shards listed in its raw_sha256 from the pinned
official DeepMath revision. Run:

    python -m grpo_dynamics.prepare --raw DATA/raw --out DATA --tokenizer QWEN3_1P7B_PATH
    cmp DATA/manifest.json grpo_dynamics/manifests/deepmath-v1.json

The split order is SHA-256(seed, namespace, normalized question ID). Question
normalization uses Unicode NFKC, whitespace folding, and casefold. Exact question
duplicates share an ID; conflicting answers for duplicate IDs are excluded.
No fuzzy or semantic deduplication is claimed. Validation (256), probe (256)
and test (1024) are disjoint from each other and the training pool. The 128-question
preflight is a separate deterministic subset of the training pool. SHA-256 of
each ordered ID list and its parquet are tracked independently. Share the
canonical parquet/ID/manifest files and their byte hashes with 4B.

Use the same merge commit, dataset files, verifier, prompt policy and optimizer
objective on 4B. The 1.7B launch lock is intentionally model-specific. For 4B,
pin and verify its own original model/tokenizer files before enabling its launch.
Run that model's difficulty preflight on the same preflight IDs. Do not prepare
a second model-dependent split. No downstream dataset or benchmark result is used
to select training questions in this version.

## Gates and independent run IDs

    bash grpo_dynamics/train.sh config config-audit
    bash grpo_dynamics/preflight.sh deepmath-preflight-v1
    bash grpo_dynamics/run_gated.sh v1

The controller requires at least 25% mixed-correctness groups, then runs an
8 GPU smoke test, verifies HF export and resume to update 2, and runs a distinct
two-update throughput probe at full batch/length. A profile with over 20%
truncation or rollout/actor log-probability disagreement fails the gate. Only
when these pass does it launch 100 updates. The smoke uses a smaller prompt
batch and the candidate preflight set. These samples do not contribute to the
formal model. Use fresh run IDs for any retry.

Hydra's data and PPO mini batch fields count **prompts** at entry. The FSDP
actor scales the latter by rollout.n and divides by 8. The formal training
dispatch therefore has 1024 rollout trajectories, 128 per rank and one
optimizer update. A non-finite gradient or extra mini batch aborts the run.
The actor uses the stock vanilla clipped PPO loss and stock standard-deviation
normalized GRPO advantages. The token-mean loss uses a global denominator so
dynamic packing does not change the mathematical objective.

The synchronized driver uses seed 42, a seeded sampler, one PPO epoch,
non-thinking Qwen3 chat template, no KL and no entropy bonuses. vLLM sampling
seeds are keyed by step and pre-dispatch request index. This makes the random
draw identifiers recoverable after resumption. It does not imply bitwise
equality across GPU architectures or different parallel runtime kernels.

## Artifacts

- logs/resolved_config-*.yaml, launch.json, git_commit.txt, environment.freeze.txt,
  nvidia-smi.txt and data_manifest.json record the actual run.
- steps/step_NNN.json records each successfully completed update. Group std
  uses sample std as in verl. Advantage aggregates are over responses. Peak memory
  reports the cross-rank maximum of PyTorch CUDA reserved memory after rollout/actor
  phases; it is not a measurement of all driver/device allocations.
- rollouts/step_NNN.parquet stores all raw prompts/responses, sample and group IDs,
  gold/extracted answers, reward, advantage, finish reasons, IDs, masks, vLLM
  log probabilities and actor recomputed old log probabilities. Sampling config
  is Parquet schema metadata. It is written before the update, so a parquet
  without the corresponding step JSON is not evidence of optimizer completion.
- checkpoints/global_step_N retain rank shards, optimizer, scheduler, RNG and
  StatefulDataLoader state. This branch can retain every tenth full state.
- exports/step_NNN are independent HF BF16 inference exports (Step 0 references
  the pinned original files), with SHA-256, counts, config/data hash and inference
  checks in checkpoint_manifest.json.
- validation/step_NNN holds four trajectories per validation prompt, raw sample
  records and question-level paired bootstrap CIs. Validation is run independently
  using the completed exports, and never affects the training reward.
- Existing optimizer runs are never overwritten. To continue a verified step
  100 on the same world size and commit, use train.sh resume RUN_ID and the same
  GRPO_SMOKE_RUN and GRPO_PROFILE_RUN environment variables as for initial launch.

## Offline dynamics

python -m grpo_dynamics.dynamics --input teacher.npz --interval 0 30
--interval 30 60 --student-probe probe.npz --out OUT

Input checkpoints are ordered in steps; teacher_logp has shape
[checkpoints, prefixes, candidates]. Upstream scorers are responsible for using
the same candidate IDs and prefixes at every step. Provide checkpoint_reward
from the fixed validation set. Provide observed_index, trajectory_id and
trajectory_reward together for trajectory alignment. The optional Student file
has identical interface on the fixed Student probe and student_logp
[prefixes, candidates]. Output teacher_only.json and student_probe.json separately.
A truncated candidate distribution is labeled explicitly and its KL is
conditioned on the candidate support. The Student-weighted cosine compares each
requested interval with the next specified interval. Alignment and covariance
are descriptive quantities.

## Completed run audit and Teacher pilot

    python -m grpo_dynamics.audit_run --run runs/RUN_ID
    bash grpo_dynamics/teacher_pilot.sh RUN_ID

The audit verifies all 100 rollout row counts and cumulative token counts,
matches data/config provenance on every HF export, rehashes the export and
validation files, and inventories the full Step 50/100 recovery state. The
small Teacher pilot uses two fixed prefix positions from each of 32 Step 0
validation trajectories, balanced across correct and incorrect outcomes.
It scores the entire vocabulary with all 11 exported checkpoints and reports
adjacent plus selected non-adjacent interval metrics in
dynamics/teacher_pilot/teacher_only.json. Pilot labels describe the archived
Step 0 validation trajectories. They do not measure a Student's distillation
benefit or substitute for a separate Student Probe.
