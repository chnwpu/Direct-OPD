# Horizon fidelity experiment

Source: `chnwpu/Direct-OPD`, with the patched official `verl` implementation.
The run wrapper keeps all training hyperparameters equal across H2, H4 and H6;
only `MAX_RESP_LENGTH` changes. It overrides the repository's adaptive KL
default to fixed KL coefficient 1 and uses only AIME 2024/2025 for validation.

Run order after the 1-step smoke and 5-step profiling gates:

```bash
bash scripts/run_horizon_fidelity.sh H2 smoke
bash scripts/run_horizon_fidelity.sh H2 profile
bash scripts/run_horizon_fidelity.sh H2 train
bash scripts/run_horizon_fidelity.sh H6 train
bash scripts/run_horizon_fidelity.sh H4 train
```

Each invocation creates a separate run directory under `runs/horizon-fidelity/`
with the code commit, local diff, installed package list, resolved parameters,
rollouts, logs and checkpoints. Checkpoints are requested every 5 training steps;
the full 32-sample AIME evaluation is requested at steps 0, 20 and 40.

At launch, `select_idle_gpus.py` finds every GPU with no compute process,
at most 512 MiB allocated and at most 10% utilization. The official verl
batch-balancing path requires the effective 256-rollout batch to be divisible
by the GPU count. The launcher selects the largest valid group of idle cards,
exports their physical indices as `CUDA_VISIBLE_DEVICES`, and passes the count
to verl as `GPUS_PER_NODE`. The snapshot records idle cards left available for
separate frozen-checkpoint analysis processes.

The paper's Figure 8 uses raw student probabilities on its top-16 set in Eq. 20.
The training implementation normalizes student weights within that set. Keep
the training loss untouched; use raw probabilities for the Figure 8 analysis.

No experimental result is recorded until a GPU run and verifier evaluation
actually finish. The paper's reported values are reference targets, not results
of this reproduction.
