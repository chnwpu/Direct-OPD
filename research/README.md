# Direct-OPD shift amplification study

This branch is the E0 vanilla OPD control. The `shift-amplification` branch is the E1 Direct-OPD experiment. E0 writes its own checkpoints and diagnostics under this branch's `runs/` directory and reads the same hash-verified models, dataset, and cloned Python environment as E1.

## E0 objective

E0 minimizes the exact Student-top-16 reverse KL of paper Eq. (4) against T1 on the Student rollout top-16 support. Student and T1 probabilities are each renormalized on that fixed support during the actor update. The code uses full-parameter FSDP training, the same rollout and evaluation settings as E1, and a zero coefficient on the Student-reference KL term. Reference log probabilities are still scored for measurement, and T0 is queried for per-step policy shift measurement. The original `opd_kl` reward path is retained for batch construction, but its approximate detached reward is not used as E0's actor loss. `research/test_e0_loss.py` verifies the exact KL gradient, including zero gradient outside the fixed top-16 support.

## Pinned sources

- Code: `chnwpu/Direct-OPD` at `5205aba17cd97c98398fd18af1246544e9239b00`, identical to upstream `BytedTsinghua-SIA/Direct-OPD` at the start of the study.
- Student: `Qwen/Qwen3-1.7B`, Hugging Face revision `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`; both local safetensors shards match the release SHA-256 values.
- T0: `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`, revision `ad9f0ae0864d7fbcd1cd905e3c6c5b069cc8b562`.
- T1: `hbx/JustRL-DeepSeek-1.5B`, revision `0637e4096c789c67f9eecbe8355e0bdeddede1c2`.
- The E1 launcher verifies SHA-256 for all four weight files before training, so partial downloads cannot be used.
- Training data: `Skywork/Skywork-OR1-RL-Data`, math split, 105,055 rows; converted SHA-256 `8a2b134901a995adaac486c2ba5daea7aeeaeb4a0f29c235e103cd6451633393`.

## Implementation audit

- `scripts/train_justrl_qwen.sh` defaults to `MINI_BATCH_SIZE=128`, and passes it as both `data.train_batch_size` and `actor.ppo_mini_batch_size`. The paper's table says 64. The reproduction must set `MINI_BATCH_SIZE=64` explicitly. `data.train_batch_size=64` counts prompts; with `rollout.n=4`, the actor sees 256 response trajectories per step, divided into mini batches of 64.
- The script defaults to `MAX_VAL_RESP_LENGTH=32768`, while Appendix A specifies 31744. Set `MAX_VAL_RESP_LENGTH=31744` explicitly.
- Appendix A says the DAPO `Answer:` template is used for evaluation, but the repository's AIME24/AIME25 parquet files use a `\\boxed{}` instruction and its supplied `ttrl_math.extract_answer` only accepts boxed answers. The primary reproduction follows the repository's prompt files and answer extractor as a matched pair; this paper/code discrepancy must be reported with the results.
- Direct-OPD uses `reward_mode=delta_opd`, `top_k_strategy=only_stu`, `log_prob_top_k=16`, `reward_weight_mode=student_p`, `adv_estimator=token_reward_direct`, `loss_agg_mode=token-mean`, and full FSDP actor training.
- `dp_actor._compute_delta_opd_rm_scores` computes `A_i=stop_gradient(softmax(log p_student on top-k)_i * (log p_T1_i - log p_T0_i))`. Actor updates use a 3D clipped PPO loss. With 64 prompts and four rollouts each, the 256 trajectories form four actor mini batches of 64; `on_policy` in `dp_actor` is therefore false and the code uses stored rollout log probabilities in the PPO ratio. Later mini batches can encounter a non-unit ratio. The separate sampled-token student-reference KL uses `low_var_kl` and an adaptive coefficient.
- Adaptive KL starts at 2.5, multiplies by `1 + 0.01 * sign(delta_opd/weighted_reward_mean)` before actor update, and clips to [0.5, 2.5]. The metric `weighted_reward_mean` is a mean over all valid top-k candidates, not a per-token sum.
- For the PG term when all PPO ratios equal 1 and clipping is inactive, the full-vocabulary logit gradient is `-A_j + p_student(j) * sum_i A_i` for a top-k candidate, and `p_student(j) * sum_i A_i` outside top-k. In the actual run, the clipped PPO ratio, mini batch order, and sampled-token KL must be included in measured gradient contributions. Thus `p_student * reward` alone is not the actual logit gradient.
- Every ten steps, the deterministic 32-trajectory/128-position snapshot stores full-vocabulary-normalized probabilities. The actor captures `d(actual scaled PPO+KL loss)/d logit` for the same positions after each micro batch forward pass, including PPO clipping and the current adaptive KL coefficient. The recorded Teacher rank is exact only when a candidate lies within the Teacher top-k; deeper ranks require offline rescoring.
- The launch script's `MANAGE_RAY=True` calls `ray stop --force` globally; use an isolated Ray process to protect unrelated jobs.
- The requested `research-bootstrap` Skill is unavailable in the installed skill catalog and on the local filesystem. This directory was initialized manually with pinned sources, isolated run paths, and an independent Python environment.
- The project environment at `.envs/direct-opd` is a local clone of `/data2/ch/projects/opd/vOPD-research/.envs/opd`, so it can run independently of that experiment. It contains Python 3.10.22, PyTorch 2.8.0+cu128, vLLM 0.11.0, Transformers 4.57.1, Ray 2.60.0, and FlashAttention 2.8.3; the run launcher saves the complete package list.

## Pending empirical gates

1. Finish pinned Teacher downloads and verify checksums and tokenizer compatibility.
2. Run E1's 1-step smoke test and 5-step profiling, including an end-to-end check of actor gradient capture.
3. Recheck available GPUs, start E1 on all idle cards, and start E0 after E1 is stable.
4. Evaluate checkpoints and report the three requested diagnostics. No empirical conclusion is available before these runs.

The E0 command is `bash research/run_e0.sh`; it writes the E0 configuration, code commit, environment, checkpoints, and diagnostics to `runs/e0_vanilla_opd`.

After E1 produces diagnostic snapshots, run `PYTHONNOUSERSITE=1 .envs/direct-opd/bin/python research/analyze_on_policy.py --diagnostic-dir runs/e1_direct_opd/diagnostics --output-dir runs/e1_direct_opd/analysis` to write candidate-level parquet, support-bin statistics, and probability/shift heatmaps. A positive logged gradient means the optimizer step tends to lower that candidate's logit.

The two Teacher tokenizers agree on the thinking tag IDs. Shared Student/Teacher vocabulary entries match except `<think>` and `</think>`. The official scorer uses raw Student IDs, which this reproduction preserves. On-policy analysis exports a sensitivity summary excluding candidate IDs 151648, 151649, 151667 and 151668. This filter is used only in reporting.
