# Direct-OPD shift amplification study

This branch is the E1 Direct-OPD experiment. The `vanilla-opd` branch is the E0 control. Raw weights, training data, checkpoints, and diagnostics belong under `models/`, `datasets/`, and `runs/` and are excluded from Git.

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
- Direct-OPD uses `reward_mode=delta_opd`, `top_k_strategy=only_stu`, `log_prob_top_k=16`, `reward_weight_mode=student_p`, `adv_estimator=token_reward_direct`, `loss_agg_mode=token-mean`, and full FSDP actor training.
- `dp_actor._compute_delta_opd_rm_scores` computes `A_i=stop_gradient(softmax(log p_student on top-k)_i * (log p_T1_i - log p_T0_i))`. Actor updates use a 3D clipped PPO loss. With 64 prompts and four rollouts each, the 256 trajectories form four actor mini batches of 64; `on_policy` in `dp_actor` is therefore false and the code uses stored rollout log probabilities in the PPO ratio. Later mini batches can encounter a non-unit ratio. The separate sampled-token student-reference KL uses `low_var_kl` and an adaptive coefficient.
- Adaptive KL starts at 2.5, multiplies by `1 + 0.01 * sign(delta_opd/weighted_reward_mean)` before actor update, and clips to [0.5, 2.5]. The metric `weighted_reward_mean` is a mean over all valid top-k candidates, not a per-token sum.
- For the PG term when all PPO ratios equal 1 and clipping is inactive, the full-vocabulary logit gradient is `-A_j + p_student(j) * sum_i A_i` for a top-k candidate, and `p_student(j) * sum_i A_i` outside top-k. In the actual run, the clipped PPO ratio, mini batch order, and sampled-token KL must be included in measured gradient contributions. Thus `p_student * reward` alone is not the actual logit gradient.
- The launch script's `MANAGE_RAY=True` calls `ray stop --force` globally; use an isolated Ray process to protect unrelated jobs.
- The requested `research-bootstrap` Skill is unavailable in the installed skill catalog and on the local filesystem. This directory was initialized manually with pinned sources, isolated run paths, and an independent Python environment.

## Pending empirical gates

1. Finish pinned Teacher downloads and verify checksums and tokenizer compatibility.
2. Install official dependencies in `.envs/direct-opd` and run the Loss unit test.
3. Implement diagnostics without changing the original optimization objective; run 1-step smoke test and 5-step profiling.
4. Recheck available GPUs, then start E1 on all idle cards and E0 after E1 is stable.
5. Evaluate checkpoints and report the three requested diagnostics. No empirical conclusion is available before these runs.
