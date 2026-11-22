"""Lossless per-step scientific records for the synchronous GRPO driver."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from grpo_dynamics.prepare import PROTOCOL


def sampling_seeds(step, count, validation=False):
    key = "validation" if validation else f"train:{step}"
    return np.array([
        int.from_bytes(hashlib.sha256(f"{PROTOCOL['seed']}:{key}:{i}".encode()).digest()[:8], "big") & ((1 << 63) - 1)
        for i in range(count)
    ], dtype=np.int64)


ROLLOUT_SCHEMA = pa.schema([
    ("global_step", pa.int32()), ("policy_step", pa.int32()), ("problem_id", pa.string()),
    ("sample_id", pa.string()), ("group_id", pa.string()), ("sampling_seed", pa.int64()),
    ("prompt", pa.string()), ("prompt_token_ids", pa.list_(pa.int32())),
    ("response_text", pa.string()), ("response_token_ids", pa.list_(pa.int32())),
    ("ground_truth", pa.string()), ("extracted_answer", pa.string()), ("reward", pa.float32()),
    ("group_mean_reward", pa.float64()), ("group_std_reward", pa.float64()),
    ("normalized_advantage", pa.float32()), ("response_length", pa.int32()),
    ("eos", pa.bool_()), ("truncated", pa.bool_()), ("finish_reason", pa.string()),
    ("rollout_logp", pa.list_(pa.float32())), ("actor_old_logp", pa.list_(pa.float32())),
    ("response_mask", pa.list_(pa.bool_())),
])


def atomic_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, allow_nan=False, sort_keys=True, indent=2) + "\n")
    tmp.replace(path)


def capture(batch, extra, trainer, dirname):
    step = trainer.global_steps
    folder = Path(dirname) / "rollouts"
    folder.mkdir(parents=True, exist_ok=True)
    b, meta = batch.batch, batch.non_tensor_batch
    samples = trainer.config.actor_rollout_ref.rollout.n
    expected = trainer.config.data.train_batch_size * samples
    if len(batch) != expected:
        raise ValueError(f"Expected {expected} rollouts, got {len(batch)}")
    scores = b["token_level_scores"].sum(-1).float().cpu().numpy()
    groups = {}
    for i, group in enumerate(meta["uid"]):
        groups.setdefault(str(group), []).append(i)
    if not np.isin(scores, [0, 1]).all() or any(len(g) != samples for g in groups.values()):
        raise ValueError("Expected binary rewards and complete GRPO groups")
    grouped = np.stack([scores[g] for g in groups.values()])
    means = grouped.mean(1)
    deviations = grouped.std(1, ddof=1)
    stats = {k: (float(means[i]), float(deviations[i])) for i, k in enumerate(groups)}
    masks = b["response_mask"].bool().cpu()
    advantages = b["advantages"].float().cpu()
    responses = b["responses"].cpu()
    old = b["old_log_probs"].float().cpu()
    sampling = b["rollout_log_probs"].float().cpu()
    prompts = b["prompts"].cpu()
    pmask = b["attention_mask"][:, :prompts.shape[-1]].bool().cpu()
    eos = trainer.tokenizer.eos_token_id
    lengths, scalar_adv, eos_flags, trunc_flags = [], [], [], []
    writer_path = folder / f"step_{step:03}.parquet"
    temporary = writer_path.with_suffix(".tmp")
    rollout_config = trainer.config.actor_rollout_ref.rollout
    sampling_config = {
        "temperature": float(rollout_config.temperature), "top_p": float(rollout_config.top_p),
        "top_k": int(rollout_config.top_k), "max_tokens": int(rollout_config.response_length),
        "samples_per_prompt": int(samples), "enable_thinking": False,
    }
    schema = ROLLOUT_SCHEMA.with_metadata({"sampling_config": json.dumps(sampling_config)})
    with pq.ParquetWriter(temporary, schema, compression="zstd") as writer:
        records = []
        for i, uid in enumerate(meta["uid"]):
            group = str(uid)
            mean, std = stats[group]
            valid = masks[i]
            ids = responses[i][valid].tolist()
            if not ids or len(ids) != int(meta["grpo_output_length"][i]):
                raise ValueError("Actor response mask differs from vLLM response length")
            adv = float(advantages[i][valid][0])
            target = (float(scores[i]) - mean) / (std + 1e-6)
            if not np.allclose(advantages[i][valid].numpy(), target, atol=1e-5, rtol=1e-5):
                raise ValueError("Recorded GRPO advantage does not reconstruct")
            finish = str(meta["grpo_finish_reason"][i])
            lengths.append(len(ids))
            scalar_adv.append(adv)
            eos_flags.append(ids[-1] == eos)
            trunc_flags.append(finish == "length")
            info = meta["extra_info"][i]
            pid = info["problem_id"]
            seed = int(meta["grpo_sampling_seed"][i])
            prompt_ids = prompts[i][pmask[i]].tolist()
            records.append({
                "global_step": step, "policy_step": step - 1, "problem_id": pid,
                "sample_id": f"{step}:{seed}", "group_id": group, "sampling_seed": seed,
                "prompt": trainer.tokenizer.decode(prompt_ids, skip_special_tokens=False),
                "prompt_token_ids": prompt_ids, "response_text": trainer.tokenizer.decode(ids, skip_special_tokens=True),
                "response_token_ids": ids, "ground_truth": meta["reward_model"][i]["ground_truth"],
                "extracted_answer": str(extra["extracted_answer"][i]), "reward": float(scores[i]),
                "group_mean_reward": mean, "group_std_reward": std, "normalized_advantage": adv,
                "response_length": len(ids), "eos": eos_flags[-1], "truncated": trunc_flags[-1],
                "finish_reason": finish, "rollout_logp": sampling[i][valid].tolist(),
                "actor_old_logp": old[i][valid].tolist(), "response_mask": valid.tolist(),
            })
            if len(records) == 32:
                writer.write_table(pa.Table.from_pylist(records, schema=schema))
                records = []
        if records:
            writer.write_table(pa.Table.from_pylist(records, schema=schema))
    temporary.replace(writer_path)
    length = np.array(lengths)
    av = np.array(scalar_adv)
    logp_delta = (old - sampling)[masks].abs()
    return {
        "train_reward_mean": float(scores.mean()), "train_reward_std": float(scores.std()),
        "train_accuracy": float(scores.mean()), "effective_group_ratio": float((deviations > 0).mean()),
        "all_correct_group_ratio": float((means == 1).mean()), "all_wrong_group_ratio": float((means == 0).mean()),
        "group_reward_variance": float((deviations ** 2).mean()),
        "advantage_mean": float(av.mean()), "advantage_std": float(av.std()), "advantage_abs_mean": float(np.abs(av).mean()),
        "response_length_mean": float(length.mean()), "response_length_p50": float(np.quantile(length, .50)),
        "response_length_p90": float(np.quantile(length, .90)), "response_length_p99": float(np.quantile(length, .99)),
        "eos_rate": float(np.mean(eos_flags)), "truncation_rate": float(np.mean(trunc_flags)),
        "generated_tokens": int(length.sum()), "generated_rollouts": len(batch), "prompt_count": len(groups),
        "rollout_actor_logp_abs_mean_difference": float(logp_delta.mean()),
        "rollout_actor_logp_abs_max_difference": float(logp_delta.max()),
    }


def save_step(dirname, step, rollouts, metrics, timings):
    prefix = Path(dirname)
    previous = prefix / "steps" / f"step_{step-1:03}.json"
    if step > 1 and not previous.is_file():
        raise FileNotFoundError(f"Missing preceding step record {previous}")
    prev = json.loads(previous.read_text()) if step > 1 else {}
    output = {**rollouts, "global_step": step}
    output["cumulative_generated_tokens"] = prev.get("cumulative_generated_tokens", 0) + output["generated_tokens"]
    output["cumulative_prompt_count"] = prev.get("cumulative_prompt_count", 0) + output["prompt_count"]
    names = {
        "actor_loss": "actor/pg_loss", "policy_entropy": "actor/entropy", "approximate_policy_kl": "actor/ppo_kl",
        "clip_fraction": "actor/pg_clipfrac", "gradient_norm": "actor/grad_norm", "learning_rate": "actor/lr",
        "peak_gpu_memory_gib": "grpo/actor_rollout_peak_reserved_gib",
    }
    for key, source in names.items():
        output[key] = float(metrics[source])
    output.update(rollout_time=timings["gen"], actor_update_time=timings["update_actor"],
                  checkpoint_time=timings.get("save_checkpoint", 0.0), total_step_time=timings["step"])
    atomic_json(prefix / "steps" / f"step_{step:03}.json", output)
