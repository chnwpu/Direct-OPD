"""Materialize deterministic disjoint splits from the pinned DeepMath snapshot."""

import argparse
import hashlib
import json
import re
import unicodedata
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

PROTOCOL_PATH = Path(__file__).parent / "config/protocol.json"
PROTOCOL = json.loads(PROTOCOL_PATH.read_text())


def canonical_question(text):
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text)).strip().casefold()


def digest(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path):
    with Path(path).open("rb") as stream:
        h = hashlib.sha256()
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def rank(row, namespace):
    return digest(f"{PROTOCOL['seed']}:{namespace}:{row['extra_info']['problem_id']}")


def create_splits(raw, eligible=None):
    by_id = {}
    conflicts = set()
    for row in raw:
        q = row["question"]
        answer = row["final_answer"]
        if not isinstance(q, str) or not isinstance(answer, str) or not q.strip() or not answer.strip():
            continue
        key = digest(canonical_question(q))
        if key in by_id and by_id[key]["reward_model"]["ground_truth"] != answer.strip():
            conflicts.add(key)
        value = {
            "data_source": PROTOCOL["dataset"],
            "prompt": [{"role": "user", "content": PROTOCOL["prompt"] + "\n\n" + q.strip()}],
            "reward_model": {"style": "rule", "ground_truth": answer.strip()},
            "extra_info": {
                "problem_id": key, "question": q.strip(), "difficulty": row["difficulty"],
                "topic": row["topic"], "source": PROTOCOL["dataset"],
            },
        }
        if key not in by_id or json.dumps(value, sort_keys=True) < json.dumps(by_id[key], sort_keys=True):
            by_id[key] = value
    ordered = sorted((v for k,v in by_id.items() if k not in conflicts), key=lambda x: rank(x, "split"))
    if eligible is not None:
        ordered = eligible(ordered)
    splits = {}
    for name in ("validation", "probe", "test"):
        count = PROTOCOL[f"{name}_size"]
        splits[name], ordered = ordered[:count], ordered[count:]
    if len(ordered) < PROTOCOL["preflight_size"]:
        raise ValueError("Insufficient distinct DeepMath prompts")
    splits["train"] = ordered
    splits["preflight"] = sorted(ordered, key=lambda x: rank(x, "preflight"))[:PROTOCOL["preflight_size"]]
    return splits, len(conflicts)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--raw", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--tokenizer", type=Path, required=True)
    args = p.parse_args()
    paths = sorted(args.raw.glob("train-*-of-00010.parquet"))
    if len(paths) != 10 or list(args.raw.glob("*.aria2")):
        raise SystemExit("Expected ten completed official DeepMath shards")
    def source_rows():
        for path in paths:
            for batch in pq.ParquetFile(path).iter_batches(columns=["question", "final_answer", "difficulty", "topic"]):
                yield from batch.to_pylist()
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    discarded = []
    def eligible(records):
        retained = []
        for i in range(0, len(records), 512):
            batch = records[i:i+512]
            encoded = tokenizer.apply_chat_template(
                [r["prompt"] for r in batch], add_generation_prompt=True, enable_thinking=False,
                tokenize=True, return_dict=False,
            )
            for row, ids in zip(batch, encoded, strict=True):
                if len(ids) <= PROTOCOL["prompt_max"]:
                    retained.append(row)
                else:
                    discarded.append(row["extra_info"]["problem_id"])
        return retained

    splits, conflicts = create_splits(source_rows(), eligible=eligible)
    args.out.mkdir(parents=True, exist_ok=True)
    manifest = {
        "source": PROTOCOL["dataset"], "revision": PROTOCOL["dataset_revision"], "seed": PROTOCOL["seed"],
        "protocol_sha256": sha256_file(PROTOCOL_PATH), "removed_conflicting_question_ids": conflicts,
        "removed_overlong_prompts": len(discarded), "raw_sha256": {p.name: sha256_file(p) for p in paths}, "splits": {},
    }
    for name, records in splits.items():
        ids = args.out / f"{name}.ids"
        ids.write_text("".join(x["extra_info"]["problem_id"] + "\n" for x in records))
        path = args.out / f"{name}.parquet"
        pq.write_table(pa.Table.from_pylist(records), path)
        manifest["splits"][name] = {
            "rows": len(records), "ids_sha256": sha256_file(ids), "parquet_sha256": sha256_file(path),
        }
    path = args.out / "manifest.json"
    path.write_text(json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n")
    print(path, sha256_file(path))


if __name__ == "__main__":
    main()
