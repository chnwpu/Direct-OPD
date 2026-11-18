#!/usr/bin/env python3
"""Select every GPU with no compute process and negligible activity."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path


GPU_QUERY = [
    "nvidia-smi",
    "--query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu",
    "--format=csv,noheader,nounits",
]
PROCESS_QUERY = [
    "nvidia-smi",
    "--query-compute-apps=gpu_uuid,pid,used_gpu_memory",
    "--format=csv,noheader,nounits",
]


def rows(output: str) -> list[list[str]]:
    return [[cell.strip() for cell in row] for row in csv.reader(output.splitlines()) if row]


def select(gpu_output: str, process_output: str, batch_size: int) -> dict:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    busy_uuids = {row[0] for row in rows(process_output) if len(row) >= 2}
    devices = []
    for row in rows(gpu_output):
        if len(row) != 6:
            raise ValueError(f"unexpected nvidia-smi GPU row: {row!r}")
        index, uuid, name, total, used, utilization = row
        reasons = []
        if uuid in busy_uuids:
            reasons.append("compute_process")
        if int(used) > 512:
            reasons.append("memory_used_gt_512_MiB")
        if utilization == "[N/A]" or int(utilization) > 10:
            reasons.append("utilization_gt_10_percent_or_unknown")
        devices.append(
            {
                "index": int(index),
                "uuid": uuid,
                "name": name,
                "memory_total_mib": int(total),
                "memory_used_mib": int(used),
                "utilization_percent": utilization,
                "idle": not reasons,
                "excluded_reasons": reasons,
            }
        )
    if not devices:
        raise ValueError("nvidia-smi returned no GPUs")
    idle_indices = [device["index"] for device in devices if device["idle"]]
    valid_group_sizes = [size for size in range(1, len(idle_indices) + 1) if batch_size % size == 0]
    group_size = max(valid_group_sizes, default=0)
    return {
        "devices": devices,
        "batch_size": batch_size,
        "idle_indices": idle_indices,
        "selected_indices": idle_indices[:group_size],
        "idle_but_unselected_indices": idle_indices[group_size:],
        "selection_rule": "largest idle GPU group dividing effective rollout batch",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, required=True)
    args = parser.parse_args()
    try:
        gpu_output = subprocess.check_output(GPU_QUERY, text=True)
        process_output = subprocess.check_output(PROCESS_QUERY, text=True)
        selection = select(gpu_output, process_output, args.batch_size)
    except (OSError, subprocess.CalledProcessError, ValueError) as error:
        print(f"GPU discovery failed: {error}", file=sys.stderr)
        return 3
    args.snapshot.write_text(json.dumps(selection, indent=2) + "\n")
    if not selection["selected_indices"]:
        print("No idle NVIDIA GPUs found", file=sys.stderr)
        return 3
    print(",".join(str(index) for index in selection["selected_indices"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
