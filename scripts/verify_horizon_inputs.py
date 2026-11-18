#!/usr/bin/env python3
"""Verify all locked horizon-fidelity inputs before a run starts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--only", nargs="+", help="Verify only the listed locked input paths")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    lock = json.loads((root / "experiments/horizon-fidelity/inputs.lock.json").read_text())
    results = {}
    selected = set(args.only) if args.only else set(lock["files"])
    unknown = selected.difference(lock["files"])
    if unknown:
        parser.error(f"unknown locked input paths: {', '.join(sorted(unknown))}")
    for name, expected in lock["files"].items():
        if name not in selected:
            continue
        path = root / name
        if not path.is_file():
            results[name] = {"ok": False, "reason": "missing"}
        elif path.stat().st_size != expected["bytes"]:
            results[name] = {
                "ok": False,
                "reason": "size_mismatch",
                "actual_bytes": path.stat().st_size,
                "expected_bytes": expected["bytes"],
            }
        else:
            actual_hash = sha256(path)
            results[name] = {
                "ok": actual_hash == expected["sha256"],
                "actual_sha256": actual_hash,
            }
    args.report.write_text(json.dumps({"sources": lock["sources"], "results": results}, indent=2) + "\n")
    failed = [name for name, result in results.items() if not result["ok"]]
    if failed:
        print("input verification failed: " + ", ".join(failed))
        return 4
    print("all locked inputs verified")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
