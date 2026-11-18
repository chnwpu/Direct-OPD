#!/usr/bin/env python3
"""Release Ray workers whose private training job has already exited."""

import os
import signal
import sys
import time
from pathlib import Path


def main():
    marker = f"DIRECT_OPD_WORKER_RUN_DIR={sys.argv[1]}".encode()

    def belongs_to_run(pid):
        try:
            return marker in (Path("/proc") / str(pid) / "environ").read_bytes().split(b"\0")
        except OSError:
            return False

    pids = [
        int(child.name) for child in Path("/proc").iterdir()
        if child.name.isdecimal() and belongs_to_run(int(child.name))
    ]
    if not pids:
        return
    print(f"Releasing remaining processes for {sys.argv[1]}: {pids}", file=sys.stderr)
    for sig in (signal.SIGTERM, signal.SIGKILL):
        for pid in pids:
            if belongs_to_run(pid):
                try:
                    os.kill(pid, sig)
                except ProcessLookupError:
                    pass
        if sig == signal.SIGTERM:
            time.sleep(3)


if __name__ == "__main__":
    main()
