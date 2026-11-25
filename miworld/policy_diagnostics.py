"""Lightweight per-process progress records for M3 policy data and training."""

import json
import os
from pathlib import Path
import time


_files = {}


def progress(stage, **fields):
    """Append a stage record when ``POLICY_DIAGNOSTIC_DIR`` is configured."""
    directory = os.environ.get("POLICY_DIAGNOSTIC_DIR")
    if not directory:
        return
    pid = os.getpid()
    if pid not in _files:
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        handle = (
            path / f"rank{os.environ.get('RANK', '0')}_pid{pid}.log"
        ).open("a", buffering=1)
        _files[pid] = handle
    handle = _files[pid]
    handle.write(json.dumps({"time": time.time(), "stage": stage, **fields}) + "\n")
