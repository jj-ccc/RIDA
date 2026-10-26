"""Console and JSON-lines logging."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

__all__ = ["Logger"]


class Logger:
    def __init__(self, work_dir: str | Path):
        self.work_dir = Path(work_dir)
        (self.work_dir / "logs").mkdir(parents=True, exist_ok=True)
        self.metrics_path = self.work_dir / "logs" / "metrics.jsonl"
        self.log_path = self.work_dir / "logs" / "train.log"

    def info(self, message: str) -> None:
        line = time.strftime("%Y-%m-%d %H:%M:%S ") + message
        print(line, flush=True)
        with open(self.log_path, "a") as f:
            f.write(line + "\n")

    def record(self, row: dict) -> None:
        with open(self.metrics_path, "a") as f:
            f.write(json.dumps(row) + "\n")

    def scalars(self, row: dict, keys=None) -> None:
        self.record(row)
        keys = keys or [k for k in row if isinstance(row[k], float)]
        text = " ".join(f"{k}={row[k]:.4f}" for k in keys if k in row and isinstance(row[k], (int, float)))
        self.info(f"[iter {row.get('iter', '')}] {text}")
        sys.stdout.flush()
