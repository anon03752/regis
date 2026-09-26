"""Write a run's configuration, status, and scalar events to local JSON files.

config.json and status.json are overwritten; events.jsonl is appended to."""
import json
import time
from pathlib import Path


class Logger:
    def __init__(self, run_dir, config=None):
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        if config is not None:
            (self.run_dir / "config.json").write_text(json.dumps(config, indent=2, default=str))
        self._t0 = time.time()
        (self.run_dir / "status.json").write_text(
            json.dumps({"status": "running", "started_at": self._t0}, indent=2))
        self._events_fh = (self.run_dir / "events.jsonl").open("a", buffering=1)

    def scalars(self, values, step):
        ts = time.time()
        for name, v in values.items():
            self._events_fh.write(json.dumps(
                {"step": step, "kind": "scalar", "name": name, "value": float(v), "ts": ts}) + "\n")

    def close(self, status="completed"):
        (self.run_dir / "status.json").write_text(json.dumps(
            {"status": status, "started_at": self._t0, "ended_at": time.time()}, indent=2))
        self._events_fh.close()
