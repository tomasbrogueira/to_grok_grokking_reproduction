"""Run directories: resolved config, log, metrics, checkpoints and results."""
import json
import os
import platform
import sys
import time
from pathlib import Path

import torch
import yaml


def pick_device(requested="auto"):
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def write_json(path, obj):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj))
    tmp.replace(path)


class Run:
    def __init__(self, base, name, resolved, resume):
        self.dir = Path(base) / name
        self.results = self.dir / "results"
        self.ckpts = self.dir / "checkpoints"
        cfg_path = self.dir / "config.yaml"
        text = yaml.safe_dump(resolved, sort_keys=True)
        if self.dir.exists():
            if not resume:
                raise FileExistsError(f"{self.dir} exists. Use --resume or a new --run-name.")
            if yaml.safe_load(cfg_path.read_text()) != yaml.safe_load(text):
                raise ValueError(f"--resume with a different config than {cfg_path}")
        else:
            if resume:
                raise FileNotFoundError(f"nothing to resume at {self.dir}")
            self.results.mkdir(parents=True)
            self.ckpts.mkdir()
            cfg_path.write_text(text)
        self._log = open(self.dir / "log.txt", "a", buffering=1)
        self._metrics = open(self.dir / "metrics.jsonl", "a", buffering=1)

    def log(self, message):
        line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {message}"
        print(line, flush=True)
        self._log.write(line + "\n")
        self._log.flush()

    def metric(self, **row):
        self._metrics.write(json.dumps(row) + "\n")
        self._metrics.flush()
        os.fsync(self._metrics.fileno())

    def result_path(self, key):
        return self.results / f"{key}.json"

    def has_result(self, key):
        return self.result_path(key).exists()

    def save_result(self, key, obj):
        write_json(self.result_path(key), obj)

    def env(self):
        return dict(python=sys.version.split()[0], torch=torch.__version__, platform=platform.platform(),
                    cuda=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                    threads=torch.get_num_threads())
