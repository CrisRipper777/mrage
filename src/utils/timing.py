from __future__ import annotations

import time
from collections import defaultdict
from contextlib import contextmanager
from typing import Iterator

import torch


class PhaseProfiler:
    """Opt-in wall-clock phase timing with CUDA synchronization only when enabled."""

    def __init__(self, enabled: bool, device: torch.device):
        self.enabled = bool(enabled)
        self.device = device
        self.seconds: dict[str, float] = defaultdict(float)
        self._epoch_started = 0.0

    def _sync(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def start_epoch(self) -> None:
        if not self.enabled:
            return
        self.seconds.clear()
        self._sync()
        self._epoch_started = time.perf_counter()

    def finish_epoch(self) -> float:
        if not self.enabled:
            return 0.0
        self._sync()
        elapsed = time.perf_counter() - self._epoch_started
        self.seconds["epoch_total"] = elapsed
        return elapsed

    @contextmanager
    def measure(self, name: str) -> Iterator[None]:
        if not self.enabled:
            yield
            return
        self._sync()
        started = time.perf_counter()
        try:
            yield
        finally:
            self._sync()
            self.seconds[name] += time.perf_counter() - started

    def format_epoch(self) -> str:
        if not self.enabled:
            return ""
        return " | ".join(
            f"{name}={seconds:.4f}s"
            for name, seconds in sorted(self.seconds.items())
        )
