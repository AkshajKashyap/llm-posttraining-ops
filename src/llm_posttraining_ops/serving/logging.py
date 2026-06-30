"""Append-only JSONL inference logging."""

from __future__ import annotations

import json
import threading
from collections.abc import Callable, Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DEFAULT_INFERENCE_LOG_PATH = Path("artifacts/logs/inference_logs.jsonl")


def utc_timestamp() -> str:
    """Return an ISO-8601 UTC timestamp."""

    return datetime.now(timezone.utc).isoformat()


class InferenceLogger:
    """Thread-safe JSONL logger with injectable timestamps for tests."""

    def __init__(
        self,
        path: str | Path = DEFAULT_INFERENCE_LOG_PATH,
        *,
        timestamp_factory: Callable[[], str] = utc_timestamp,
    ) -> None:
        self.path = Path(path)
        self._timestamp_factory = timestamp_factory
        self._lock = threading.Lock()

    def write(self, record: Mapping[str, Any]) -> Path:
        """Append one structured inference event."""

        payload = {"timestamp": self._timestamp_factory(), **dict(record)}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        with self._lock, self.path.open("a", encoding="utf-8", newline="\n") as log_file:
            log_file.write(encoded)
            log_file.write("\n")
        return self.path
