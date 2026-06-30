"""Durable experiment registry with per-stage status tracking."""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

StageStatus = Literal["running", "pass", "fail", "skipped"]
RunStatus = Literal["running", "pass", "fail"]
RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
REGISTRY_SCHEMA_VERSION = "1.0"


def utc_timestamp() -> str:
    """Return a UTC ISO-8601 timestamp."""

    return datetime.now(timezone.utc).isoformat()


def create_run_id(now: datetime | None = None) -> str:
    """Create a sortable run ID with microsecond precision."""

    timestamp = now or datetime.now(timezone.utc)
    normalized = timestamp.astimezone(timezone.utc)
    return f"run-{normalized.strftime('%Y%m%dT%H%M%S%fZ')}"


def validate_run_id(run_id: str) -> str:
    """Reject empty, unsafe, or path-like run identifiers."""

    if not RUN_ID_PATTERN.fullmatch(run_id):
        raise ValueError(
            "run_id must start with an alphanumeric character and contain only "
            "letters, numbers, '.', '_', or '-' (maximum 128 characters)"
        )
    return run_id


@dataclass(slots=True)
class StageRecord:
    """Lifecycle and artifacts for one workflow stage."""

    name: str
    status: StageStatus
    started_at: str
    ended_at: str | None = None
    artifacts: dict[str, str] = field(default_factory=dict)
    error: str | None = None
    reason: str | None = None


class ExperimentRegistry:
    """Append stage outcomes to a stable per-run JSON registry."""

    def __init__(
        self,
        run_id: str,
        path: str | Path,
        *,
        settings: Mapping[str, Any],
        timestamp_factory: Callable[[], str] = utc_timestamp,
    ) -> None:
        self.run_id = validate_run_id(run_id)
        self.path = Path(path)
        self.settings = dict(settings)
        self._timestamp_factory = timestamp_factory
        self.started_at = timestamp_factory()
        self.ended_at: str | None = None
        self.status: RunStatus = "running"
        self.stages: list[StageRecord] = []
        self.artifacts: dict[str, str] = {}
        self.write()

    def begin_stage(self, name: str) -> StageRecord:
        """Record and persist a running stage."""

        stage = StageRecord(
            name=name,
            status="running",
            started_at=self._timestamp_factory(),
        )
        self.stages.append(stage)
        self.write()
        return stage

    def finish_stage(
        self,
        stage: StageRecord,
        status: Literal["pass", "fail"],
        *,
        artifacts: Mapping[str, str] | None = None,
        error: str | None = None,
    ) -> None:
        """Complete a running stage and persist its outcome."""

        stage.status = status
        stage.ended_at = self._timestamp_factory()
        stage.artifacts = dict(artifacts or {})
        stage.error = error
        self.artifacts.update(stage.artifacts)
        self.write()

    def skip_stage(self, name: str, reason: str) -> StageRecord:
        """Persist an explicitly skipped stage."""

        timestamp = self._timestamp_factory()
        stage = StageRecord(
            name=name,
            status="skipped",
            started_at=timestamp,
            ended_at=timestamp,
            reason=reason,
        )
        self.stages.append(stage)
        self.write()
        return stage

    def add_artifacts(self, artifacts: Mapping[str, str]) -> None:
        """Register final workflow-level artifacts."""

        self.artifacts.update(artifacts)
        self.write()

    def finalize(self, status: Literal["pass", "fail"]) -> None:
        """Close the experiment registry."""

        self.status = status
        self.ended_at = self._timestamp_factory()
        self.write()

    def to_dict(self) -> dict[str, Any]:
        """Return the complete JSON-serializable registry."""

        return {
            "schema_version": REGISTRY_SCHEMA_VERSION,
            "run_id": self.run_id,
            "status": self.status,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "settings": self.settings,
            "stages": [asdict(stage) for stage in self.stages],
            "artifacts": dict(sorted(self.artifacts.items())),
        }

    def write(self) -> Path:
        """Atomically replace the registry JSON."""

        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = self.path.with_suffix(f"{self.path.suffix}.tmp")
        with temporary_path.open("w", encoding="utf-8", newline="\n") as output_file:
            json.dump(self.to_dict(), output_file, indent=2, sort_keys=True)
            output_file.write("\n")
        temporary_path.replace(self.path)
        return self.path
