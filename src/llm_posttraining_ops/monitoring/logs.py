"""Loading and validation for structured inference JSONL logs."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

LogStatus = Literal["ok", "error"]


class MonitoringError(ValueError):
    """Raised when monitoring inputs are missing or malformed."""


@dataclass(frozen=True, slots=True)
class InferenceLogRecord:
    """Fields required to calculate operational service metrics."""

    request_id: str
    timestamp: str
    endpoint: str
    status: LogStatus
    model_name: str
    mock: bool
    latency_seconds: float
    response_length_tokens: int


def _required_string(
    record: dict[str, object],
    field: str,
    location: str,
) -> str:
    value = record.get(field)
    if not isinstance(value, str) or not value.strip():
        raise MonitoringError(f"{location}: field '{field}' must be a non-empty string")
    return value


def _parse_record(
    record: object,
    *,
    location: str,
) -> InferenceLogRecord:
    if not isinstance(record, dict):
        raise MonitoringError(f"{location}: expected a JSON object")

    status = _required_string(record, "status", location)
    if status not in {"ok", "error"}:
        raise MonitoringError(f"{location}: unsupported status '{status}'")

    mock = record.get("mock")
    if not isinstance(mock, bool):
        raise MonitoringError(f"{location}: field 'mock' must be a boolean")

    latency = record.get("latency_seconds")
    if not isinstance(latency, (int, float)) or isinstance(latency, bool) or latency < 0:
        raise MonitoringError(
            f"{location}: field 'latency_seconds' must be a non-negative number"
        )

    response_length = record.get("response_length_tokens")
    if (
        not isinstance(response_length, int)
        or isinstance(response_length, bool)
        or response_length < 0
    ):
        raise MonitoringError(
            f"{location}: field 'response_length_tokens' "
            "must be a non-negative integer"
        )

    return InferenceLogRecord(
        request_id=_required_string(record, "request_id", location),
        timestamp=_required_string(record, "timestamp", location),
        endpoint=_required_string(record, "endpoint", location),
        status=cast(LogStatus, status),
        model_name=_required_string(record, "model_name", location),
        mock=mock,
        latency_seconds=float(latency),
        response_length_tokens=response_length,
    )


def load_inference_logs(path: str | Path) -> list[InferenceLogRecord]:
    """Load validated inference events from an append-only JSONL file."""

    input_path = Path(path)
    records: list[InferenceLogRecord] = []
    try:
        with input_path.open(encoding="utf-8") as input_file:
            for line_number, line in enumerate(input_file, start=1):
                if not line.strip():
                    continue
                location = f"{input_path}:line {line_number}"
                try:
                    raw_record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise MonitoringError(
                        f"{location}: invalid JSON: {exc.msg}"
                    ) from exc
                records.append(_parse_record(raw_record, location=location))
    except OSError as exc:
        raise MonitoringError(f"Could not read inference logs at {input_path}: {exc}") from exc

    if not records:
        raise MonitoringError(f"{input_path}: inference log is empty")
    return records
