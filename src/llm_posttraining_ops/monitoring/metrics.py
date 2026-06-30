"""Deterministic operational metrics and threshold evaluation."""

from __future__ import annotations

import json
import math
from collections import Counter
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

from llm_posttraining_ops.monitoring.logs import InferenceLogRecord, load_inference_logs

MonitoringStatus = Literal["pass", "warn", "fail"]
MONITORING_SCHEMA_VERSION = "1.0"
DEFAULT_MONITORING_SUMMARY_PATH = Path("artifacts/evals/monitoring_summary.json")


@dataclass(frozen=True, slots=True)
class MonitoringThresholds:
    """Hard operational limits; warning begins at 80% utilization."""

    max_error_rate: float = 0.05
    max_p95_latency: float = 5.0
    min_average_response_length: float = 1.0
    max_empty_response_rate: float = 0.05

    def __post_init__(self) -> None:
        rate_fields = (
            ("max_error_rate", self.max_error_rate),
            ("max_empty_response_rate", self.max_empty_response_rate),
        )
        for name, value in rate_fields:
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1")
        if self.max_p95_latency < 0:
            raise ValueError("max_p95_latency must be non-negative")
        if self.min_average_response_length < 0:
            raise ValueError("min_average_response_length must be non-negative")


@dataclass(frozen=True, slots=True)
class MonitoringMetrics:
    """Aggregate request, latency, response, and routing metrics."""

    request_count: int
    error_rate: float
    average_latency_seconds: float
    p50_latency_seconds: float
    p95_latency_seconds: float
    average_response_length_tokens: float
    empty_response_rate: float
    mock_request_count: int
    real_request_count: int
    requests_by_endpoint: dict[str, int]
    requests_by_model: dict[str, int]


@dataclass(frozen=True, slots=True)
class ThresholdCheck:
    """Outcome for one monitored threshold."""

    metric: str
    value: float
    threshold: float
    comparison: str
    status: MonitoringStatus


@dataclass(frozen=True, slots=True)
class MonitoringResult:
    """Versioned monitoring output ready for JSON and Markdown reports."""

    schema_version: str
    logs_path: str
    status: MonitoringStatus
    metrics: MonitoringMetrics
    thresholds: MonitoringThresholds
    checks: list[ThresholdCheck]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def percentile(values: Sequence[float], quantile: float) -> float:
    """Calculate a linearly interpolated percentile deterministically."""

    if not values:
        raise ValueError("Cannot calculate a percentile of an empty sequence")
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("quantile must be between 0 and 1")

    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return round(ordered[lower], 6)
    weight = position - lower
    return round(ordered[lower] + (ordered[upper] - ordered[lower]) * weight, 6)


def calculate_monitoring_metrics(
    records: Sequence[InferenceLogRecord],
) -> MonitoringMetrics:
    """Aggregate validated inference events."""

    if not records:
        raise ValueError("Cannot monitor an empty record sequence")

    count = len(records)
    latencies = [record.latency_seconds for record in records]
    response_lengths = [record.response_length_tokens for record in records]
    error_count = sum(record.status == "error" for record in records)
    mock_count = sum(record.mock for record in records)
    return MonitoringMetrics(
        request_count=count,
        error_rate=round(error_count / count, 6),
        average_latency_seconds=round(sum(latencies) / count, 6),
        p50_latency_seconds=percentile(latencies, 0.5),
        p95_latency_seconds=percentile(latencies, 0.95),
        average_response_length_tokens=round(sum(response_lengths) / count, 6),
        empty_response_rate=round(
            sum(length == 0 for length in response_lengths) / count,
            6,
        ),
        mock_request_count=mock_count,
        real_request_count=count - mock_count,
        requests_by_endpoint=dict(
            sorted(Counter(record.endpoint for record in records).items())
        ),
        requests_by_model=dict(
            sorted(Counter(record.model_name for record in records).items())
        ),
    )


def _maximum_check(
    metric: str,
    value: float,
    threshold: float,
) -> ThresholdCheck:
    if value > threshold:
        status: MonitoringStatus = "fail"
    elif threshold > 0 and value >= threshold * 0.8:
        status = "warn"
    else:
        status = "pass"
    return ThresholdCheck(
        metric=metric,
        value=value,
        threshold=threshold,
        comparison="<=",
        status=status,
    )


def _minimum_check(
    metric: str,
    value: float,
    threshold: float,
) -> ThresholdCheck:
    if value < threshold:
        status: MonitoringStatus = "fail"
    elif threshold > 0 and value < threshold * 1.2:
        status = "warn"
    else:
        status = "pass"
    return ThresholdCheck(
        metric=metric,
        value=value,
        threshold=threshold,
        comparison=">=",
        status=status,
    )


def evaluate_thresholds(
    metrics: MonitoringMetrics,
    thresholds: MonitoringThresholds,
) -> tuple[MonitoringStatus, list[ThresholdCheck]]:
    """Evaluate hard limits and deterministic 80% warning bands."""

    checks = [
        _maximum_check("error_rate", metrics.error_rate, thresholds.max_error_rate),
        _maximum_check(
            "p95_latency_seconds",
            metrics.p95_latency_seconds,
            thresholds.max_p95_latency,
        ),
        _minimum_check(
            "average_response_length_tokens",
            metrics.average_response_length_tokens,
            thresholds.min_average_response_length,
        ),
        _maximum_check(
            "empty_response_rate",
            metrics.empty_response_rate,
            thresholds.max_empty_response_rate,
        ),
    ]
    statuses = {check.status for check in checks}
    status: MonitoringStatus
    if "fail" in statuses:
        status = "fail"
    elif "warn" in statuses:
        status = "warn"
    else:
        status = "pass"
    return status, checks


def write_monitoring_summary(
    result: MonitoringResult,
    path: str | Path = DEFAULT_MONITORING_SUMMARY_PATH,
) -> Path:
    """Write a stable monitoring summary JSON artifact."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as output_file:
        json.dump(result.to_dict(), output_file, indent=2, sort_keys=True)
        output_file.write("\n")
    return output_path


def monitor_inference_logs(
    logs_path: str | Path,
    *,
    thresholds: MonitoringThresholds | None = None,
    output_path: str | Path = DEFAULT_MONITORING_SUMMARY_PATH,
) -> MonitoringResult:
    """Load inference logs, calculate metrics, apply thresholds, and save JSON."""

    active_thresholds = thresholds or MonitoringThresholds()
    metrics = calculate_monitoring_metrics(load_inference_logs(logs_path))
    status, checks = evaluate_thresholds(metrics, active_thresholds)
    result = MonitoringResult(
        schema_version=MONITORING_SCHEMA_VERSION,
        logs_path=str(logs_path),
        status=status,
        metrics=metrics,
        thresholds=active_thresholds,
        checks=checks,
    )
    write_monitoring_summary(result, output_path)
    return result
