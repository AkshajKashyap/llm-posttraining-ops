"""Deterministic regression gates for model evaluation artifacts."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

from llm_posttraining_ops.monitoring.logs import MonitoringError

RELEASE_GATE_SCHEMA_VERSION = "1.0"
DEFAULT_RELEASE_GATE_PATH = Path("artifacts/evals/release_gate.json")
DEFAULT_MAX_RELEASE_P95_LATENCY = 5.0
ReleaseStatus = Literal["pass", "fail"]


@dataclass(frozen=True, slots=True)
class GateMetrics:
    """Release-critical quality and optional latency metrics."""

    required_fact_coverage: float
    forbidden_term_violation_rate: float
    empty_response_rate: float
    p95_latency_seconds: float | None


@dataclass(frozen=True, slots=True)
class ReleaseGateCheck:
    """One auditable release comparison."""

    metric: str
    baseline: float | None
    current: float | None
    requirement: str
    passed: bool


@dataclass(frozen=True, slots=True)
class ReleaseGateResult:
    """Versioned release decision."""

    schema_version: str
    baseline_eval_path: str
    current_eval_path: str
    status: ReleaseStatus
    max_p95_latency_seconds: float
    baseline_metrics: GateMetrics
    current_metrics: GateMetrics
    checks: list[ReleaseGateCheck]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _load_json_object(path: str | Path) -> dict[str, Any]:
    input_path = Path(path)
    try:
        with input_path.open(encoding="utf-8") as input_file:
            payload = json.load(input_file)
    except (OSError, json.JSONDecodeError) as exc:
        raise MonitoringError(f"Could not load evaluation JSON at {input_path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise MonitoringError(f"{input_path}: evaluation artifact must be a JSON object")
    return payload


def _number(
    mapping: dict[str, Any],
    key: str,
    *,
    location: str,
) -> float:
    value = mapping.get(key)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise MonitoringError(f"{location}: metric '{key}' must be numeric")
    return float(value)


def _optional_p95(payload: dict[str, Any], location: str) -> float | None:
    metrics = payload.get("metrics")
    latency = payload.get("latency")
    candidates: list[object] = []
    if isinstance(metrics, dict):
        candidates.extend(
            [
                metrics.get("p95_latency_seconds"),
                metrics.get("p95_latency"),
            ]
        )
    if isinstance(latency, dict):
        candidates.extend(
            [
                latency.get("p95_latency_seconds"),
                latency.get("p95_seconds"),
                latency.get("p95_latency"),
            ]
        )
    candidates.extend(
        [
            payload.get("p95_latency_seconds"),
            payload.get("p95_latency"),
        ]
    )
    for candidate in candidates:
        if candidate is None:
            continue
        if not isinstance(candidate, (int, float)) or isinstance(candidate, bool):
            raise MonitoringError(f"{location}: p95 latency must be numeric")
        if candidate < 0:
            raise MonitoringError(f"{location}: p95 latency must be non-negative")
        return float(candidate)
    return None


def load_gate_metrics(path: str | Path) -> GateMetrics:
    """Load release-critical metrics from a rigorous evaluation artifact."""

    payload = _load_json_object(path)
    metrics = payload.get("metrics")
    if not isinstance(metrics, dict):
        raise MonitoringError(f"{path}: evaluation artifact is missing 'metrics'")
    return GateMetrics(
        required_fact_coverage=_number(
            metrics,
            "required_fact_coverage",
            location=str(path),
        ),
        forbidden_term_violation_rate=_number(
            metrics,
            "forbidden_term_violation_rate",
            location=str(path),
        ),
        empty_response_rate=_number(
            metrics,
            "empty_response_rate",
            location=str(path),
        ),
        p95_latency_seconds=_optional_p95(payload, str(path)),
    )


def compare_gate_metrics(
    baseline: GateMetrics,
    current: GateMetrics,
    *,
    max_p95_latency_seconds: float,
) -> tuple[ReleaseStatus, list[ReleaseGateCheck]]:
    """Compare current quality with baseline and enforce a latency ceiling."""

    if max_p95_latency_seconds < 0:
        raise ValueError("max_p95_latency_seconds must be non-negative")

    checks = [
        ReleaseGateCheck(
            metric="required_fact_coverage",
            baseline=baseline.required_fact_coverage,
            current=current.required_fact_coverage,
            requirement="current >= baseline",
            passed=current.required_fact_coverage >= baseline.required_fact_coverage,
        ),
        ReleaseGateCheck(
            metric="forbidden_term_violation_rate",
            baseline=baseline.forbidden_term_violation_rate,
            current=current.forbidden_term_violation_rate,
            requirement="current <= baseline",
            passed=(
                current.forbidden_term_violation_rate
                <= baseline.forbidden_term_violation_rate
            ),
        ),
        ReleaseGateCheck(
            metric="empty_response_rate",
            baseline=baseline.empty_response_rate,
            current=current.empty_response_rate,
            requirement="current <= baseline",
            passed=current.empty_response_rate <= baseline.empty_response_rate,
        ),
    ]
    if current.p95_latency_seconds is not None:
        checks.append(
            ReleaseGateCheck(
                metric="p95_latency_seconds",
                baseline=baseline.p95_latency_seconds,
                current=current.p95_latency_seconds,
                requirement=f"current <= {max_p95_latency_seconds:g}",
                passed=current.p95_latency_seconds <= max_p95_latency_seconds,
            )
        )
    status: ReleaseStatus = "pass" if all(check.passed for check in checks) else "fail"
    return status, checks


def write_release_gate_result(
    result: ReleaseGateResult,
    path: str | Path = DEFAULT_RELEASE_GATE_PATH,
) -> Path:
    """Write a stable release gate JSON artifact."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as output_file:
        json.dump(result.to_dict(), output_file, indent=2, sort_keys=True)
        output_file.write("\n")
    return output_path


def run_release_gate(
    baseline_eval_path: str | Path,
    current_eval_path: str | Path,
    *,
    max_p95_latency_seconds: float = DEFAULT_MAX_RELEASE_P95_LATENCY,
    output_path: str | Path = DEFAULT_RELEASE_GATE_PATH,
) -> ReleaseGateResult:
    """Compare two eval artifacts, save the decision, and return it."""

    baseline = load_gate_metrics(baseline_eval_path)
    current = load_gate_metrics(current_eval_path)
    status, checks = compare_gate_metrics(
        baseline,
        current,
        max_p95_latency_seconds=max_p95_latency_seconds,
    )
    result = ReleaseGateResult(
        schema_version=RELEASE_GATE_SCHEMA_VERSION,
        baseline_eval_path=str(baseline_eval_path),
        current_eval_path=str(current_eval_path),
        status=status,
        max_p95_latency_seconds=max_p95_latency_seconds,
        baseline_metrics=baseline,
        current_metrics=current,
        checks=checks,
    )
    write_release_gate_result(result, output_path)
    return result
