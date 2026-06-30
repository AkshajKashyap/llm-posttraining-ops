"""Markdown reports for monitoring summaries and release gates."""

from __future__ import annotations

from pathlib import Path

from llm_posttraining_ops.monitoring.metrics import MonitoringResult
from llm_posttraining_ops.monitoring.release_gate import ReleaseGateResult

DEFAULT_MONITORING_REPORT_PATH = Path("reports/monitoring_report.md")
DEFAULT_RELEASE_GATE_REPORT_PATH = Path("reports/release_gate_report.md")


def render_monitoring_report(result: MonitoringResult) -> str:
    """Render operational metrics, threshold checks, and routing counts."""

    metrics = result.metrics
    lines = [
        "# Inference Monitoring Report",
        "",
        f"- Status: **{result.status.upper()}**",
        f"- Logs: `{result.logs_path}`",
        f"- Requests: {metrics.request_count}",
        "",
        "## Service metrics",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| Error rate | {metrics.error_rate:.3f} |",
        f"| Average latency (seconds) | {metrics.average_latency_seconds:.3f} |",
        f"| p50 latency (seconds) | {metrics.p50_latency_seconds:.3f} |",
        f"| p95 latency (seconds) | {metrics.p95_latency_seconds:.3f} |",
        (
            "| Average response length (tokens) "
            f"| {metrics.average_response_length_tokens:.3f} |"
        ),
        f"| Empty response rate | {metrics.empty_response_rate:.3f} |",
        f"| Mock requests | {metrics.mock_request_count} |",
        f"| Real requests | {metrics.real_request_count} |",
        "",
        "## Threshold checks",
        "",
        "| Metric | Value | Requirement | Status |",
        "| --- | ---: | ---: | --- |",
    ]
    lines.extend(
        f"| {check.metric} | {check.value:.3f} "
        f"| {check.comparison} {check.threshold:.3f} | {check.status} |"
        for check in result.checks
    )
    lines.extend(
        [
            "",
            "## Requests by endpoint",
            "",
            "| Endpoint | Count |",
            "| --- | ---: |",
        ]
    )
    lines.extend(
        f"| `{endpoint}` | {count} |"
        for endpoint, count in metrics.requests_by_endpoint.items()
    )
    lines.extend(
        [
            "",
            "## Requests by model",
            "",
            "| Model | Count |",
            "| --- | ---: |",
        ]
    )
    lines.extend(
        f"| `{model}` | {count} |"
        for model, count in metrics.requests_by_model.items()
    )
    lines.extend(
        [
            "",
            "Warning status begins at 80% of a maximum limit or within 20% of a",
            "minimum requirement. A breached threshold produces failure.",
            "",
        ]
    )
    return "\n".join(lines)


def write_monitoring_report(
    result: MonitoringResult,
    path: str | Path = DEFAULT_MONITORING_REPORT_PATH,
) -> Path:
    """Write a monitoring Markdown report."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        render_monitoring_report(result),
        encoding="utf-8",
        newline="\n",
    )
    return output_path


def render_release_gate_report(result: ReleaseGateResult) -> str:
    """Render an auditable baseline-to-current release decision."""

    lines = [
        "# Model Release Gate Report",
        "",
        f"- Result: **{result.status.upper()}**",
        f"- Baseline evaluation: `{result.baseline_eval_path}`",
        f"- Current evaluation: `{result.current_eval_path}`",
        "",
        "## Regression checks",
        "",
        "| Metric | Baseline | Current | Requirement | Result |",
        "| --- | ---: | ---: | --- | --- |",
    ]
    for check in result.checks:
        baseline = "n/a" if check.baseline is None else f"{check.baseline:.3f}"
        current = "n/a" if check.current is None else f"{check.current:.3f}"
        lines.append(
            f"| {check.metric} | {baseline} | {current} "
            f"| {check.requirement} | {'pass' if check.passed else 'fail'} |"
        )
    lines.extend(
        [
            "",
            "The release passes only when every quality regression and available",
            "latency check passes.",
            "",
        ]
    )
    return "\n".join(lines)


def write_release_gate_report(
    result: ReleaseGateResult,
    path: str | Path = DEFAULT_RELEASE_GATE_REPORT_PATH,
) -> Path:
    """Write the release gate Markdown report."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        render_release_gate_report(result),
        encoding="utf-8",
        newline="\n",
    )
    return output_path
