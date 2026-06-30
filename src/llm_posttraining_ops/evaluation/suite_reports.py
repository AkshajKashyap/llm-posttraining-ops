"""Markdown reports for deterministic suite and pairwise evaluation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from llm_posttraining_ops.evaluation.pairwise import PairwiseResult
from llm_posttraining_ops.evaluation.suite import EvalSuiteResult

DEFAULT_EVAL_SUITE_REPORT_PATH = Path("reports/eval_suite_report.md")
DEFAULT_PAIRWISE_REPORT_PATH = Path("reports/pairwise_comparison_report.md")


def load_suite_metrics(path: str | Path) -> dict[str, Any]:
    """Load aggregate suite metrics for integration into other reports."""

    with Path(path).open(encoding="utf-8") as input_file:
        result = json.load(input_file)
    if not isinstance(result, dict) or not isinstance(result.get("metrics"), dict):
        raise ValueError("Evaluation suite result is missing aggregate metrics")
    return result["metrics"]


def compact_suite_section(metrics: dict[str, Any]) -> list[str]:
    """Render the most decision-relevant suite metrics as Markdown lines."""

    return [
        "## Rigorous evaluation summary",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| Required fact coverage | {metrics['required_fact_coverage']:.3f} |",
        (
            "| Forbidden term violation rate "
            f"| {metrics['forbidden_term_violation_rate']:.3f} |"
        ),
        f"| Instruction copying rate | {metrics['instruction_copying_rate']:.3f} |",
        f"| Refusal rate | {metrics['refusal_rate']:.3f} |",
        f"| Format compliance rate | {metrics['format_compliance_rate']:.3f} |",
        (
            "| Unsupported named entity rate "
            f"| {metrics['unsupported_named_entity_rate']:.3f} |"
        ),
        f"| Numeric mismatch rate | {metrics['numeric_mismatch_rate']:.3f} |",
        f"| Contradiction rate | {metrics['contradiction_rate']:.3f} |",
        "",
    ]


def render_eval_suite_report(result: EvalSuiteResult) -> str:
    """Render aggregate metrics and per-example failures."""

    metrics = result.metrics
    lines = [
        "# Rigorous Evaluation Suite Report",
        "",
        f"- Generations: `{result.generations_path}`",
        f"- Evaluation data: `{result.eval_data_path}`",
        f"- Records: {result.record_count}",
        "",
        "## Aggregate metrics",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| Exact match | {metrics.exact_match:.3f} |",
        f"| Token overlap F1 | {metrics.token_overlap_f1:.3f} |",
        f"| Required fact coverage | {metrics.required_fact_coverage:.3f} |",
        (
            "| Forbidden term violation rate "
            f"| {metrics.forbidden_term_violation_rate:.3f} |"
        ),
        f"| Instruction copying rate | {metrics.instruction_copying_rate:.3f} |",
        f"| Empty response rate | {metrics.empty_response_rate:.3f} |",
        f"| Refusal rate | {metrics.refusal_rate:.3f} |",
        f"| Format compliance rate | {metrics.format_compliance_rate:.3f} |",
        (
            "| Unsupported named entity rate "
            f"| {metrics.unsupported_named_entity_rate:.3f} |"
        ),
        f"| Numeric mismatch rate | {metrics.numeric_mismatch_rate:.3f} |",
        f"| Contradiction rate | {metrics.contradiction_rate:.3f} |",
        f"| Average response length | {metrics.response_length.average:.3f} |",
        f"| Minimum response length | {metrics.response_length.minimum} |",
        f"| Maximum response length | {metrics.response_length.maximum} |",
        "",
        "## Per-example diagnostics",
        "",
        "| ID | Facts | Forbidden | Copy | Refusal | Format | Entity | Number | Contradiction |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for item in result.examples:
        diagnostics = item.diagnostics
        lines.append(
            f"| {item.id} "
            f"| {diagnostics.required_fact_coverage:.3f} "
            f"| {diagnostics.forbidden_term_violation:.0f} "
            f"| {diagnostics.instruction_copying:.0f} "
            f"| {diagnostics.refusal_detected:.0f} "
            f"| {diagnostics.format_compliant:.0f} "
            f"| {int(bool(diagnostics.unsupported_named_entities))} "
            f"| {diagnostics.numeric_mismatch:.0f} "
            f"| {diagnostics.contradiction_detected:.0f} |"
        )
    lines.append("")
    return "\n".join(lines)


def write_eval_suite_report(
    result: EvalSuiteResult,
    path: str | Path = DEFAULT_EVAL_SUITE_REPORT_PATH,
) -> Path:
    """Write the evaluation-suite Markdown report."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        render_eval_suite_report(result),
        encoding="utf-8",
        newline="\n",
    )
    return output_path


def render_pairwise_report(result: PairwiseResult) -> str:
    """Render pairwise counts and decision reasons."""

    lines = [
        "# Pairwise Generation Comparison",
        "",
        f"- Left: `{result.left_path}`",
        f"- Right: `{result.right_path}`",
        f"- Evaluation data: `{result.eval_data_path}`",
        f"- Records: {result.record_count}",
        "",
        "## Outcome",
        "",
        "| Result | Count |",
        "| --- | ---: |",
        f"| Left wins | {result.counts.left_wins} |",
        f"| Right wins | {result.counts.right_wins} |",
        f"| Ties | {result.counts.ties} |",
        "",
        "## Decisions",
        "",
        "| ID | Winner | Deterministic reason |",
        "| --- | --- | --- |",
    ]
    lines.extend(
        f"| {decision.id} | {decision.winner} | {decision.reason} |"
        for decision in result.decisions
    )
    lines.extend(
        [
            "",
            "Decisions use forbidden terms, required facts, instruction copying,",
            "length sanity, format compliance, refusal/hallucination checks, and",
            "lexical metrics in a fixed order. Equal signals produce a tie.",
            "",
        ]
    )
    return "\n".join(lines)


def write_pairwise_report(
    result: PairwiseResult,
    path: str | Path = DEFAULT_PAIRWISE_REPORT_PATH,
) -> Path:
    """Write the pairwise Markdown report."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        render_pairwise_report(result),
        encoding="utf-8",
        newline="\n",
    )
    return output_path
