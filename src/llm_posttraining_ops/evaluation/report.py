"""Markdown reporting for baseline evaluation artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from llm_posttraining_ops.evaluation.evaluator import (
    DEFAULT_EVALUATION_PATH,
    EVALUATION_SCHEMA_VERSION,
)
from llm_posttraining_ops.inference.evaluation import (
    MODEL_EVALUATION_SCHEMA_VERSION,
)

DEFAULT_REPORT_PATH = Path("reports/baseline_eval_report.md")


class ReportError(ValueError):
    """Raised when an evaluation artifact cannot produce a report."""


def _load_result(path: Path) -> dict[str, Any]:
    try:
        with path.open(encoding="utf-8") as result_file:
            result = json.load(result_file)
    except FileNotFoundError as exc:
        raise ReportError(f"Evaluation result not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ReportError(f"Invalid evaluation JSON in {path}: {exc.msg}") from exc

    if not isinstance(result, dict):
        raise ReportError("Evaluation result must be a JSON object")
    if result.get("schema_version") != EVALUATION_SCHEMA_VERSION:
        raise ReportError(
            f"Unsupported evaluation schema version: {result.get('schema_version')!r}"
        )
    if not isinstance(result.get("dataset"), dict) or not isinstance(
        result.get("baselines"), list
    ):
        raise ReportError("Evaluation result is missing dataset or baseline data")
    return result


def _format_rate(value: object) -> str:
    if not isinstance(value, (int, float)):
        raise ReportError("Evaluation metric values must be numeric")
    return f"{value:.3f}"


def _load_model_result(path: Path) -> dict[str, Any]:
    try:
        with path.open(encoding="utf-8") as result_file:
            result = json.load(result_file)
    except FileNotFoundError as exc:
        raise ReportError(f"Model evaluation result not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ReportError(f"Invalid model evaluation JSON in {path}: {exc.msg}") from exc

    if not isinstance(result, dict):
        raise ReportError("Model evaluation result must be a JSON object")
    if result.get("schema_version") != MODEL_EVALUATION_SCHEMA_VERSION:
        raise ReportError(
            f"Unsupported model evaluation schema version: {result.get('schema_version')!r}"
        )
    if not isinstance(result.get("model"), dict) or not isinstance(
        result.get("metrics"), dict
    ):
        raise ReportError("Model evaluation result is missing model or metric data")
    return result


def render_markdown_report(
    result: dict[str, Any],
    model_result: dict[str, Any] | None = None,
) -> str:
    """Render model-free and optional model-backed evaluation results."""

    dataset = result["dataset"]
    baselines = result["baselines"]
    split_counts = ", ".join(
        f"{split}={count}" for split, count in sorted(dataset["split_counts"].items())
    )

    lines = [
        "# Baseline Evaluation Report",
        "",
        f"- Dataset: `{dataset['path']}`",
        f"- Records: {dataset['record_count']}",
        f"- Splits: {split_counts}",
        f"- Schema version: {result['schema_version']}",
        "",
        "## Summary",
        "",
        (
            "| Baseline | Exact match | Token overlap F1 | Contains expected key terms "
            "| Average response length | Empty response rate |"
        ),
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for baseline in baselines:
        metrics = baseline["metrics"]
        lines.append(
            f"| {baseline['name']} "
            f"| {_format_rate(metrics['exact_match'])} "
            f"| {_format_rate(metrics['token_overlap_f1'])} "
            f"| {_format_rate(metrics['contains_expected_key_terms'])} "
            f"| {_format_rate(metrics['average_response_length'])} "
            f"| {_format_rate(metrics['empty_response_rate'])} |"
        )
    if model_result is not None:
        model = model_result["model"]
        metrics = model_result["metrics"]
        lines.append(
            f"| hf:{model['name']} "
            f"| {_format_rate(metrics['exact_match'])} "
            f"| {_format_rate(metrics['token_overlap_f1'])} "
            f"| {_format_rate(metrics['contains_expected_key_terms'])} "
            f"| {_format_rate(metrics['average_response_length'])} "
            f"| {_format_rate(metrics['empty_response_rate'])} |"
        )

        latency = model_result["latency"]
        lines.extend(
            [
                "",
                "## Model latency",
                "",
                f"- Model: `{model['name']}`",
                f"- Device: `{model['device']}`",
                (
                    "- Total generation time: "
                    f"{_format_rate(latency['total_generation_seconds'])} seconds"
                ),
                (
                    "- Average seconds per example: "
                    f"{_format_rate(latency['average_seconds_per_example'])}"
                ),
                (
                    "- Average generated tokens: "
                    f"{_format_rate(latency['average_generated_tokens'])}"
                ),
            ]
        )

    lines.extend(
        [
            "",
            "## Metric definitions",
            "",
            "- **Exact match:** normalized generated text equals the expected output.",
            "- **Token overlap F1:** multiset token precision/recall F1.",
            "- **Contains expected key terms:** all expected-output tokens appear in the response.",
            "- **Average response length:** mean generated response length in tokens.",
            "- **Empty response rate:** fraction of responses that are empty or whitespace-only.",
            "",
            (
                "Model-free baselines are deterministic; Hugging Face generation uses "
                "the recorded settings and seed."
                if model_result is not None
                else "All baselines are deterministic and run locally without a model or GPU."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def generate_baseline_report(
    evaluation_path: str | Path = DEFAULT_EVALUATION_PATH,
    output_path: str | Path = DEFAULT_REPORT_PATH,
    *,
    model_evaluation_path: str | Path | None = None,
) -> Path:
    """Read baseline and optional model artifacts, then write a Markdown report."""

    result = _load_result(Path(evaluation_path))
    model_result = (
        _load_model_result(Path(model_evaluation_path))
        if model_evaluation_path is not None
        else None
    )
    report_path = Path(output_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        render_markdown_report(result, model_result),
        encoding="utf-8",
        newline="\n",
    )
    return report_path
