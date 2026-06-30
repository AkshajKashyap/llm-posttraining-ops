"""Markdown comparison of pre-SFT and post-SFT model evaluation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from llm_posttraining_ops.inference.evaluation import DEFAULT_MODEL_EVALUATION_PATH
from llm_posttraining_ops.evaluation.suite import DEFAULT_EVAL_SUITE_PATH
from llm_posttraining_ops.evaluation.suite_reports import compact_suite_section, load_suite_metrics
from llm_posttraining_ops.training.evaluation import DEFAULT_SFT_EVALUATION_PATH
from llm_posttraining_ops.training.sft import DEFAULT_SFT_SUMMARY_PATH

DEFAULT_SFT_REPORT_PATH = Path("reports/sft_report.md")


class SFTReportError(ValueError):
    """Raised when SFT report inputs are unavailable or malformed."""


def _load_json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        with path.open(encoding="utf-8") as input_file:
            value = json.load(input_file)
    except FileNotFoundError as exc:
        raise SFTReportError(f"{label} not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise SFTReportError(f"Invalid {label} JSON in {path}: {exc.msg}") from exc
    if not isinstance(value, dict):
        raise SFTReportError(f"{label} must be a JSON object")
    return value


def _metric(value: object) -> str:
    if not isinstance(value, (int, float)):
        raise SFTReportError("SFT report metrics must be numeric")
    return f"{value:.3f}"


def render_sft_report(
    training_summary: dict[str, Any],
    pre_sft: dict[str, Any],
    post_sft: dict[str, Any],
    suite_metrics: dict[str, Any] | None = None,
) -> str:
    """Render training settings, metrics, and latency as Markdown."""

    pre_metrics = pre_sft["metrics"]
    post_metrics = post_sft["metrics"]
    settings = training_summary["settings"]
    training_metrics = training_summary.get("metrics", {})
    loss = training_metrics.get("training_loss", training_metrics.get("train_loss"))
    lines = [
        "# Supervised Fine-Tuning Report",
        "",
        f"- Base model: `{training_summary['model_name']}`",
        f"- Checkpoint: `{training_summary['checkpoint_path']}`",
        f"- Training records: {training_summary['training_record_count']}",
        f"- LoRA enabled: {training_summary['use_lora']}",
        f"- Training loss: {_metric(loss) if loss is not None else 'not available'}",
        "",
        "## Pre-SFT vs post-SFT metrics",
        "",
        "| Stage | Exact match | Token overlap F1 | Contains expected key terms "
        "| Average response length | Empty response rate |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
        (
            f"| Pre-SFT | {_metric(pre_metrics['exact_match'])} "
            f"| {_metric(pre_metrics['token_overlap_f1'])} "
            f"| {_metric(pre_metrics['contains_expected_key_terms'])} "
            f"| {_metric(pre_metrics['average_response_length'])} "
            f"| {_metric(pre_metrics['empty_response_rate'])} |"
        ),
        (
            f"| Post-SFT | {_metric(post_metrics['exact_match'])} "
            f"| {_metric(post_metrics['token_overlap_f1'])} "
            f"| {_metric(post_metrics['contains_expected_key_terms'])} "
            f"| {_metric(post_metrics['average_response_length'])} "
            f"| {_metric(post_metrics['empty_response_rate'])} |"
        ),
        "",
        "## Training settings",
        "",
        "| Setting | Value |",
        "| --- | --- |",
    ]
    lines.extend(f"| {key} | `{value}` |" for key, value in settings.items())

    lines.extend(
        [
            "",
            "## Generation latency",
            "",
            "| Stage | Total seconds | Seconds/example | Average generated tokens |",
            "| --- | ---: | ---: | ---: |",
            (
                f"| Pre-SFT | {_metric(pre_sft['latency']['total_generation_seconds'])} "
                f"| {_metric(pre_sft['latency']['average_seconds_per_example'])} "
                f"| {_metric(pre_sft['latency']['average_generated_tokens'])} |"
            ),
            (
                f"| Post-SFT | {_metric(post_sft['latency']['total_generation_seconds'])} "
                f"| {_metric(post_sft['latency']['average_seconds_per_example'])} "
                f"| {_metric(post_sft['latency']['average_generated_tokens'])} |"
            ),
            "",
        ]
    )
    if suite_metrics is not None:
        lines.extend(compact_suite_section(suite_metrics))
    return "\n".join(lines)


def generate_sft_report(
    *,
    summary_path: str | Path = DEFAULT_SFT_SUMMARY_PATH,
    pre_sft_path: str | Path = DEFAULT_MODEL_EVALUATION_PATH,
    post_sft_path: str | Path = DEFAULT_SFT_EVALUATION_PATH,
    output_path: str | Path = DEFAULT_SFT_REPORT_PATH,
    suite_result_path: str | Path | None = DEFAULT_EVAL_SUITE_PATH,
) -> Path:
    """Load SFT artifacts and write the comparison report."""

    training_summary = _load_json_object(Path(summary_path), "training summary")
    pre_sft = _load_json_object(Path(pre_sft_path), "pre-SFT evaluation")
    post_sft = _load_json_object(Path(post_sft_path), "post-SFT evaluation")
    suite_path = Path(suite_result_path) if suite_result_path is not None else None
    suite_metrics = (
        load_suite_metrics(suite_path) if suite_path is not None and suite_path.exists() else None
    )
    report_path = Path(output_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        render_sft_report(training_summary, pre_sft, post_sft, suite_metrics),
        encoding="utf-8",
        newline="\n",
    )
    return report_path
