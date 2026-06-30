"""Markdown report comparing base, optional SFT, and DPO models."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from llm_posttraining_ops.inference.evaluation import DEFAULT_MODEL_EVALUATION_PATH
from llm_posttraining_ops.evaluation.suite import DEFAULT_EVAL_SUITE_PATH
from llm_posttraining_ops.evaluation.suite_reports import compact_suite_section, load_suite_metrics
from llm_posttraining_ops.training.dpo import DEFAULT_DPO_SUMMARY_PATH
from llm_posttraining_ops.training.dpo_evaluation import DEFAULT_DPO_EVALUATION_PATH
from llm_posttraining_ops.training.evaluation import DEFAULT_SFT_EVALUATION_PATH

DEFAULT_DPO_REPORT_PATH = Path("reports/dpo_report.md")


class DPOReportError(ValueError):
    """Raised when DPO report artifacts are missing or malformed."""


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        with path.open(encoding="utf-8") as input_file:
            value = json.load(input_file)
    except FileNotFoundError as exc:
        raise DPOReportError(f"{label} not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise DPOReportError(f"Invalid {label} JSON in {path}: {exc.msg}") from exc
    if not isinstance(value, dict):
        raise DPOReportError(f"{label} must be a JSON object")
    return value


def _number(value: object) -> str:
    if not isinstance(value, (int, float)):
        raise DPOReportError("DPO report metrics must be numeric")
    return f"{value:.3f}"


def _metric_row(label: str, evaluation: dict[str, Any]) -> str:
    metrics = evaluation["metrics"]
    return (
        f"| {label} | {_number(metrics['exact_match'])} "
        f"| {_number(metrics['token_overlap_f1'])} "
        f"| {_number(metrics['contains_expected_key_terms'])} "
        f"| {_number(metrics['average_response_length'])} "
        f"| {_number(metrics['empty_response_rate'])} |"
    )


def _latency_row(label: str, evaluation: dict[str, Any]) -> str:
    latency = evaluation["latency"]
    return (
        f"| {label} | {_number(latency['total_generation_seconds'])} "
        f"| {_number(latency['average_seconds_per_example'])} "
        f"| {_number(latency['average_generated_tokens'])} |"
    )


def render_dpo_report(
    training_summary: dict[str, Any],
    base_evaluation: dict[str, Any],
    dpo_evaluation: dict[str, Any],
    sft_evaluation: dict[str, Any] | None = None,
    suite_metrics: dict[str, Any] | None = None,
) -> str:
    """Render base/SFT/DPO metrics, settings, loss, and latency."""

    training_metrics = training_summary.get("metrics", {})
    loss = training_metrics.get("training_loss", training_metrics.get("train_loss"))
    lines = [
        "# Direct Preference Optimization Report",
        "",
        f"- Starting model: `{training_summary['starting_model']}`",
        f"- DPO checkpoint: `{training_summary['checkpoint_path']}`",
        f"- Preference training records: {training_summary['training_record_count']}",
        f"- LoRA enabled: {training_summary['use_lora']}",
        f"- DPO loss: {_number(loss) if loss is not None else 'not available'}",
        "",
        "## Model comparison",
        "",
        "| Model | Exact match | Token overlap F1 | Contains expected key terms "
        "| Average response length | Empty response rate |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
        _metric_row("Base", base_evaluation),
    ]
    if sft_evaluation is not None:
        lines.append(_metric_row("SFT", sft_evaluation))
    lines.append(_metric_row("DPO", dpo_evaluation))

    lines.extend(
        [
            "",
            "## Training settings",
            "",
            "| Setting | Value |",
            "| --- | --- |",
        ]
    )
    lines.extend(
        f"| {key} | `{value}` |" for key, value in training_summary["settings"].items()
    )
    lines.extend(
        [
            "",
            "## Generation latency",
            "",
            "| Model | Total seconds | Seconds/example | Average generated tokens |",
            "| --- | ---: | ---: | ---: |",
            _latency_row("Base", base_evaluation),
        ]
    )
    if sft_evaluation is not None:
        lines.append(_latency_row("SFT", sft_evaluation))
    lines.extend([_latency_row("DPO", dpo_evaluation), ""])
    if suite_metrics is not None:
        lines.extend(compact_suite_section(suite_metrics))
    return "\n".join(lines)


def generate_dpo_report(
    *,
    summary_path: str | Path = DEFAULT_DPO_SUMMARY_PATH,
    base_evaluation_path: str | Path = DEFAULT_MODEL_EVALUATION_PATH,
    sft_evaluation_path: str | Path | None = DEFAULT_SFT_EVALUATION_PATH,
    dpo_evaluation_path: str | Path = DEFAULT_DPO_EVALUATION_PATH,
    output_path: str | Path = DEFAULT_DPO_REPORT_PATH,
    suite_result_path: str | Path | None = DEFAULT_EVAL_SUITE_PATH,
) -> Path:
    """Load DPO artifacts and write the comparison report."""

    summary = _load_json(Path(summary_path), "DPO training summary")
    base = _load_json(Path(base_evaluation_path), "base model evaluation")
    dpo = _load_json(Path(dpo_evaluation_path), "DPO model evaluation")
    sft_path = Path(sft_evaluation_path) if sft_evaluation_path is not None else None
    sft = _load_json(sft_path, "SFT model evaluation") if sft_path and sft_path.exists() else None
    suite_path = Path(suite_result_path) if suite_result_path is not None else None
    suite_metrics = (
        load_suite_metrics(suite_path) if suite_path is not None and suite_path.exists() else None
    )
    report_path = Path(output_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        render_dpo_report(summary, base, dpo, sft, suite_metrics),
        encoding="utf-8",
        newline="\n",
    )
    return report_path
