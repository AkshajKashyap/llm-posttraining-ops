"""Command-line interface for data preparation and baseline evaluation."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from llm_posttraining_ops.config import ConfigError, load_config
from llm_posttraining_ops.data.generation import prepare_demo_data
from llm_posttraining_ops.data.ingestion import (
    DEFAULT_INGEST_MIN_OUTPUT_LENGTH,
    ingest_sft_data,
)
from llm_posttraining_ops.data.jsonl import JsonlError
from llm_posttraining_ops.data.normalization import NormalizationError, SFTFormat
from llm_posttraining_ops.data.profiling import (
    DEFAULT_DATASET_CARD_PATH,
    DEFAULT_PROFILE_PATH,
    profile_data_directory,
)
from llm_posttraining_ops.data.validation import DataValidationError, validate_data_directory
from llm_posttraining_ops.evaluation.evaluator import (
    DEFAULT_EVALUATION_PATH,
    run_baseline_evaluation,
)
from llm_posttraining_ops.evaluation.report import (
    DEFAULT_REPORT_PATH,
    ReportError,
    generate_baseline_report,
)

app = typer.Typer(
    no_args_is_help=True,
    help="Reproducible utilities for LLM post-training operations.",
)


@app.command("prepare-demo-data")
def prepare_demo_data_command(
    config: Annotated[
        Path,
        typer.Option(
            "--config",
            exists=True,
            dir_okay=False,
            readable=True,
            help="YAML configuration file.",
        ),
    ] = Path("configs/default.yaml"),
) -> None:
    """Generate deterministic toy SFT and preference JSONL files."""

    try:
        paths = prepare_demo_data(load_config(config))
    except (ConfigError, OSError, DataValidationError) as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    for kind, path in paths.items():
        typer.echo(f"Wrote {kind} data to {path}")


@app.command("validate-data")
def validate_data_command(
    data_dir: Annotated[
        Path,
        typer.Option(
            "--data-dir",
            exists=True,
            file_okay=False,
            readable=True,
            help="Directory containing sft.jsonl and optional preference.jsonl.",
        ),
    ] = Path("data/processed/demo"),
    minimum_output_length: Annotated[
        int,
        typer.Option(
            "--minimum-output-length",
            min=1,
            help="Minimum SFT output length in characters.",
        ),
    ] = 1,
) -> None:
    """Validate normalized SFT and optional preference datasets."""

    try:
        counts = validate_data_directory(
            data_dir,
            minimum_output_length=minimum_output_length,
        )
    except (DataValidationError, JsonlError) as exc:
        typer.echo(f"Validation failed:\n{exc}", err=True)
        raise typer.Exit(code=1) from exc

    summary = [f"{counts['sft']} SFT records"]
    if "preference" in counts:
        summary.append(f"{counts['preference']} preference records")
    typer.echo(f"Validation passed: {', '.join(summary)}")


@app.command("ingest-sft-data")
def ingest_sft_data_command(
    input_path: Annotated[
        Path,
        typer.Option(
            "--input-path",
            exists=True,
            dir_okay=False,
            readable=True,
            help="Local raw JSONL dataset.",
        ),
    ],
    output_dir: Annotated[
        Path,
        typer.Option("--output-dir", file_okay=False, help="Normalized data directory."),
    ],
    format_name: Annotated[
        SFTFormat,
        typer.Option("--format", help="Raw dataset format: alpaca or messages."),
    ],
    minimum_output_length: Annotated[
        int,
        typer.Option(
            "--minimum-output-length",
            min=1,
            help="Minimum normalized output length in characters.",
        ),
    ] = DEFAULT_INGEST_MIN_OUTPUT_LENGTH,
) -> None:
    """Normalize and validate a local instruction dataset."""

    try:
        output_path, records = ingest_sft_data(
            input_path,
            output_dir,
            format_name,
            minimum_output_length=minimum_output_length,
        )
    except (
        DataValidationError,
        JsonlError,
        NormalizationError,
        OSError,
        ValueError,
    ) as exc:
        typer.echo(f"Ingestion failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(f"Ingested {len(records)} SFT records to {output_path}")


@app.command("profile-data")
def profile_data_command(
    data_dir: Annotated[
        Path,
        typer.Option(
            "--data-dir",
            exists=True,
            file_okay=False,
            readable=True,
            help="Directory containing normalized sft.jsonl.",
        ),
    ],
    output: Annotated[
        Path,
        typer.Option("--output", dir_okay=False, help="Dataset profile JSON output."),
    ] = DEFAULT_PROFILE_PATH,
    card_output: Annotated[
        Path,
        typer.Option("--card-output", dir_okay=False, help="Dataset card Markdown output."),
    ] = DEFAULT_DATASET_CARD_PATH,
) -> None:
    """Profile normalized SFT data and generate a dataset card."""

    try:
        profile, profile_path, card_path = profile_data_directory(
            data_dir,
            profile_path=output,
            card_path=card_output,
        )
    except (DataValidationError, JsonlError, OSError, ValueError) as exc:
        typer.echo(f"Profiling failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(f"Profiled {profile.record_count} SFT records")
    typer.echo(f"Wrote dataset profile to {profile_path}")
    typer.echo(f"Wrote dataset card to {card_path}")


@app.command("run-baseline-eval")
def run_baseline_eval_command(
    data_dir: Annotated[
        Path,
        typer.Option(
            "--data-dir",
            exists=True,
            file_okay=False,
            readable=True,
            help="Directory containing sft.jsonl.",
        ),
    ] = Path("data/processed/demo"),
    output: Annotated[
        Path,
        typer.Option("--output", dir_okay=False, help="Evaluation JSON output path."),
    ] = DEFAULT_EVALUATION_PATH,
) -> None:
    """Evaluate local deterministic baselines on the SFT dataset."""

    try:
        result = run_baseline_evaluation(data_dir, output)
    except (DataValidationError, JsonlError, OSError, ValueError) as exc:
        typer.echo(f"Evaluation failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    typer.echo(
        f"Evaluated {result.dataset.record_count} records with "
        f"{len(result.baselines)} baselines"
    )
    typer.echo(f"Wrote evaluation results to {output}")


@app.command("generate-baseline-report")
def generate_baseline_report_command(
    evaluation_path: Annotated[
        Path,
        typer.Option(
            "--evaluation-path",
            exists=True,
            dir_okay=False,
            readable=True,
            help="Baseline evaluation JSON artifact.",
        ),
    ] = DEFAULT_EVALUATION_PATH,
    output: Annotated[
        Path,
        typer.Option("--output", dir_okay=False, help="Markdown report output path."),
    ] = DEFAULT_REPORT_PATH,
) -> None:
    """Generate a Markdown report from baseline evaluation JSON."""

    try:
        report_path = generate_baseline_report(evaluation_path, output)
    except (ReportError, OSError) as exc:
        typer.echo(f"Report generation failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(f"Wrote baseline report to {report_path}")


if __name__ == "__main__":
    app()
