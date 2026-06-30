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
from llm_posttraining_ops.inference.config import (
    DEFAULT_MODEL_NAME,
    GenerationConfigError,
    GenerationSettings,
)
from llm_posttraining_ops.inference.evaluation import (
    DEFAULT_MODEL_EVALUATION_PATH,
    run_model_evaluation,
)
from llm_posttraining_ops.inference.huggingface import ModelInferenceError

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


@app.command("run-model-eval")
def run_model_eval_command(
    data_dir: Annotated[
        Path,
        typer.Option(
            "--data-dir",
            exists=True,
            file_okay=False,
            readable=True,
            help="Directory containing normalized sft.jsonl.",
        ),
    ] = Path("data/processed/custom"),
    model_name: Annotated[
        str,
        typer.Option("--model-name", help="Hugging Face causal LM name or local path."),
    ] = DEFAULT_MODEL_NAME,
    max_new_tokens: Annotated[
        int,
        typer.Option("--max-new-tokens", min=1, help="Maximum generated tokens."),
    ] = 32,
    temperature: Annotated[
        float,
        typer.Option("--temperature", min=0.0, help="Sampling temperature; 0 is greedy."),
    ] = 0.0,
    top_p: Annotated[
        float,
        typer.Option("--top-p", min=0.0, max=1.0, help="Nucleus sampling probability."),
    ] = 1.0,
    seed: Annotated[
        int,
        typer.Option("--seed", min=0, help="Generation seed."),
    ] = 42,
    output: Annotated[
        Path,
        typer.Option("--output", dir_okay=False, help="Model evaluation JSON output."),
    ] = DEFAULT_MODEL_EVALUATION_PATH,
    generations_output: Annotated[
        Path | None,
        typer.Option(
            "--generations-output",
            dir_okay=False,
            help="Optional generations JSONL output path.",
        ),
    ] = None,
) -> None:
    """Run CPU-first Hugging Face model inference and evaluation."""

    try:
        settings = GenerationSettings(
            model_name=model_name,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            seed=seed,
        )
        result = run_model_evaluation(
            data_dir,
            settings,
            output_path=output,
            generations_path=generations_output,
        )
    except (
        DataValidationError,
        GenerationConfigError,
        JsonlError,
        ModelInferenceError,
        OSError,
        ValueError,
    ) as exc:
        typer.echo(f"Model evaluation failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    typer.echo(
        f"Evaluated {result.dataset.record_count} records with model {result.model.name}"
    )
    typer.echo(f"Wrote model generations to {result.generations_path}")
    typer.echo(f"Wrote model evaluation to {output}")
    typer.echo(
        f"Generation latency: {result.latency.total_generation_seconds:.3f}s total, "
        f"{result.latency.average_seconds_per_example:.3f}s/example"
    )


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
    model_evaluation_path: Annotated[
        Path,
        typer.Option(
            "--model-evaluation-path",
            dir_okay=False,
            help="Optional model evaluation JSON artifact.",
        ),
    ] = DEFAULT_MODEL_EVALUATION_PATH,
) -> None:
    """Generate a combined baseline and optional model Markdown report."""

    try:
        report_path = generate_baseline_report(
            evaluation_path,
            output,
            model_evaluation_path=(
                model_evaluation_path if model_evaluation_path.exists() else None
            ),
        )
    except (ReportError, OSError) as exc:
        typer.echo(f"Report generation failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(f"Wrote baseline report to {report_path}")


if __name__ == "__main__":
    app()
