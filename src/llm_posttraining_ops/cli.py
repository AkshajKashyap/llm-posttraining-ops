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
from llm_posttraining_ops.data.preference_ingestion import (
    DEFAULT_PREFERENCE_MIN_LENGTH,
    ingest_preference_data,
)
from llm_posttraining_ops.data.preference_normalization import (
    PreferenceFormat,
    PreferenceNormalizationError,
)
from llm_posttraining_ops.data.preference_profiling import (
    DEFAULT_PREFERENCE_PROFILE_PATH,
    profile_preference_directory,
)
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
from llm_posttraining_ops.training.dpo import (
    DEFAULT_DPO_SUMMARY_PATH,
    DPOTrainingError,
    run_dpo_training,
)
from llm_posttraining_ops.training.dpo_config import DPOConfigError, DPOTrainingConfig
from llm_posttraining_ops.training.dpo_evaluation import (
    DEFAULT_DPO_EVALUATION_PATH,
    DEFAULT_DPO_GENERATIONS_PATH,
    run_dpo_evaluation,
)
from llm_posttraining_ops.training.dpo_report import (
    DEFAULT_DPO_REPORT_PATH,
    DPOReportError,
    generate_dpo_report,
)
from llm_posttraining_ops.training.config import SFTConfigError, SFTTrainingConfig
from llm_posttraining_ops.training.evaluation import (
    DEFAULT_SFT_EVALUATION_PATH,
    DEFAULT_SFT_GENERATIONS_PATH,
    SFTCheckpointError,
    run_sft_evaluation,
)
from llm_posttraining_ops.training.report import (
    DEFAULT_SFT_REPORT_PATH,
    SFTReportError,
    generate_sft_report,
)
from llm_posttraining_ops.training.sft import (
    DEFAULT_SFT_SUMMARY_PATH,
    SFTTrainingError,
    run_sft_training,
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


@app.command("ingest-preference-data")
def ingest_preference_data_command(
    input_path: Annotated[
        Path,
        typer.Option(
            "--input-path",
            exists=True,
            dir_okay=False,
            readable=True,
            help="Local raw preference JSONL.",
        ),
    ],
    output_dir: Annotated[
        Path,
        typer.Option("--output-dir", file_okay=False),
    ],
    format_name: Annotated[
        PreferenceFormat,
        typer.Option("--format", help="Preference format: direct or messages."),
    ],
    minimum_response_length: Annotated[
        int,
        typer.Option("--minimum-response-length", min=1),
    ] = DEFAULT_PREFERENCE_MIN_LENGTH,
) -> None:
    """Normalize and validate a local preference dataset."""

    try:
        output_path, records = ingest_preference_data(
            input_path,
            output_dir,
            format_name,
            minimum_response_length=minimum_response_length,
        )
    except (
        DataValidationError,
        JsonlError,
        OSError,
        PreferenceNormalizationError,
        ValueError,
    ) as exc:
        typer.echo(f"Preference ingestion failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(f"Ingested {len(records)} preference records to {output_path}")


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


@app.command("profile-preference-data")
def profile_preference_data_command(
    data_dir: Annotated[
        Path,
        typer.Option(
            "--data-dir",
            exists=True,
            file_okay=False,
            readable=True,
        ),
    ] = Path("data/processed/preferences"),
    output: Annotated[
        Path,
        typer.Option("--output", dir_okay=False),
    ] = DEFAULT_PREFERENCE_PROFILE_PATH,
) -> None:
    """Profile a normalized preference dataset."""

    try:
        profile, profile_path = profile_preference_directory(
            data_dir,
            output_path=output,
        )
    except (DataValidationError, JsonlError, OSError, ValueError) as exc:
        typer.echo(f"Preference profiling failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(f"Profiled {profile.record_count} preference records")
    typer.echo(f"Wrote preference profile to {profile_path}")


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


@app.command("train-sft")
def train_sft_command(
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
        typer.Option("--model-name", help="Base Hugging Face causal LM."),
    ] = DEFAULT_MODEL_NAME,
    output_dir: Annotated[
        Path | None,
        typer.Option("--output-dir", file_okay=False, help="Model or adapter output."),
    ] = None,
    max_steps: Annotated[
        int,
        typer.Option("--max-steps", min=1, help="Maximum optimizer steps."),
    ] = 1,
    learning_rate: Annotated[
        float,
        typer.Option("--learning-rate", min=0.0, help="Optimizer learning rate."),
    ] = 5e-5,
    batch_size: Annotated[
        int,
        typer.Option("--batch-size", min=1, help="Per-device training batch size."),
    ] = 1,
    gradient_accumulation_steps: Annotated[
        int,
        typer.Option("--gradient-accumulation-steps", min=1),
    ] = 1,
    max_seq_length: Annotated[
        int,
        typer.Option("--max-seq-length", min=1),
    ] = 128,
    seed: Annotated[int, typer.Option("--seed", min=0)] = 42,
    use_lora: Annotated[
        bool,
        typer.Option("--use-lora/--no-use-lora", help="Train PEFT LoRA adapters."),
    ] = False,
    lora_r: Annotated[int, typer.Option("--lora-r", min=1)] = 8,
    lora_alpha: Annotated[int, typer.Option("--lora-alpha", min=1)] = 16,
    lora_dropout: Annotated[
        float,
        typer.Option("--lora-dropout", min=0.0, max=0.999),
    ] = 0.05,
    summary_output: Annotated[
        Path,
        typer.Option("--summary-output", dir_okay=False),
    ] = DEFAULT_SFT_SUMMARY_PATH,
) -> None:
    """Run a small CPU supervised fine-tuning job."""

    resolved_output_dir = output_dir or Path(
        "artifacts/adapters/sft" if use_lora else "artifacts/models/sft"
    )
    try:
        config = SFTTrainingConfig(
            model_name=model_name,
            output_dir=resolved_output_dir,
            max_steps=max_steps,
            learning_rate=learning_rate,
            batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            max_seq_length=max_seq_length,
            seed=seed,
            use_lora=use_lora,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
        )
        summary = run_sft_training(
            data_dir,
            config,
            summary_path=summary_output,
        )
    except (
        DataValidationError,
        JsonlError,
        OSError,
        SFTConfigError,
        SFTTrainingError,
        ValueError,
    ) as exc:
        typer.echo(f"SFT training failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    loss = summary.metrics.get("training_loss", summary.metrics.get("train_loss"))
    typer.echo(
        f"Trained {summary.training_record_count} records for "
        f"{summary.settings['max_steps']} step(s)"
    )
    typer.echo(f"Saved {'adapter' if summary.use_lora else 'model'} to {summary.checkpoint_path}")
    typer.echo(f"Wrote training summary to {summary_output}")
    if loss is not None:
        typer.echo(f"Training loss: {loss:.6f}")


@app.command("evaluate-sft")
def evaluate_sft_command(
    data_dir: Annotated[
        Path,
        typer.Option(
            "--data-dir",
            exists=True,
            file_okay=False,
            readable=True,
        ),
    ] = Path("data/processed/custom"),
    model_path: Annotated[
        Path,
        typer.Option(
            "--model-path",
            exists=True,
            file_okay=False,
            readable=True,
            help="Saved full model or PEFT adapter directory.",
        ),
    ] = Path("artifacts/models/sft"),
    max_new_tokens: Annotated[int, typer.Option("--max-new-tokens", min=1)] = 32,
    temperature: Annotated[float, typer.Option("--temperature", min=0.0)] = 0.0,
    top_p: Annotated[float, typer.Option("--top-p", min=0.0, max=1.0)] = 1.0,
    seed: Annotated[int, typer.Option("--seed", min=0)] = 42,
    output: Annotated[
        Path,
        typer.Option("--output", dir_okay=False),
    ] = DEFAULT_SFT_EVALUATION_PATH,
    generations_output: Annotated[
        Path,
        typer.Option("--generations-output", dir_okay=False),
    ] = DEFAULT_SFT_GENERATIONS_PATH,
    report_output: Annotated[
        Path,
        typer.Option("--report-output", dir_okay=False),
    ] = DEFAULT_SFT_REPORT_PATH,
) -> None:
    """Evaluate a trained SFT model and generate its comparison report."""

    try:
        result = run_sft_evaluation(
            data_dir,
            model_path,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            seed=seed,
            output_path=output,
            generations_path=generations_output,
        )
        report_path = generate_sft_report(
            post_sft_path=output,
            output_path=report_output,
        )
    except (
        DataValidationError,
        GenerationConfigError,
        JsonlError,
        ModelInferenceError,
        OSError,
        SFTCheckpointError,
        SFTReportError,
        ValueError,
    ) as exc:
        typer.echo(f"SFT evaluation failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    typer.echo(
        f"Evaluated {result.dataset.record_count} records with SFT model {model_path}"
    )
    typer.echo(f"Wrote SFT generations to {result.generations_path}")
    typer.echo(f"Wrote SFT evaluation to {output}")
    typer.echo(f"Wrote SFT report to {report_path}")
    typer.echo(
        f"Generation latency: {result.latency.total_generation_seconds:.3f}s total, "
        f"{result.latency.average_seconds_per_example:.3f}s/example"
    )


@app.command("train-dpo")
def train_dpo_command(
    preference_data_dir: Annotated[
        Path,
        typer.Option(
            "--preference-data-dir",
            exists=True,
            file_okay=False,
            readable=True,
        ),
    ] = Path("data/processed/preferences"),
    model_name: Annotated[str, typer.Option("--model-name")] = DEFAULT_MODEL_NAME,
    sft_model_path: Annotated[
        Path | None,
        typer.Option("--sft-model-path", file_okay=False),
    ] = None,
    output_dir: Annotated[
        Path | None,
        typer.Option("--output-dir", file_okay=False),
    ] = None,
    max_steps: Annotated[int, typer.Option("--max-steps", min=1)] = 1,
    learning_rate: Annotated[
        float,
        typer.Option("--learning-rate", min=0.0),
    ] = 1e-6,
    batch_size: Annotated[int, typer.Option("--batch-size", min=1)] = 1,
    gradient_accumulation_steps: Annotated[
        int,
        typer.Option("--gradient-accumulation-steps", min=1),
    ] = 1,
    max_seq_length: Annotated[int, typer.Option("--max-seq-length", min=1)] = 128,
    beta: Annotated[float, typer.Option("--beta", min=0.0)] = 0.1,
    seed: Annotated[int, typer.Option("--seed", min=0)] = 42,
    use_lora: Annotated[
        bool,
        typer.Option("--use-lora/--no-use-lora"),
    ] = False,
    lora_r: Annotated[int, typer.Option("--lora-r", min=1)] = 8,
    lora_alpha: Annotated[int, typer.Option("--lora-alpha", min=1)] = 16,
    lora_dropout: Annotated[
        float,
        typer.Option("--lora-dropout", min=0.0, max=0.999),
    ] = 0.05,
    summary_output: Annotated[
        Path,
        typer.Option("--summary-output", dir_okay=False),
    ] = DEFAULT_DPO_SUMMARY_PATH,
) -> None:
    """Run a tiny CPU direct preference optimization job."""

    resolved_output_dir = output_dir or Path(
        "artifacts/adapters/dpo" if use_lora else "artifacts/models/dpo"
    )
    try:
        config = DPOTrainingConfig(
            model_name=model_name,
            sft_model_path=sft_model_path,
            output_dir=resolved_output_dir,
            max_steps=max_steps,
            learning_rate=learning_rate,
            batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            max_seq_length=max_seq_length,
            beta=beta,
            seed=seed,
            use_lora=use_lora,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
        )
        summary = run_dpo_training(
            preference_data_dir,
            config,
            summary_path=summary_output,
        )
    except (
        DPOConfigError,
        DPOTrainingError,
        DataValidationError,
        JsonlError,
        OSError,
        SFTCheckpointError,
        ValueError,
    ) as exc:
        typer.echo(f"DPO training failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    loss = summary.metrics.get("training_loss", summary.metrics.get("train_loss"))
    typer.echo(
        f"Trained {summary.training_record_count} preference records for "
        f"{summary.settings['max_steps']} step(s)"
    )
    typer.echo(f"Saved {'adapter' if summary.use_lora else 'model'} to {summary.checkpoint_path}")
    typer.echo(f"Wrote DPO training summary to {summary_output}")
    if loss is not None:
        typer.echo(f"DPO loss: {loss:.6f}")


@app.command("evaluate-dpo")
def evaluate_dpo_command(
    data_dir: Annotated[
        Path,
        typer.Option("--data-dir", exists=True, file_okay=False, readable=True),
    ] = Path("data/processed/custom"),
    model_path: Annotated[
        Path,
        typer.Option(
            "--model-path",
            exists=True,
            file_okay=False,
            readable=True,
        ),
    ] = Path("artifacts/models/dpo"),
    max_new_tokens: Annotated[int, typer.Option("--max-new-tokens", min=1)] = 32,
    temperature: Annotated[float, typer.Option("--temperature", min=0.0)] = 0.0,
    top_p: Annotated[float, typer.Option("--top-p", min=0.0, max=1.0)] = 1.0,
    seed: Annotated[int, typer.Option("--seed", min=0)] = 42,
    output: Annotated[
        Path,
        typer.Option("--output", dir_okay=False),
    ] = DEFAULT_DPO_EVALUATION_PATH,
    generations_output: Annotated[
        Path,
        typer.Option("--generations-output", dir_okay=False),
    ] = DEFAULT_DPO_GENERATIONS_PATH,
    report_output: Annotated[
        Path,
        typer.Option("--report-output", dir_okay=False),
    ] = DEFAULT_DPO_REPORT_PATH,
) -> None:
    """Evaluate a DPO model and generate its comparison report."""

    try:
        result = run_dpo_evaluation(
            data_dir,
            model_path,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            seed=seed,
            output_path=output,
            generations_path=generations_output,
        )
        report_path = generate_dpo_report(
            dpo_evaluation_path=output,
            output_path=report_output,
        )
    except (
        DPOReportError,
        DataValidationError,
        GenerationConfigError,
        JsonlError,
        ModelInferenceError,
        OSError,
        SFTCheckpointError,
        ValueError,
    ) as exc:
        typer.echo(f"DPO evaluation failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    typer.echo(
        f"Evaluated {result.dataset.record_count} records with DPO model {model_path}"
    )
    typer.echo(f"Wrote DPO generations to {result.generations_path}")
    typer.echo(f"Wrote DPO evaluation to {output}")
    typer.echo(f"Wrote DPO report to {report_path}")
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
