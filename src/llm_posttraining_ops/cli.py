"""Command-line interface for data preparation and validation."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from llm_posttraining_ops.config import ConfigError, load_config
from llm_posttraining_ops.data.generation import prepare_demo_data
from llm_posttraining_ops.data.validation import DataValidationError, validate_data_directory

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
            help="Directory containing sft.jsonl and preference.jsonl.",
        ),
    ] = Path("data/processed/demo"),
) -> None:
    """Validate prepared SFT and preference datasets."""

    try:
        counts = validate_data_directory(data_dir)
    except DataValidationError as exc:
        typer.echo(f"Validation failed:\n{exc}", err=True)
        raise typer.Exit(code=1) from exc

    typer.echo(
        "Validation passed: "
        f"{counts['sft']} SFT records, {counts['preference']} preference records"
    )


if __name__ == "__main__":
    app()
