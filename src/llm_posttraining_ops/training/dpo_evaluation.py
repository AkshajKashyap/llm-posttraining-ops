"""Evaluation wrapper for trained DPO models and adapters."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from llm_posttraining_ops.inference.evaluation import ModelEvaluationResult, ModelGenerator
from llm_posttraining_ops.training.evaluation import (
    ResolvedModelArtifact,
    resolve_model_artifact,
    run_sft_evaluation,
)

DEFAULT_DPO_EVALUATION_PATH = Path("artifacts/evals/dpo_model_eval.json")
DEFAULT_DPO_GENERATIONS_PATH = Path("artifacts/evals/generations/dpo.jsonl")


def resolve_dpo_checkpoint(model_path: str | Path) -> ResolvedModelArtifact:
    """Resolve a DPO full-model or PEFT-adapter path."""

    return resolve_model_artifact(model_path)


def run_dpo_evaluation(
    data_dir: str | Path,
    model_path: str | Path,
    *,
    max_new_tokens: int = 32,
    temperature: float = 0.0,
    top_p: float = 1.0,
    seed: int = 42,
    output_path: str | Path = DEFAULT_DPO_EVALUATION_PATH,
    generations_path: str | Path = DEFAULT_DPO_GENERATIONS_PATH,
    generator: ModelGenerator | None = None,
    clock: Any = None,
) -> ModelEvaluationResult:
    """Evaluate a DPO model through the shared model harness."""

    return run_sft_evaluation(
        data_dir,
        model_path,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        seed=seed,
        output_path=output_path,
        generations_path=generations_path,
        generator=generator,
        clock=clock,
    )
