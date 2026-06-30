"""Checkpoint resolution and evaluation for trained SFT models or adapters."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from llm_posttraining_ops.inference.config import GenerationSettings
from llm_posttraining_ops.inference.evaluation import (
    ModelEvaluationResult,
    ModelGenerator,
    run_model_evaluation,
)
from llm_posttraining_ops.inference.huggingface import HuggingFaceCausalLMGenerator

DEFAULT_SFT_EVALUATION_PATH = Path("artifacts/evals/sft_model_eval.json")
DEFAULT_SFT_GENERATIONS_PATH = Path("artifacts/evals/generations/sft.jsonl")


class SFTCheckpointError(ValueError):
    """Raised when a trained model or adapter checkpoint is invalid."""


@dataclass(frozen=True, slots=True)
class ResolvedModelArtifact:
    """Resolved full-model or PEFT-adapter metadata."""

    kind: str
    path: Path
    base_model_name: str | None = None


def resolve_model_artifact(model_path: str | Path) -> ResolvedModelArtifact:
    """Identify whether a local path contains a full model or PEFT adapter."""

    path = Path(model_path)
    if not path.is_dir():
        raise SFTCheckpointError(f"Model path is not a directory: {path}")
    adapter_config_path = path / "adapter_config.json"
    if not adapter_config_path.exists():
        return ResolvedModelArtifact(kind="full_model", path=path)

    try:
        with adapter_config_path.open(encoding="utf-8") as config_file:
            adapter_config = json.load(config_file)
    except json.JSONDecodeError as exc:
        raise SFTCheckpointError(
            f"Invalid adapter configuration in {adapter_config_path}: {exc.msg}"
        ) from exc
    base_model_name = adapter_config.get("base_model_name_or_path")
    if not isinstance(base_model_name, str) or not base_model_name.strip():
        raise SFTCheckpointError("Adapter config must define base_model_name_or_path")
    return ResolvedModelArtifact(
        kind="adapter",
        path=path,
        base_model_name=base_model_name,
    )


def load_trained_generator(model_path: str | Path) -> HuggingFaceCausalLMGenerator:
    """Load a local full model or compose a PEFT adapter with its base model."""

    artifact = resolve_model_artifact(model_path)
    if artifact.kind == "full_model":
        return HuggingFaceCausalLMGenerator(str(artifact.path))

    assert artifact.base_model_name is not None
    try:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
    except ImportError as exc:
        raise SFTCheckpointError(
            "Adapter evaluation requires torch, transformers, and peft"
        ) from exc

    try:
        tokenizer = AutoTokenizer.from_pretrained(artifact.path)
        base_model = AutoModelForCausalLM.from_pretrained(artifact.base_model_name)
        model = PeftModel.from_pretrained(base_model, artifact.path)
        model.to("cpu")
        model.eval()
    except (OSError, ValueError) as exc:
        raise SFTCheckpointError(f"Unable to load SFT adapter: {exc}") from exc
    return HuggingFaceCausalLMGenerator(
        str(artifact.path),
        tokenizer=tokenizer,
        model=model,
        torch_module=torch,
        seed_function=set_seed,
    )


def run_sft_evaluation(
    data_dir: str | Path,
    model_path: str | Path,
    *,
    max_new_tokens: int = 32,
    temperature: float = 0.0,
    top_p: float = 1.0,
    seed: int = 42,
    output_path: str | Path = DEFAULT_SFT_EVALUATION_PATH,
    generations_path: str | Path = DEFAULT_SFT_GENERATIONS_PATH,
    generator: ModelGenerator | None = None,
    clock: Any = None,
) -> ModelEvaluationResult:
    """Evaluate a trained full model or adapter with the existing harness."""

    resolved_path = Path(model_path)
    active_generator = generator or load_trained_generator(resolved_path)
    settings = GenerationSettings(
        model_name=str(resolved_path),
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        seed=seed,
    )
    kwargs: dict[str, Any] = {}
    if clock is not None:
        kwargs["clock"] = clock
    return run_model_evaluation(
        data_dir,
        settings,
        output_path=output_path,
        generations_path=generations_path,
        generator=active_generator,
        **kwargs,
    )
