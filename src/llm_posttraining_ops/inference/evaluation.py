"""Model-backed evaluation on normalized SFT records."""

from __future__ import annotations

import json
import re
import time
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

from llm_posttraining_ops.data.ingestion import load_normalized_sft
from llm_posttraining_ops.data.jsonl import write_jsonl
from llm_posttraining_ops.data.schemas import SFTRecord
from llm_posttraining_ops.evaluation.metrics import (
    AggregateMetrics,
    ExampleMetrics,
    aggregate_metrics,
    calculate_example_metrics,
)
from llm_posttraining_ops.inference.config import GenerationSettings
from llm_posttraining_ops.inference.huggingface import (
    GenerationOutput,
    HuggingFaceCausalLMGenerator,
)
from llm_posttraining_ops.inference.prompts import format_instruction_prompt

MODEL_EVALUATION_SCHEMA_VERSION = "1.0"
DEFAULT_MODEL_EVALUATION_PATH = Path("artifacts/evals/model_eval.json")
DEFAULT_GENERATIONS_DIR = Path("artifacts/evals/generations")


class ModelGenerator(Protocol):
    """Generator interface used by the model evaluation harness."""

    model_name: str

    def generate(self, prompt: str, settings: GenerationSettings) -> GenerationOutput:
        """Generate one model response."""


@dataclass(frozen=True, slots=True)
class ModelEvaluationExample:
    """One prompt, generation, metric set, and latency measurement."""

    id: str
    split: str
    instruction: str
    input: str
    prompt: str
    expected_output: str
    generated_response: str
    generated_tokens: int
    generation_seconds: float
    metrics: ExampleMetrics

    def generation_record(self, model_name: str) -> dict[str, Any]:
        """Return the stable JSONL generation schema."""

        return {
            "id": self.id,
            "split": self.split,
            "model_name": model_name,
            "prompt": self.prompt,
            "expected_output": self.expected_output,
            "generated_response": self.generated_response,
            "generated_tokens": self.generated_tokens,
            "generation_seconds": self.generation_seconds,
        }


@dataclass(frozen=True, slots=True)
class ModelLatency:
    """Aggregate generation latency and token statistics."""

    total_generation_seconds: float
    average_seconds_per_example: float
    average_generated_tokens: float


@dataclass(frozen=True, slots=True)
class ModelDatasetSummary:
    """Metadata about the evaluated normalized dataset."""

    path: str
    record_count: int
    split_counts: dict[str, int]


@dataclass(frozen=True, slots=True)
class ModelSummary:
    """Model identity and generation settings."""

    name: str
    device: str
    generation: GenerationSettings


@dataclass(frozen=True, slots=True)
class ModelEvaluationResult:
    """Versioned model-evaluation artifact."""

    schema_version: str
    dataset: ModelDatasetSummary
    model: ModelSummary
    metrics: AggregateMetrics
    latency: ModelLatency
    generations_path: str
    examples: list[ModelEvaluationExample]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def default_generations_path(model_name: str) -> Path:
    """Build a filesystem-safe generation path from a model identifier."""

    slug = re.sub(r"[^a-zA-Z0-9._-]+", "_", model_name).strip("_") or "model"
    return DEFAULT_GENERATIONS_DIR / f"{slug}.jsonl"


def evaluate_model_records(
    records: Sequence[SFTRecord],
    *,
    generator: ModelGenerator,
    settings: GenerationSettings,
    dataset_path: str,
    generations_path: str | Path,
    clock: Callable[[], float] = time.perf_counter,
) -> ModelEvaluationResult:
    """Generate and score model responses with per-example timing."""

    if not records:
        raise ValueError("Cannot evaluate an empty dataset")

    examples: list[ModelEvaluationExample] = []
    for record in records:
        prompt = format_instruction_prompt(record.instruction, record.input)
        start = clock()
        output = generator.generate(prompt, settings)
        elapsed = round(clock() - start, 6)
        examples.append(
            ModelEvaluationExample(
                id=record.id,
                split=record.split,
                instruction=record.instruction,
                input=record.input,
                prompt=prompt,
                expected_output=record.output,
                generated_response=output.text,
                generated_tokens=output.generated_tokens,
                generation_seconds=elapsed,
                metrics=calculate_example_metrics(record.output, output.text),
            )
        )

    count = len(examples)
    total_seconds = round(sum(example.generation_seconds for example in examples), 6)
    generation_path = Path(generations_path)
    device = getattr(generator, "device", "cpu")
    return ModelEvaluationResult(
        schema_version=MODEL_EVALUATION_SCHEMA_VERSION,
        dataset=ModelDatasetSummary(
            path=dataset_path,
            record_count=count,
            split_counts=dict(sorted(Counter(record.split for record in records).items())),
        ),
        model=ModelSummary(
            name=settings.model_name,
            device=device,
            generation=settings,
        ),
        metrics=aggregate_metrics([example.metrics for example in examples]),
        latency=ModelLatency(
            total_generation_seconds=total_seconds,
            average_seconds_per_example=round(total_seconds / count, 6),
            average_generated_tokens=round(
                sum(example.generated_tokens for example in examples) / count,
                6,
            ),
        ),
        generations_path=str(generation_path),
        examples=examples,
    )


def write_model_evaluation(result: ModelEvaluationResult, path: str | Path) -> Path:
    """Write a model evaluation as formatted JSON."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as output_file:
        json.dump(result.to_dict(), output_file, indent=2, sort_keys=True)
        output_file.write("\n")
    return output_path


def write_model_generations(result: ModelEvaluationResult) -> Path:
    """Write per-example model generations as JSONL."""

    return write_jsonl(
        result.generations_path,
        [example.generation_record(result.model.name) for example in result.examples],
    )


def run_model_evaluation(
    data_dir: str | Path,
    settings: GenerationSettings,
    *,
    output_path: str | Path = DEFAULT_MODEL_EVALUATION_PATH,
    generations_path: str | Path | None = None,
    generator: ModelGenerator | None = None,
    clock: Callable[[], float] = time.perf_counter,
) -> ModelEvaluationResult:
    """Run Hugging Face inference, then save generations and evaluation metrics."""

    sft_path = Path(data_dir) / "sft.jsonl"
    resolved_generations_path = generations_path or default_generations_path(
        settings.model_name
    )
    active_generator = generator or HuggingFaceCausalLMGenerator(settings.model_name)
    result = evaluate_model_records(
        load_normalized_sft(sft_path),
        generator=active_generator,
        settings=settings,
        dataset_path=str(sft_path),
        generations_path=resolved_generations_path,
        clock=clock,
    )
    write_model_generations(result)
    write_model_evaluation(result, output_path)
    return result
