"""Evaluation harness for deterministic baseline response generators."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

from llm_posttraining_ops.data.jsonl import read_jsonl
from llm_posttraining_ops.data.schemas import SFTRecord, SplitName
from llm_posttraining_ops.data.validation import validate_records
from llm_posttraining_ops.evaluation.baselines import (
    DEFAULT_BASELINES,
    BaselineGenerator,
)
from llm_posttraining_ops.evaluation.metrics import (
    AggregateMetrics,
    ExampleMetrics,
    aggregate_metrics,
    calculate_example_metrics,
)

EVALUATION_SCHEMA_VERSION = "1.0"
DEFAULT_EVALUATION_PATH = Path("artifacts/evals/baseline_eval.json")


@dataclass(frozen=True, slots=True)
class EvaluationExample:
    """Inputs, output, and scores for one generated response."""

    id: str
    split: str
    instruction: str
    input: str
    expected_output: str
    generated_response: str
    metrics: ExampleMetrics


@dataclass(frozen=True, slots=True)
class BaselineEvaluation:
    """Aggregate and per-example results for one baseline."""

    name: str
    metrics: AggregateMetrics
    examples: list[EvaluationExample]


@dataclass(frozen=True, slots=True)
class DatasetSummary:
    """Metadata about the evaluated SFT dataset."""

    path: str
    record_count: int
    split_counts: dict[str, int]


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    """Versioned JSON-serializable evaluation output."""

    schema_version: str
    dataset: DatasetSummary
    baselines: list[BaselineEvaluation]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_sft_records(path: str | Path) -> list[SFTRecord]:
    """Load and validate SFT records from JSONL."""

    input_path = Path(path)
    raw_records = read_jsonl(input_path)
    validate_records(raw_records, "sft", source=str(input_path))
    return [
        SFTRecord(
            id=record["id"],
            split=cast(SplitName, record["split"]),
            instruction=record["instruction"],
            input=record["input"],
            output=record["output"],
        )
        for record in raw_records
    ]


def evaluate_records(
    records: Sequence[SFTRecord],
    *,
    dataset_path: str,
    baselines: Sequence[BaselineGenerator] = DEFAULT_BASELINES,
) -> EvaluationResult:
    """Generate and score responses for each baseline."""

    if not records:
        raise ValueError("Cannot evaluate an empty dataset")
    if not baselines:
        raise ValueError("At least one baseline is required")

    baseline_results: list[BaselineEvaluation] = []
    for baseline in baselines:
        examples: list[EvaluationExample] = []
        for record in records:
            response = baseline.generate(record.instruction, record.input)
            metrics = calculate_example_metrics(record.output, response)
            examples.append(
                EvaluationExample(
                    id=record.id,
                    split=record.split,
                    instruction=record.instruction,
                    input=record.input,
                    expected_output=record.output,
                    generated_response=response,
                    metrics=metrics,
                )
            )
        baseline_results.append(
            BaselineEvaluation(
                name=baseline.name,
                metrics=aggregate_metrics([example.metrics for example in examples]),
                examples=examples,
            )
        )

    return EvaluationResult(
        schema_version=EVALUATION_SCHEMA_VERSION,
        dataset=DatasetSummary(
            path=dataset_path,
            record_count=len(records),
            split_counts=dict(sorted(Counter(record.split for record in records).items())),
        ),
        baselines=baseline_results,
    )


def write_evaluation_result(result: EvaluationResult, path: str | Path) -> Path:
    """Write an evaluation result as stable, human-readable JSON."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as output_file:
        json.dump(result.to_dict(), output_file, indent=2, sort_keys=True)
        output_file.write("\n")
    return output_path


def run_baseline_evaluation(
    data_dir: str | Path,
    output_path: str | Path = DEFAULT_EVALUATION_PATH,
) -> EvaluationResult:
    """Evaluate all default baselines on the SFT dataset and save the result."""

    sft_path = Path(data_dir) / "sft.jsonl"
    result = evaluate_records(
        load_sft_records(sft_path),
        dataset_path=str(sft_path),
    )
    write_evaluation_result(result, output_path)
    return result
