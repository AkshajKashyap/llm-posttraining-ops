"""Deterministic pairwise comparison for two generation files."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, TypeAlias

from llm_posttraining_ops.evaluation.suite import (
    EvaluationExample,
    ResponseEvaluation,
    evaluate_response,
    load_evaluation_examples,
    load_generation_responses,
)

PairwiseWinner: TypeAlias = Literal["left", "right", "tie"]
PAIRWISE_SCHEMA_VERSION = "1.0"
DEFAULT_PAIRWISE_PATH = Path("artifacts/evals/pairwise_comparison.json")


@dataclass(frozen=True, slots=True)
class PairwiseDecision:
    """One deterministic left/right/tie decision."""

    id: str
    winner: PairwiseWinner
    reason: str
    left_response: str
    right_response: str
    left_diagnostics: ResponseEvaluation
    right_diagnostics: ResponseEvaluation


@dataclass(frozen=True, slots=True)
class PairwiseCounts:
    """Aggregate win/loss/tie counts from the left perspective."""

    left_wins: int
    right_wins: int
    ties: int


@dataclass(frozen=True, slots=True)
class PairwiseResult:
    """Versioned pairwise comparison artifact."""

    schema_version: str
    left_path: str
    right_path: str
    eval_data_path: str
    record_count: int
    counts: PairwiseCounts
    decisions: list[PairwiseDecision]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def length_sane(example: EvaluationExample, diagnostics: ResponseEvaluation) -> float:
    """Return 1 for non-empty responses within a reference-relative limit."""

    configured_max = example.metadata.get("max_tokens")
    default_max = max(10, len(example.reference_output.split()) * 3)
    maximum = (
        configured_max
        if isinstance(configured_max, int) and configured_max > 0
        else default_max
    )
    return float(0 < diagnostics.response_length <= maximum)


def _quality_components(
    example: EvaluationExample,
    diagnostics: ResponseEvaluation,
) -> tuple[tuple[float, ...], tuple[str, ...]]:
    values = (
        -diagnostics.forbidden_term_violation,
        diagnostics.required_fact_coverage,
        -diagnostics.instruction_copying,
        length_sane(example, diagnostics),
        diagnostics.format_compliant,
        -diagnostics.refusal_detected,
        -float(bool(diagnostics.unsupported_named_entities)),
        -diagnostics.numeric_mismatch,
        -diagnostics.contradiction_detected,
        diagnostics.token_overlap_f1,
        diagnostics.exact_match,
    )
    labels = (
        "fewer forbidden-term violations",
        "higher required-fact coverage",
        "less instruction copying",
        "saner response length",
        "better format compliance",
        "fewer refusal signals",
        "fewer unsupported named entities",
        "fewer numeric mismatches",
        "fewer contradiction signals",
        "higher token overlap",
        "exact match",
    )
    return values, labels


def compare_responses(
    example: EvaluationExample,
    left_response: str,
    right_response: str,
) -> PairwiseDecision:
    """Choose a response using a deterministic lexicographic quality order."""

    left = evaluate_response(example, left_response)
    right = evaluate_response(example, right_response)
    left_values, labels = _quality_components(example, left)
    right_values, _ = _quality_components(example, right)
    winner: PairwiseWinner = "tie"
    reason = "all deterministic quality signals are equal"
    for left_value, right_value, label in zip(left_values, right_values, labels):
        if left_value == right_value:
            continue
        winner = "left" if left_value > right_value else "right"
        reason = label
        break
    return PairwiseDecision(
        id=example.id,
        winner=winner,
        reason=reason,
        left_response=left_response,
        right_response=right_response,
        left_diagnostics=left,
        right_diagnostics=right,
    )


def compare_generation_files(
    left_path: str | Path,
    right_path: str | Path,
    eval_data_path: str | Path,
) -> PairwiseResult:
    """Compare aligned generation files on every evaluation example."""

    examples = load_evaluation_examples(eval_data_path)
    left_responses = load_generation_responses(left_path)
    right_responses = load_generation_responses(right_path)
    missing_left = [example.id for example in examples if example.id not in left_responses]
    missing_right = [example.id for example in examples if example.id not in right_responses]
    if missing_left or missing_right:
        missing_parts = []
        if missing_left:
            missing_parts.append(f"left missing: {', '.join(missing_left)}")
        if missing_right:
            missing_parts.append(f"right missing: {', '.join(missing_right)}")
        raise ValueError("; ".join(missing_parts))

    decisions = [
        compare_responses(
            example,
            left_responses[example.id],
            right_responses[example.id],
        )
        for example in examples
    ]
    return PairwiseResult(
        schema_version=PAIRWISE_SCHEMA_VERSION,
        left_path=str(left_path),
        right_path=str(right_path),
        eval_data_path=str(eval_data_path),
        record_count=len(decisions),
        counts=PairwiseCounts(
            left_wins=sum(decision.winner == "left" for decision in decisions),
            right_wins=sum(decision.winner == "right" for decision in decisions),
            ties=sum(decision.winner == "tie" for decision in decisions),
        ),
        decisions=decisions,
    )


def write_pairwise_result(
    result: PairwiseResult,
    path: str | Path = DEFAULT_PAIRWISE_PATH,
) -> Path:
    """Write pairwise comparison results as formatted JSON."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as output_file:
        json.dump(result.to_dict(), output_file, indent=2, sort_keys=True)
        output_file.write("\n")
    return output_path
