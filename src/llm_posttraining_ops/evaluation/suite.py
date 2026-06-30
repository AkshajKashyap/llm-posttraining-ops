"""Deterministic instruction-following evaluation suite."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, TypeAlias, cast

from llm_posttraining_ops.data.jsonl import read_jsonl
from llm_posttraining_ops.data.schemas import SPLIT_NAMES, SplitName
from llm_posttraining_ops.evaluation.metrics import (
    exact_match,
    response_length,
    token_overlap_f1,
    tokenize,
)

TaskType: TypeAlias = Literal["freeform", "json", "list", "short_answer"]
TASK_TYPES: tuple[TaskType, ...] = ("freeform", "json", "list", "short_answer")
EVAL_SUITE_SCHEMA_VERSION = "1.0"
DEFAULT_EVAL_SUITE_PATH = Path("artifacts/evals/eval_suite_results.json")

REFUSAL_PHRASES = (
    "i cannot",
    "i can't",
    "i will not",
    "i won't",
    "unable to",
    "cannot assist",
    "as an ai",
)
NEGATION_TERMS = {"no", "not", "never", "cannot", "without"}
IGNORED_ENTITY_WORDS = {"A", "An", "I", "It", "The", "This"}


class EvalSuiteError(ValueError):
    """Raised when evaluation data or generations are invalid."""


@dataclass(frozen=True, slots=True)
class EvaluationExample:
    """Reference-backed instruction-following evaluation example."""

    id: str
    split: SplitName
    instruction: str
    input: str
    reference_output: str
    required_facts: list[str]
    forbidden_terms: list[str]
    task_type: TaskType
    metadata: dict[str, Any]


@dataclass(frozen=True, slots=True)
class ResponseEvaluation:
    """All deterministic diagnostics for one generated response."""

    exact_match: float
    token_overlap_f1: float
    required_fact_coverage: float
    forbidden_term_violation: float
    instruction_copying: float
    empty_response: float
    response_length: int
    refusal_detected: float
    format_compliant: float
    unsupported_named_entities: list[str]
    numeric_mismatch: float
    contradiction_detected: float


@dataclass(frozen=True, slots=True)
class EvaluatedExample:
    """Response plus diagnostics for one evaluation ID."""

    id: str
    split: str
    task_type: str
    reference_output: str
    generated_response: str
    diagnostics: ResponseEvaluation


@dataclass(frozen=True, slots=True)
class ResponseLengthStatistics:
    """Aggregate response length in tokens."""

    average: float
    minimum: int
    maximum: int


@dataclass(frozen=True, slots=True)
class SuiteMetrics:
    """Aggregate deterministic evaluation-suite metrics."""

    exact_match: float
    token_overlap_f1: float
    required_fact_coverage: float
    forbidden_term_violation_rate: float
    instruction_copying_rate: float
    empty_response_rate: float
    refusal_rate: float
    format_compliance_rate: float
    unsupported_named_entity_rate: float
    numeric_mismatch_rate: float
    contradiction_rate: float
    response_length: ResponseLengthStatistics


@dataclass(frozen=True, slots=True)
class EvalSuiteResult:
    """Versioned evaluation-suite artifact."""

    schema_version: str
    generations_path: str
    eval_data_path: str
    record_count: int
    metrics: SuiteMetrics
    examples: list[EvaluatedExample]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _string_list(value: object, field: str, location: str) -> list[str]:
    if not isinstance(value, list) or not all(
        isinstance(item, str) and item.strip() for item in value
    ):
        raise EvalSuiteError(f"{location}: field '{field}' must be a list of strings")
    return list(value)


def load_evaluation_examples(path: str | Path) -> list[EvaluationExample]:
    """Load and validate deterministic evaluation examples from JSONL."""

    input_path = Path(path)
    raw_records = read_jsonl(input_path)
    examples: list[EvaluationExample] = []
    seen_ids: set[str] = set()
    for index, record in enumerate(raw_records, start=1):
        location = f"{input_path}:record {index}"
        required = (
            "id",
            "split",
            "instruction",
            "input",
            "reference_output",
            "required_facts",
            "forbidden_terms",
            "task_type",
            "metadata",
        )
        missing = [field for field in required if field not in record]
        if missing:
            raise EvalSuiteError(f"{location}: missing required fields: {', '.join(missing)}")

        for field in ("id", "split", "instruction", "reference_output", "task_type"):
            if not isinstance(record[field], str) or not record[field].strip():
                raise EvalSuiteError(f"{location}: field '{field}' must be non-empty")
        if not isinstance(record["input"], str):
            raise EvalSuiteError(f"{location}: field 'input' must be a string")
        if record["split"] not in SPLIT_NAMES:
            raise EvalSuiteError(f"{location}: unsupported split '{record['split']}'")
        if record["task_type"] not in TASK_TYPES:
            raise EvalSuiteError(
                f"{location}: unsupported task type '{record['task_type']}'"
            )
        if not isinstance(record["metadata"], Mapping):
            raise EvalSuiteError(f"{location}: field 'metadata' must be an object")
        record_id = record["id"]
        if record_id in seen_ids:
            raise EvalSuiteError(f"{location}: duplicate id '{record_id}'")
        seen_ids.add(record_id)

        examples.append(
            EvaluationExample(
                id=record_id,
                split=cast(SplitName, record["split"]),
                instruction=record["instruction"],
                input=record["input"],
                reference_output=record["reference_output"],
                required_facts=_string_list(
                    record["required_facts"], "required_facts", location
                ),
                forbidden_terms=_string_list(
                    record["forbidden_terms"], "forbidden_terms", location
                ),
                task_type=cast(TaskType, record["task_type"]),
                metadata=dict(record["metadata"]),
            )
        )
    if not examples:
        raise EvalSuiteError(f"{input_path}: evaluation dataset is empty")
    return examples


def load_generation_responses(path: str | Path) -> dict[str, str]:
    """Load unique ID-to-response mappings from a generation JSONL file."""

    input_path = Path(path)
    records = read_jsonl(input_path)
    responses: dict[str, str] = {}
    for index, record in enumerate(records, start=1):
        location = f"{input_path}:record {index}"
        record_id = record.get("id")
        response = record.get("generated_response", record.get("response"))
        if not isinstance(record_id, str) or not record_id.strip():
            raise EvalSuiteError(f"{location}: field 'id' must be non-empty")
        if not isinstance(response, str):
            raise EvalSuiteError(
                f"{location}: expected 'generated_response' or 'response' string"
            )
        if record_id in responses:
            raise EvalSuiteError(f"{location}: duplicate generation id '{record_id}'")
        responses[record_id] = response
    return responses


def _contains_token_phrase(response_tokens: list[str], phrase: str) -> bool:
    phrase_tokens = tokenize(phrase)
    if not phrase_tokens:
        return False
    width = len(phrase_tokens)
    return any(
        response_tokens[index : index + width] == phrase_tokens
        for index in range(len(response_tokens) - width + 1)
    )


def required_fact_coverage(required_facts: Sequence[str], response: str) -> float:
    """Return the fraction of normalized required phrases present."""

    if not required_facts:
        return 1.0
    response_tokens = tokenize(response)
    covered = sum(_contains_token_phrase(response_tokens, fact) for fact in required_facts)
    return covered / len(required_facts)


def forbidden_term_violation(forbidden_terms: Sequence[str], response: str) -> float:
    """Return 1 when any forbidden normalized phrase appears."""

    response_tokens = tokenize(response)
    return float(
        any(_contains_token_phrase(response_tokens, term) for term in forbidden_terms)
    )


def instruction_copying(instruction: str, response: str) -> float:
    """Detect responses that reproduce the instruction text."""

    normalized_instruction = " ".join(tokenize(instruction))
    normalized_response = " ".join(tokenize(response))
    if not normalized_instruction or not normalized_response:
        return 0.0
    return float(
        normalized_response == normalized_instruction
        or (
            len(tokenize(instruction)) >= 4
            and normalized_instruction in normalized_response
        )
    )


def refusal_detected(response: str) -> float:
    """Detect a small deterministic set of refusal phrases."""

    normalized = response.casefold()
    return float(any(phrase in normalized for phrase in REFUSAL_PHRASES))


def format_compliance(example: EvaluationExample, response: str) -> float:
    """Check response shape for JSON, list, short-answer, or freeform tasks."""

    if not response.strip():
        return 0.0
    if example.task_type == "freeform":
        return 1.0
    if example.task_type == "json":
        try:
            parsed = json.loads(response)
        except json.JSONDecodeError:
            return 0.0
        return float(isinstance(parsed, (dict, list)))
    if example.task_type == "list":
        lines = [line.strip() for line in response.splitlines() if line.strip()]
        return float(
            len(lines) >= 2
            and all(re.match(r"^(?:[-*]|\d+[.)])\s+\S", line) for line in lines)
        )
    max_tokens = example.metadata.get("max_tokens", 10)
    if not isinstance(max_tokens, int) or max_tokens < 1:
        max_tokens = 10
    return float(response_length(response) <= max_tokens)


def unsupported_named_entities(example: EvaluationExample, response: str) -> list[str]:
    """Find capitalized terms absent from reference output and required facts."""

    allowed_tokens = set(
        tokenize(f"{example.reference_output} {' '.join(example.required_facts)}")
    )
    entities = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b", response)
    unsupported: list[str] = []
    for entity in entities:
        if entity in IGNORED_ENTITY_WORDS:
            continue
        if not set(tokenize(entity)).issubset(allowed_tokens) and entity not in unsupported:
            unsupported.append(entity)
    return unsupported


def numeric_mismatch(reference: str, response: str) -> float:
    """Flag differing numeric terms when the reference contains numbers."""

    reference_numbers = re.findall(r"[-+]?\d+(?:\.\d+)?", reference)
    if not reference_numbers:
        return 0.0
    response_numbers = re.findall(r"[-+]?\d+(?:\.\d+)?", response)
    return float(sorted(reference_numbers) != sorted(response_numbers))


def contradiction_detected(reference: str, response: str) -> float:
    """Flag a simple mismatch in explicit negation."""

    reference_has_negation = bool(set(tokenize(reference)) & NEGATION_TERMS)
    response_has_negation = bool(set(tokenize(response)) & NEGATION_TERMS)
    return float(reference_has_negation != response_has_negation)


def evaluate_response(example: EvaluationExample, response: str) -> ResponseEvaluation:
    """Calculate all suite diagnostics for one response."""

    return ResponseEvaluation(
        exact_match=exact_match(example.reference_output, response),
        token_overlap_f1=token_overlap_f1(example.reference_output, response),
        required_fact_coverage=required_fact_coverage(example.required_facts, response),
        forbidden_term_violation=forbidden_term_violation(
            example.forbidden_terms, response
        ),
        instruction_copying=instruction_copying(example.instruction, response),
        empty_response=float(not response.strip()),
        response_length=response_length(response),
        refusal_detected=refusal_detected(response),
        format_compliant=format_compliance(example, response),
        unsupported_named_entities=unsupported_named_entities(example, response),
        numeric_mismatch=numeric_mismatch(example.reference_output, response),
        contradiction_detected=contradiction_detected(
            example.reference_output, response
        ),
    )


def _mean(values: Sequence[float | int]) -> float:
    return round(sum(values) / len(values), 6)


def evaluate_generation_file(
    generations_path: str | Path,
    eval_data_path: str | Path,
) -> EvalSuiteResult:
    """Evaluate a generation JSONL file against reference-backed examples."""

    examples = load_evaluation_examples(eval_data_path)
    responses = load_generation_responses(generations_path)
    missing = [example.id for example in examples if example.id not in responses]
    if missing:
        raise EvalSuiteError(f"Missing generation IDs: {', '.join(missing)}")

    evaluated = [
        EvaluatedExample(
            id=example.id,
            split=example.split,
            task_type=example.task_type,
            reference_output=example.reference_output,
            generated_response=responses[example.id],
            diagnostics=evaluate_response(example, responses[example.id]),
        )
        for example in examples
    ]
    diagnostics = [item.diagnostics for item in evaluated]
    lengths = [item.response_length for item in diagnostics]
    return EvalSuiteResult(
        schema_version=EVAL_SUITE_SCHEMA_VERSION,
        generations_path=str(generations_path),
        eval_data_path=str(eval_data_path),
        record_count=len(evaluated),
        metrics=SuiteMetrics(
            exact_match=_mean([item.exact_match for item in diagnostics]),
            token_overlap_f1=_mean([item.token_overlap_f1 for item in diagnostics]),
            required_fact_coverage=_mean(
                [item.required_fact_coverage for item in diagnostics]
            ),
            forbidden_term_violation_rate=_mean(
                [item.forbidden_term_violation for item in diagnostics]
            ),
            instruction_copying_rate=_mean(
                [item.instruction_copying for item in diagnostics]
            ),
            empty_response_rate=_mean([item.empty_response for item in diagnostics]),
            refusal_rate=_mean([item.refusal_detected for item in diagnostics]),
            format_compliance_rate=_mean(
                [item.format_compliant for item in diagnostics]
            ),
            unsupported_named_entity_rate=_mean(
                [float(bool(item.unsupported_named_entities)) for item in diagnostics]
            ),
            numeric_mismatch_rate=_mean(
                [item.numeric_mismatch for item in diagnostics]
            ),
            contradiction_rate=_mean(
                [item.contradiction_detected for item in diagnostics]
            ),
            response_length=ResponseLengthStatistics(
                average=_mean(lengths),
                minimum=min(lengths),
                maximum=max(lengths),
            ),
        ),
        examples=evaluated,
    )


def write_eval_suite_result(
    result: EvalSuiteResult,
    path: str | Path = DEFAULT_EVAL_SUITE_PATH,
) -> Path:
    """Write suite results as formatted JSON."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as output_file:
        json.dump(result.to_dict(), output_file, indent=2, sort_keys=True)
        output_file.write("\n")
    return output_path
