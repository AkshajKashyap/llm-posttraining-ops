"""Validation for SFT and preference JSONL datasets."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal, TypeAlias

from llm_posttraining_ops.data.jsonl import JsonlError, read_jsonl
from llm_posttraining_ops.data.schemas import SPLIT_NAMES

DatasetKind: TypeAlias = Literal["sft", "preference"]

REQUIRED_FIELDS: dict[DatasetKind, tuple[str, ...]] = {
    "sft": ("id", "split", "instruction", "input", "output", "source", "metadata"),
    "preference": ("id", "split", "instruction", "input", "chosen", "rejected"),
}


class DataValidationError(ValueError):
    """Raised with all validation issues found in a dataset."""

    def __init__(self, issues: Sequence[str]) -> None:
        self.issues = tuple(issues)
        super().__init__("\n".join(self.issues))


def _normalized_tokens(value: str) -> list[str]:
    return re.findall(r"\w+", value.casefold())


def _is_suspiciously_repetitive(value: str) -> bool:
    tokens = _normalized_tokens(value)
    if len(tokens) < 4:
        return False
    dominant_count = Counter(tokens).most_common(1)[0][1]
    return dominant_count / len(tokens) >= 0.75


def validate_records(
    records: Sequence[Mapping[str, Any]],
    kind: DatasetKind,
    *,
    source: str = "<records>",
    minimum_output_length: int = 1,
) -> None:
    """Validate schema, IDs, splits, and response quality for one dataset."""

    if minimum_output_length < 1:
        raise ValueError("minimum_output_length must be at least 1")
    issues: list[str] = []
    seen_ids: set[str] = set()
    if not records:
        issues.append(f"{source}: dataset is empty")

    for index, record in enumerate(records, start=1):
        location = f"{source}:record {index}"
        for field in REQUIRED_FIELDS[kind]:
            if field not in record:
                issues.append(f"{location}: missing required field '{field}'")

        if kind == "sft":
            for field in ("id", "split", "instruction", "output", "source"):
                value = record.get(field)
                if not isinstance(value, str) or not value.strip():
                    issues.append(f"{location}: field '{field}' must be a non-empty string")
            input_text = record.get("input")
            if not isinstance(input_text, str):
                issues.append(f"{location}: field 'input' must be a string")
            if not isinstance(record.get("metadata"), Mapping):
                issues.append(f"{location}: field 'metadata' must be an object")

            instruction = record.get("instruction")
            output = record.get("output")
            if isinstance(output, str) and output.strip():
                if len(output.strip()) < minimum_output_length:
                    issues.append(
                        f"{location}: output is shorter than minimum length "
                        f"{minimum_output_length}"
                    )
                if _is_suspiciously_repetitive(output):
                    issues.append(f"{location}: output is suspiciously repetitive")
                if (
                    isinstance(instruction, str)
                    and _normalized_tokens(output) == _normalized_tokens(instruction)
                ):
                    issues.append(f"{location}: output copies the instruction")
        else:
            for field in REQUIRED_FIELDS[kind]:
                value = record.get(field)
                if not isinstance(value, str) or not value.strip():
                    issues.append(f"{location}: field '{field}' must be a non-empty string")

        record_id = record.get("id")
        if isinstance(record_id, str) and record_id.strip():
            if record_id in seen_ids:
                issues.append(f"{location}: duplicate id '{record_id}'")
            seen_ids.add(record_id)

        split = record.get("split")
        if isinstance(split, str) and split.strip() and split not in SPLIT_NAMES:
            allowed = ", ".join(SPLIT_NAMES)
            issues.append(
                f"{location}: unsupported split '{split}' (expected one of: {allowed})"
            )

    if issues:
        raise DataValidationError(issues)


def validate_data_directory(
    data_dir: str | Path,
    *,
    minimum_output_length: int = 1,
) -> dict[str, int]:
    """Validate required SFT and optional preference files in a directory."""

    directory = Path(data_dir)
    counts: dict[str, int] = {}
    issues: list[str] = []
    for kind in ("sft", "preference"):
        path = directory / f"{kind}.jsonl"
        if kind == "preference" and not path.exists():
            continue
        try:
            records = read_jsonl(path)
            validate_records(
                records,
                kind,
                source=str(path),
                minimum_output_length=minimum_output_length if kind == "sft" else 1,
            )
        except (JsonlError, DataValidationError) as exc:
            issues.extend(str(exc).splitlines())
        else:
            counts[kind] = len(records)

    if issues:
        raise DataValidationError(issues)
    return counts
