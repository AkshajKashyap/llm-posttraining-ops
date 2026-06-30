"""Validation for SFT and preference JSONL datasets."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal, TypeAlias

from llm_posttraining_ops.data.jsonl import JsonlError, read_jsonl
from llm_posttraining_ops.data.schemas import SPLIT_NAMES

DatasetKind: TypeAlias = Literal["sft", "preference"]

REQUIRED_FIELDS: dict[DatasetKind, tuple[str, ...]] = {
    "sft": ("id", "split", "instruction", "input", "output"),
    "preference": ("id", "split", "instruction", "input", "chosen", "rejected"),
}


class DataValidationError(ValueError):
    """Raised with all validation issues found in a dataset."""

    def __init__(self, issues: Sequence[str]) -> None:
        self.issues = tuple(issues)
        super().__init__("\n".join(self.issues))


def validate_records(
    records: Sequence[Mapping[str, Any]],
    kind: DatasetKind,
    *,
    source: str = "<records>",
) -> None:
    """Validate required strings, IDs, and split names for one dataset."""

    issues: list[str] = []
    seen_ids: set[str] = set()
    if not records:
        issues.append(f"{source}: dataset is empty")

    for index, record in enumerate(records, start=1):
        location = f"{source}:record {index}"
        for field in REQUIRED_FIELDS[kind]:
            if field not in record:
                issues.append(f"{location}: missing required field '{field}'")
                continue
            value = record[field]
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


def validate_data_directory(data_dir: str | Path) -> dict[str, int]:
    """Load and validate the expected SFT and preference files in a directory."""

    directory = Path(data_dir)
    counts: dict[str, int] = {}
    issues: list[str] = []
    for kind in ("sft", "preference"):
        path = directory / f"{kind}.jsonl"
        try:
            records = read_jsonl(path)
            validate_records(records, kind, source=str(path))
        except (JsonlError, DataValidationError) as exc:
            issues.extend(str(exc).splitlines())
        else:
            counts[kind] = len(records)

    if issues:
        raise DataValidationError(issues)
    return counts
