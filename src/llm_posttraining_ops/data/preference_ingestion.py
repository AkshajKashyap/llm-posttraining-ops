"""Local JSONL ingestion for normalized preference datasets."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

from llm_posttraining_ops.data.jsonl import read_jsonl, write_jsonl
from llm_posttraining_ops.data.preference_normalization import (
    PreferenceFormat,
    normalize_preference_records,
)
from llm_posttraining_ops.data.schemas import PreferenceRecord, SplitName
from llm_posttraining_ops.data.validation import validate_records

DEFAULT_PREFERENCE_MIN_LENGTH = 10


def load_normalized_preferences(
    path: str | Path,
    *,
    minimum_response_length: int = 1,
) -> list[PreferenceRecord]:
    """Load and validate normalized preference JSONL."""

    input_path = Path(path)
    records = read_jsonl(input_path)
    validate_records(
        records,
        "preference",
        source=str(input_path),
        minimum_output_length=minimum_response_length,
    )
    return [
        PreferenceRecord(
            id=record["id"],
            split=cast(SplitName, record["split"]),
            instruction=record["instruction"],
            input=record["input"],
            chosen=record["chosen"],
            rejected=record["rejected"],
            source=record["source"],
            metadata=dict(cast(Mapping[str, Any], record["metadata"])),
        )
        for record in records
    ]


def ingest_preference_data(
    input_path: str | Path,
    output_dir: str | Path,
    format_name: PreferenceFormat,
    *,
    minimum_response_length: int = DEFAULT_PREFERENCE_MIN_LENGTH,
) -> tuple[Path, list[PreferenceRecord]]:
    """Normalize, validate, and save a local preference dataset."""

    raw_path = Path(input_path)
    raw_records = read_jsonl(raw_path)
    normalized = normalize_preference_records(
        raw_records,
        format_name=format_name,
        source=raw_path.stem,
    )
    serialized = [record.to_dict() for record in normalized]
    validate_records(
        serialized,
        "preference",
        source=str(raw_path),
        minimum_output_length=minimum_response_length,
    )
    output_path = Path(output_dir) / "preference.jsonl"
    write_jsonl(output_path, serialized)
    return output_path, normalized
