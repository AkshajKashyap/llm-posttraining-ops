"""Local JSONL ingestion for normalized SFT datasets."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

from llm_posttraining_ops.data.jsonl import read_jsonl, write_jsonl
from llm_posttraining_ops.data.normalization import SFTFormat, normalize_sft_records
from llm_posttraining_ops.data.schemas import SFTRecord, SplitName
from llm_posttraining_ops.data.validation import validate_records

DEFAULT_INGEST_MIN_OUTPUT_LENGTH = 10


def load_normalized_sft(
    path: str | Path,
    *,
    minimum_output_length: int = 1,
) -> list[SFTRecord]:
    """Load and validate normalized SFT JSONL."""

    input_path = Path(path)
    records = read_jsonl(input_path)
    validate_records(
        records,
        "sft",
        source=str(input_path),
        minimum_output_length=minimum_output_length,
    )
    return [
        SFTRecord(
            id=record["id"],
            split=cast(SplitName, record["split"]),
            instruction=record["instruction"],
            input=record["input"],
            output=record["output"],
            source=record["source"],
            metadata=dict(cast(Mapping[str, Any], record["metadata"])),
        )
        for record in records
    ]


def ingest_sft_data(
    input_path: str | Path,
    output_dir: str | Path,
    format_name: SFTFormat,
    *,
    minimum_output_length: int = DEFAULT_INGEST_MIN_OUTPUT_LENGTH,
) -> tuple[Path, list[SFTRecord]]:
    """Normalize, validate, and save a local raw instruction dataset."""

    raw_path = Path(input_path)
    raw_records = read_jsonl(raw_path)
    normalized = normalize_sft_records(
        raw_records,
        format_name=format_name,
        source=raw_path.stem,
    )
    serialized = [record.to_dict() for record in normalized]
    validate_records(
        serialized,
        "sft",
        source=str(raw_path),
        minimum_output_length=minimum_output_length,
    )
    output_path = Path(output_dir) / "sft.jsonl"
    write_jsonl(output_path, serialized)
    return output_path, normalized
