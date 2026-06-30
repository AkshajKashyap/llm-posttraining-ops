"""Small, dependency-free JSON Lines helpers."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any


class JsonlError(ValueError):
    """Raised when JSONL content cannot be read."""


def write_jsonl(path: str | Path, records: Iterable[Mapping[str, Any]]) -> Path:
    """Write records to UTF-8 JSONL, creating the parent directory if needed."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as output_file:
        for record in records:
            output_file.write(json.dumps(dict(record), ensure_ascii=False, sort_keys=True))
            output_file.write("\n")
    return output_path


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Read JSON objects from a UTF-8 JSONL file."""

    input_path = Path(path)
    records: list[dict[str, Any]] = []
    try:
        with input_path.open(encoding="utf-8") as input_file:
            for line_number, line in enumerate(input_file, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise JsonlError(
                        f"{input_path}:{line_number}: invalid JSON: {exc.msg}"
                    ) from exc
                if not isinstance(record, dict):
                    raise JsonlError(f"{input_path}:{line_number}: expected a JSON object")
                records.append(record)
    except FileNotFoundError as exc:
        raise JsonlError(f"JSONL file not found: {input_path}") from exc
    return records
