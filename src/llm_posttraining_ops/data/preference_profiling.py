"""Deterministic profiling for normalized preference data."""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from llm_posttraining_ops.data.preference_ingestion import load_normalized_preferences
from llm_posttraining_ops.data.schemas import PreferenceRecord

PREFERENCE_PROFILE_SCHEMA_VERSION = "1.0"
DEFAULT_PREFERENCE_PROFILE_PATH = Path("artifacts/evals/preference_profile.json")


def _token_count(value: str) -> int:
    return len(re.findall(r"\w+", value.casefold()))


def _mean(values: list[int]) -> float:
    return round(sum(values) / len(values), 6)


@dataclass(frozen=True, slots=True)
class PreferenceProfile:
    """Versioned preference-dataset profile."""

    schema_version: str
    dataset_path: str
    record_count: int
    split_counts: dict[str, int]
    source_counts: dict[str, int]
    average_prompt_length: float
    average_chosen_length: float
    average_rejected_length: float
    chosen_rejected_length_ratio: float
    duplicate_chosen_rate: float
    duplicate_rejected_rate: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_preference_profile(
    records: list[PreferenceRecord],
    *,
    dataset_path: str,
) -> PreferenceProfile:
    """Calculate length and duplication statistics."""

    if not records:
        raise ValueError("Cannot profile an empty preference dataset")
    count = len(records)
    prompt_lengths = [
        _token_count(f"{record.instruction} {record.input}") for record in records
    ]
    chosen_lengths = [_token_count(record.chosen) for record in records]
    rejected_lengths = [_token_count(record.rejected) for record in records]
    average_chosen = _mean(chosen_lengths)
    average_rejected = _mean(rejected_lengths)
    normalized_chosen = [" ".join(record.chosen.casefold().split()) for record in records]
    normalized_rejected = [
        " ".join(record.rejected.casefold().split()) for record in records
    ]
    return PreferenceProfile(
        schema_version=PREFERENCE_PROFILE_SCHEMA_VERSION,
        dataset_path=dataset_path,
        record_count=count,
        split_counts=dict(sorted(Counter(record.split for record in records).items())),
        source_counts=dict(sorted(Counter(record.source for record in records).items())),
        average_prompt_length=_mean(prompt_lengths),
        average_chosen_length=average_chosen,
        average_rejected_length=average_rejected,
        chosen_rejected_length_ratio=round(average_chosen / average_rejected, 6),
        duplicate_chosen_rate=round(
            (count - len(set(normalized_chosen))) / count,
            6,
        ),
        duplicate_rejected_rate=round(
            (count - len(set(normalized_rejected))) / count,
            6,
        ),
    )


def write_preference_profile(profile: PreferenceProfile, path: str | Path) -> Path:
    """Write a stable preference profile JSON."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as output_file:
        json.dump(profile.to_dict(), output_file, indent=2, sort_keys=True)
        output_file.write("\n")
    return output_path


def profile_preference_directory(
    data_dir: str | Path,
    *,
    output_path: str | Path = DEFAULT_PREFERENCE_PROFILE_PATH,
) -> tuple[PreferenceProfile, Path]:
    """Profile normalized preference JSONL in a directory."""

    preference_path = Path(data_dir) / "preference.jsonl"
    profile = build_preference_profile(
        load_normalized_preferences(preference_path),
        dataset_path=str(preference_path),
    )
    return profile, write_preference_profile(profile, output_path)
