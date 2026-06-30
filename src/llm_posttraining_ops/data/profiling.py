"""Deterministic profiling and dataset-card generation for normalized SFT data."""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from llm_posttraining_ops.data.ingestion import load_normalized_sft
from llm_posttraining_ops.data.schemas import SFTRecord

PROFILE_SCHEMA_VERSION = "1.0"
DEFAULT_PROFILE_PATH = Path("artifacts/evals/dataset_profile.json")
DEFAULT_DATASET_CARD_PATH = Path("reports/dataset_card.md")
STARTING_PHRASE_TOKENS = 3
TOP_PHRASE_LIMIT = 5


def _tokens(value: str) -> list[str]:
    return re.findall(r"\w+", value.casefold())


@dataclass(frozen=True, slots=True)
class RepeatedPhrase:
    """One repeated output prefix and its frequency."""

    phrase: str
    count: int


@dataclass(frozen=True, slots=True)
class ProfileMetrics:
    """Aggregate quality and length statistics."""

    average_instruction_length: float
    average_output_length: float
    empty_input_rate: float
    duplicate_output_rate: float
    top_repeated_starting_phrases: list[RepeatedPhrase]


@dataclass(frozen=True, slots=True)
class DatasetProfile:
    """Versioned profile of a normalized SFT dataset."""

    schema_version: str
    dataset_path: str
    record_count: int
    split_counts: dict[str, int]
    source_counts: dict[str, int]
    metrics: ProfileMetrics

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _mean(values: list[int]) -> float:
    return round(sum(values) / len(values), 6)


def build_dataset_profile(
    records: list[SFTRecord],
    *,
    dataset_path: str,
) -> DatasetProfile:
    """Calculate a deterministic profile from normalized records."""

    if not records:
        raise ValueError("Cannot profile an empty dataset")

    normalized_outputs = [" ".join(record.output.casefold().split()) for record in records]
    repeated_phrases = Counter(
        " ".join(_tokens(record.output)[:STARTING_PHRASE_TOKENS])
        for record in records
        if _tokens(record.output)
    )
    top_phrases = [
        RepeatedPhrase(phrase=phrase, count=count)
        for phrase, count in sorted(
            repeated_phrases.items(),
            key=lambda item: (-item[1], item[0]),
        )
        if count > 1
    ][:TOP_PHRASE_LIMIT]

    count = len(records)
    return DatasetProfile(
        schema_version=PROFILE_SCHEMA_VERSION,
        dataset_path=dataset_path,
        record_count=count,
        split_counts=dict(sorted(Counter(record.split for record in records).items())),
        source_counts=dict(sorted(Counter(record.source for record in records).items())),
        metrics=ProfileMetrics(
            average_instruction_length=_mean(
                [len(_tokens(record.instruction)) for record in records]
            ),
            average_output_length=_mean([len(_tokens(record.output)) for record in records]),
            empty_input_rate=round(
                sum(not record.input.strip() for record in records) / count,
                6,
            ),
            duplicate_output_rate=round(
                (count - len(set(normalized_outputs))) / count,
                6,
            ),
            top_repeated_starting_phrases=top_phrases,
        ),
    )


def write_dataset_profile(profile: DatasetProfile, path: str | Path) -> Path:
    """Write a stable profile JSON artifact."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as output_file:
        json.dump(profile.to_dict(), output_file, indent=2, sort_keys=True)
        output_file.write("\n")
    return output_path


def render_dataset_card(profile: DatasetProfile) -> str:
    """Render a deterministic Markdown dataset card."""

    split_counts = ", ".join(
        f"{name}={count}" for name, count in profile.split_counts.items()
    )
    source_counts = ", ".join(
        f"{name}={count}" for name, count in profile.source_counts.items()
    )
    metrics = profile.metrics
    lines = [
        "# SFT Dataset Card",
        "",
        f"- Dataset: `{profile.dataset_path}`",
        f"- Records: {profile.record_count}",
        f"- Splits: {split_counts}",
        f"- Sources: {source_counts}",
        f"- Profile schema: {profile.schema_version}",
        "",
        "## Profile",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| Average instruction length (tokens) | {metrics.average_instruction_length:.3f} |",
        f"| Average output length (tokens) | {metrics.average_output_length:.3f} |",
        f"| Empty input rate | {metrics.empty_input_rate:.3f} |",
        f"| Duplicate output rate | {metrics.duplicate_output_rate:.3f} |",
        "",
        "## Repeated output starting phrases",
        "",
    ]
    if metrics.top_repeated_starting_phrases:
        lines.extend(
            f"- `{item.phrase}`: {item.count}"
            for item in metrics.top_repeated_starting_phrases
        )
    else:
        lines.append("- None repeated.")

    lines.extend(
        [
            "",
            "## Quality checks",
            "",
            "Records were checked for schema completeness, duplicate IDs, valid splits,",
            "minimum output length, repetitive outputs, and copied instructions.",
            "",
            "This card was generated deterministically from local normalized JSONL.",
            "",
        ]
    )
    return "\n".join(lines)


def write_dataset_card(profile: DatasetProfile, path: str | Path) -> Path:
    """Write a Markdown dataset card."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(render_dataset_card(profile), encoding="utf-8", newline="\n")
    return output_path


def profile_data_directory(
    data_dir: str | Path,
    *,
    profile_path: str | Path = DEFAULT_PROFILE_PATH,
    card_path: str | Path = DEFAULT_DATASET_CARD_PATH,
) -> tuple[DatasetProfile, Path, Path]:
    """Profile normalized SFT JSONL and save JSON plus Markdown outputs."""

    sft_path = Path(data_dir) / "sft.jsonl"
    profile = build_dataset_profile(
        load_normalized_sft(sft_path),
        dataset_path=str(sft_path),
    )
    return (
        profile,
        write_dataset_profile(profile, profile_path),
        write_dataset_card(profile, card_path),
    )
