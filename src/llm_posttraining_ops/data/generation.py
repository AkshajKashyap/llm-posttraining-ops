"""Deterministic toy dataset generation."""

from __future__ import annotations

import random
from collections.abc import Mapping
from pathlib import Path
from typing import cast

from llm_posttraining_ops.config import AppConfig
from llm_posttraining_ops.data.jsonl import write_jsonl
from llm_posttraining_ops.data.schemas import PreferenceRecord, SFTRecord, SplitName
from llm_posttraining_ops.data.validation import validate_records


def generate_sft_records(
    split_sizes: Mapping[str, int], seed: int
) -> list[SFTRecord]:
    """Create deterministic arithmetic SFT examples."""

    rng = random.Random(seed)
    records: list[SFTRecord] = []
    for split, size in split_sizes.items():
        split_name = cast(SplitName, split)
        for index in range(size):
            left = rng.randint(2, 50)
            right = rng.randint(2, 30)
            operation = rng.choice(("add", "multiply"))
            if operation == "add":
                instruction = "Add the two integers and return only the result."
                answer = left + right
            else:
                instruction = "Multiply the two integers and return only the result."
                answer = left * right
            records.append(
                SFTRecord(
                    id=f"sft-{split}-{index:04d}",
                    split=split_name,
                    instruction=instruction,
                    input=f"First integer: {left}. Second integer: {right}.",
                    output=str(answer),
                    source="deterministic-demo",
                    metadata={
                        "generator": "arithmetic-v1",
                        "operation": operation,
                        "seed": seed,
                    },
                )
            )
    return records


def generate_preference_records(
    split_sizes: Mapping[str, int], seed: int
) -> list[PreferenceRecord]:
    """Create deterministic preference pairs with one correct response."""

    rng = random.Random(seed)
    records: list[PreferenceRecord] = []
    for split, size in split_sizes.items():
        split_name = cast(SplitName, split)
        for index in range(size):
            value = rng.randint(3, 40)
            increment = rng.randint(2, 20)
            answer = value + increment
            wrong_answer = answer + rng.choice((-3, -2, -1, 1, 2, 3))
            records.append(
                PreferenceRecord(
                    id=f"preference-{split}-{index:04d}",
                    split=split_name,
                    instruction="Answer the arithmetic question accurately and concisely.",
                    input=f"What is {value} plus {increment}?",
                    chosen=f"The answer is {answer}.",
                    rejected=f"The answer is {wrong_answer}.",
                    source="deterministic-demo",
                    metadata={"generator": "arithmetic-preference-v1", "seed": seed},
                )
            )
    return records


def prepare_demo_data(config: AppConfig) -> dict[str, Path]:
    """Generate, validate, and write both demo datasets."""

    sft_records = generate_sft_records(config.data.sft_split_sizes, config.seed)
    preference_records = generate_preference_records(
        config.data.preference_split_sizes, config.seed + 1
    )
    sft_dicts = [record.to_dict() for record in sft_records]
    preference_dicts = [record.to_dict() for record in preference_records]
    validate_records(sft_dicts, "sft")
    validate_records(preference_dicts, "preference")

    return {
        "sft": write_jsonl(config.data.output_dir / "sft.jsonl", sft_dicts),
        "preference": write_jsonl(
            config.data.output_dir / "preference.jsonl", preference_dicts
        ),
    }
