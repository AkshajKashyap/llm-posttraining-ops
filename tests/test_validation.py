from pathlib import Path

import pytest

from llm_posttraining_ops.config import AppConfig, DataConfig
from llm_posttraining_ops.data.generation import prepare_demo_data
from llm_posttraining_ops.data.validation import (
    DataValidationError,
    validate_data_directory,
    validate_records,
)


def test_validation_succeeds_for_prepared_data(tmp_path: Path) -> None:
    split_sizes = {"train": 2, "validation": 1, "test": 1}
    config = AppConfig(
        seed=12,
        data=DataConfig(
            output_dir=tmp_path,
            sft_split_sizes=split_sizes,
            preference_split_sizes=split_sizes,
        ),
    )
    prepare_demo_data(config)

    assert validate_data_directory(tmp_path) == {"sft": 4, "preference": 4}


def test_normalized_sft_validation_allows_empty_input() -> None:
    validate_records(
        [
            {
                "id": "one",
                "split": "train",
                "instruction": "Explain the term briefly.",
                "input": "",
                "output": "A concise and useful explanation.",
                "source": "test",
                "metadata": {},
            }
        ],
        "sft",
        minimum_output_length=10,
    )


@pytest.mark.parametrize(
    ("records", "message"),
    [
        (
            [{"id": "1", "split": "train", "instruction": "Do it", "input": "x"}],
            "missing required field 'output'",
        ),
        (
            [
                {
                    "id": "1",
                    "split": "train",
                    "instruction": " ",
                    "input": "x",
                    "output": "y",
                }
            ],
            "must be a non-empty string",
        ),
        (
            [
                {
                    "id": "same",
                    "split": "train",
                    "instruction": "Do it",
                    "input": "x",
                    "output": "y",
                },
                {
                    "id": "same",
                    "split": "test",
                    "instruction": "Do it",
                    "input": "z",
                    "output": "w",
                },
            ],
            "duplicate id 'same'",
        ),
        (
            [
                {
                    "id": "1",
                    "split": "dev",
                    "instruction": "Do it",
                    "input": "x",
                    "output": "y",
                }
            ],
            "unsupported split 'dev'",
        ),
    ],
)
def test_validation_reports_invalid_records(
    records: list[dict[str, str]], message: str
) -> None:
    with pytest.raises(DataValidationError, match=message):
        validate_records(records, "sft")


@pytest.mark.parametrize(
    ("instruction", "output", "minimum_length", "message"),
    [
        ("Give an answer.", "short", 10, "shorter than minimum length"),
        ("Give an answer.", "yes yes yes yes", 1, "suspiciously repetitive"),
        ("Repeat this instruction.", "Repeat this instruction!", 1, "copies the instruction"),
    ],
)
def test_stronger_output_quality_failures(
    instruction: str,
    output: str,
    minimum_length: int,
    message: str,
) -> None:
    record = {
        "id": "one",
        "split": "train",
        "instruction": instruction,
        "input": "",
        "output": output,
        "source": "test",
        "metadata": {},
    }

    with pytest.raises(DataValidationError, match=message):
        validate_records(
            [record],
            "sft",
            minimum_output_length=minimum_length,
        )
