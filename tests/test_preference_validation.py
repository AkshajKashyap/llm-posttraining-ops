import pytest

from llm_posttraining_ops.data.validation import DataValidationError, validate_records


def _valid_record() -> dict[str, object]:
    return {
        "id": "one",
        "split": "train",
        "instruction": "Explain the concept.",
        "input": "",
        "chosen": "This is a clear and accurate explanation.",
        "rejected": "This explanation contains an important factual error.",
        "source": "test",
        "metadata": {},
    }


def test_preference_validation_success() -> None:
    validate_records(
        [_valid_record()],
        "preference",
        minimum_output_length=10,
    )


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"chosen": ""}, "must be a non-empty string"),
        (
            {
                "chosen": "The same response is used here.",
                "rejected": "The same response is used here.",
            },
            "chosen and rejected responses are identical",
        ),
        ({"chosen": "short"}, "shorter than minimum length"),
        ({"chosen": "yes yes yes yes"}, "suspiciously repetitive"),
        ({"chosen": "Explain the concept!"}, "chosen copies the prompt"),
        ({"split": "dev"}, "unsupported split"),
    ],
)
def test_preference_validation_failures(
    change: dict[str, object],
    message: str,
) -> None:
    record = _valid_record()
    record.update(change)

    with pytest.raises(DataValidationError, match=message):
        validate_records(
            [record],
            "preference",
            minimum_output_length=10,
        )


def test_preference_validation_rejects_duplicate_ids() -> None:
    first = _valid_record()
    second = _valid_record()
    second["chosen"] = "A second preferred answer with enough detail."

    with pytest.raises(DataValidationError, match="duplicate id"):
        validate_records([first, second], "preference")
