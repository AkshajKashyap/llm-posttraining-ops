from llm_posttraining_ops.data.generation import (
    generate_preference_records,
    generate_sft_records,
)

SPLIT_SIZES = {"train": 3, "validation": 1, "test": 1}


def test_sft_generation_is_deterministic() -> None:
    first = generate_sft_records(SPLIT_SIZES, seed=7)
    second = generate_sft_records(SPLIT_SIZES, seed=7)
    different = generate_sft_records(SPLIT_SIZES, seed=8)

    assert first == second
    assert first != different
    assert len(first) == 5


def test_preference_generation_has_distinct_responses() -> None:
    records = generate_preference_records(SPLIT_SIZES, seed=7)

    assert len(records) == 5
    assert all(record.chosen != record.rejected for record in records)
