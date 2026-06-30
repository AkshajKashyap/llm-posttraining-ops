from llm_posttraining_ops.data.preference_normalization import (
    normalize_direct_preference,
    normalize_messages_preference,
)


def test_normalize_direct_preference() -> None:
    record = normalize_direct_preference(
        {
            "instruction": "Answer accurately.",
            "input": "A question",
            "chosen": "This is the accurate answer.",
            "rejected": "This answer is inaccurate.",
        },
        source="local preferences",
        index=3,
    )

    assert record.id == "local-preferences-direct-000003"
    assert record.split == "train"
    assert record.source == "local preferences"


def test_normalize_messages_preference() -> None:
    record = normalize_messages_preference(
        {
            "prompt": [
                {"role": "system", "content": "Be factual."},
                {"role": "user", "content": "Why is the sky blue?"},
            ],
            "chosen": {
                "role": "assistant",
                "content": "The atmosphere scatters blue wavelengths efficiently.",
            },
            "rejected": {
                "role": "assistant",
                "content": "The ocean paints the sky blue each morning.",
            },
        },
        source="chat",
        index=0,
    )

    assert record.instruction == "Why is the sky blue?"
    assert record.input == ""
    assert record.metadata == {
        "prompt_message_count": 2,
        "system_prompt": "Be factual.",
    }
