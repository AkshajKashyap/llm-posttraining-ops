from llm_posttraining_ops.data.normalization import (
    normalize_alpaca_record,
    normalize_messages_record,
)


def test_normalize_alpaca_record_with_generated_fields() -> None:
    record = normalize_alpaca_record(
        {
            "instruction": "Summarize the input.",
            "input": "A short passage.",
            "output": "This is a short summary.",
        },
        source="local sample",
        index=2,
    )

    assert record.id == "local-sample-alpaca-000002"
    assert record.split == "train"
    assert record.source == "local sample"
    assert record.metadata == {}


def test_normalize_messages_record_extracts_turns_and_system_prompt() -> None:
    record = normalize_messages_record(
        {
            "messages": [
                {"role": "system", "content": "Be concise."},
                {"role": "user", "content": "What is a mutex?"},
                {
                    "role": "assistant",
                    "content": "A mutex coordinates exclusive access to a shared resource.",
                },
            ]
        },
        source="chat",
        index=0,
    )

    assert record.instruction == "What is a mutex?"
    assert record.input == ""
    assert record.output.startswith("A mutex")
    assert record.metadata == {"message_count": 3, "system_prompt": "Be concise."}
