from pathlib import Path

from llm_posttraining_ops.data.jsonl import read_jsonl
from llm_posttraining_ops.data.preference_ingestion import (
    ingest_preference_data,
    load_normalized_preferences,
)


def test_ingest_direct_preference_jsonl(tmp_path: Path) -> None:
    output_path, records = ingest_preference_data(
        "tests/fixtures/preference_direct_sample.jsonl",
        tmp_path,
        "direct",
    )

    assert output_path == tmp_path / "preference.jsonl"
    assert len(records) == 4
    assert records[0].source == "preference_direct_sample"
    assert set(read_jsonl(output_path)[0]) == {
        "id",
        "split",
        "instruction",
        "input",
        "chosen",
        "rejected",
        "source",
        "metadata",
    }


def test_ingest_messages_preference_jsonl(tmp_path: Path) -> None:
    output_path, records = ingest_preference_data(
        "tests/fixtures/preference_messages_sample.jsonl",
        tmp_path,
        "messages",
    )

    assert len(records) == 2
    assert records[0].instruction == "Why does ice float on liquid water?"
    assert records[0].chosen.startswith("Ice floats because")
    assert records[0].metadata["prompt_message_count"] == 2
    assert load_normalized_preferences(output_path) == records
