from pathlib import Path

from llm_posttraining_ops.data.ingestion import ingest_sft_data, load_normalized_sft
from llm_posttraining_ops.data.jsonl import read_jsonl


def test_ingest_alpaca_jsonl(tmp_path: Path) -> None:
    output_path, records = ingest_sft_data(
        "tests/fixtures/alpaca_sample.jsonl",
        tmp_path,
        "alpaca",
    )

    assert output_path == tmp_path / "sft.jsonl"
    assert len(records) == 4
    assert records[0].source == "alpaca_sample"
    assert records[0].metadata == {"category": "geography"}
    assert set(read_jsonl(output_path)[0]) == {
        "id",
        "split",
        "instruction",
        "input",
        "output",
        "source",
        "metadata",
    }


def test_ingest_messages_jsonl(tmp_path: Path) -> None:
    output_path, records = ingest_sft_data(
        "tests/fixtures/messages_sample.jsonl",
        tmp_path,
        "messages",
    )

    assert len(records) == 2
    assert records[0].instruction == "Why do leaves often look green?"
    assert records[0].input == ""
    assert records[0].metadata["message_count"] == 3
    assert load_normalized_sft(output_path) == records
