from pathlib import Path

from llm_posttraining_ops.data.jsonl import read_jsonl, write_jsonl


def test_jsonl_round_trip(tmp_path: Path) -> None:
    records = [
        {"id": "one", "text": "hello"},
        {"id": "two", "text": "unicode: café"},
    ]
    path = tmp_path / "nested" / "records.jsonl"

    returned_path = write_jsonl(path, records)

    assert returned_path == path
    assert read_jsonl(path) == records
    assert path.read_bytes().endswith(b"\n")
