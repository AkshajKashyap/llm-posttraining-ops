import json
from pathlib import Path

from llm_posttraining_ops.serving.logging import InferenceLogger


def test_inference_logger_appends_structured_jsonl(tmp_path: Path) -> None:
    path = tmp_path / "logs" / "inference.jsonl"
    logger = InferenceLogger(path, timestamp_factory=lambda: "fixed-time")

    logger.write({"request_id": "one", "status": "ok"})
    logger.write({"request_id": "two", "status": "error"})
    records = [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
    ]

    assert records == [
        {"request_id": "one", "status": "ok", "timestamp": "fixed-time"},
        {"request_id": "two", "status": "error", "timestamp": "fixed-time"},
    ]
