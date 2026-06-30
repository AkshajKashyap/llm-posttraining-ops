import json
from pathlib import Path

from llm_posttraining_ops.data.schemas import SFTRecord
from llm_posttraining_ops.evaluation.evaluator import (
    EVALUATION_SCHEMA_VERSION,
    evaluate_records,
    write_evaluation_result,
)


def test_evaluation_output_schema_and_metrics(tmp_path: Path) -> None:
    records = [
        SFTRecord(
            id="sft-test-0000",
            split="test",
            instruction="Add the two integers.",
            input="First integer: 2. Second integer: 3.",
            output="5",
        ),
        SFTRecord(
            id="sft-test-0001",
            split="test",
            instruction="Multiply the two integers.",
            input="First integer: 3. Second integer: 4.",
            output="12",
        ),
    ]

    result = evaluate_records(records, dataset_path="demo/sft.jsonl")
    output_path = write_evaluation_result(result, tmp_path / "evaluation.json")
    payload = json.loads(output_path.read_text(encoding="utf-8"))

    assert payload["schema_version"] == EVALUATION_SCHEMA_VERSION
    assert payload["dataset"] == {
        "path": "demo/sft.jsonl",
        "record_count": 2,
        "split_counts": {"test": 2},
    }
    assert [baseline["name"] for baseline in payload["baselines"]] == [
        "echo",
        "template",
        "keyword_rule",
    ]
    keyword_result = payload["baselines"][2]
    assert keyword_result["metrics"]["exact_match"] == 1.0
    assert len(keyword_result["examples"]) == 2
    assert keyword_result["examples"][0]["generated_response"] == "5"
    assert set(keyword_result["examples"][0]["metrics"]) == {
        "contains_expected_key_terms",
        "empty_response",
        "exact_match",
        "response_length",
        "token_overlap_f1",
    }
