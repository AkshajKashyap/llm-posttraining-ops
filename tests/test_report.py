import json
from pathlib import Path

from llm_posttraining_ops.data.schemas import SFTRecord
from llm_posttraining_ops.evaluation.evaluator import (
    evaluate_records,
    write_evaluation_result,
)
from llm_posttraining_ops.evaluation.report import generate_baseline_report


def test_generate_markdown_report(tmp_path: Path) -> None:
    records = [
        SFTRecord(
            id="one",
            split="test",
            instruction="Add the two integers.",
            input="First integer: 8. Second integer: 4.",
            output="12",
        )
    ]
    result = evaluate_records(records, dataset_path="demo/sft.jsonl")
    evaluation_path = write_evaluation_result(result, tmp_path / "evaluation.json")

    report_path = generate_baseline_report(
        evaluation_path,
        tmp_path / "reports" / "baseline.md",
    )
    report = report_path.read_text(encoding="utf-8")

    assert report.startswith("# Baseline Evaluation Report\n")
    assert "| keyword_rule | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 |" in report
    assert "All baselines are deterministic" in report


def test_generate_report_includes_hugging_face_model(tmp_path: Path) -> None:
    records = [
        SFTRecord(
            id="one",
            split="test",
            instruction="Give a greeting.",
            input="",
            output="Hello",
        )
    ]
    baseline_path = write_evaluation_result(
        evaluate_records(records, dataset_path="custom/sft.jsonl"),
        tmp_path / "baseline.json",
    )
    model_path = tmp_path / "model.json"
    model_path.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "model": {
                    "name": "mock/model",
                    "device": "cpu",
                    "generation": {},
                },
                "metrics": {
                    "exact_match": 0.5,
                    "token_overlap_f1": 0.6,
                    "contains_expected_key_terms": 0.75,
                    "average_response_length": 4.0,
                    "empty_response_rate": 0.0,
                },
                "latency": {
                    "total_generation_seconds": 1.25,
                    "average_seconds_per_example": 0.25,
                    "average_generated_tokens": 3.0,
                },
            }
        ),
        encoding="utf-8",
    )

    report_path = generate_baseline_report(
        baseline_path,
        tmp_path / "combined.md",
        model_evaluation_path=model_path,
    )
    report = report_path.read_text(encoding="utf-8")

    assert "| hf:mock/model | 0.500 | 0.600 | 0.750 | 4.000 | 0.000 |" in report
    assert "## Model latency" in report
    assert "Total generation time: 1.250 seconds" in report
