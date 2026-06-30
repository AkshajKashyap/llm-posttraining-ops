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
