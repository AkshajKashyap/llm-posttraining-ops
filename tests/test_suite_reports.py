from pathlib import Path

from llm_posttraining_ops.evaluation.pairwise import compare_generation_files
from llm_posttraining_ops.evaluation.suite import evaluate_generation_file
from llm_posttraining_ops.evaluation.suite_reports import (
    write_eval_suite_report,
    write_pairwise_report,
)


def test_suite_and_pairwise_report_generation(tmp_path: Path) -> None:
    suite_result = evaluate_generation_file(
        "tests/fixtures/generations_good.jsonl",
        "tests/fixtures/eval_suite_sample.jsonl",
    )
    pairwise_result = compare_generation_files(
        "tests/fixtures/generations_bad.jsonl",
        "tests/fixtures/generations_good.jsonl",
        "tests/fixtures/eval_suite_sample.jsonl",
    )

    suite_path = write_eval_suite_report(suite_result, tmp_path / "suite.md")
    pairwise_path = write_pairwise_report(pairwise_result, tmp_path / "pairwise.md")
    suite_report = suite_path.read_text(encoding="utf-8")
    pairwise_report = pairwise_path.read_text(encoding="utf-8")

    assert suite_report.startswith("# Rigorous Evaluation Suite Report\n")
    assert "Required fact coverage | 1.000" in suite_report
    assert pairwise_report.startswith("# Pairwise Generation Comparison\n")
    assert "| Right wins | 3 |" in pairwise_report
