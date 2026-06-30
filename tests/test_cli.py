from pathlib import Path

from typer.testing import CliRunner

from llm_posttraining_ops.cli import app
from llm_posttraining_ops.data.jsonl import write_jsonl

runner = CliRunner()


def test_baseline_evaluation_and_report_cli(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    write_jsonl(
        data_dir / "sft.jsonl",
        [
            {
                "id": "sft-test-0000",
                "split": "test",
                "instruction": "Add the two integers.",
                "input": "First integer: 2. Second integer: 5.",
                "output": "7",
                "source": "test",
                "metadata": {},
            }
        ],
    )
    evaluation_path = tmp_path / "evaluation.json"
    report_path = tmp_path / "report.md"

    evaluation_result = runner.invoke(
        app,
        [
            "run-baseline-eval",
            "--data-dir",
            str(data_dir),
            "--output",
            str(evaluation_path),
        ],
    )
    assert evaluation_result.exit_code == 0
    assert "Evaluated 1 records with 3 baselines" in evaluation_result.output
    assert evaluation_path.is_file()

    report_result = runner.invoke(
        app,
        [
            "generate-baseline-report",
            "--evaluation-path",
            str(evaluation_path),
            "--output",
            str(report_path),
        ],
    )
    assert report_result.exit_code == 0
    assert "Wrote baseline report" in report_result.output
    assert report_path.is_file()


def test_ingestion_and_profiling_cli(tmp_path: Path) -> None:
    data_dir = tmp_path / "custom"
    profile_path = tmp_path / "profile.json"
    card_path = tmp_path / "dataset_card.md"

    ingestion_result = runner.invoke(
        app,
        [
            "ingest-sft-data",
            "--input-path",
            "tests/fixtures/alpaca_sample.jsonl",
            "--output-dir",
            str(data_dir),
            "--format",
            "alpaca",
        ],
    )
    assert ingestion_result.exit_code == 0
    assert "Ingested 4 SFT records" in ingestion_result.output

    validation_result = runner.invoke(
        app,
        ["validate-data", "--data-dir", str(data_dir)],
    )
    assert validation_result.exit_code == 0
    assert "Validation passed: 4 SFT records" in validation_result.output

    profile_result = runner.invoke(
        app,
        [
            "profile-data",
            "--data-dir",
            str(data_dir),
            "--output",
            str(profile_path),
            "--card-output",
            str(card_path),
        ],
    )
    assert profile_result.exit_code == 0
    assert "Profiled 4 SFT records" in profile_result.output
    assert profile_path.is_file()
    assert card_path.is_file()

    evaluation_path = tmp_path / "custom_evaluation.json"
    evaluation_result = runner.invoke(
        app,
        [
            "run-baseline-eval",
            "--data-dir",
            str(data_dir),
            "--output",
            str(evaluation_path),
        ],
    )
    assert evaluation_result.exit_code == 0
    assert "Evaluated 4 records with 3 baselines" in evaluation_result.output
