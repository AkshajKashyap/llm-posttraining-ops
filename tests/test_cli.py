from pathlib import Path
from types import SimpleNamespace
from typing import Any

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


def test_model_evaluation_cli_uses_mock_without_download(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    data_dir = tmp_path / "custom"
    data_dir.mkdir()
    captured: dict[str, Any] = {}

    def fake_run_model_evaluation(
        passed_data_dir: Path,
        settings: Any,
        **kwargs: Any,
    ) -> Any:
        captured["data_dir"] = passed_data_dir
        captured["settings"] = settings
        captured["kwargs"] = kwargs
        return SimpleNamespace(
            dataset=SimpleNamespace(record_count=2),
            model=SimpleNamespace(name=settings.model_name),
            generations_path="mock-generations.jsonl",
            latency=SimpleNamespace(
                total_generation_seconds=0.4,
                average_seconds_per_example=0.2,
            ),
        )

    monkeypatch.setattr(
        "llm_posttraining_ops.cli.run_model_evaluation",
        fake_run_model_evaluation,
    )
    result = runner.invoke(
        app,
        [
            "run-model-eval",
            "--data-dir",
            str(data_dir),
            "--model-name",
            "mock/model",
            "--max-new-tokens",
            "7",
            "--seed",
            "5",
            "--output",
            str(tmp_path / "model_eval.json"),
        ],
    )

    assert result.exit_code == 0
    assert "Evaluated 2 records with model mock/model" in result.output
    assert "0.200s/example" in result.output
    assert captured["data_dir"] == data_dir
    assert captured["settings"].max_new_tokens == 7
    assert captured["settings"].seed == 5


def test_sft_training_and_evaluation_cli_use_mocks(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    data_dir = tmp_path / "custom"
    model_path = tmp_path / "model"
    data_dir.mkdir()
    model_path.mkdir()
    captured: dict[str, Any] = {}

    def fake_training(data_dir: Path, config: Any, **kwargs: Any) -> Any:
        captured["training_config"] = config
        return SimpleNamespace(
            training_record_count=2,
            settings=config.to_dict(),
            checkpoint_path=str(config.output_dir),
            use_lora=config.use_lora,
            metrics={"training_loss": 1.25},
        )

    def fake_evaluation(data_dir: Path, passed_model_path: Path, **kwargs: Any) -> Any:
        captured["model_path"] = passed_model_path
        return SimpleNamespace(
            dataset=SimpleNamespace(record_count=4),
            generations_path="sft-generations.jsonl",
            latency=SimpleNamespace(
                total_generation_seconds=0.8,
                average_seconds_per_example=0.2,
            ),
        )

    monkeypatch.setattr("llm_posttraining_ops.cli.run_sft_training", fake_training)
    monkeypatch.setattr("llm_posttraining_ops.cli.run_sft_evaluation", fake_evaluation)
    monkeypatch.setattr(
        "llm_posttraining_ops.cli.generate_sft_report",
        lambda **kwargs: tmp_path / "sft_report.md",
    )

    train_result = runner.invoke(
        app,
        [
            "train-sft",
            "--data-dir",
            str(data_dir),
            "--model-name",
            "mock/model",
            "--max-steps",
            "1",
            "--use-lora",
        ],
    )
    assert train_result.exit_code == 0
    assert "Trained 2 records for 1 step(s)" in train_result.output
    assert "Saved adapter" in train_result.output
    assert captured["training_config"].use_lora is True

    evaluation_result = runner.invoke(
        app,
        [
            "evaluate-sft",
            "--data-dir",
            str(data_dir),
            "--model-path",
            str(model_path),
        ],
    )
    assert evaluation_result.exit_code == 0
    assert "Evaluated 4 records with SFT model" in evaluation_result.output
    assert "0.200s/example" in evaluation_result.output
    assert captured["model_path"] == model_path


def test_preference_ingestion_and_profiling_cli(tmp_path: Path) -> None:
    data_dir = tmp_path / "preferences"
    profile_path = tmp_path / "preference_profile.json"

    ingestion_result = runner.invoke(
        app,
        [
            "ingest-preference-data",
            "--input-path",
            "tests/fixtures/preference_direct_sample.jsonl",
            "--output-dir",
            str(data_dir),
            "--format",
            "direct",
        ],
    )
    assert ingestion_result.exit_code == 0
    assert "Ingested 4 preference records" in ingestion_result.output

    profile_result = runner.invoke(
        app,
        [
            "profile-preference-data",
            "--data-dir",
            str(data_dir),
            "--output",
            str(profile_path),
        ],
    )
    assert profile_result.exit_code == 0
    assert "Profiled 4 preference records" in profile_result.output
    assert profile_path.is_file()


def test_dpo_training_and_evaluation_cli_use_mocks(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    preference_dir = tmp_path / "preferences"
    data_dir = tmp_path / "custom"
    model_path = tmp_path / "dpo-model"
    preference_dir.mkdir()
    data_dir.mkdir()
    model_path.mkdir()
    captured: dict[str, Any] = {}

    def fake_training(data_dir: Path, config: Any, **kwargs: Any) -> Any:
        captured["dpo_config"] = config
        return SimpleNamespace(
            training_record_count=2,
            settings=config.to_dict(),
            checkpoint_path=str(config.output_dir),
            use_lora=config.use_lora,
            metrics={"training_loss": 0.69},
        )

    def fake_evaluation(data_dir: Path, passed_model_path: Path, **kwargs: Any) -> Any:
        captured["dpo_model_path"] = passed_model_path
        return SimpleNamespace(
            dataset=SimpleNamespace(record_count=4),
            generations_path="dpo-generations.jsonl",
            latency=SimpleNamespace(
                total_generation_seconds=0.4,
                average_seconds_per_example=0.1,
            ),
        )

    monkeypatch.setattr("llm_posttraining_ops.cli.run_dpo_training", fake_training)
    monkeypatch.setattr("llm_posttraining_ops.cli.run_dpo_evaluation", fake_evaluation)
    monkeypatch.setattr(
        "llm_posttraining_ops.cli.generate_dpo_report",
        lambda **kwargs: tmp_path / "dpo_report.md",
    )

    train_result = runner.invoke(
        app,
        [
            "train-dpo",
            "--preference-data-dir",
            str(preference_dir),
            "--model-name",
            "mock/model",
            "--max-steps",
            "1",
            "--use-lora",
        ],
    )
    assert train_result.exit_code == 0
    assert "Trained 2 preference records for 1 step(s)" in train_result.output
    assert "Saved adapter" in train_result.output
    assert captured["dpo_config"].use_lora is True

    evaluation_result = runner.invoke(
        app,
        [
            "evaluate-dpo",
            "--data-dir",
            str(data_dir),
            "--model-path",
            str(model_path),
        ],
    )
    assert evaluation_result.exit_code == 0
    assert "Evaluated 4 records with DPO model" in evaluation_result.output
    assert "0.100s/example" in evaluation_result.output
    assert captured["dpo_model_path"] == model_path


def test_eval_suite_and_pairwise_cli(tmp_path: Path) -> None:
    suite_output = tmp_path / "suite.json"
    suite_report = tmp_path / "suite.md"
    suite_result = runner.invoke(
        app,
        [
            "run-eval-suite",
            "--generations-path",
            "tests/fixtures/generations_good.jsonl",
            "--eval-data-path",
            "tests/fixtures/eval_suite_sample.jsonl",
            "--output",
            str(suite_output),
            "--report-output",
            str(suite_report),
        ],
    )
    assert suite_result.exit_code == 0
    assert "Evaluated 4 generation records" in suite_result.output
    assert suite_output.is_file()
    assert suite_report.is_file()

    pairwise_output = tmp_path / "pairwise.json"
    pairwise_report = tmp_path / "pairwise.md"
    pairwise_result = runner.invoke(
        app,
        [
            "compare-generations",
            "--left-path",
            "tests/fixtures/generations_bad.jsonl",
            "--right-path",
            "tests/fixtures/generations_good.jsonl",
            "--eval-data-path",
            "tests/fixtures/eval_suite_sample.jsonl",
            "--output",
            str(pairwise_output),
            "--report-output",
            str(pairwise_report),
        ],
    )
    assert pairwise_result.exit_code == 0
    assert "left=0, right=3, ties=1" in pairwise_result.output
    assert pairwise_output.is_file()
    assert pairwise_report.is_file()


def test_serve_cli_smoke_uses_mock_app(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    captured: dict[str, Any] = {}

    def fake_run(api: Any, *, host: str, port: int) -> None:
        captured["api"] = api
        captured["host"] = host
        captured["port"] = port

    monkeypatch.setattr("uvicorn.run", fake_run)
    result = runner.invoke(
        app,
        [
            "serve",
            "--mock",
            "--host",
            "0.0.0.0",
            "--port",
            "8123",
            "--log-path",
            str(tmp_path / "inference.jsonl"),
        ],
    )

    assert result.exit_code == 0
    assert captured["host"] == "0.0.0.0"
    assert captured["port"] == 8123
    assert captured["api"].state.model_manager.mock is True
    assert captured["api"].state.model_manager.loaded is False


def test_monitoring_and_release_gate_cli(tmp_path: Path) -> None:
    monitoring_output = tmp_path / "monitoring.json"
    monitoring_report = tmp_path / "monitoring.md"
    monitoring_result = runner.invoke(
        app,
        [
            "monitor-logs",
            "--logs-path",
            "tests/fixtures/inference_logs_sample.jsonl",
            "--output",
            str(monitoring_output),
            "--report-output",
            str(monitoring_report),
        ],
    )

    assert monitoring_result.exit_code == 0
    assert "Monitoring status: pass (5 requests" in monitoring_result.output
    assert monitoring_output.is_file()
    assert monitoring_report.is_file()

    gate_output = tmp_path / "gate.json"
    gate_report = tmp_path / "gate.md"
    gate_result = runner.invoke(
        app,
        [
            "run-release-gate",
            "--baseline-eval",
            "tests/fixtures/baseline_eval_gate_sample.json",
            "--current-eval",
            "tests/fixtures/current_eval_gate_sample.json",
            "--output",
            str(gate_output),
            "--report-output",
            str(gate_report),
        ],
    )

    assert gate_result.exit_code == 0
    assert "Release gate: pass (4/4 checks passed)" in gate_result.output
    assert gate_output.is_file()
    assert gate_report.is_file()


def test_demo_workflow_cli_without_models(tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        [
            "run-demo-workflow",
            "--run-id",
            "cli-smoke",
            "--output-dir",
            str(tmp_path / "runs"),
            "--skip-model",
            "--skip-sft",
            "--skip-dpo",
            "--report-output",
            str(tmp_path / "workflow_report.md"),
        ],
    )

    assert result.exit_code == 0
    assert "Workflow cli-smoke: pass (8 passed, 0 failed, 3 skipped)" in result.output
    assert (tmp_path / "runs/cli-smoke/workflow_summary.json").is_file()
    assert (tmp_path / "workflow_report.md").is_file()
