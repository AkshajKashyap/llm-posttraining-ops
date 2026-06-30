import json
from pathlib import Path

from llm_posttraining_ops.workflows.runner import (
    STAGE_NAMES,
    DemoWorkflowConfig,
    run_demo_workflow,
)


def test_demo_workflow_skip_flags_and_report(tmp_path: Path) -> None:
    result = run_demo_workflow(
        DemoWorkflowConfig(
            run_id="unit-smoke",
            output_dir=tmp_path / "runs",
            skip_model=True,
            skip_sft=True,
            skip_dpo=True,
            report_path=tmp_path / "workflow_report.md",
        ),
        timestamp_factory=lambda: "2026-06-30T00:00:00+00:00",
    )

    assert result.status == "pass"
    assert [stage.name for stage in result.stages] == list(STAGE_NAMES)
    statuses = {stage.name: stage.status for stage in result.stages}
    assert statuses["base_model_evaluation"] == "skipped"
    assert statuses["sft_training"] == "skipped"
    assert statuses["dpo_training"] == "skipped"
    assert statuses["evaluation_suite"] == "pass"
    assert statuses["release_gate"] == "pass"

    run_dir = Path(result.run_dir)
    assert (run_dir / "experiment_registry.json").is_file()
    assert (run_dir / "reproducibility_manifest.json").is_file()
    assert (run_dir / "workflow_summary.json").is_file()
    assert (run_dir / "workflow_report.md").is_file()
    assert (run_dir / "data/sft/sft.jsonl").is_file()
    assert (run_dir / "data/preferences/preference.jsonl").is_file()

    summary = json.loads(Path(result.summary_path).read_text(encoding="utf-8"))
    report = Path(result.report_path).read_text(encoding="utf-8")
    assert summary["status"] == "pass"
    assert summary["run_id"] == "unit-smoke"
    assert report.startswith("# End-to-End Workflow Report\n")
    assert "Status: **PASS**" in report
    assert "| base_model_evaluation | skipped |" in report


def test_workflow_records_failure_and_still_writes_final_artifacts(
    tmp_path: Path,
) -> None:
    result = run_demo_workflow(
        DemoWorkflowConfig(
            run_id="failure-test",
            output_dir=tmp_path / "runs",
            skip_model=True,
            skip_sft=True,
            skip_dpo=True,
            continue_on_error=False,
            sft_input_path=tmp_path / "missing.jsonl",
            report_path=tmp_path / "failure_report.md",
        ),
        timestamp_factory=lambda: "2026-06-30T00:00:00+00:00",
    )

    assert result.status == "fail"
    assert result.stages[0].status == "fail"
    assert result.stages[0].error is not None
    assert all(stage.status == "skipped" for stage in result.stages[1:])
    assert Path(result.registry_path).is_file()
    assert Path(result.manifest_path).is_file()
    assert Path(result.summary_path).is_file()
    assert Path(result.report_path).is_file()
