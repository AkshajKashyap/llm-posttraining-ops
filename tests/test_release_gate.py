import json
from pathlib import Path

from llm_posttraining_ops.monitoring.release_gate import (
    GateMetrics,
    compare_gate_metrics,
    load_gate_metrics,
    run_release_gate,
)
from llm_posttraining_ops.monitoring.reports import write_release_gate_report

BASELINE_PATH = Path("tests/fixtures/baseline_eval_gate_sample.json")
CURRENT_PATH = Path("tests/fixtures/current_eval_gate_sample.json")


def test_release_gate_passes_for_improvement(tmp_path: Path) -> None:
    output_path = tmp_path / "gate.json"

    result = run_release_gate(
        BASELINE_PATH,
        CURRENT_PATH,
        max_p95_latency_seconds=1.0,
        output_path=output_path,
    )

    assert result.status == "pass"
    assert len(result.checks) == 4
    assert all(check.passed for check in result.checks)
    assert json.loads(output_path.read_text(encoding="utf-8"))["status"] == "pass"


def test_release_gate_fails_quality_and_latency_regressions() -> None:
    baseline = load_gate_metrics(BASELINE_PATH)
    current = GateMetrics(
        required_fact_coverage=0.7,
        forbidden_term_violation_rate=0.3,
        empty_response_rate=0.2,
        p95_latency_seconds=1.5,
    )

    status, checks = compare_gate_metrics(
        baseline,
        current,
        max_p95_latency_seconds=1.0,
    )

    assert status == "fail"
    assert [check.passed for check in checks] == [False, False, False, False]


def test_release_gate_skips_latency_when_unavailable() -> None:
    baseline = GateMetrics(0.8, 0.1, 0.0, None)
    current = GateMetrics(0.8, 0.1, 0.0, None)

    status, checks = compare_gate_metrics(
        baseline,
        current,
        max_p95_latency_seconds=1.0,
    )

    assert status == "pass"
    assert len(checks) == 3


def test_release_gate_report_generation(tmp_path: Path) -> None:
    result = run_release_gate(
        BASELINE_PATH,
        CURRENT_PATH,
        output_path=tmp_path / "gate.json",
    )
    report_path = write_release_gate_report(result, tmp_path / "gate.md")
    report = report_path.read_text(encoding="utf-8")

    assert report.startswith("# Model Release Gate Report\n")
    assert "Result: **PASS**" in report
    assert "required_fact_coverage" in report
    assert "p95_latency_seconds" in report
