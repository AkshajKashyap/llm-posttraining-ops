import json
from pathlib import Path

import pytest

from llm_posttraining_ops.monitoring.logs import MonitoringError, load_inference_logs
from llm_posttraining_ops.monitoring.metrics import (
    MonitoringThresholds,
    calculate_monitoring_metrics,
    evaluate_thresholds,
    monitor_inference_logs,
    percentile,
)
from llm_posttraining_ops.monitoring.reports import write_monitoring_report

LOGS_PATH = Path("tests/fixtures/inference_logs_sample.jsonl")


def test_load_logs_and_calculate_monitoring_metrics() -> None:
    records = load_inference_logs(LOGS_PATH)
    metrics = calculate_monitoring_metrics(records)

    assert len(records) == 5
    assert records[0].request_id == "req-001"
    assert metrics.request_count == 5
    assert metrics.error_rate == 0.0
    assert metrics.average_latency_seconds == 0.3
    assert metrics.p50_latency_seconds == 0.3
    assert metrics.p95_latency_seconds == 0.48
    assert metrics.average_response_length_tokens == 8.0
    assert metrics.empty_response_rate == 0.0
    assert metrics.mock_request_count == 3
    assert metrics.real_request_count == 2
    assert metrics.requests_by_endpoint == {
        "/batch-generate": 1,
        "/evaluate-generation": 1,
        "/generate": 3,
    }
    assert metrics.requests_by_model == {"local/sft": 2, "mock/model": 3}


def test_log_parser_rejects_invalid_json(tmp_path: Path) -> None:
    path = tmp_path / "broken.jsonl"
    path.write_text('{"request_id":\n', encoding="utf-8")

    with pytest.raises(MonitoringError, match="invalid JSON"):
        load_inference_logs(path)


def test_percentile_uses_linear_interpolation() -> None:
    values = [0.1, 0.2, 0.3, 0.4, 0.5]

    assert percentile(values, 0.5) == 0.3
    assert percentile(values, 0.95) == 0.48


def test_threshold_status_pass_warn_and_fail() -> None:
    metrics = calculate_monitoring_metrics(load_inference_logs(LOGS_PATH))

    passed, pass_checks = evaluate_thresholds(metrics, MonitoringThresholds())
    warned, warn_checks = evaluate_thresholds(
        metrics,
        MonitoringThresholds(max_p95_latency=0.5),
    )
    failed, fail_checks = evaluate_thresholds(
        metrics,
        MonitoringThresholds(max_p95_latency=0.4),
    )

    assert passed == "pass"
    assert all(check.status == "pass" for check in pass_checks)
    assert warned == "warn"
    assert next(
        check for check in warn_checks if check.metric == "p95_latency_seconds"
    ).status == "warn"
    assert failed == "fail"
    assert next(
        check for check in fail_checks if check.metric == "p95_latency_seconds"
    ).status == "fail"


def test_monitoring_outputs_json_and_markdown(tmp_path: Path) -> None:
    output_path = tmp_path / "monitoring.json"
    report_path = tmp_path / "monitoring.md"

    result = monitor_inference_logs(LOGS_PATH, output_path=output_path)
    write_monitoring_report(result, report_path)

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    report = report_path.read_text(encoding="utf-8")
    assert payload["status"] == "pass"
    assert payload["metrics"]["p95_latency_seconds"] == 0.48
    assert report.startswith("# Inference Monitoring Report\n")
    assert "Status: **PASS**" in report
    assert "| `/generate` | 3 |" in report
