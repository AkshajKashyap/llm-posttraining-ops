import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from llm_posttraining_ops.workflows.registry import (
    ExperimentRegistry,
    create_run_id,
    validate_run_id,
)


def test_run_id_creation_and_validation() -> None:
    timestamp = datetime(2026, 6, 30, 12, 34, 56, 123456, tzinfo=timezone.utc)

    assert create_run_id(timestamp) == "run-20260630T123456123456Z"
    assert validate_run_id("smoke-01") == "smoke-01"
    with pytest.raises(ValueError, match="run_id"):
        validate_run_id("../unsafe")


def test_registry_writes_stage_pass_fail_and_skip(tmp_path: Path) -> None:
    registry_path = tmp_path / "registry.json"
    registry = ExperimentRegistry(
        "registry-test",
        registry_path,
        settings={"seed": 42},
        timestamp_factory=lambda: "2026-06-30T00:00:00+00:00",
    )

    passed = registry.begin_stage("passed")
    registry.finish_stage(passed, "pass", artifacts={"result": "result.json"})
    failed = registry.begin_stage("failed")
    registry.finish_stage(failed, "fail", error="ValueError: broken")
    registry.skip_stage("skipped", "disabled by test")
    registry.finalize("fail")

    payload = json.loads(registry_path.read_text(encoding="utf-8"))
    assert payload["status"] == "fail"
    assert [stage["status"] for stage in payload["stages"]] == [
        "pass",
        "fail",
        "skipped",
    ]
    assert payload["stages"][1]["error"] == "ValueError: broken"
    assert payload["artifacts"] == {"result": "result.json"}
