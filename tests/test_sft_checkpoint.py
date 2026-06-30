import json
from pathlib import Path

import pytest

from llm_posttraining_ops.training.evaluation import (
    SFTCheckpointError,
    resolve_model_artifact,
)


def test_resolve_full_model_checkpoint(tmp_path: Path) -> None:
    model_path = tmp_path / "model"
    model_path.mkdir()

    artifact = resolve_model_artifact(model_path)

    assert artifact.kind == "full_model"
    assert artifact.path == model_path
    assert artifact.base_model_name is None


def test_resolve_adapter_checkpoint(tmp_path: Path) -> None:
    adapter_path = tmp_path / "adapter"
    adapter_path.mkdir()
    (adapter_path / "adapter_config.json").write_text(
        json.dumps({"base_model_name_or_path": "base/model"}),
        encoding="utf-8",
    )

    artifact = resolve_model_artifact(adapter_path)

    assert artifact.kind == "adapter"
    assert artifact.base_model_name == "base/model"


def test_resolve_checkpoint_rejects_missing_path(tmp_path: Path) -> None:
    with pytest.raises(SFTCheckpointError, match="not a directory"):
        resolve_model_artifact(tmp_path / "missing")
