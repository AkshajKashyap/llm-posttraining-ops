import json
from pathlib import Path

from llm_posttraining_ops.training.dpo_evaluation import resolve_dpo_checkpoint


def test_dpo_checkpoint_supports_full_model_and_adapter(tmp_path: Path) -> None:
    model_path = tmp_path / "model"
    model_path.mkdir()
    assert resolve_dpo_checkpoint(model_path).kind == "full_model"

    adapter_path = tmp_path / "adapter"
    adapter_path.mkdir()
    (adapter_path / "adapter_config.json").write_text(
        json.dumps({"base_model_name_or_path": "base/model"}),
        encoding="utf-8",
    )
    artifact = resolve_dpo_checkpoint(adapter_path)

    assert artifact.kind == "adapter"
    assert artifact.base_model_name == "base/model"
