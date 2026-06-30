import json
from pathlib import Path

from llm_posttraining_ops import __version__
from llm_posttraining_ops.workflows.manifest import (
    build_reproducibility_manifest,
    write_reproducibility_manifest,
)


def test_manifest_contains_environment_models_and_data(tmp_path: Path) -> None:
    manifest = build_reproducibility_manifest(
        run_id="manifest-test",
        seed=7,
        models={"base_model": "tiny/model", "sft_model": None},
        data_paths={"sft": "data/sft.jsonl"},
    )
    path = write_reproducibility_manifest(manifest, tmp_path / "manifest.json")
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert payload["package_version"] == __version__
    assert payload["run_id"] == "manifest-test"
    assert payload["seed"] == 7
    assert payload["python_version"]
    assert payload["platform"]
    assert payload["models"]["base_model"] == "tiny/model"
    assert payload["data_paths"] == {"sft": "data/sft.jsonl"}
    assert isinstance(payload["dependencies"], dict)
    assert payload["git_commit"] is None or len(payload["git_commit"]) == 40
