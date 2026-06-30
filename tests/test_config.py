from pathlib import Path

import pytest

from llm_posttraining_ops.config import ConfigError, load_config


def test_load_default_config() -> None:
    config = load_config("configs/default.yaml")

    assert config.seed == 42
    assert config.data.output_dir == Path("data/processed/demo")
    assert config.data.sft_split_sizes == {"train": 8, "validation": 2, "test": 2}
    assert config.data.preference_split_sizes["train"] == 8


def test_load_config_rejects_unknown_split(tmp_path: Path) -> None:
    path = tmp_path / "invalid.yaml"
    path.write_text(
        """
seed: 1
data:
  output_dir: output
  sft_split_sizes: {train: 1, validation: 1, test: 1, surprise: 1}
  preference_split_sizes: {train: 1, validation: 1, test: 1}
""",
        encoding="utf-8",
    )

    with pytest.raises(ConfigError, match="unsupported split"):
        load_config(path)
