from pathlib import Path

import pytest

from llm_posttraining_ops.training.dpo_config import DPOConfigError, DPOTrainingConfig


def test_dpo_config_parsing() -> None:
    config = DPOTrainingConfig.from_mapping(
        {
            "model_name": "base/model",
            "sft_model_path": "artifacts/models/sft",
            "output_dir": "artifacts/adapters/dpo",
            "max_steps": 2,
            "learning_rate": 2e-6,
            "batch_size": 2,
            "gradient_accumulation_steps": 4,
            "max_seq_length": 64,
            "beta": 0.2,
            "seed": 8,
            "use_lora": True,
            "lora_r": 4,
            "lora_alpha": 8,
            "lora_dropout": 0.1,
        }
    )

    assert config.sft_model_path == Path("artifacts/models/sft")
    assert config.output_dir == Path("artifacts/adapters/dpo")
    assert config.beta == 0.2
    assert config.use_lora is True


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("max_steps", 0),
        ("learning_rate", 0),
        ("beta", 0),
        ("lora_dropout", 1.0),
    ],
)
def test_dpo_config_rejects_invalid_values(field: str, value: object) -> None:
    values = DPOTrainingConfig().to_dict()
    values[field] = value  # type: ignore[assignment]

    with pytest.raises(DPOConfigError):
        DPOTrainingConfig.from_mapping(values)
