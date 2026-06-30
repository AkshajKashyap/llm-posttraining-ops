from pathlib import Path

import pytest

from llm_posttraining_ops.training.config import SFTConfigError, SFTTrainingConfig


def test_sft_config_parsing() -> None:
    config = SFTTrainingConfig.from_mapping(
        {
            "model_name": "mock/model",
            "output_dir": "artifacts/adapters/test",
            "max_steps": 3,
            "learning_rate": 0.001,
            "batch_size": 2,
            "gradient_accumulation_steps": 4,
            "max_seq_length": 64,
            "seed": 7,
            "use_lora": True,
            "lora_r": 4,
            "lora_alpha": 8,
            "lora_dropout": 0.1,
        }
    )

    assert config.output_dir == Path("artifacts/adapters/test")
    assert config.use_lora is True
    assert config.to_dict()["gradient_accumulation_steps"] == 4


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("max_steps", 0),
        ("learning_rate", 0),
        ("max_seq_length", -1),
        ("lora_dropout", 1.0),
    ],
)
def test_sft_config_rejects_invalid_values(field: str, value: object) -> None:
    values = SFTTrainingConfig().to_dict()
    values[field] = value  # type: ignore[assignment]

    with pytest.raises(SFTConfigError):
        SFTTrainingConfig.from_mapping(values)
