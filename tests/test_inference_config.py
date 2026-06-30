import pytest

from llm_posttraining_ops.inference.config import (
    DEFAULT_MODEL_NAME,
    GenerationConfigError,
    GenerationSettings,
)


def test_generation_settings_from_mapping() -> None:
    settings = GenerationSettings.from_mapping(
        {
            "model_name": "local/model",
            "max_new_tokens": 12,
            "temperature": 0.7,
            "top_p": 0.8,
            "seed": 9,
        }
    )

    assert settings.to_dict() == {
        "model_name": "local/model",
        "max_new_tokens": 12,
        "temperature": 0.7,
        "top_p": 0.8,
        "seed": 9,
    }
    assert settings.do_sample is True
    assert GenerationSettings().model_name == DEFAULT_MODEL_NAME


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("model_name", ""),
        ("max_new_tokens", 0),
        ("temperature", -0.1),
        ("top_p", 0.0),
        ("seed", -1),
    ],
)
def test_generation_settings_reject_invalid_values(field: str, value: object) -> None:
    values = GenerationSettings().to_dict()
    values[field] = value  # type: ignore[assignment]

    with pytest.raises(GenerationConfigError):
        GenerationSettings.from_mapping(values)
