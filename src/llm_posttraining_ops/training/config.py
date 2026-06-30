"""Validated configuration for supervised fine-tuning."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from llm_posttraining_ops.inference.config import DEFAULT_MODEL_NAME


class SFTConfigError(ValueError):
    """Raised when SFT settings are invalid."""


@dataclass(frozen=True, slots=True)
class SFTTrainingConfig:
    """Settings for a small Hugging Face Trainer SFT run."""

    model_name: str = DEFAULT_MODEL_NAME
    output_dir: Path = Path("artifacts/models/sft")
    max_steps: int = 1
    learning_rate: float = 5e-5
    batch_size: int = 1
    gradient_accumulation_steps: int = 1
    max_seq_length: int = 128
    seed: int = 42
    use_lora: bool = False
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05

    def __post_init__(self) -> None:
        if not self.model_name.strip():
            raise SFTConfigError("model_name must be non-empty")
        if not str(self.output_dir).strip():
            raise SFTConfigError("output_dir must be non-empty")
        for field_name in (
            "max_steps",
            "batch_size",
            "gradient_accumulation_steps",
            "max_seq_length",
            "lora_r",
            "lora_alpha",
        ):
            value = getattr(self, field_name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise SFTConfigError(f"{field_name} must be a positive integer")
        if not isinstance(self.learning_rate, (int, float)) or self.learning_rate <= 0:
            raise SFTConfigError("learning_rate must be positive")
        if not isinstance(self.seed, int) or isinstance(self.seed, bool) or self.seed < 0:
            raise SFTConfigError("seed must be a non-negative integer")
        if not isinstance(self.use_lora, bool):
            raise SFTConfigError("use_lora must be a boolean")
        if (
            not isinstance(self.lora_dropout, (int, float))
            or not 0 <= self.lora_dropout < 1
        ):
            raise SFTConfigError("lora_dropout must be at least 0 and less than 1")

    def to_dict(self) -> dict[str, str | int | float | bool]:
        """Return a JSON-serializable representation."""

        return {
            "model_name": self.model_name,
            "output_dir": str(self.output_dir),
            "max_steps": self.max_steps,
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "max_seq_length": self.max_seq_length,
            "seed": self.seed,
            "use_lora": self.use_lora,
            "lora_r": self.lora_r,
            "lora_alpha": self.lora_alpha,
            "lora_dropout": self.lora_dropout,
        }

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> SFTTrainingConfig:
        """Parse SFT settings from a mapping while applying defaults."""

        return cls(
            model_name=values.get("model_name", DEFAULT_MODEL_NAME),
            output_dir=Path(values.get("output_dir", "artifacts/models/sft")),
            max_steps=values.get("max_steps", 1),
            learning_rate=values.get("learning_rate", 5e-5),
            batch_size=values.get("batch_size", 1),
            gradient_accumulation_steps=values.get("gradient_accumulation_steps", 1),
            max_seq_length=values.get("max_seq_length", 128),
            seed=values.get("seed", 42),
            use_lora=values.get("use_lora", False),
            lora_r=values.get("lora_r", 8),
            lora_alpha=values.get("lora_alpha", 16),
            lora_dropout=values.get("lora_dropout", 0.05),
        )
