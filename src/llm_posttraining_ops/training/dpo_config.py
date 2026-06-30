"""Validated configuration for direct preference optimization."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from llm_posttraining_ops.inference.config import DEFAULT_MODEL_NAME


class DPOConfigError(ValueError):
    """Raised when DPO settings are invalid."""


@dataclass(frozen=True, slots=True)
class DPOTrainingConfig:
    """Settings for a tiny TRL DPO run."""

    model_name: str = DEFAULT_MODEL_NAME
    sft_model_path: Path | None = None
    output_dir: Path = Path("artifacts/models/dpo")
    max_steps: int = 1
    learning_rate: float = 1e-6
    batch_size: int = 1
    gradient_accumulation_steps: int = 1
    max_seq_length: int = 128
    beta: float = 0.1
    seed: int = 42
    use_lora: bool = False
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05

    def __post_init__(self) -> None:
        if not self.model_name.strip():
            raise DPOConfigError("model_name must be non-empty")
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
                raise DPOConfigError(f"{field_name} must be a positive integer")
        if not isinstance(self.learning_rate, (int, float)) or self.learning_rate <= 0:
            raise DPOConfigError("learning_rate must be positive")
        if not isinstance(self.beta, (int, float)) or self.beta <= 0:
            raise DPOConfigError("beta must be positive")
        if not isinstance(self.seed, int) or isinstance(self.seed, bool) or self.seed < 0:
            raise DPOConfigError("seed must be a non-negative integer")
        if not isinstance(self.use_lora, bool):
            raise DPOConfigError("use_lora must be a boolean")
        if (
            not isinstance(self.lora_dropout, (int, float))
            or not 0 <= self.lora_dropout < 1
        ):
            raise DPOConfigError("lora_dropout must be at least 0 and less than 1")

    def to_dict(self) -> dict[str, str | int | float | bool | None]:
        """Return JSON-serializable settings."""

        return {
            "model_name": self.model_name,
            "sft_model_path": (
                str(self.sft_model_path) if self.sft_model_path is not None else None
            ),
            "output_dir": str(self.output_dir),
            "max_steps": self.max_steps,
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "max_seq_length": self.max_seq_length,
            "beta": self.beta,
            "seed": self.seed,
            "use_lora": self.use_lora,
            "lora_r": self.lora_r,
            "lora_alpha": self.lora_alpha,
            "lora_dropout": self.lora_dropout,
        }

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> DPOTrainingConfig:
        """Parse DPO settings from a mapping while applying defaults."""

        sft_path = values.get("sft_model_path")
        return cls(
            model_name=values.get("model_name", DEFAULT_MODEL_NAME),
            sft_model_path=Path(sft_path) if sft_path is not None else None,
            output_dir=Path(values.get("output_dir", "artifacts/models/dpo")),
            max_steps=values.get("max_steps", 1),
            learning_rate=values.get("learning_rate", 1e-6),
            batch_size=values.get("batch_size", 1),
            gradient_accumulation_steps=values.get("gradient_accumulation_steps", 1),
            max_seq_length=values.get("max_seq_length", 128),
            beta=values.get("beta", 0.1),
            seed=values.get("seed", 42),
            use_lora=values.get("use_lora", False),
            lora_r=values.get("lora_r", 8),
            lora_alpha=values.get("lora_alpha", 16),
            lora_dropout=values.get("lora_dropout", 0.05),
        )
