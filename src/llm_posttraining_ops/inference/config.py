"""Validated settings for causal language-model generation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping

DEFAULT_MODEL_NAME = "sshleifer/tiny-gpt2"


class GenerationConfigError(ValueError):
    """Raised when generation settings are invalid."""


@dataclass(frozen=True, slots=True)
class GenerationSettings:
    """Configuration shared by CLI and Hugging Face generation."""

    model_name: str = DEFAULT_MODEL_NAME
    max_new_tokens: int = 32
    temperature: float = 0.0
    top_p: float = 1.0
    seed: int = 42

    def __post_init__(self) -> None:
        if not self.model_name.strip():
            raise GenerationConfigError("model_name must be non-empty")
        if (
            not isinstance(self.max_new_tokens, int)
            or isinstance(self.max_new_tokens, bool)
            or self.max_new_tokens < 1
        ):
            raise GenerationConfigError("max_new_tokens must be a positive integer")
        if not isinstance(self.temperature, (int, float)) or self.temperature < 0:
            raise GenerationConfigError("temperature must be non-negative")
        if not isinstance(self.top_p, (int, float)) or not 0 < self.top_p <= 1:
            raise GenerationConfigError("top_p must be greater than 0 and at most 1")
        if not isinstance(self.seed, int) or isinstance(self.seed, bool) or self.seed < 0:
            raise GenerationConfigError("seed must be a non-negative integer")

    @property
    def do_sample(self) -> bool:
        """Whether generation should use stochastic sampling."""

        return self.temperature > 0

    def to_dict(self) -> dict[str, str | int | float]:
        """Return JSON-serializable settings."""

        return asdict(self)

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> GenerationSettings:
        """Parse settings from a mapping while applying defaults."""

        return cls(
            model_name=values.get("model_name", DEFAULT_MODEL_NAME),
            max_new_tokens=values.get("max_new_tokens", 32),
            temperature=values.get("temperature", 0.0),
            top_p=values.get("top_p", 1.0),
            seed=values.get("seed", 42),
        )
