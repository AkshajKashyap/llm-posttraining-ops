"""Lazy, CPU-first model management with deterministic mock mode."""

from __future__ import annotations

import re
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Protocol

from llm_posttraining_ops.evaluation.metrics import response_length
from llm_posttraining_ops.inference.config import GenerationSettings
from llm_posttraining_ops.inference.huggingface import (
    GenerationOutput,
    HuggingFaceCausalLMGenerator,
)
from llm_posttraining_ops.training.evaluation import load_trained_generator


class ServingGenerator(Protocol):
    """Generator interface needed by the API."""

    model_name: str

    def generate(self, prompt: str, settings: GenerationSettings) -> GenerationOutput:
        """Generate one response."""


class MockGenerator:
    """Deterministic generator for tests and local API smoke checks."""

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name

    def generate(self, prompt: str, settings: GenerationSettings) -> GenerationOutput:
        del settings
        match = re.search(
            r"### Instruction:\n(?P<instruction>.*?)(?:\n\n### Input:|\n\n### Response:)",
            prompt,
            flags=re.DOTALL,
        )
        instruction = match.group("instruction").strip() if match else prompt.strip()
        text = f"Mock response: {instruction}"
        return GenerationOutput(text=text, generated_tokens=response_length(text))


class ModelManager:
    """Load once on first generation and serialize CPU generation calls."""

    def __init__(
        self,
        model_name: str,
        *,
        mock: bool = False,
        generator_factory: Callable[[], ServingGenerator] | None = None,
    ) -> None:
        self.model_name = model_name
        self.mock = mock
        self.device = "cpu"
        self._generator_factory = generator_factory
        self._generator: ServingGenerator | None = None
        self._loaded = False
        self._lock = threading.Lock()

    @property
    def loaded(self) -> bool:
        """Whether a generation has completed successfully."""

        return self._loaded

    def _create_generator(self) -> ServingGenerator:
        if self._generator_factory is not None:
            return self._generator_factory()
        if self.mock:
            return MockGenerator(self.model_name)
        model_path = Path(self.model_name)
        if model_path.is_dir():
            return load_trained_generator(model_path)
        return HuggingFaceCausalLMGenerator(self.model_name)

    def generate(self, prompt: str, settings: GenerationSettings) -> GenerationOutput:
        """Lazily create/load the generator and produce one response."""

        with self._lock:
            if self._generator is None:
                self._generator = self._create_generator()
            output = self._generator.generate(prompt, settings)
            self._loaded = True
            return output
