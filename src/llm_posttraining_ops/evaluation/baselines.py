"""Local, deterministic response generators used as evaluation baselines."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Protocol


class BaselineGenerator(Protocol):
    """Interface implemented by all baseline response generators."""

    name: str

    def generate(self, instruction: str, input_text: str) -> str:
        """Generate one response from an instruction and its input."""


@dataclass(frozen=True, slots=True)
class EchoBaseline:
    """Return the input verbatim."""

    name: str = "echo"

    def generate(self, instruction: str, input_text: str) -> str:
        del instruction
        return input_text


@dataclass(frozen=True, slots=True)
class TemplateBaseline:
    """Wrap the input in a fixed response template."""

    name: str = "template"

    def generate(self, instruction: str, input_text: str) -> str:
        del instruction
        return f"Based on the provided input: {input_text}"


@dataclass(frozen=True, slots=True)
class KeywordRuleBaseline:
    """Apply simple arithmetic rules selected from instruction keywords."""

    name: str = "keyword_rule"

    def generate(self, instruction: str, input_text: str) -> str:
        numbers = [int(value) for value in re.findall(r"-?\d+", input_text)]
        normalized_instruction = instruction.casefold()
        if len(numbers) < 2:
            return "Unable to determine an answer."
        if "multiply" in normalized_instruction or "product" in normalized_instruction:
            return str(math.prod(numbers))
        if "add" in normalized_instruction or "sum" in normalized_instruction:
            return str(sum(numbers))
        return "Unable to determine an answer."


DEFAULT_BASELINES: tuple[BaselineGenerator, ...] = (
    EchoBaseline(),
    TemplateBaseline(),
    KeywordRuleBaseline(),
)
