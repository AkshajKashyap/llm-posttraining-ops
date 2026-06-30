"""Prompt and target construction for supervised fine-tuning."""

from __future__ import annotations

from dataclasses import dataclass

from llm_posttraining_ops.data.schemas import SFTRecord
from llm_posttraining_ops.inference.prompts import format_instruction_prompt


@dataclass(frozen=True, slots=True)
class SFTText:
    """Separate prompt and expected response text."""

    prompt: str
    response: str

    def full_text(self, eos_token: str = "") -> str:
        """Return the concatenated training sequence."""

        return f"{self.prompt}{self.response}{eos_token}"


def build_sft_text(record: SFTRecord) -> SFTText:
    """Construct a prompt ending immediately before the expected output."""

    prompt = f"{format_instruction_prompt(record.instruction, record.input)}\n"
    return SFTText(prompt=prompt, response=record.output.strip())
