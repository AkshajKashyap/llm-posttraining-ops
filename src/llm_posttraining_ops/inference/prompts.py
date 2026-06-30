"""Prompt templates for normalized instruction examples."""

from __future__ import annotations


def format_instruction_prompt(instruction: str, input_text: str) -> str:
    """Format an instruction and optional input for a causal language model."""

    sections = ["### Instruction:", instruction.strip()]
    if input_text.strip():
        sections.extend(["", "### Input:", input_text.strip()])
    sections.extend(["", "### Response:"])
    return "\n".join(sections)
