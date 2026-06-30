from llm_posttraining_ops.inference.prompts import format_instruction_prompt


def test_prompt_formatting_with_input() -> None:
    assert format_instruction_prompt("Summarize this.", "A long passage.") == (
        "### Instruction:\n"
        "Summarize this.\n\n"
        "### Input:\n"
        "A long passage.\n\n"
        "### Response:"
    )


def test_prompt_formatting_without_input() -> None:
    assert format_instruction_prompt("Say hello.", "  ") == (
        "### Instruction:\nSay hello.\n\n### Response:"
    )
