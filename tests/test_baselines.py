from llm_posttraining_ops.evaluation.baselines import (
    EchoBaseline,
    KeywordRuleBaseline,
    TemplateBaseline,
)


def test_echo_and_template_baselines() -> None:
    instruction = "Add the values."
    input_text = "First integer: 2. Second integer: 3."

    assert EchoBaseline().generate(instruction, input_text) == input_text
    assert TemplateBaseline().generate(instruction, input_text) == (
        f"Based on the provided input: {input_text}"
    )


def test_keyword_rule_baseline_handles_arithmetic() -> None:
    baseline = KeywordRuleBaseline()

    assert baseline.generate("Add both integers.", "Values: 4 and 7.") == "11"
    assert baseline.generate("Multiply both integers.", "Values: 4 and 7.") == "28"
    assert baseline.generate("Describe the integers.", "Values: 4 and 7.") == (
        "Unable to determine an answer."
    )
