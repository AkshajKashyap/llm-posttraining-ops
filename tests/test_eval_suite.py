from llm_posttraining_ops.evaluation.suite import (
    EvaluationExample,
    evaluate_response,
    forbidden_term_violation,
    format_compliance,
    instruction_copying,
    load_evaluation_examples,
    numeric_mismatch,
    required_fact_coverage,
)


def _example(task_type: str, **metadata: object) -> EvaluationExample:
    return EvaluationExample(
        id="one",
        split="test",
        instruction="Return the requested result.",
        input="",
        reference_output="The result is 42.",
        required_facts=["42"],
        forbidden_terms=["unknown"],
        task_type=task_type,  # type: ignore[arg-type]
        metadata=dict(metadata),
    )


def test_evaluation_schema_loading() -> None:
    examples = load_evaluation_examples("tests/fixtures/eval_suite_sample.jsonl")

    assert len(examples) == 4
    assert examples[0].id == "alpaca-001"
    assert examples[0].required_facts == ["Paris", "capital city of France"]
    assert examples[0].task_type == "short_answer"


def test_required_facts_and_forbidden_terms() -> None:
    assert required_fact_coverage(
        ["Paris", "capital city of France"],
        "Paris is the capital city of France.",
    ) == 1.0
    assert required_fact_coverage(["Paris", "France"], "Paris") == 0.5
    assert forbidden_term_violation(["London"], "The answer is London.") == 1.0
    assert forbidden_term_violation(["London"], "The answer is Paris.") == 0.0


def test_format_compliance_for_json_list_and_short_answer() -> None:
    assert format_compliance(_example("json"), '{"answer": 42}') == 1.0
    assert format_compliance(_example("json"), "answer: 42") == 0.0
    assert format_compliance(_example("list"), "- first\n- second") == 1.0
    assert format_compliance(_example("list"), "first, second") == 0.0
    assert format_compliance(_example("short_answer", max_tokens=3), "forty two") == 1.0
    assert (
        format_compliance(
            _example("short_answer", max_tokens=3),
            "This response is much too long.",
        )
        == 0.0
    )


def test_instruction_copying_detection() -> None:
    instruction = "Return the requested result exactly."

    assert instruction_copying(instruction, instruction) == 1.0
    assert instruction_copying(instruction, f"{instruction} The result is 42.") == 1.0
    assert instruction_copying(instruction, "The result is 42.") == 0.0


def test_numeric_mismatch_detection() -> None:
    assert numeric_mismatch("The result is 42.", "The result is 41.") == 1.0
    assert numeric_mismatch("The result is 42.", "The result is 42.") == 0.0
    assert numeric_mismatch("No number is required.", "The result is 41.") == 0.0


def test_hallucination_and_contradiction_diagnostics() -> None:
    example = EvaluationExample(
        id="entity",
        split="test",
        instruction="Name the capital.",
        input="France",
        reference_output="Paris is the capital of France.",
        required_facts=["Paris"],
        forbidden_terms=[],
        task_type="freeform",
        metadata={},
    )

    diagnostics = evaluate_response(
        example,
        "London is not the capital of France.",
    )

    assert diagnostics.unsupported_named_entities == ["London"]
    assert diagnostics.contradiction_detected == 1.0
