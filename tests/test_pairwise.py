from llm_posttraining_ops.evaluation.pairwise import (
    compare_generation_files,
    compare_responses,
)
from llm_posttraining_ops.evaluation.suite import EvaluationExample


def test_pairwise_fixture_win_loss_tie_counts() -> None:
    result = compare_generation_files(
        "tests/fixtures/generations_bad.jsonl",
        "tests/fixtures/generations_good.jsonl",
        "tests/fixtures/eval_suite_sample.jsonl",
    )

    assert result.counts.left_wins == 0
    assert result.counts.right_wins == 3
    assert result.counts.ties == 1
    assert [decision.winner for decision in result.decisions] == [
        "right",
        "right",
        "right",
        "tie",
    ]


def test_pairwise_can_prefer_left_response() -> None:
    example = EvaluationExample(
        id="one",
        split="test",
        instruction="Name the capital of France.",
        input="",
        reference_output="Paris.",
        required_facts=["Paris"],
        forbidden_terms=["London"],
        task_type="short_answer",
        metadata={"max_tokens": 3},
    )

    decision = compare_responses(example, "Paris.", "London.")

    assert decision.winner == "left"
    assert decision.reason == "fewer forbidden-term violations"
