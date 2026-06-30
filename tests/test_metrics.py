import pytest

from llm_posttraining_ops.evaluation.metrics import (
    aggregate_metrics,
    calculate_example_metrics,
    contains_expected_key_terms,
    exact_match,
    token_overlap_f1,
)


def test_exact_match_normalizes_case_and_whitespace() -> None:
    assert exact_match("The Answer", "  the   answer ") == 1.0
    assert exact_match("42", "The answer is 42") == 0.0


def test_token_overlap_f1() -> None:
    assert token_overlap_f1("red blue", "red green") == pytest.approx(0.5)
    assert token_overlap_f1("", "") == 1.0
    assert token_overlap_f1("red", "") == 0.0


def test_contains_expected_key_terms_respects_token_counts() -> None:
    assert contains_expected_key_terms("go go", "Go, then go!") == 1.0
    assert contains_expected_key_terms("go go", "go once") == 0.0


def test_example_and_aggregate_response_metrics() -> None:
    first = calculate_example_metrics("42", "42")
    second = calculate_example_metrics("7", " ")
    aggregate = aggregate_metrics([first, second])

    assert first.response_length == 1
    assert second.empty_response == 1.0
    assert aggregate.exact_match == 0.5
    assert aggregate.average_response_length == 0.5
    assert aggregate.empty_response_rate == 0.5
