"""Deterministic text metrics for instruction-following responses."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass


def tokenize(text: str) -> list[str]:
    """Return case-insensitive word and number tokens."""

    return re.findall(r"\w+", text.casefold())


def exact_match(expected: str, response: str) -> float:
    """Return 1 when case and surrounding whitespace-normalized texts match."""

    def normalize(value: str) -> str:
        return " ".join(value.casefold().split())

    return float(normalize(expected) == normalize(response))


def token_overlap_f1(expected: str, response: str) -> float:
    """Compute multiset token overlap F1."""

    expected_tokens = tokenize(expected)
    response_tokens = tokenize(response)
    if not expected_tokens and not response_tokens:
        return 1.0
    if not expected_tokens or not response_tokens:
        return 0.0

    overlap = sum((Counter(expected_tokens) & Counter(response_tokens)).values())
    precision = overlap / len(response_tokens)
    recall = overlap / len(expected_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def contains_expected_key_terms(expected: str, response: str) -> float:
    """Return 1 when the response contains every expected token."""

    expected_counts = Counter(tokenize(expected))
    response_counts = Counter(tokenize(response))
    if not expected_counts:
        return 1.0
    return float(all(response_counts[token] >= count for token, count in expected_counts.items()))


def response_length(response: str) -> int:
    """Return response length in word/number tokens."""

    return len(tokenize(response))


def empty_response(response: str) -> float:
    """Return 1 when a response is empty or whitespace-only."""

    return float(not response.strip())


@dataclass(frozen=True, slots=True)
class ExampleMetrics:
    """Metrics calculated for one expected/generated response pair."""

    exact_match: float
    token_overlap_f1: float
    contains_expected_key_terms: float
    response_length: int
    empty_response: float

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class AggregateMetrics:
    """Mean metrics across one baseline's examples."""

    exact_match: float
    token_overlap_f1: float
    contains_expected_key_terms: float
    average_response_length: float
    empty_response_rate: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def calculate_example_metrics(expected: str, response: str) -> ExampleMetrics:
    """Calculate every supported metric for one response."""

    return ExampleMetrics(
        exact_match=exact_match(expected, response),
        token_overlap_f1=token_overlap_f1(expected, response),
        contains_expected_key_terms=contains_expected_key_terms(expected, response),
        response_length=response_length(response),
        empty_response=empty_response(response),
    )


def aggregate_metrics(metrics: Sequence[ExampleMetrics]) -> AggregateMetrics:
    """Average per-example metrics using a stable six-decimal representation."""

    if not metrics:
        raise ValueError("Cannot aggregate an empty metrics sequence")

    count = len(metrics)

    def mean(values: Iterable[float | int]) -> float:
        return round(sum(values) / count, 6)

    return AggregateMetrics(
        exact_match=mean(item.exact_match for item in metrics),
        token_overlap_f1=mean(item.token_overlap_f1 for item in metrics),
        contains_expected_key_terms=mean(
            item.contains_expected_key_terms for item in metrics
        ),
        average_response_length=mean(item.response_length for item in metrics),
        empty_response_rate=mean(item.empty_response for item in metrics),
    )
