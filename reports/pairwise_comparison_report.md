# Pairwise Generation Comparison

- Left: `tests/fixtures/generations_bad.jsonl`
- Right: `tests/fixtures/generations_good.jsonl`
- Evaluation data: `tests/fixtures/eval_suite_sample.jsonl`
- Records: 4

## Outcome

| Result | Count |
| --- | ---: |
| Left wins | 0 |
| Right wins | 3 |
| Ties | 1 |

## Decisions

| ID | Winner | Deterministic reason |
| --- | --- | --- |
| alpaca-001 | right | fewer forbidden-term violations |
| alpaca-002 | right | higher required-fact coverage |
| alpaca-003 | right | higher required-fact coverage |
| alpaca-004 | tie | all deterministic quality signals are equal |

Decisions use forbidden terms, required facts, instruction copying,
length sanity, format compliance, refusal/hallucination checks, and
lexical metrics in a fixed order. Equal signals produce a tie.
