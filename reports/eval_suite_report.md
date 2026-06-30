# Rigorous Evaluation Suite Report

- Generations: `artifacts/evals/generations/sshleifer_tiny-gpt2.jsonl`
- Evaluation data: `tests/fixtures/eval_suite_sample.jsonl`
- Records: 4

## Aggregate metrics

| Metric | Value |
| --- | ---: |
| Exact match | 0.000 |
| Token overlap F1 | 0.000 |
| Required fact coverage | 0.000 |
| Forbidden term violation rate | 0.000 |
| Instruction copying rate | 0.000 |
| Empty response rate | 0.000 |
| Refusal rate | 0.000 |
| Format compliance rate | 0.500 |
| Unsupported named entity rate | 0.000 |
| Numeric mismatch rate | 0.000 |
| Contradiction rate | 0.000 |
| Average response length | 32.000 |
| Minimum response length | 32 |
| Maximum response length | 32 |

## Per-example diagnostics

| ID | Facts | Forbidden | Copy | Refusal | Format | Entity | Number | Contradiction |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| alpaca-001 | 0.000 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| alpaca-002 | 0.000 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| alpaca-003 | 0.000 | 0 | 0 | 0 | 1 | 0 | 0 | 0 |
| alpaca-004 | 0.000 | 0 | 0 | 0 | 1 | 0 | 0 | 0 |
