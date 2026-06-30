# Baseline Evaluation Report

- Dataset: `data/processed/demo/sft.jsonl`
- Records: 12
- Splits: test=2, train=8, validation=2
- Schema version: 1.0

## Summary

| Baseline | Exact match | Token overlap F1 | Contains expected key terms | Average response length | Empty response rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| echo | 0.000 | 0.000 | 0.000 | 6.000 | 0.000 |
| template | 0.000 | 0.000 | 0.000 | 11.000 | 0.000 |
| keyword_rule | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 |

## Metric definitions

- **Exact match:** normalized generated text equals the expected output.
- **Token overlap F1:** multiset token precision/recall F1.
- **Contains expected key terms:** all expected-output tokens appear in the response.
- **Average response length:** mean generated response length in tokens.
- **Empty response rate:** fraction of responses that are empty or whitespace-only.

All baselines are deterministic and run locally without a model or GPU.
