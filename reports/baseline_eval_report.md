# Baseline Evaluation Report

- Dataset: `data/processed/custom/sft.jsonl`
- Records: 4
- Splits: test=1, train=2, validation=1
- Schema version: 1.0

## Summary

| Baseline | Exact match | Token overlap F1 | Contains expected key terms | Average response length | Empty response rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| echo | 0.000 | 0.223 | 0.000 | 3.500 | 0.000 |
| template | 0.000 | 0.192 | 0.000 | 8.500 | 0.000 |
| keyword_rule | 0.000 | 0.000 | 0.000 | 5.000 | 0.000 |
| hf:sshleifer/tiny-gpt2 | 0.000 | 0.000 | 0.000 | 32.000 | 0.000 |

## Model latency

- Model: `sshleifer/tiny-gpt2`
- Device: `cpu`
- Total generation time: 3.475 seconds
- Average seconds per example: 0.869
- Average generated tokens: 32.000

## Rigorous evaluation summary

| Metric | Value |
| --- | ---: |
| Required fact coverage | 0.000 |
| Forbidden term violation rate | 0.000 |
| Instruction copying rate | 0.000 |
| Refusal rate | 0.000 |
| Format compliance rate | 0.500 |
| Unsupported named entity rate | 0.000 |
| Numeric mismatch rate | 0.000 |
| Contradiction rate | 0.000 |


## Metric definitions

- **Exact match:** normalized generated text equals the expected output.
- **Token overlap F1:** multiset token precision/recall F1.
- **Contains expected key terms:** all expected-output tokens appear in the response.
- **Average response length:** mean generated response length in tokens.
- **Empty response rate:** fraction of responses that are empty or whitespace-only.

Model-free baselines are deterministic; Hugging Face generation uses the recorded settings and seed.
