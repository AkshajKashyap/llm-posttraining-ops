# SFT Dataset Card

- Dataset: `data/processed/custom/sft.jsonl`
- Records: 4
- Splits: test=1, train=2, validation=1
- Sources: alpaca_sample=4
- Profile schema: 1.0

## Profile

| Metric | Value |
| --- | ---: |
| Average instruction length (tokens) | 8.500 |
| Average output length (tokens) | 7.750 |
| Empty input rate | 0.000 |
| Duplicate output rate | 0.000 |

## Repeated output starting phrases

- `a practical benefit`: 2

## Quality checks

Records were checked for schema completeness, duplicate IDs, valid splits,
minimum output length, repetitive outputs, and copied instructions.

This card was generated deterministically from local normalized JSONL.
