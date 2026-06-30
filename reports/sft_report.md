# Supervised Fine-Tuning Report

- Base model: `sshleifer/tiny-gpt2`
- Checkpoint: `artifacts/models/sft`
- Training records: 2
- LoRA enabled: False
- Training loss: 10.835

## Pre-SFT vs post-SFT metrics

| Stage | Exact match | Token overlap F1 | Contains expected key terms | Average response length | Empty response rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| Pre-SFT | 0.000 | 0.000 | 0.000 | 32.000 | 0.000 |
| Post-SFT | 0.000 | 0.000 | 0.000 | 32.000 | 0.000 |

## Training settings

| Setting | Value |
| --- | --- |
| batch_size | `1` |
| gradient_accumulation_steps | `1` |
| learning_rate | `5e-05` |
| lora_alpha | `16` |
| lora_dropout | `0.05` |
| lora_r | `8` |
| max_seq_length | `128` |
| max_steps | `1` |
| model_name | `sshleifer/tiny-gpt2` |
| output_dir | `artifacts/models/sft` |
| seed | `42` |
| use_lora | `False` |

## Generation latency

| Stage | Total seconds | Seconds/example | Average generated tokens |
| --- | ---: | ---: | ---: |
| Pre-SFT | 3.475 | 0.869 | 32.000 |
| Post-SFT | 2.034 | 0.508 | 32.000 |

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
