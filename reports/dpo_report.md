# Direct Preference Optimization Report

- Starting model: `sshleifer/tiny-gpt2`
- DPO checkpoint: `artifacts/models/dpo`
- Preference training records: 2
- LoRA enabled: False
- DPO loss: 0.693

## Model comparison

| Model | Exact match | Token overlap F1 | Contains expected key terms | Average response length | Empty response rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| Base | 0.000 | 0.000 | 0.000 | 32.000 | 0.000 |
| SFT | 0.000 | 0.000 | 0.000 | 32.000 | 0.000 |
| DPO | 0.000 | 0.000 | 0.000 | 32.000 | 0.000 |

## Training settings

| Setting | Value |
| --- | --- |
| batch_size | `1` |
| beta | `0.1` |
| gradient_accumulation_steps | `1` |
| learning_rate | `1e-06` |
| lora_alpha | `16` |
| lora_dropout | `0.05` |
| lora_r | `8` |
| max_seq_length | `128` |
| max_steps | `1` |
| model_name | `sshleifer/tiny-gpt2` |
| output_dir | `artifacts/models/dpo` |
| seed | `42` |
| sft_model_path | `None` |
| use_lora | `False` |

## Generation latency

| Model | Total seconds | Seconds/example | Average generated tokens |
| --- | ---: | ---: | ---: |
| Base | 3.475 | 0.869 | 32.000 |
| SFT | 2.034 | 0.508 | 32.000 |
| DPO | 4.068 | 1.017 | 32.000 |

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
