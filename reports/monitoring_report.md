# Inference Monitoring Report

- Status: **PASS**
- Logs: `tests/fixtures/inference_logs_sample.jsonl`
- Requests: 5

## Service metrics

| Metric | Value |
| --- | ---: |
| Error rate | 0.000 |
| Average latency (seconds) | 0.300 |
| p50 latency (seconds) | 0.300 |
| p95 latency (seconds) | 0.480 |
| Average response length (tokens) | 8.000 |
| Empty response rate | 0.000 |
| Mock requests | 3 |
| Real requests | 2 |

## Threshold checks

| Metric | Value | Requirement | Status |
| --- | ---: | ---: | --- |
| error_rate | 0.000 | <= 0.050 | pass |
| p95_latency_seconds | 0.480 | <= 5.000 | pass |
| average_response_length_tokens | 8.000 | >= 1.000 | pass |
| empty_response_rate | 0.000 | <= 0.050 | pass |

## Requests by endpoint

| Endpoint | Count |
| --- | ---: |
| `/batch-generate` | 1 |
| `/evaluate-generation` | 1 |
| `/generate` | 3 |

## Requests by model

| Model | Count |
| --- | ---: |
| `local/sft` | 2 |
| `mock/model` | 3 |

Warning status begins at 80% of a maximum limit or within 20% of a
minimum requirement. A breached threshold produces failure.
