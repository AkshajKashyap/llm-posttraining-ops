# Model Release Gate Report

- Result: **PASS**
- Baseline evaluation: `tests/fixtures/baseline_eval_gate_sample.json`
- Current evaluation: `tests/fixtures/current_eval_gate_sample.json`

## Regression checks

| Metric | Baseline | Current | Requirement | Result |
| --- | ---: | ---: | --- | --- |
| required_fact_coverage | 0.750 | 0.800 | current >= baseline | pass |
| forbidden_term_violation_rate | 0.250 | 0.200 | current <= baseline | pass |
| empty_response_rate | 0.100 | 0.050 | current <= baseline | pass |
| p95_latency_seconds | 0.800 | 0.900 | current <= 5 | pass |

The release passes only when every quality regression and available
latency check passes.
