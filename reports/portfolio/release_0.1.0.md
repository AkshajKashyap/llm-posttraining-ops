# Portfolio Release Report: v0.1.0

## Release outcome

Version 0.1.0 is a complete, CPU-compatible reference implementation of an LLM
post-training operations lifecycle. The release smoke workflow passed with
eight stages completed, zero failed, and three model-dependent stages
explicitly skipped.

## Implemented milestones

1. Configuration, schemas, deterministic demo data, and validation.
2. Model-free baselines and evaluation metrics.
3. Local SFT ingestion, profiling, and dataset cards.
4. Hugging Face model inference and latency-aware evaluation.
5. Full-model and LoRA supervised fine-tuning.
6. Preference ingestion plus full-model and LoRA DPO.
7. Rigorous deterministic evaluation and pairwise comparison.
8. FastAPI serving, lazy model loading, mock mode, and inference logs.
9. Monitoring metrics and evaluation regression gates.
10. Workflow orchestration, experiment registry, and reproducibility manifest.
11. Release metadata, CI, Docker smoke support, and portfolio documentation.

## Verification

```text
Python 3.11
113 tests passed
ruff check . passed
release-smoke workflow: 8 passed, 0 failed, 3 skipped
```

Tests do not download models or require a GPU. The tracked smoke path exercises
real ingestion, validation, profiling, baseline evaluation, rigorous
evaluation, release gating, registry updates, and manifest generation.

## Supported CLI surface

- `project-info`
- `run-demo-workflow`
- `prepare-demo-data`
- `validate-data`
- `ingest-sft-data`
- `ingest-preference-data`
- `profile-data`
- `profile-preference-data`
- `run-baseline-eval`
- `run-model-eval`
- `run-eval-suite`
- `compare-generations`
- `train-sft`
- `evaluate-sft`
- `train-dpo`
- `evaluate-dpo`
- `serve`
- `monitor-logs`
- `run-release-gate`
- `generate-baseline-report`
- global `--version`

## Key artifacts

| Artifact | Purpose |
| --- | --- |
| `sft.jsonl`, `preference.jsonl` | Normalized training contracts |
| Dataset profiles and cards | Length, split, duplication, and provenance review |
| Generation JSONL | Model outputs decoupled from evaluation |
| Eval and pairwise JSON | Aggregate plus per-example evidence |
| SFT/DPO summaries | Settings, loss, parameters, and checkpoint location |
| Inference logs | Request, model, settings, latency, response, and errors |
| Monitoring/release reports | Operational thresholds and regression decisions |
| Experiment registry | Durable pass/fail/skip status for every workflow stage |
| Reproducibility manifest | Version, environment, model, data, and Git provenance |

## Honest scope

The repository demonstrates production-shaped interfaces and release practices,
not a production-quality trained model. Tiny fixtures, `sshleifer/tiny-gpt2`,
and one-step training are deliberately chosen to validate infrastructure.
Meaningful model release claims would require governed datasets,
representative held-out evaluation, human review, calibrated judges, and
deployment-specific safety and performance testing.
