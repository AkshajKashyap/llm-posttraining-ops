# Interview Notes

## Thirty-second summary

`llm-posttraining-ops` is a CPU-testable reference system for the lifecycle
around LLM post-training. It normalizes SFT and preference data, supports
Trainer-based SFT and TRL DPO with optional LoRA, evaluates saved generations,
serves checkpoints lazily through FastAPI, monitors inference logs, gates
releases, and records reproducible workflow runs.

## Design decisions worth discussing

### Artifacts before orchestration

Each module first established an explicit JSON/JSONL contract. The workflow
runner composes those stable modules instead of hiding logic in a monolithic
pipeline.

### Model-free tests

Generators and training backends use small protocols and dependency injection.
Tests exercise prompt construction, masking, summaries, checkpoint resolution,
API behavior, and orchestration without downloading models.

### Deterministic evaluation

Local heuristics are transparent and repeatable. They are useful for CI
regression detection, while the docs clearly state where human or calibrated
LLM-judge evaluation is still needed.

### Lazy serving

Health and model-info endpoints do not load weights. The first generation
request resolves a base model, full checkpoint, or PEFT adapter. Mock mode
provides deterministic API and Docker smoke testing.

### Failure as data

Workflow stages are persisted before and after execution. Early failure does
not erase context: registry, manifest, summary, and report are still finalized.

## Tradeoffs

- JSONL is simple and inspectable, but large-scale datasets would need columnar
  storage, streaming, and stronger lineage.
- A process-local lock makes generation safe for a local CPU service, but
  production throughput needs worker and queue design.
- Heuristic evaluation is cheap and auditable, but semantically limited.
- One-step training keeps smoke tests practical, but says nothing about learning
  curves or final quality.
- A single repository clarifies the lifecycle, while larger teams may split
  data, training, evaluation, and serving ownership.

## Useful walkthrough

1. Start with `tests/fixtures/alpaca_sample.jsonl`.
2. Show normalized data and profile artifacts.
3. Show a generation JSONL and per-example eval diagnostics.
4. Open a workflow `experiment_registry.json`.
5. Trace its model/data provenance in `reproducibility_manifest.json`.
6. Show the release gate and explain why its thresholds are intentionally explicit.
7. Start `serve --mock` and inspect the inference log.

## Likely follow-up questions

- How would you scale training? Add distributed configuration, streaming data,
  checkpoint storage, and scheduler integration behind the existing interfaces.
- How would you improve evaluation? Build representative slices, add calibrated
  model judges, human review, confidence intervals, and task-specific policies.
- How would you deploy safely? Add authentication, rate limits, request size
  limits, observability export, load tests, rollout strategy, and rollback.
- Why not claim improvement? Tiny fixtures and one-step runs prove system wiring,
  not statistical or practical model gains.
