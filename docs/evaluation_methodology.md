# Evaluation Methodology

## Evaluation layers

The project deliberately separates fast plumbing checks from stronger release
evidence.

### Baseline metrics

Model-free and model-backed evaluation records:

- exact match;
- token-overlap F1;
- expected key-term coverage;
- response length;
- empty-response rate;
- generation latency and token counts where available.

These metrics catch obvious regressions and schema failures but are not
sufficient measures of instruction-following quality.

### Rigorous deterministic suite

Evaluation fixtures add required facts, forbidden terms, task type, and
metadata. Per-response diagnostics include:

- required-fact coverage;
- forbidden-term violations;
- instruction copying;
- refusal detection;
- JSON, list, short-answer, or freeform format compliance;
- unsupported named-entity heuristics;
- numeric mismatch;
- simple negation-based contradiction checks.

Aggregate results retain per-example evidence so a score can be traced back to
the responsible output.

### Pairwise comparison

Two aligned generation files are compared in a fixed order: forbidden terms,
required facts, copying, length sanity, format compliance, refusal and
hallucination signals, token F1, then exact match. Equal signals produce a tie.
No random or hidden judge is involved.

### Operational checks

Inference monitoring aggregates request count, error rate, average/p50/p95
latency, response length, empty responses, endpoint traffic, model traffic, and
mock versus real calls. Thresholds produce pass, warn, or fail.

The release gate fails when:

- required-fact coverage drops from baseline;
- forbidden-term violation rate rises;
- empty-response rate rises;
- current p95 latency exceeds its configured ceiling when available.

## Reproducibility

Fixtures and metrics are local and deterministic. Evaluation artifacts include
input paths, record counts, aggregate metrics, and per-example outputs. Workflow
manifests capture environment and source provenance.

## Interpretation

Passing the current fixture gate means the software correctly detected the
expected fixture relationship. It is not evidence that a tiny model is useful.
Before a real release, replace fixtures with representative held-out cases,
define task-specific thresholds, inspect slices and failures, and add human or
model-judge evaluation with calibration and review.
