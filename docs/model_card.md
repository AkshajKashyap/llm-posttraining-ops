# Model Card

## Scope

Version 0.1.0 does not publish a trained model or claim a quality-improving
checkpoint. It provides infrastructure for evaluating a base causal language
model and producing local SFT or DPO checkpoints.

## Supported model path

- Architecture class: Hugging Face causal language models.
- Default plumbing model: `sshleifer/tiny-gpt2`.
- Training methods: full-model SFT, LoRA SFT, full-model DPO, and LoRA DPO.
- Starting points: a base model, local full checkpoint, or compatible PEFT
  adapter.
- Runtime: CPU-compatible; GPU is optional and not required by tests.

## Intended uses

- Verify post-training data and artifact contracts.
- Exercise one-step training and checkpoint-loading paths.
- Compare base, SFT, and DPO generations with deterministic diagnostics.
- Demonstrate local serving, monitoring, and release workflow integration.
- Provide a foundation for replacing tiny fixtures with governed datasets and
  appropriately selected models.

## Out-of-scope uses

- Production deployment of `sshleifer/tiny-gpt2`.
- Claims about instruction-following, safety, factuality, or preference quality.
- High-stakes or autonomous decision making.
- Treating a one-step smoke checkpoint as a meaningful trained release.

## Training data

Tracked examples are tiny synthetic or hand-authored fixtures. The ingestion
layer supports normalized local SFT and preference JSONL, but no external
dataset is downloaded automatically.

## Evaluation

The project measures exact match, token overlap, required facts, forbidden
terms, format compliance, copying, refusals, simple hallucination heuristics,
latency, and response health. See
[`evaluation_methodology.md`](evaluation_methodology.md).

## Risks and limitations

Metrics are lexical and heuristic. They may miss paraphrases, subtle
contradictions, unsafe content, cultural context, or domain-specific errors.
Real releases require representative held-out data, human review, stronger
model-based evaluation where appropriate, and application-specific safety
testing.
