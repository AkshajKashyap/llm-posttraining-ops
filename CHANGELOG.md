# Changelog

All notable changes to this project are documented in this file. The format is
based on Keep a Changelog, and the project uses semantic versioning.

## [0.1.0] - 2026-06-30

### Added

- Deterministic configuration, SFT/preference schemas, JSONL utilities, and demo data.
- Local Alpaca, messages, and preference-data normalization, validation, and profiling.
- Model-free baselines and Hugging Face causal-LM inference.
- Trainer-based SFT and TRL DPO with full-model and optional LoRA paths.
- Deterministic evaluation suite, pairwise comparison, hallucination heuristics, and reports.
- FastAPI serving with lazy model loading, mock mode, structured logs, and health checks.
- Inference monitoring, thresholds, and evaluation regression gates.
- End-to-end workflow orchestration, experiment registry, and reproducibility manifest.
- GitHub Actions CI, CPU/mock Docker support, Make targets, smoke scripts, and portfolio docs.

### Notes

- Tiny fixtures and one-step training validate infrastructure only.
- No trained model checkpoint is distributed with this release.
