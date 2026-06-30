# LLM Post-Training Ops

An incremental, reproducible foundation for LLM post-training workflows.

## Milestone 1

Milestone 1 provides typed configuration loading, dataset schemas, deterministic toy
SFT and preference data, JSONL utilities, validation, and a Typer CLI. It does not
download models or datasets and does not start training.

Install the package and development tools:

```bash
python -m pip install -e ".[dev]"
```

Generate the demo datasets using `configs/default.yaml`:

```bash
llm-posttraining-ops prepare-demo-data
# Equivalent module invocation:
python -m llm_posttraining_ops.cli prepare-demo-data
```

The command writes `sft.jsonl` and `preference.jsonl` under
`data/processed/demo/`. Validate both files with:

```bash
llm-posttraining-ops validate-data --data-dir data/processed/demo
# Equivalent module invocation:
python -m llm_posttraining_ops.cli validate-data --data-dir data/processed/demo
```

Run the quality checks:

```bash
pytest -q
ruff check .
```

Use a different deterministic seed or output directory by copying
`configs/default.yaml`, changing its values, and passing
`--config path/to/config.yaml` to `prepare-demo-data`.

## Milestone 2

Milestone 2 adds a model-free inference and evaluation harness for the demo SFT
dataset. It compares three deterministic response generators:

- `echo` returns the input unchanged.
- `template` wraps the input in a fixed response.
- `keyword_rule` applies arithmetic rules selected from instruction keywords.

Run the baselines and write `artifacts/evals/baseline_eval.json`:

```bash
llm-posttraining-ops run-baseline-eval --data-dir data/processed/demo
# Equivalent module invocation:
python -m llm_posttraining_ops.cli run-baseline-eval --data-dir data/processed/demo
```

Generate `reports/baseline_eval_report.md` from the JSON artifact:

```bash
llm-posttraining-ops generate-baseline-report
# Equivalent module invocation:
python -m llm_posttraining_ops.cli generate-baseline-report
```

The evaluation records exact match, token overlap F1, expected-key-term coverage,
average response length in tokens, and empty response rate. It evaluates every
demo SFT split and stores both aggregate metrics and per-example outputs. The
artifact intentionally omits timestamps so identical data produces identical
results.

## Milestone 3

Milestone 3 adds local instruction-dataset ingestion and a normalized SFT schema:
`id`, `split`, `instruction`, `input`, `output`, `source`, and `metadata`.
Supported raw JSONL formats are:

- `alpaca`: records with `instruction`, optional `input`, and `output`.
- `messages`: records containing user and assistant turns in `messages`.

Ingest the tiny local Alpaca fixture:

```bash
python -m llm_posttraining_ops.cli ingest-sft-data \
  --input-path tests/fixtures/alpaca_sample.jsonl \
  --output-dir data/processed/custom \
  --format alpaca
```

Ingestion is deterministic and checks required fields, IDs and splits, output
length, repetitive responses, and outputs that simply copy their instruction.
Empty input strings are valid.

Validate and profile the normalized dataset:

```bash
python -m llm_posttraining_ops.cli validate-data \
  --data-dir data/processed/custom
python -m llm_posttraining_ops.cli profile-data \
  --data-dir data/processed/custom
```

Profiling writes `artifacts/evals/dataset_profile.json` and
`reports/dataset_card.md`. The profile includes split/source counts, average
instruction and output token lengths, empty-input rate, duplicate-output rate,
and repeated output starting phrases.

The baseline harness accepts the same normalized custom directory:

```bash
python -m llm_posttraining_ops.cli run-baseline-eval \
  --data-dir data/processed/custom
python -m llm_posttraining_ops.cli generate-baseline-report
```

Milestones 1–3 do not download a model or dataset.

## Milestone 4

Milestone 4 adds CPU-first causal language-model inference through Hugging Face
Transformers. Instruction examples use a consistent prompt with `Instruction`,
optional `Input`, and `Response` sections.

Run the default tiny model against normalized custom data:

```bash
python -m llm_posttraining_ops.cli run-model-eval \
  --data-dir data/processed/custom \
  --model-name sshleifer/tiny-gpt2
```

The first real run may download the tiny model from Hugging Face. Tests use mocks
and never download a model. Generation defaults to deterministic greedy decoding
on CPU and supports:

```text
--max-new-tokens 32
--temperature 0.0
--top-p 1.0
--seed 42
```

Model generations are written under `artifacts/evals/generations/`, and aggregate
metrics plus latency are written to `artifacts/evals/model_eval.json`. Latency
includes total generation time, average seconds per example, and average generated
tokens.

Regenerate the report after model evaluation:

```bash
python -m llm_posttraining_ops.cli generate-baseline-report
```

When `model_eval.json` is present, the report compares the Hugging Face model with
the echo, template, and keyword/rule baselines and includes a model latency section.
No fine-tuning is performed.

## Milestone 5

Milestone 5 adds supervised fine-tuning for causal language models. SFT reuses the
normalized instruction schema and the inference prompt template. Prompt and padding
tokens receive label `-100`, so training loss is computed on expected response
tokens.

Run a one-step CPU smoke fine-tune:

```bash
python -m llm_posttraining_ops.cli train-sft \
  --data-dir data/processed/custom \
  --model-name sshleifer/tiny-gpt2 \
  --max-steps 1
```

The default full-model checkpoint is written to `artifacts/models/sft`, and the
training summary is written to `artifacts/evals/sft_training_summary.json`.
Training defaults are intentionally small:

```text
--max-steps 1
--learning-rate 0.00005
--batch-size 1
--gradient-accumulation-steps 1
--max-seq-length 128
--seed 42
```

LoRA is supported through PEFT:

```bash
python -m llm_posttraining_ops.cli train-sft \
  --data-dir data/processed/custom \
  --model-name sshleifer/tiny-gpt2 \
  --max-steps 1 \
  --use-lora
```

Without an explicit `--output-dir`, LoRA adapters are saved under
`artifacts/adapters/sft`.

Evaluate the trained full model:

```bash
python -m llm_posttraining_ops.cli evaluate-sft \
  --data-dir data/processed/custom \
  --model-path artifacts/models/sft
```

`evaluate-sft` accepts either a full checkpoint or PEFT adapter. It writes
`artifacts/evals/sft_model_eval.json`, saves generations under
`artifacts/evals/generations/sft.jsonl`, and generates `reports/sft_report.md`
with pre/post metrics, settings, training loss, and latency. Tests mock training
and model loading, so they remain CPU-only and download-free.

## Milestone 6

Milestone 6 adds local preference-data ingestion and direct preference
optimization with TRL. Normalized preference records contain `instruction`,
optional `input`, `chosen`, `rejected`, split/provenance fields, and metadata.

Ingest direct chosen/rejected JSONL and profile it:

```bash
python -m llm_posttraining_ops.cli ingest-preference-data \
  --input-path tests/fixtures/preference_direct_sample.jsonl \
  --output-dir data/processed/preferences \
  --format direct

python -m llm_posttraining_ops.cli profile-preference-data \
  --data-dir data/processed/preferences
```

The `messages` format is also supported for prompt messages plus chosen/rejected
assistant responses. Validation rejects empty or identical responses, invalid
splits, duplicate IDs, short/repetitive responses, and responses that simply copy
the prompt.

Run a one-step CPU DPO smoke train from the base model:

```bash
python -m llm_posttraining_ops.cli train-dpo \
  --preference-data-dir data/processed/preferences \
  --model-name sshleifer/tiny-gpt2 \
  --max-steps 1
```

Pass `--sft-model-path artifacts/models/sft` to start from an SFT checkpoint.
Use `--use-lora` to save a PEFT adapter under `artifacts/adapters/dpo`; otherwise
the full model is written under `artifacts/models/dpo`.

Evaluate the trained DPO model:

```bash
python -m llm_posttraining_ops.cli evaluate-dpo \
  --data-dir data/processed/custom \
  --model-path artifacts/models/dpo
```

Training writes `artifacts/evals/dpo_training_summary.json`. Evaluation writes
the DPO metric/generation artifacts and creates `reports/dpo_report.md`, comparing
the base model, optional SFT model, and DPO model with settings, loss, and latency.
Tests mock TRL training and model loading, so no test downloads or trains a model.

## Milestone 7

Milestone 7 adds a deterministic, reference-backed evaluation suite that does not
use an external model, API, or judge. Evaluation examples specify required facts,
forbidden terms, task type, and metadata alongside the instruction and reference.

Run the suite on saved model generations:

```bash
python -m llm_posttraining_ops.cli run-eval-suite \
  --generations-path artifacts/evals/generations/sshleifer_tiny-gpt2.jsonl \
  --eval-data-path tests/fixtures/eval_suite_sample.jsonl
```

The suite measures exact match, token F1, required-fact coverage, forbidden-term
violations, instruction copying, empty/refusal rates, response lengths, and
JSON/list/short-answer format compliance. It also flags unsupported capitalized
entities, numeric mismatches, and simple contradiction signals.

Compare two aligned generation files:

```bash
python -m llm_posttraining_ops.cli compare-generations \
  --left-path tests/fixtures/generations_bad.jsonl \
  --right-path tests/fixtures/generations_good.jsonl \
  --eval-data-path tests/fixtures/eval_suite_sample.jsonl
```

Pairwise decisions use a fixed order: forbidden terms, required facts, instruction
copying, length sanity, format compliance, refusal/hallucination checks, token F1,
and exact match. Equal signals produce a tie.

JSON results are written under `artifacts/evals/`. Markdown reports are generated
at `reports/eval_suite_report.md` and
`reports/pairwise_comparison_report.md`. When present, rigorous-suite metrics can
also be included in regenerated baseline, SFT, and DPO reports.

## Milestone 8

Milestone 8 adds a local FastAPI serving layer with lazy checkpoint loading,
deterministic mock mode, generation/evaluation endpoints, health checks, and
structured JSONL inference logs.

Start a real model service:

```bash
python -m llm_posttraining_ops.cli serve \
  --model-name sshleifer/tiny-gpt2 \
  --host 0.0.0.0 \
  --port 8000
```

The model is not loaded by `/health` or `/model-info`; loading occurs on the first
generation request. A local full-model checkpoint or PEFT SFT/DPO adapter directory
can be passed to `--model-name`.

Start deterministic mock mode without loading a model:

```bash
python -m llm_posttraining_ops.cli serve --mock --host 0.0.0.0 --port 8000
```

Example requests:

```bash
curl http://localhost:8000/health
curl http://localhost:8000/model-info

curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"instruction":"Explain SFT in one sentence.","input":"","max_new_tokens":32}'

curl -X POST http://localhost:8000/batch-generate \
  -H "Content-Type: application/json" \
  -d '{"items":[{"instruction":"Define SFT."},{"instruction":"Define DPO."}],"seed":42}'

curl -X POST http://localhost:8000/evaluate-generation \
  -H "Content-Type: application/json" \
  -d '{"instruction":"Name the capital of France.","reference_output":"Paris.","generated_response":"London.","required_facts":["Paris"],"forbidden_terms":["London"],"task_type":"short_answer"}'
```

The convenience script `scripts/smoke_api.sh` calls health, model information, and
single generation against `http://localhost:8000` (override with `BASE_URL`).

Inference events are appended to `artifacts/logs/inference_logs.jsonl`. Each event
contains a request ID, UTC timestamp, endpoint, model, mock flag, prompt and caller
metadata, generation settings, latency, token/character response lengths, status,
and any error. Generated logs remain ignored by Git.

## Milestone 9

Milestone 9 turns inference logs and rigorous evaluation artifacts into
deterministic operational checks. It calculates request volume, errors, latency
percentiles, response health, mock/real traffic, and endpoint/model routing.

Monitor the default service log:

```bash
python -m llm_posttraining_ops.cli monitor-logs \
  --logs-path artifacts/logs/inference_logs.jsonl
```

Run the deterministic fixture:

```bash
python -m llm_posttraining_ops.cli monitor-logs \
  --logs-path tests/fixtures/inference_logs_sample.jsonl
```

Thresholds are configurable:

```bash
python -m llm_posttraining_ops.cli monitor-logs \
  --logs-path artifacts/logs/inference_logs.jsonl \
  --max-error-rate 0.05 \
  --max-p95-latency 5.0 \
  --min-average-response-length 1 \
  --max-empty-response-rate 0.05
```

A breached threshold produces `fail`. Values at 80% of a maximum limit, or
within 20% of a minimum requirement, produce `warn`; otherwise monitoring
passes. Results are saved to `artifacts/evals/monitoring_summary.json` and
`reports/monitoring_report.md`.

Gate a candidate evaluation against a baseline:

```bash
python -m llm_posttraining_ops.cli run-release-gate \
  --baseline-eval tests/fixtures/baseline_eval_gate_sample.json \
  --current-eval tests/fixtures/current_eval_gate_sample.json
```

The gate fails if required-fact coverage drops, forbidden-term violations rise,
empty responses rise, or current p95 latency exceeds `--max-p95-latency` when
that metric is present. The JSON decision is written to
`artifacts/evals/release_gate.json`, with an auditable Markdown report at
`reports/release_gate_report.md`. Failed monitoring and release gates return a
non-zero CLI exit code for CI use.

## Milestone 10

Milestone 10 connects data preparation, evaluation, optional training, and
release checks through a reproducible workflow runner. Every run has an
experiment registry, manifest, stage outcomes, and isolated artifacts.

Run the deterministic no-download workflow:

```bash
python -m llm_posttraining_ops.cli run-demo-workflow \
  --run-id smoke \
  --skip-model \
  --skip-sft \
  --skip-dpo
```

This still executes local SFT and preference ingestion, validation, profiling,
model-free baseline evaluation, the rigorous evaluation suite, and the release
gate. It skips only Hugging Face inference and training.

Run the tiny base-model path without training:

```bash
python -m llm_posttraining_ops.cli run-demo-workflow \
  --run-id tiny-model \
  --skip-sft \
  --skip-dpo
```

Run all stages, including one-step SFT and DPO:

```bash
python -m llm_posttraining_ops.cli run-demo-workflow \
  --run-id full-tiny \
  --model-name sshleifer/tiny-gpt2 \
  --seed 42
```

Run artifacts are isolated under `artifacts/runs/<run_id>/`:

```text
artifacts/runs/<run_id>/
├── data/
│   ├── sft/sft.jsonl
│   └── preferences/preference.jsonl
├── evals/
│   ├── generations/
│   ├── baseline_eval.json
│   ├── eval_suite.json
│   ├── release_gate.json
│   ├── sft_profile.json
│   └── preference_profile.json
├── models/
├── reports/
├── experiment_registry.json
├── reproducibility_manifest.json
├── workflow_summary.json
└── workflow_report.md
```

The manifest records package, Python, platform, dependency and Git versions;
the seed and run ID; model names/checkpoints; and source/normalized data paths.
The final report is also written to `reports/workflow_report.md`.

Stages stop after the first failure by default, while the registry, manifest,
summary, and report are still finalized. Pass `--continue-on-error` to attempt
later stages and return a zero CLI exit code after recording failures:

```bash
python -m llm_posttraining_ops.cli run-demo-workflow \
  --run-id diagnostic \
  --continue-on-error \
  --skip-model \
  --skip-sft \
  --skip-dpo
```
