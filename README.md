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

No command downloads a model or dataset.
