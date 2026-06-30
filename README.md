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
