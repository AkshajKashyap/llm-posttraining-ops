.PHONY: install test lint prepare-demo workflow-smoke serve-mock clean-artifacts

PYTHON ?= python

install:
	$(PYTHON) -m pip install -e ".[dev]"

test:
	$(PYTHON) -m pytest -q

lint:
	$(PYTHON) -m ruff check .

prepare-demo:
	$(PYTHON) -m llm_posttraining_ops.cli prepare-demo-data

workflow-smoke:
	scripts/run_workflow_smoke.sh

serve-mock:
	$(PYTHON) -m llm_posttraining_ops.cli serve --mock --host 0.0.0.0 --port 8000

clean-artifacts:
	rm -rf artifacts/logs artifacts/models artifacts/adapters artifacts/runs
	rm -f artifacts/evals/monitoring_summary.json artifacts/evals/release_gate.json
