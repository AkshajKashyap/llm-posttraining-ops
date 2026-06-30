#!/usr/bin/env bash
set -euo pipefail

RUN_ID="${RUN_ID:-workflow-smoke}"
PYTHON="${PYTHON:-python}"

"${PYTHON}" -m llm_posttraining_ops.cli run-demo-workflow \
  --run-id "${RUN_ID}" \
  --skip-model \
  --skip-sft \
  --skip-dpo
