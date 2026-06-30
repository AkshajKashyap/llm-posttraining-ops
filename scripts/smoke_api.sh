#!/usr/bin/env bash
set -euo pipefail

BASE_URL="${BASE_URL:-http://localhost:8000}"

curl -fsS "${BASE_URL}/health"
printf '\n'
curl -fsS "${BASE_URL}/model-info"
printf '\n'
curl -fsS -X POST "${BASE_URL}/generate" \
  -H "Content-Type: application/json" \
  -d '{"instruction":"Explain SFT in one sentence.","input":"","max_new_tokens":32}'
printf '\n'
