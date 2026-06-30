#!/usr/bin/env bash
set -euo pipefail

IMAGE="${IMAGE:-llm-posttraining-ops:smoke}"
CONTAINER="${CONTAINER:-llm-posttraining-ops-smoke}"
PORT="${PORT:-18000}"
BASE_URL="http://127.0.0.1:${PORT}"

cleanup() {
  docker rm -f "${CONTAINER}" >/dev/null 2>&1 || true
}
trap cleanup EXIT

docker build -t "${IMAGE}" .
docker run --rm -d \
  --name "${CONTAINER}" \
  -p "${PORT}:8000" \
  "${IMAGE}" >/dev/null

for _ in $(seq 1 30); do
  if curl -fsS "${BASE_URL}/health" >/dev/null; then
    break
  fi
  sleep 1
done

curl -fsS "${BASE_URL}/health"
printf '\n'
curl -fsS "${BASE_URL}/model-info"
printf '\n'
curl -fsS -X POST "${BASE_URL}/generate" \
  -H "Content-Type: application/json" \
  -d '{"instruction":"Explain SFT in one sentence.","input":"","max_new_tokens":32}'
printf '\n'
