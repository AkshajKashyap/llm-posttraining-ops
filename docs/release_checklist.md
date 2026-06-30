# Release Checklist

## Version 0.1.0 automated checks

- [x] Package and CLI report version `0.1.0`.
- [x] Python 3.11 compatibility is declared.
- [x] Ruff passes.
- [x] Pytest passes without GPU or model downloads.
- [x] Deterministic workflow smoke passes with model and training stages skipped.
- [x] Generated logs, runs, checkpoints, and adapters are ignored.
- [x] FastAPI mock serving has endpoint and error-path tests.
- [x] Monitoring and release gates have passing and failing tests.
- [x] GitHub Actions runs lint, tests, and workflow smoke.
- [x] CPU/mock Docker build and smoke scripts are present.
- [x] Architecture, model, evaluation, interview, and limitation docs are present.

## Manual checks before tagging

- [ ] Confirm the working tree contains only intended release files.
- [ ] Confirm `CHANGELOG.md` matches the tag.
- [ ] Run `scripts/run_workflow_smoke.sh`.
- [ ] Run `scripts/docker_smoke_test.sh` where Docker is available.
- [ ] Inspect the workflow registry, manifest, and release-gate report.
- [ ] Confirm no credentials, private data, model weights, or generated logs are tracked.
- [ ] Create and push tag `v0.1.0`.

## Real-model release additions

- [ ] Document the exact governed training and evaluation datasets.
- [ ] Record model-license and dataset-license compatibility.
- [ ] Evaluate representative held-out slices and safety cases.
- [ ] Calibrate automated judges against human review.
- [ ] Review latency, memory, throughput, and failure behavior under load.
- [ ] Publish a checkpoint-specific model card and rollback plan.
