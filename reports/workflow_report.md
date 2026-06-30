# End-to-End Workflow Report

- Run ID: `smoke`
- Status: **PASS**
- Run directory: `artifacts/runs/smoke`
- Started: 2026-06-30T17:33:25.802264+00:00
- Ended: 2026-06-30T17:33:25.832858+00:00
- Reproducibility manifest: `artifacts/runs/smoke/reproducibility_manifest.json`
- Workflow summary: `artifacts/runs/smoke/workflow_summary.json`

## Stage results

| Stage | Status | Artifacts | Error or reason |
| --- | --- | ---: | --- |
| ingest_sft_data | pass | 1 |  |
| ingest_preference_data | pass | 1 |  |
| validate_sft_data | pass | 1 |  |
| profile_sft_data | pass | 2 |  |
| profile_preference_data | pass | 1 |  |
| baseline_evaluation | pass | 2 |  |
| base_model_evaluation | skipped | 0 | disabled by --skip-model |
| sft_training | skipped | 0 | disabled by --skip-sft |
| dpo_training | skipped | 0 | disabled by --skip-dpo |
| evaluation_suite | pass | 2 |  |
| release_gate | pass | 2 |  |

## Artifact registry

| Name | Path |
| --- | --- |
| baseline_evaluation | `artifacts/runs/smoke/evals/baseline_eval.json` |
| baseline_generations | `artifacts/runs/smoke/evals/generations/keyword_rule.jsonl` |
| evaluation_suite | `artifacts/runs/smoke/evals/eval_suite.json` |
| evaluation_suite_report | `artifacts/runs/smoke/reports/eval_suite_report.md` |
| experiment_registry | `artifacts/runs/smoke/experiment_registry.json` |
| preference_data | `artifacts/runs/smoke/data/preferences/preference.jsonl` |
| preference_profile | `artifacts/runs/smoke/evals/preference_profile.json` |
| release_gate | `artifacts/runs/smoke/evals/release_gate.json` |
| release_gate_report | `artifacts/runs/smoke/reports/release_gate_report.md` |
| reproducibility_manifest | `artifacts/runs/smoke/reproducibility_manifest.json` |
| run_workflow_report | `artifacts/runs/smoke/workflow_report.md` |
| sft_data | `artifacts/runs/smoke/data/sft/sft.jsonl` |
| sft_dataset_card | `artifacts/runs/smoke/reports/sft_dataset_card.md` |
| sft_profile | `artifacts/runs/smoke/evals/sft_profile.json` |
| validated_sft_data | `artifacts/runs/smoke/data/sft/sft.jsonl` |
| workflow_report | `reports/workflow_report.md` |
| workflow_summary | `artifacts/runs/smoke/workflow_summary.json` |

Skipped stages are explicit and do not fail the workflow. Any failed
stage marks the final workflow result as failed.
