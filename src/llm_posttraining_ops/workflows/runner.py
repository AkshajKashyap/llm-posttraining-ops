"""End-to-end demo workflow built from existing post-training modules."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

from llm_posttraining_ops.data.ingestion import ingest_sft_data
from llm_posttraining_ops.data.jsonl import write_jsonl
from llm_posttraining_ops.data.preference_ingestion import ingest_preference_data
from llm_posttraining_ops.data.preference_profiling import (
    profile_preference_directory,
)
from llm_posttraining_ops.data.profiling import profile_data_directory
from llm_posttraining_ops.data.validation import validate_data_directory
from llm_posttraining_ops.evaluation.evaluator import run_baseline_evaluation
from llm_posttraining_ops.evaluation.suite import (
    evaluate_generation_file,
    write_eval_suite_result,
)
from llm_posttraining_ops.evaluation.suite_reports import write_eval_suite_report
from llm_posttraining_ops.inference.config import DEFAULT_MODEL_NAME, GenerationSettings
from llm_posttraining_ops.inference.evaluation import run_model_evaluation
from llm_posttraining_ops.monitoring.release_gate import run_release_gate
from llm_posttraining_ops.monitoring.reports import write_release_gate_report
from llm_posttraining_ops.training.config import SFTTrainingConfig
from llm_posttraining_ops.training.dpo import run_dpo_training
from llm_posttraining_ops.training.dpo_config import DPOTrainingConfig
from llm_posttraining_ops.training.sft import run_sft_training
from llm_posttraining_ops.workflows.manifest import (
    build_reproducibility_manifest,
    write_reproducibility_manifest,
)
from llm_posttraining_ops.workflows.registry import (
    ExperimentRegistry,
    RunStatus,
    StageRecord,
    create_run_id,
    utc_timestamp,
    validate_run_id,
)
from llm_posttraining_ops.workflows.report import (
    DEFAULT_WORKFLOW_REPORT_PATH,
    write_workflow_report,
)

WORKFLOW_SCHEMA_VERSION = "1.0"
DEFAULT_RUNS_DIR = Path("artifacts/runs")
DEFAULT_SFT_FIXTURE = Path("tests/fixtures/alpaca_sample.jsonl")
DEFAULT_PREFERENCE_FIXTURE = Path("tests/fixtures/preference_direct_sample.jsonl")
DEFAULT_EVAL_DATA_FIXTURE = Path("tests/fixtures/eval_suite_sample.jsonl")
DEFAULT_GATE_BASELINE_FIXTURE = Path(
    "tests/fixtures/baseline_eval_gate_sample.json"
)
DEFAULT_GATE_CURRENT_FIXTURE = Path("tests/fixtures/current_eval_gate_sample.json")

STAGE_NAMES = (
    "ingest_sft_data",
    "ingest_preference_data",
    "validate_sft_data",
    "profile_sft_data",
    "profile_preference_data",
    "baseline_evaluation",
    "base_model_evaluation",
    "sft_training",
    "dpo_training",
    "evaluation_suite",
    "release_gate",
)


@dataclass(frozen=True, slots=True)
class DemoWorkflowConfig:
    """Settings and local fixture inputs for the demo workflow."""

    run_id: str | None = None
    output_dir: Path = DEFAULT_RUNS_DIR
    seed: int = 42
    model_name: str = DEFAULT_MODEL_NAME
    skip_model: bool = False
    skip_sft: bool = False
    skip_dpo: bool = False
    continue_on_error: bool = False
    sft_input_path: Path = DEFAULT_SFT_FIXTURE
    preference_input_path: Path = DEFAULT_PREFERENCE_FIXTURE
    eval_data_path: Path = DEFAULT_EVAL_DATA_FIXTURE
    gate_baseline_eval_path: Path = DEFAULT_GATE_BASELINE_FIXTURE
    gate_current_eval_path: Path = DEFAULT_GATE_CURRENT_FIXTURE
    report_path: Path = DEFAULT_WORKFLOW_REPORT_PATH

    def __post_init__(self) -> None:
        if self.run_id is not None:
            validate_run_id(self.run_id)
        if self.seed < 0:
            raise ValueError("seed must be non-negative")
        if not self.model_name.strip():
            raise ValueError("model_name must be non-empty")

    def settings(self, resolved_run_id: str) -> dict[str, Any]:
        """Return all command-relevant settings for the registry."""

        return {
            "run_id": resolved_run_id,
            "output_dir": str(self.output_dir),
            "seed": self.seed,
            "model_name": self.model_name,
            "skip_model": self.skip_model,
            "skip_sft": self.skip_sft,
            "skip_dpo": self.skip_dpo,
            "continue_on_error": self.continue_on_error,
            "sft_input_path": str(self.sft_input_path),
            "preference_input_path": str(self.preference_input_path),
            "eval_data_path": str(self.eval_data_path),
            "gate_baseline_eval_path": str(self.gate_baseline_eval_path),
            "gate_current_eval_path": str(self.gate_current_eval_path),
            "report_path": str(self.report_path),
        }


@dataclass(frozen=True, slots=True)
class WorkflowResult:
    """Final workflow outcome and the durable artifact locations."""

    schema_version: str
    run_id: str
    status: RunStatus
    run_dir: str
    registry_path: str
    manifest_path: str
    summary_path: str
    report_path: str
    stages: list[StageRecord]
    artifacts: dict[str, str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "status": self.status,
            "run_dir": self.run_dir,
            "registry_path": self.registry_path,
            "manifest_path": self.manifest_path,
            "summary_path": self.summary_path,
            "report_path": self.report_path,
            "stages": [asdict(stage) for stage in self.stages],
            "artifacts": dict(sorted(self.artifacts.items())),
        }


def _write_workflow_summary(result: WorkflowResult, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as output_file:
        json.dump(result.to_dict(), output_file, indent=2, sort_keys=True)
        output_file.write("\n")
    return path


def _baseline_generations(
    evaluation: Any,
    path: Path,
) -> Path:
    selected = next(
        baseline for baseline in evaluation.baselines if baseline.name == "keyword_rule"
    )
    return write_jsonl(
        path,
        [
            {
                "id": example.id,
                "generated_response": example.generated_response,
            }
            for example in selected.examples
        ],
    )


def run_demo_workflow(
    config: DemoWorkflowConfig | None = None,
    *,
    timestamp_factory: Callable[[], str] = utc_timestamp,
) -> WorkflowResult:
    """Run the local demo pipeline while durably recording every stage."""

    active_config = config or DemoWorkflowConfig()
    run_id = active_config.run_id or create_run_id()
    run_dir = active_config.output_dir / run_id
    registry_path = run_dir / "experiment_registry.json"
    manifest_path = run_dir / "reproducibility_manifest.json"
    summary_path = run_dir / "workflow_summary.json"
    local_report_path = run_dir / "workflow_report.md"
    sft_data_dir = run_dir / "data" / "sft"
    preference_data_dir = run_dir / "data" / "preferences"
    evals_dir = run_dir / "evals"
    reports_dir = run_dir / "reports"
    generations_dir = evals_dir / "generations"
    models_dir = run_dir / "models"
    for directory in (
        sft_data_dir,
        preference_data_dir,
        generations_dir,
        reports_dir,
        models_dir,
    ):
        directory.mkdir(parents=True, exist_ok=True)

    registry = ExperimentRegistry(
        run_id,
        registry_path,
        settings=active_config.settings(run_id),
        timestamp_factory=timestamp_factory,
    )
    state: dict[str, str | None] = {
        "sft_data": None,
        "preference_data": None,
        "baseline_generations": None,
        "model_generations": None,
        "sft_checkpoint": None,
        "dpo_checkpoint": None,
    }
    aborted = False

    def execute_stage(
        name: str,
        action: Callable[[], Mapping[str, str]],
        *,
        skip_reason: str | None = None,
    ) -> None:
        nonlocal aborted
        if aborted:
            registry.skip_stage(name, "blocked by an earlier failed stage")
            return
        if skip_reason is not None:
            registry.skip_stage(name, skip_reason)
            return

        stage = registry.begin_stage(name)
        try:
            artifacts = dict(action())
        except Exception as exc:
            registry.finish_stage(
                stage,
                "fail",
                error=f"{type(exc).__name__}: {exc}",
            )
            if not active_config.continue_on_error:
                aborted = True
        else:
            registry.finish_stage(stage, "pass", artifacts=artifacts)

    def ingest_sft() -> Mapping[str, str]:
        path, _ = ingest_sft_data(
            active_config.sft_input_path,
            sft_data_dir,
            "alpaca",
        )
        state["sft_data"] = str(path)
        return {"sft_data": str(path)}

    def ingest_preferences() -> Mapping[str, str]:
        path, _ = ingest_preference_data(
            active_config.preference_input_path,
            preference_data_dir,
            "direct",
        )
        state["preference_data"] = str(path)
        return {"preference_data": str(path)}

    def validate_sft() -> Mapping[str, str]:
        validate_data_directory(sft_data_dir)
        return {"validated_sft_data": str(sft_data_dir / "sft.jsonl")}

    def profile_sft() -> Mapping[str, str]:
        _, profile_path, card_path = profile_data_directory(
            sft_data_dir,
            profile_path=evals_dir / "sft_profile.json",
            card_path=reports_dir / "sft_dataset_card.md",
        )
        return {
            "sft_profile": str(profile_path),
            "sft_dataset_card": str(card_path),
        }

    def profile_preferences() -> Mapping[str, str]:
        _, profile_path = profile_preference_directory(
            preference_data_dir,
            output_path=evals_dir / "preference_profile.json",
        )
        return {"preference_profile": str(profile_path)}

    def evaluate_baselines() -> Mapping[str, str]:
        evaluation_path = evals_dir / "baseline_eval.json"
        result = run_baseline_evaluation(sft_data_dir, evaluation_path)
        generations_path = _baseline_generations(
            result,
            generations_dir / "keyword_rule.jsonl",
        )
        state["baseline_generations"] = str(generations_path)
        return {
            "baseline_evaluation": str(evaluation_path),
            "baseline_generations": str(generations_path),
        }

    def evaluate_base_model() -> Mapping[str, str]:
        evaluation_path = evals_dir / "base_model_eval.json"
        generations_path = generations_dir / "base_model.jsonl"
        run_model_evaluation(
            sft_data_dir,
            GenerationSettings(
                model_name=active_config.model_name,
                max_new_tokens=32,
                temperature=0.0,
                top_p=1.0,
                seed=active_config.seed,
            ),
            output_path=evaluation_path,
            generations_path=generations_path,
        )
        state["model_generations"] = str(generations_path)
        return {
            "base_model_evaluation": str(evaluation_path),
            "base_model_generations": str(generations_path),
        }

    def train_sft() -> Mapping[str, str]:
        summary = run_sft_training(
            sft_data_dir,
            SFTTrainingConfig(
                model_name=active_config.model_name,
                output_dir=models_dir / "sft",
                max_steps=1,
                seed=active_config.seed,
            ),
            summary_path=evals_dir / "sft_training_summary.json",
        )
        state["sft_checkpoint"] = summary.checkpoint_path
        return {
            "sft_checkpoint": summary.checkpoint_path,
            "sft_training_summary": str(evals_dir / "sft_training_summary.json"),
        }

    def train_dpo() -> Mapping[str, str]:
        sft_path = state["sft_checkpoint"]
        summary = run_dpo_training(
            preference_data_dir,
            DPOTrainingConfig(
                model_name=active_config.model_name,
                sft_model_path=Path(sft_path) if sft_path is not None else None,
                output_dir=models_dir / "dpo",
                max_steps=1,
                seed=active_config.seed,
            ),
            summary_path=evals_dir / "dpo_training_summary.json",
        )
        state["dpo_checkpoint"] = summary.checkpoint_path
        return {
            "dpo_checkpoint": summary.checkpoint_path,
            "dpo_training_summary": str(evals_dir / "dpo_training_summary.json"),
        }

    def run_suite() -> Mapping[str, str]:
        generations = state["model_generations"] or state["baseline_generations"]
        if generations is None:
            raise ValueError("No generation artifact is available for the eval suite")
        result = evaluate_generation_file(generations, active_config.eval_data_path)
        result_path = write_eval_suite_result(result, evals_dir / "eval_suite.json")
        report_path = write_eval_suite_report(
            result,
            reports_dir / "eval_suite_report.md",
        )
        return {
            "evaluation_suite": str(result_path),
            "evaluation_suite_report": str(report_path),
        }

    def gate_release() -> Mapping[str, str]:
        result = run_release_gate(
            active_config.gate_baseline_eval_path,
            active_config.gate_current_eval_path,
            output_path=evals_dir / "release_gate.json",
        )
        report_path = write_release_gate_report(
            result,
            reports_dir / "release_gate_report.md",
        )
        if result.status != "pass":
            raise ValueError("Release gate did not pass")
        return {
            "release_gate": str(evals_dir / "release_gate.json"),
            "release_gate_report": str(report_path),
        }

    execute_stage("ingest_sft_data", ingest_sft)
    execute_stage("ingest_preference_data", ingest_preferences)
    execute_stage("validate_sft_data", validate_sft)
    execute_stage("profile_sft_data", profile_sft)
    execute_stage("profile_preference_data", profile_preferences)
    execute_stage("baseline_evaluation", evaluate_baselines)
    execute_stage(
        "base_model_evaluation",
        evaluate_base_model,
        skip_reason="disabled by --skip-model" if active_config.skip_model else None,
    )
    execute_stage(
        "sft_training",
        train_sft,
        skip_reason="disabled by --skip-sft" if active_config.skip_sft else None,
    )
    execute_stage(
        "dpo_training",
        train_dpo,
        skip_reason="disabled by --skip-dpo" if active_config.skip_dpo else None,
    )
    execute_stage("evaluation_suite", run_suite)
    execute_stage("release_gate", gate_release)

    final_status: Literal["pass", "fail"] = (
        "fail" if any(stage.status == "fail" for stage in registry.stages) else "pass"
    )
    manifest = build_reproducibility_manifest(
        run_id=run_id,
        seed=active_config.seed,
        models={
            "base_model": active_config.model_name,
            "sft_model": state["sft_checkpoint"],
            "dpo_model": state["dpo_checkpoint"],
        },
        data_paths={
            "sft_input": str(active_config.sft_input_path),
            "normalized_sft": str(state["sft_data"] or sft_data_dir / "sft.jsonl"),
            "preference_input": str(active_config.preference_input_path),
            "normalized_preference": str(
                state["preference_data"]
                or preference_data_dir / "preference.jsonl"
            ),
            "evaluation_data": str(active_config.eval_data_path),
            "release_gate_baseline": str(active_config.gate_baseline_eval_path),
            "release_gate_current": str(active_config.gate_current_eval_path),
        },
    )
    write_reproducibility_manifest(manifest, manifest_path)
    registry.add_artifacts(
        {
            "experiment_registry": str(registry_path),
            "reproducibility_manifest": str(manifest_path),
            "workflow_summary": str(summary_path),
            "workflow_report": str(active_config.report_path),
            "run_workflow_report": str(local_report_path),
        }
    )
    registry.finalize(final_status)

    result = WorkflowResult(
        schema_version=WORKFLOW_SCHEMA_VERSION,
        run_id=run_id,
        status=registry.status,
        run_dir=str(run_dir),
        registry_path=str(registry_path),
        manifest_path=str(manifest_path),
        summary_path=str(summary_path),
        report_path=str(active_config.report_path),
        stages=list(registry.stages),
        artifacts=dict(registry.artifacts),
    )
    _write_workflow_summary(result, summary_path)
    write_workflow_report(
        registry,
        run_dir=run_dir,
        manifest_path=manifest_path,
        summary_path=summary_path,
        path=local_report_path,
    )
    write_workflow_report(
        registry,
        run_dir=run_dir,
        manifest_path=manifest_path,
        summary_path=summary_path,
        path=active_config.report_path,
    )
    return result
