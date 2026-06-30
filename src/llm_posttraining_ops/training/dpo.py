"""Mockable TRL pipeline for direct preference optimization."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from llm_posttraining_ops.data.preference_ingestion import load_normalized_preferences
from llm_posttraining_ops.data.schemas import PreferenceRecord
from llm_posttraining_ops.inference.prompts import format_instruction_prompt
from llm_posttraining_ops.training.dpo_config import DPOTrainingConfig
from llm_posttraining_ops.training.evaluation import resolve_model_artifact

DPO_SUMMARY_SCHEMA_VERSION = "1.0"
DEFAULT_DPO_SUMMARY_PATH = Path("artifacts/evals/dpo_training_summary.json")


class DPOTrainingError(RuntimeError):
    """Raised when the TRL DPO backend cannot train."""


@dataclass(frozen=True, slots=True)
class DPOBackendResult:
    """Artifacts and metrics returned by a DPO backend."""

    checkpoint_path: Path
    metrics: dict[str, float]
    trainable_parameters: int
    total_parameters: int


class DPOBackend(Protocol):
    """Backend interface keeping tests independent from TRL."""

    def train(
        self,
        records: Sequence[PreferenceRecord],
        config: DPOTrainingConfig,
    ) -> DPOBackendResult:
        """Train and save a DPO model or adapter."""


@dataclass(frozen=True, slots=True)
class DPOTrainingSummary:
    """Versioned DPO training summary."""

    schema_version: str
    dataset_path: str
    training_record_count: int
    model_name: str
    starting_model: str
    checkpoint_path: str
    use_lora: bool
    settings: dict[str, str | int | float | bool | None]
    metrics: dict[str, float]
    trainable_parameters: int
    total_parameters: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "dataset_path": self.dataset_path,
            "training_record_count": self.training_record_count,
            "model_name": self.model_name,
            "starting_model": self.starting_model,
            "checkpoint_path": self.checkpoint_path,
            "use_lora": self.use_lora,
            "settings": self.settings,
            "metrics": self.metrics,
            "trainable_parameters": self.trainable_parameters,
            "total_parameters": self.total_parameters,
        }


def _preference_rows(records: Sequence[PreferenceRecord]) -> list[dict[str, str]]:
    return [
        {
            "prompt": f"{format_instruction_prompt(record.instruction, record.input)}\n",
            "chosen": record.chosen,
            "rejected": record.rejected,
        }
        for record in records
    ]


class TRLDPOBackend:
    """CPU-first TRL DPO backend with optional PEFT LoRA."""

    def train(
        self,
        records: Sequence[PreferenceRecord],
        config: DPOTrainingConfig,
    ) -> DPOBackendResult:
        try:
            from datasets import Dataset
            from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
            from trl import DPOConfig as TRLDPOConfig
            from trl import DPOTrainer
        except ImportError as exc:
            raise DPOTrainingError(
                "DPO requires the 'trl', 'datasets', 'transformers', and 'accelerate' packages"
            ) from exc

        set_seed(config.seed)
        model_source = config.model_name
        tokenizer_source = config.model_name
        try:
            if config.sft_model_path is not None:
                artifact = resolve_model_artifact(config.sft_model_path)
                tokenizer_source = str(artifact.path)
                if artifact.kind == "adapter":
                    assert artifact.base_model_name is not None
                    from peft import PeftModel

                    base_model = AutoModelForCausalLM.from_pretrained(
                        artifact.base_model_name
                    )
                    model = PeftModel.from_pretrained(base_model, artifact.path)
                    model = model.merge_and_unload()
                else:
                    model = AutoModelForCausalLM.from_pretrained(artifact.path)
                model_source = str(artifact.path)
            else:
                model = AutoModelForCausalLM.from_pretrained(model_source)
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_source)
        except (OSError, ValueError) as exc:
            raise DPOTrainingError(f"Unable to load DPO starting model: {exc}") from exc

        if tokenizer.pad_token_id is None:
            if tokenizer.eos_token is None:
                raise DPOTrainingError("Tokenizer must define an EOS or padding token")
            tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.pad_token_id
        model.config.use_cache = False

        peft_config = None
        if config.use_lora:
            try:
                from peft import LoraConfig, TaskType
            except ImportError as exc:
                raise DPOTrainingError("LoRA DPO requires the 'peft' package") from exc
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=config.lora_r,
                lora_alpha=config.lora_alpha,
                lora_dropout=config.lora_dropout,
                bias="none",
            )

        output_dir = config.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        arguments = TRLDPOConfig(
            output_dir=str(output_dir),
            overwrite_output_dir=True,
            max_steps=config.max_steps,
            learning_rate=config.learning_rate,
            per_device_train_batch_size=config.batch_size,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            max_length=config.max_seq_length,
            beta=config.beta,
            logging_steps=1,
            save_strategy="no",
            report_to=[],
            seed=config.seed,
            data_seed=config.seed,
            use_cpu=True,
            disable_tqdm=True,
            dataloader_num_workers=0,
            gradient_checkpointing=False,
            optim="adamw_torch",
        )
        trainer = DPOTrainer(
            model=model,
            ref_model=None,
            args=arguments,
            train_dataset=Dataset.from_list(_preference_rows(records)),
            processing_class=tokenizer,
            peft_config=peft_config,
        )
        try:
            train_output = trainer.train()
            trainer.save_model(str(output_dir))
            tokenizer.save_pretrained(output_dir)
        except (RuntimeError, ValueError) as exc:
            raise DPOTrainingError(f"DPO training failed: {exc}") from exc

        trained_model = trainer.model
        trainable_parameters = sum(
            parameter.numel()
            for parameter in trained_model.parameters()
            if parameter.requires_grad
        )
        total_parameters = sum(parameter.numel() for parameter in trained_model.parameters())
        metrics = {
            key: float(value)
            for key, value in train_output.metrics.items()
            if isinstance(value, (int, float))
        }
        metrics.setdefault("training_loss", float(train_output.training_loss))
        return DPOBackendResult(
            checkpoint_path=output_dir,
            metrics=metrics,
            trainable_parameters=trainable_parameters,
            total_parameters=total_parameters,
        )


def write_dpo_summary(summary: DPOTrainingSummary, path: str | Path) -> Path:
    """Write a stable DPO summary JSON."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as output_file:
        json.dump(summary.to_dict(), output_file, indent=2, sort_keys=True)
        output_file.write("\n")
    return output_path


def run_dpo_training(
    preference_data_dir: str | Path,
    config: DPOTrainingConfig,
    *,
    summary_path: str | Path = DEFAULT_DPO_SUMMARY_PATH,
    backend: DPOBackend | None = None,
) -> DPOTrainingSummary:
    """Train DPO on the normalized train split and save a summary."""

    preference_path = Path(preference_data_dir) / "preference.jsonl"
    all_records = load_normalized_preferences(preference_path)
    training_records = [record for record in all_records if record.split == "train"]
    if not training_records:
        raise DPOTrainingError("Preference data must contain at least one train record")

    result = (backend or TRLDPOBackend()).train(training_records, config)
    starting_model = (
        str(config.sft_model_path)
        if config.sft_model_path is not None
        else config.model_name
    )
    summary = DPOTrainingSummary(
        schema_version=DPO_SUMMARY_SCHEMA_VERSION,
        dataset_path=str(preference_path),
        training_record_count=len(training_records),
        model_name=config.model_name,
        starting_model=starting_model,
        checkpoint_path=str(result.checkpoint_path),
        use_lora=config.use_lora,
        settings=config.to_dict(),
        metrics=result.metrics,
        trainable_parameters=result.trainable_parameters,
        total_parameters=result.total_parameters,
    )
    write_dpo_summary(summary, summary_path)
    return summary
