"""Mockable Hugging Face Trainer pipeline for supervised fine-tuning."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from llm_posttraining_ops.data.ingestion import load_normalized_sft
from llm_posttraining_ops.data.schemas import SFTRecord
from llm_posttraining_ops.training.config import SFTTrainingConfig
from llm_posttraining_ops.training.tokenization import tokenize_sft_record

SFT_SUMMARY_SCHEMA_VERSION = "1.0"
DEFAULT_SFT_SUMMARY_PATH = Path("artifacts/evals/sft_training_summary.json")


class SFTTrainingError(RuntimeError):
    """Raised when the Hugging Face training backend cannot run."""


@dataclass(frozen=True, slots=True)
class BackendTrainingResult:
    """Artifacts and metrics returned by a training backend."""

    checkpoint_path: Path
    metrics: dict[str, float]
    trainable_parameters: int
    total_parameters: int


class TrainingBackend(Protocol):
    """Backend interface used to keep unit tests model-free."""

    def train(
        self,
        records: Sequence[SFTRecord],
        config: SFTTrainingConfig,
    ) -> BackendTrainingResult:
        """Train and save a model or adapter."""


@dataclass(frozen=True, slots=True)
class SFTTrainingSummary:
    """Versioned summary for one SFT run."""

    schema_version: str
    dataset_path: str
    training_record_count: int
    model_name: str
    checkpoint_path: str
    use_lora: bool
    settings: dict[str, str | int | float | bool]
    metrics: dict[str, float]
    trainable_parameters: int
    total_parameters: int

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable summary."""

        return {
            "schema_version": self.schema_version,
            "dataset_path": self.dataset_path,
            "training_record_count": self.training_record_count,
            "model_name": self.model_name,
            "checkpoint_path": self.checkpoint_path,
            "use_lora": self.use_lora,
            "settings": self.settings,
            "metrics": self.metrics,
            "trainable_parameters": self.trainable_parameters,
            "total_parameters": self.total_parameters,
        }


class HuggingFaceTrainerBackend:
    """CPU-first Transformers Trainer backend with optional PEFT LoRA."""

    def train(
        self,
        records: Sequence[SFTRecord],
        config: SFTTrainingConfig,
    ) -> BackendTrainingResult:
        try:
            from transformers import (
                AutoModelForCausalLM,
                AutoTokenizer,
                Trainer,
                TrainingArguments,
                default_data_collator,
                set_seed,
            )
        except ImportError as exc:
            raise SFTTrainingError(
                "SFT requires the 'torch', 'transformers', and 'accelerate' packages"
            ) from exc

        set_seed(config.seed)
        try:
            tokenizer = AutoTokenizer.from_pretrained(config.model_name)
            model = AutoModelForCausalLM.from_pretrained(config.model_name)
        except (OSError, ValueError) as exc:
            raise SFTTrainingError(
                f"Unable to load training model '{config.model_name}': {exc}"
            ) from exc

        if tokenizer.pad_token_id is None:
            if tokenizer.eos_token is None:
                raise SFTTrainingError("Tokenizer must define an EOS or padding token")
            tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.pad_token_id
        model.config.use_cache = False

        if config.use_lora:
            try:
                from peft import LoraConfig, TaskType, get_peft_model
            except ImportError as exc:
                raise SFTTrainingError("LoRA training requires the 'peft' package") from exc
            model = get_peft_model(
                model,
                LoraConfig(
                    task_type=TaskType.CAUSAL_LM,
                    r=config.lora_r,
                    lora_alpha=config.lora_alpha,
                    lora_dropout=config.lora_dropout,
                    bias="none",
                ),
            )

        tokenized_records = [
            tokenize_sft_record(
                record,
                tokenizer,
                max_seq_length=config.max_seq_length,
            )
            for record in records
        ]
        output_dir = config.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        arguments = TrainingArguments(
            output_dir=str(output_dir),
            overwrite_output_dir=True,
            max_steps=config.max_steps,
            learning_rate=config.learning_rate,
            per_device_train_batch_size=config.batch_size,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            logging_steps=1,
            save_strategy="no",
            report_to=[],
            seed=config.seed,
            data_seed=config.seed,
            use_cpu=True,
            disable_tqdm=True,
            dataloader_num_workers=0,
            remove_unused_columns=False,
        )
        trainer = Trainer(
            model=model,
            args=arguments,
            train_dataset=tokenized_records,
            data_collator=default_data_collator,
        )
        try:
            train_output = trainer.train()
            trainer.save_model(str(output_dir))
            tokenizer.save_pretrained(output_dir)
        except (RuntimeError, ValueError) as exc:
            raise SFTTrainingError(f"SFT training failed: {exc}") from exc

        trainable_parameters = sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        )
        total_parameters = sum(parameter.numel() for parameter in model.parameters())
        metrics = {
            key: float(value)
            for key, value in train_output.metrics.items()
            if isinstance(value, (int, float))
        }
        metrics.setdefault("training_loss", float(train_output.training_loss))
        return BackendTrainingResult(
            checkpoint_path=output_dir,
            metrics=metrics,
            trainable_parameters=trainable_parameters,
            total_parameters=total_parameters,
        )


def write_sft_summary(summary: SFTTrainingSummary, path: str | Path) -> Path:
    """Write the training summary as stable JSON."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as output_file:
        json.dump(summary.to_dict(), output_file, indent=2, sort_keys=True)
        output_file.write("\n")
    return output_path


def run_sft_training(
    data_dir: str | Path,
    config: SFTTrainingConfig,
    *,
    summary_path: str | Path = DEFAULT_SFT_SUMMARY_PATH,
    backend: TrainingBackend | None = None,
) -> SFTTrainingSummary:
    """Train on the normalized train split and save a summary artifact."""

    sft_path = Path(data_dir) / "sft.jsonl"
    all_records = load_normalized_sft(sft_path)
    training_records = [record for record in all_records if record.split == "train"]
    if not training_records:
        raise SFTTrainingError("SFT data must contain at least one train record")

    result = (backend or HuggingFaceTrainerBackend()).train(training_records, config)
    summary = SFTTrainingSummary(
        schema_version=SFT_SUMMARY_SCHEMA_VERSION,
        dataset_path=str(sft_path),
        training_record_count=len(training_records),
        model_name=config.model_name,
        checkpoint_path=str(result.checkpoint_path),
        use_lora=config.use_lora,
        settings=config.to_dict(),
        metrics=result.metrics,
        trainable_parameters=result.trainable_parameters,
        total_parameters=result.total_parameters,
    )
    write_sft_summary(summary, summary_path)
    return summary
