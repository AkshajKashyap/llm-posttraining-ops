import json
from pathlib import Path
from typing import Sequence

from llm_posttraining_ops.data.jsonl import write_jsonl
from llm_posttraining_ops.data.schemas import PreferenceRecord
from llm_posttraining_ops.training.dpo import DPOBackendResult, run_dpo_training
from llm_posttraining_ops.training.dpo_config import DPOTrainingConfig


class FakeDPOBackend:
    def __init__(self, checkpoint_path: Path) -> None:
        self.checkpoint_path = checkpoint_path
        self.records: Sequence[PreferenceRecord] = []

    def train(
        self,
        records: Sequence[PreferenceRecord],
        config: DPOTrainingConfig,
    ) -> DPOBackendResult:
        self.records = records
        assert config.beta == 0.1
        return DPOBackendResult(
            checkpoint_path=self.checkpoint_path,
            metrics={"training_loss": 0.693, "train_runtime": 0.2},
            trainable_parameters=50,
            total_parameters=100,
        )


def test_mocked_dpo_training_summary(tmp_path: Path) -> None:
    data_dir = tmp_path / "preferences"
    write_jsonl(
        data_dir / "preference.jsonl",
        [
            {
                "id": "train",
                "split": "train",
                "instruction": "Answer the question.",
                "input": "",
                "chosen": "This is the preferred detailed answer.",
                "rejected": "This answer contains an important error.",
                "source": "test",
                "metadata": {},
            },
            {
                "id": "test",
                "split": "test",
                "instruction": "Answer another question.",
                "input": "",
                "chosen": "Another preferred response with useful detail.",
                "rejected": "Another response with a factual mistake.",
                "source": "test",
                "metadata": {},
            },
        ],
    )
    backend = FakeDPOBackend(tmp_path / "checkpoint")
    summary_path = tmp_path / "summary.json"

    summary = run_dpo_training(
        data_dir,
        DPOTrainingConfig(model_name="mock/model", output_dir=tmp_path / "model"),
        summary_path=summary_path,
        backend=backend,
    )
    payload = json.loads(summary_path.read_text(encoding="utf-8"))

    assert [record.id for record in backend.records] == ["train"]
    assert summary.training_record_count == 1
    assert payload["starting_model"] == "mock/model"
    assert payload["metrics"]["training_loss"] == 0.693
    assert payload["trainable_parameters"] == 50
