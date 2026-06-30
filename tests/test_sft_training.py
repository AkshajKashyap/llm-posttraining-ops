import json
from pathlib import Path
from typing import Sequence

from llm_posttraining_ops.data.jsonl import write_jsonl
from llm_posttraining_ops.data.schemas import SFTRecord
from llm_posttraining_ops.training.config import SFTTrainingConfig
from llm_posttraining_ops.training.sft import (
    BackendTrainingResult,
    run_sft_training,
)


class FakeTrainingBackend:
    def __init__(self, checkpoint_path: Path) -> None:
        self.checkpoint_path = checkpoint_path
        self.records: Sequence[SFTRecord] = []

    def train(
        self,
        records: Sequence[SFTRecord],
        config: SFTTrainingConfig,
    ) -> BackendTrainingResult:
        self.records = records
        assert config.max_steps == 1
        return BackendTrainingResult(
            checkpoint_path=self.checkpoint_path,
            metrics={"training_loss": 2.5, "train_runtime": 0.25},
            trainable_parameters=100,
            total_parameters=200,
        )


def test_mocked_training_writes_summary(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    write_jsonl(
        data_dir / "sft.jsonl",
        [
            {
                "id": "train",
                "split": "train",
                "instruction": "Give a greeting.",
                "input": "",
                "output": "Hello there.",
                "source": "test",
                "metadata": {},
            },
            {
                "id": "test",
                "split": "test",
                "instruction": "Say goodbye.",
                "input": "",
                "output": "Goodbye now.",
                "source": "test",
                "metadata": {},
            },
        ],
    )
    backend = FakeTrainingBackend(tmp_path / "checkpoint")
    summary_path = tmp_path / "summary.json"

    summary = run_sft_training(
        data_dir,
        SFTTrainingConfig(model_name="mock/model", output_dir=tmp_path / "model"),
        summary_path=summary_path,
        backend=backend,
    )
    payload = json.loads(summary_path.read_text(encoding="utf-8"))

    assert [record.id for record in backend.records] == ["train"]
    assert summary.training_record_count == 1
    assert payload["checkpoint_path"] == str(tmp_path / "checkpoint")
    assert payload["metrics"]["training_loss"] == 2.5
    assert payload["trainable_parameters"] == 100
