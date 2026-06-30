import json
from pathlib import Path

from llm_posttraining_ops.data.jsonl import read_jsonl, write_jsonl
from llm_posttraining_ops.inference.config import GenerationSettings
from llm_posttraining_ops.inference.evaluation import run_model_evaluation
from llm_posttraining_ops.inference.huggingface import GenerationOutput


class FakeGenerator:
    model_name = "mock/model"
    device = "cpu"

    def __init__(self) -> None:
        self.responses = iter(
            [
                GenerationOutput(text="Paris", generated_tokens=2),
                GenerationOutput(text="", generated_tokens=0),
            ]
        )

    def generate(
        self,
        prompt: str,
        settings: GenerationSettings,
    ) -> GenerationOutput:
        assert prompt.startswith("### Instruction:")
        assert settings.model_name == self.model_name
        return next(self.responses)


def test_model_evaluation_outputs_and_latency(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    write_jsonl(
        data_dir / "sft.jsonl",
        [
            {
                "id": "one",
                "split": "train",
                "instruction": "Name France's capital.",
                "input": "",
                "output": "Paris",
                "source": "test",
                "metadata": {},
            },
            {
                "id": "two",
                "split": "test",
                "instruction": "Give a greeting.",
                "input": "",
                "output": "Hello there",
                "source": "test",
                "metadata": {},
            },
        ],
    )
    times = iter([1.0, 1.2, 2.0, 2.6])
    evaluation_path = tmp_path / "model_eval.json"
    generations_path = tmp_path / "generations.jsonl"

    result = run_model_evaluation(
        data_dir,
        GenerationSettings(model_name="mock/model", max_new_tokens=8),
        output_path=evaluation_path,
        generations_path=generations_path,
        generator=FakeGenerator(),
        clock=lambda: next(times),
    )
    payload = json.loads(evaluation_path.read_text(encoding="utf-8"))
    generations = read_jsonl(generations_path)

    assert payload["schema_version"] == "1.0"
    assert payload["dataset"]["record_count"] == 2
    assert payload["model"]["name"] == "mock/model"
    assert payload["model"]["generation"]["max_new_tokens"] == 8
    assert payload["metrics"]["exact_match"] == 0.5
    assert payload["latency"] == {
        "average_generated_tokens": 1.0,
        "average_seconds_per_example": 0.4,
        "total_generation_seconds": 0.8,
    }
    assert result.examples[0].prompt.endswith("### Response:")
    assert len(generations) == 2
    assert set(generations[0]) == {
        "expected_output",
        "generated_response",
        "generated_tokens",
        "generation_seconds",
        "id",
        "model_name",
        "prompt",
        "split",
    }
