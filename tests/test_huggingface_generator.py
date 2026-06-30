from __future__ import annotations

from contextlib import nullcontext
from typing import Any

from llm_posttraining_ops.inference.config import GenerationSettings
from llm_posttraining_ops.inference.huggingface import HuggingFaceCausalLMGenerator


class FakeInputTensor:
    shape = (1, 3)

    def to(self, device: str) -> FakeInputTensor:
        assert device == "cpu"
        return self


class FakeGeneratedTokens:
    shape = (2,)


class FakeOutputTensor:
    def __getitem__(self, key: tuple[int, slice]) -> FakeGeneratedTokens:
        row, token_slice = key
        assert row == 0
        assert token_slice.start == 3
        return FakeGeneratedTokens()


class FakeTokenizer:
    pad_token_id = None
    eos_token_id = 99

    def __call__(self, prompt: str, *, return_tensors: str) -> dict[str, FakeInputTensor]:
        assert prompt == "formatted prompt"
        assert return_tensors == "pt"
        return {"input_ids": FakeInputTensor()}

    def decode(self, tokens: FakeGeneratedTokens, *, skip_special_tokens: bool) -> str:
        assert isinstance(tokens, FakeGeneratedTokens)
        assert skip_special_tokens is True
        return " generated text "


class FakeModel:
    def __init__(self) -> None:
        self.kwargs: dict[str, Any] = {}

    def generate(self, **kwargs: Any) -> FakeOutputTensor:
        self.kwargs = kwargs
        return FakeOutputTensor()


class FakeTorch:
    @staticmethod
    def inference_mode() -> Any:
        return nullcontext()


def test_mocked_huggingface_generation() -> None:
    model = FakeModel()
    seeds: list[int] = []
    generator = HuggingFaceCausalLMGenerator(
        "mock/model",
        tokenizer=FakeTokenizer(),
        model=model,
        torch_module=FakeTorch(),
        seed_function=seeds.append,
    )
    settings = GenerationSettings(
        model_name="mock/model",
        max_new_tokens=5,
        temperature=0.0,
        top_p=0.9,
        seed=17,
    )

    output = generator.generate("formatted prompt", settings)

    assert output.text == "generated text"
    assert output.generated_tokens == 2
    assert seeds == [17]
    assert model.kwargs["max_new_tokens"] == 5
    assert model.kwargs["do_sample"] is False
    assert model.kwargs["pad_token_id"] == 99
    assert "temperature" not in model.kwargs
