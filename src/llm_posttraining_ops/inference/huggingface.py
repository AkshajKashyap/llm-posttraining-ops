"""Hugging Face causal language-model generation with lazy dependency loading."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from llm_posttraining_ops.inference.config import GenerationSettings


class ModelInferenceError(RuntimeError):
    """Raised when a Hugging Face model cannot be loaded or used."""


@dataclass(frozen=True, slots=True)
class GenerationOutput:
    """Decoded text plus generated-token count."""

    text: str
    generated_tokens: int


class HuggingFaceCausalLMGenerator:
    """CPU-first causal LM generator backed by Transformers."""

    def __init__(
        self,
        model_name: str,
        *,
        device: str = "cpu",
        tokenizer: Any | None = None,
        model: Any | None = None,
        torch_module: Any | None = None,
        seed_function: Callable[[int], None] | None = None,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self._tokenizer = tokenizer
        self._model = model
        self._torch = torch_module
        self._seed_function = seed_function

    def _ensure_loaded(self) -> None:
        if self._tokenizer is not None and self._model is not None and self._torch is not None:
            return
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
        except ImportError as exc:
            raise ModelInferenceError(
                "Model inference requires the 'torch' and 'transformers' packages"
            ) from exc

        try:
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModelForCausalLM.from_pretrained(self.model_name)
            self._model.to(self.device)
            self._model.eval()
        except (OSError, ValueError) as exc:
            raise ModelInferenceError(
                f"Unable to load Hugging Face model '{self.model_name}': {exc}"
            ) from exc
        self._torch = torch
        self._seed_function = set_seed

    def generate(self, prompt: str, settings: GenerationSettings) -> GenerationOutput:
        """Generate one continuation and return only newly generated text."""

        self._ensure_loaded()
        assert self._tokenizer is not None
        assert self._model is not None
        assert self._torch is not None

        if self._seed_function is not None:
            self._seed_function(settings.seed)
        encoded = self._tokenizer(prompt, return_tensors="pt")
        encoded = {name: tensor.to(self.device) for name, tensor in encoded.items()}
        prompt_tokens = int(encoded["input_ids"].shape[-1])
        pad_token_id = self._tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = self._tokenizer.eos_token_id

        generation_kwargs: dict[str, Any] = {
            "max_new_tokens": settings.max_new_tokens,
            "do_sample": settings.do_sample,
            "pad_token_id": pad_token_id,
        }
        if settings.do_sample:
            generation_kwargs.update(
                temperature=settings.temperature,
                top_p=settings.top_p,
            )

        with self._torch.inference_mode():
            output_ids = self._model.generate(**encoded, **generation_kwargs)
        generated_ids = output_ids[0, prompt_tokens:]
        text = self._tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        return GenerationOutput(
            text=text,
            generated_tokens=int(generated_ids.shape[-1]),
        )
