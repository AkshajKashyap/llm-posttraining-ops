"""Causal-LM tokenization with prompt labels masked from loss."""

from __future__ import annotations

from typing import Any, Protocol

from llm_posttraining_ops.data.schemas import SFTRecord
from llm_posttraining_ops.training.formatting import build_sft_text

IGNORE_INDEX = -100


class SFTTokenizer(Protocol):
    """Tokenizer operations required by SFT preprocessing."""

    eos_token: str | None
    pad_token_id: int | None

    def __call__(
        self,
        text: str,
        *,
        add_special_tokens: bool,
    ) -> dict[str, Any]:
        """Tokenize one unpadded text string."""


def _token_ids(tokenizer: SFTTokenizer, text: str) -> list[int]:
    encoded = tokenizer(text, add_special_tokens=False)
    input_ids = encoded.get("input_ids")
    if not isinstance(input_ids, list) or not all(isinstance(item, int) for item in input_ids):
        raise ValueError("tokenizer must return input_ids as a list of integers")
    return input_ids


def tokenize_sft_record(
    record: SFTRecord,
    tokenizer: SFTTokenizer,
    *,
    max_seq_length: int,
) -> dict[str, list[int]]:
    """Tokenize an SFT example and mask every prompt/padding label."""

    if max_seq_length < 1:
        raise ValueError("max_seq_length must be positive")
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        raise ValueError("tokenizer.pad_token_id must be configured")

    text = build_sft_text(record)
    prompt_ids = _token_ids(tokenizer, text.prompt)
    response_ids = _token_ids(tokenizer, f"{text.response}{tokenizer.eos_token or ''}")
    input_ids = (prompt_ids + response_ids)[:max_seq_length]
    prompt_length = min(len(prompt_ids), len(input_ids))
    labels = [IGNORE_INDEX] * prompt_length + input_ids[prompt_length:]
    attention_mask = [1] * len(input_ids)

    padding_length = max_seq_length - len(input_ids)
    input_ids.extend([pad_token_id] * padding_length)
    attention_mask.extend([0] * padding_length)
    labels.extend([IGNORE_INDEX] * padding_length)
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }
