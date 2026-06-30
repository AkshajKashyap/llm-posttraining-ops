from typing import Any

from llm_posttraining_ops.data.schemas import SFTRecord
from llm_posttraining_ops.training.formatting import build_sft_text
from llm_posttraining_ops.training.tokenization import IGNORE_INDEX, tokenize_sft_record


class CharacterTokenizer:
    eos_token = "~"
    pad_token_id = 0

    def __call__(self, text: str, *, add_special_tokens: bool) -> dict[str, Any]:
        assert add_special_tokens is False
        return {"input_ids": [ord(character) for character in text]}


def test_sft_tokenization_masks_prompt_and_padding() -> None:
    tokenizer = CharacterTokenizer()
    record = SFTRecord(
        id="one",
        split="train",
        instruction="Answer.",
        input="Input.",
        output="OK",
    )
    prompt_length = len(build_sft_text(record).prompt)
    max_length = prompt_length + 5

    tokenized = tokenize_sft_record(
        record,
        tokenizer,
        max_seq_length=max_length,
    )

    assert len(tokenized["input_ids"]) == max_length
    assert tokenized["labels"][:prompt_length] == [IGNORE_INDEX] * prompt_length
    assert tokenized["labels"][prompt_length : prompt_length + 3] == [
        ord("O"),
        ord("K"),
        ord("~"),
    ]
    assert tokenized["labels"][-2:] == [IGNORE_INDEX, IGNORE_INDEX]
    assert tokenized["attention_mask"][-2:] == [0, 0]
