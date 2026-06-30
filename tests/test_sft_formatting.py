from llm_posttraining_ops.data.schemas import SFTRecord
from llm_posttraining_ops.training.formatting import build_sft_text


def test_sft_prompt_contains_input_and_output_boundary() -> None:
    record = SFTRecord(
        id="one",
        split="train",
        instruction="Rewrite the sentence.",
        input="meeting starts now",
        output="The meeting begins now.",
    )

    text = build_sft_text(record)

    assert text.prompt == (
        "### Instruction:\n"
        "Rewrite the sentence.\n\n"
        "### Input:\n"
        "meeting starts now\n\n"
        "### Response:\n"
    )
    assert text.response == "The meeting begins now."
    assert text.full_text("<eos>").endswith("The meeting begins now.<eos>")
