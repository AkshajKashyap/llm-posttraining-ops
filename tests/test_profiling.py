import json
from pathlib import Path

from llm_posttraining_ops.data.ingestion import ingest_sft_data
from llm_posttraining_ops.data.profiling import profile_data_directory


def test_profile_outputs_and_dataset_card(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    ingest_sft_data("tests/fixtures/alpaca_sample.jsonl", data_dir, "alpaca")
    profile_path = tmp_path / "artifacts" / "profile.json"
    card_path = tmp_path / "reports" / "dataset_card.md"

    profile, written_profile, written_card = profile_data_directory(
        data_dir,
        profile_path=profile_path,
        card_path=card_path,
    )
    payload = json.loads(written_profile.read_text(encoding="utf-8"))
    card = written_card.read_text(encoding="utf-8")

    assert profile.record_count == 4
    assert profile.split_counts == {"test": 1, "train": 2, "validation": 1}
    assert profile.source_counts == {"alpaca_sample": 4}
    assert profile.metrics.average_instruction_length == 8.5
    assert profile.metrics.average_output_length == 7.75
    assert profile.metrics.empty_input_rate == 0.0
    assert profile.metrics.duplicate_output_rate == 0.0
    assert payload["metrics"]["top_repeated_starting_phrases"] == [
        {"count": 2, "phrase": "a practical benefit"}
    ]
    assert card.startswith("# SFT Dataset Card\n")
    assert "Average instruction length (tokens) | 8.500" in card
    assert "`a practical benefit`: 2" in card
