import json
from pathlib import Path

from llm_posttraining_ops.data.preference_ingestion import ingest_preference_data
from llm_posttraining_ops.data.preference_profiling import profile_preference_directory


def test_preference_profile(tmp_path: Path) -> None:
    data_dir = tmp_path / "preferences"
    ingest_preference_data(
        "tests/fixtures/preference_direct_sample.jsonl",
        data_dir,
        "direct",
    )
    output_path = tmp_path / "preference_profile.json"

    profile, written_path = profile_preference_directory(
        data_dir,
        output_path=output_path,
    )
    payload = json.loads(written_path.read_text(encoding="utf-8"))

    assert profile.record_count == 4
    assert profile.split_counts == {"test": 1, "train": 2, "validation": 1}
    assert profile.average_prompt_length == 10.5
    assert profile.average_chosen_length == 8.75
    assert profile.average_rejected_length == 7.5
    assert profile.chosen_rejected_length_ratio == 1.166667
    assert profile.duplicate_chosen_rate == 0.0
    assert profile.duplicate_rejected_rate == 0.0
    assert payload["source_counts"] == {"preference_direct_sample": 4}
