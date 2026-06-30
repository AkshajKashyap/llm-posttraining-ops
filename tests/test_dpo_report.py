import json
from pathlib import Path

from llm_posttraining_ops.training.dpo_report import generate_dpo_report


def _evaluation(exact_match: float, seconds: float) -> dict[str, object]:
    return {
        "metrics": {
            "exact_match": exact_match,
            "token_overlap_f1": exact_match,
            "contains_expected_key_terms": exact_match,
            "average_response_length": 5.0,
            "empty_response_rate": 0.0,
        },
        "latency": {
            "total_generation_seconds": seconds,
            "average_seconds_per_example": seconds / 2,
            "average_generated_tokens": 5.0,
        },
    }


def test_generate_dpo_report_with_optional_sft_row(tmp_path: Path) -> None:
    summary_path = tmp_path / "summary.json"
    base_path = tmp_path / "base.json"
    sft_path = tmp_path / "sft.json"
    dpo_path = tmp_path / "dpo.json"
    summary_path.write_text(
        json.dumps(
            {
                "starting_model": "base/model",
                "checkpoint_path": "artifacts/models/dpo",
                "training_record_count": 2,
                "use_lora": False,
                "settings": {"max_steps": 1, "beta": 0.1},
                "metrics": {"training_loss": 0.69},
            }
        ),
        encoding="utf-8",
    )
    base_path.write_text(json.dumps(_evaluation(0.0, 2.0)), encoding="utf-8")
    sft_path.write_text(json.dumps(_evaluation(0.25, 1.8)), encoding="utf-8")
    dpo_path.write_text(json.dumps(_evaluation(0.5, 1.5)), encoding="utf-8")

    report_path = generate_dpo_report(
        summary_path=summary_path,
        base_evaluation_path=base_path,
        sft_evaluation_path=sft_path,
        dpo_evaluation_path=dpo_path,
        output_path=tmp_path / "dpo_report.md",
    )
    report = report_path.read_text(encoding="utf-8")

    assert report.startswith("# Direct Preference Optimization Report\n")
    assert "DPO loss: 0.690" in report
    assert "| Base | 0.000" in report
    assert "| SFT | 0.250" in report
    assert "| DPO | 0.500" in report
