import json
from pathlib import Path

from llm_posttraining_ops.training.report import generate_sft_report


def _evaluation_payload(exact_match: float, seconds: float) -> dict[str, object]:
    return {
        "metrics": {
            "exact_match": exact_match,
            "token_overlap_f1": exact_match,
            "contains_expected_key_terms": exact_match,
            "average_response_length": 4.0,
            "empty_response_rate": 0.0,
        },
        "latency": {
            "total_generation_seconds": seconds,
            "average_seconds_per_example": seconds / 2,
            "average_generated_tokens": 4.0,
        },
    }


def test_generate_sft_comparison_report(tmp_path: Path) -> None:
    summary_path = tmp_path / "summary.json"
    pre_path = tmp_path / "pre.json"
    post_path = tmp_path / "post.json"
    summary_path.write_text(
        json.dumps(
            {
                "model_name": "mock/model",
                "checkpoint_path": "artifacts/models/sft",
                "training_record_count": 2,
                "use_lora": False,
                "settings": {"max_steps": 1, "learning_rate": 5e-5},
                "metrics": {"training_loss": 3.25},
            }
        ),
        encoding="utf-8",
    )
    pre_path.write_text(json.dumps(_evaluation_payload(0.0, 2.0)), encoding="utf-8")
    post_path.write_text(json.dumps(_evaluation_payload(0.5, 1.5)), encoding="utf-8")

    report_path = generate_sft_report(
        summary_path=summary_path,
        pre_sft_path=pre_path,
        post_sft_path=post_path,
        output_path=tmp_path / "sft.md",
    )
    report = report_path.read_text(encoding="utf-8")

    assert report.startswith("# Supervised Fine-Tuning Report\n")
    assert "Training loss: 3.250" in report
    assert "| Pre-SFT | 0.000" in report
    assert "| Post-SFT | 0.500" in report
