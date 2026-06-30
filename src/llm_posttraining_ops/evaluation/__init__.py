"""Deterministic baseline inference, metrics, evaluation, and reporting."""

from llm_posttraining_ops.evaluation.evaluator import (
    EvaluationResult,
    run_baseline_evaluation,
)

__all__ = ["EvaluationResult", "run_baseline_evaluation"]
