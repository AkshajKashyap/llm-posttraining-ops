"""Deterministic baseline inference, metrics, evaluation, and reporting."""

from llm_posttraining_ops.evaluation.evaluator import (
    EvaluationResult,
    run_baseline_evaluation,
)
from llm_posttraining_ops.evaluation.pairwise import compare_generation_files
from llm_posttraining_ops.evaluation.suite import evaluate_generation_file

__all__ = [
    "EvaluationResult",
    "compare_generation_files",
    "evaluate_generation_file",
    "run_baseline_evaluation",
]
