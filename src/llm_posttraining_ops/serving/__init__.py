"""FastAPI serving layer for local post-trained models."""

from llm_posttraining_ops.serving.app import create_app

__all__ = ["create_app"]
