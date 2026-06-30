"""Pydantic request and response schemas for the serving API."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class StrictSchema(BaseModel):
    """Reject unknown API fields to surface client mistakes early."""

    model_config = ConfigDict(extra="forbid")


class GenerationSettingsSchema(StrictSchema):
    """Generation controls exposed through the API."""

    max_new_tokens: int = Field(default=32, ge=1, le=1024)
    temperature: float = Field(default=0.0, ge=0.0)
    top_p: float = Field(default=1.0, gt=0.0, le=1.0)
    seed: int = Field(default=42, ge=0)


class GenerationInput(StrictSchema):
    """Instruction, optional input, and caller metadata."""

    instruction: str = Field(min_length=1)
    input: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class GenerationRequest(GenerationSettingsSchema, GenerationInput):
    """Single instruction generation request."""


class GenerationResponse(StrictSchema):
    """One generated continuation and execution metadata."""

    request_id: str
    model_name: str
    response: str
    generated_tokens: int
    latency_seconds: float
    settings: GenerationSettingsSchema
    mock: bool


class BatchGenerationRequest(GenerationSettingsSchema):
    """Batch of instructions sharing generation settings."""

    items: list[GenerationInput] = Field(min_length=1, max_length=100)


class BatchGenerationResponse(StrictSchema):
    """Ordered batch generation results."""

    request_id: str
    results: list[GenerationResponse]
    total_latency_seconds: float


class EvaluationRequest(StrictSchema):
    """Evaluate a supplied generation against reference-backed constraints."""

    instruction: str = Field(min_length=1)
    input: str = ""
    reference_output: str = Field(min_length=1)
    generated_response: str
    required_facts: list[str] = Field(default_factory=list)
    forbidden_terms: list[str] = Field(default_factory=list)
    task_type: Literal["freeform", "json", "list", "short_answer"] = "freeform"
    metadata: dict[str, Any] = Field(default_factory=dict)


class EvaluationDiagnosticsResponse(StrictSchema):
    """Deterministic evaluation diagnostics returned by the API."""

    exact_match: float
    token_overlap_f1: float
    required_fact_coverage: float
    forbidden_term_violation: float
    instruction_copying: float
    empty_response: float
    response_length: int
    refusal_detected: float
    format_compliant: float
    unsupported_named_entities: list[str]
    numeric_mismatch: float
    contradiction_detected: float


class EvaluationResponse(StrictSchema):
    """Evaluation response with a traceable request ID."""

    request_id: str
    diagnostics: EvaluationDiagnosticsResponse


class HealthResponse(StrictSchema):
    """Service health without forcing model loading."""

    status: str
    model_loaded: bool
    mock: bool


class ModelInfoResponse(StrictSchema):
    """Configured model and lazy-loading state."""

    model_name: str
    model_loaded: bool
    mock: bool
    device: str


class ErrorResponse(StrictSchema):
    """Consistent API error payload."""

    error: str
    request_id: str
    detail: Any | None = None
