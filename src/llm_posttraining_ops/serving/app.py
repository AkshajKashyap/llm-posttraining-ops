"""FastAPI application for local model generation and evaluation."""

from __future__ import annotations

import time
import uuid
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from typing import cast

from fastapi import FastAPI, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from llm_posttraining_ops.data.schemas import SplitName
from llm_posttraining_ops.evaluation.suite import (
    EvaluationExample,
    TaskType,
    evaluate_response,
)
from llm_posttraining_ops.inference.config import DEFAULT_MODEL_NAME, GenerationSettings
from llm_posttraining_ops.inference.prompts import format_instruction_prompt
from llm_posttraining_ops.serving.logging import (
    DEFAULT_INFERENCE_LOG_PATH,
    InferenceLogger,
)
from llm_posttraining_ops.serving.model_manager import ModelManager
from llm_posttraining_ops.serving.schemas import (
    BatchGenerationRequest,
    BatchGenerationResponse,
    ErrorResponse,
    EvaluationDiagnosticsResponse,
    EvaluationRequest,
    EvaluationResponse,
    GenerationInput,
    GenerationRequest,
    GenerationResponse,
    GenerationSettingsSchema,
    HealthResponse,
    ModelInfoResponse,
)


def new_request_id() -> str:
    """Return a unique request identifier."""

    return str(uuid.uuid4())


class GenerationServiceError(RuntimeError):
    """Carry request context from generation failures to the API handler."""

    def __init__(self, request_id: str, detail: str) -> None:
        self.request_id = request_id
        self.detail = detail
        super().__init__(detail)


def _domain_settings(
    model_name: str,
    settings: GenerationSettingsSchema,
) -> GenerationSettings:
    return GenerationSettings(
        model_name=model_name,
        max_new_tokens=settings.max_new_tokens,
        temperature=settings.temperature,
        top_p=settings.top_p,
        seed=settings.seed,
    )


def _settings_schema(request: GenerationRequest | BatchGenerationRequest) -> GenerationSettingsSchema:
    return GenerationSettingsSchema(
        max_new_tokens=request.max_new_tokens,
        temperature=request.temperature,
        top_p=request.top_p,
        seed=request.seed,
    )


def create_app(
    *,
    model_name: str = DEFAULT_MODEL_NAME,
    mock: bool = False,
    log_path: str | Path = DEFAULT_INFERENCE_LOG_PATH,
    manager: ModelManager | None = None,
    logger: InferenceLogger | None = None,
    request_id_factory: Callable[[], str] = new_request_id,
    clock: Callable[[], float] = time.perf_counter,
) -> FastAPI:
    """Create an API app without loading model weights."""

    api = FastAPI(
        title="LLM Post-Training Ops",
        version="0.1.0",
    )
    active_manager = manager or ModelManager(model_name, mock=mock)
    active_logger = logger or InferenceLogger(log_path)
    api.state.model_manager = active_manager
    api.state.inference_logger = active_logger

    def perform_generation(
        item: GenerationInput,
        settings_schema: GenerationSettingsSchema,
        *,
        request_id: str,
        endpoint: str,
    ) -> GenerationResponse:
        settings = _domain_settings(active_manager.model_name, settings_schema)
        prompt = format_instruction_prompt(item.instruction, item.input)
        start = clock()
        try:
            output = active_manager.generate(prompt, settings)
        except Exception as exc:
            elapsed = round(clock() - start, 6)
            active_logger.write(
                {
                    "request_id": request_id,
                    "endpoint": endpoint,
                    "status": "error",
                    "model_name": active_manager.model_name,
                    "mock": active_manager.mock,
                    "prompt": prompt,
                    "instruction": item.instruction,
                    "input": item.input,
                    "metadata": item.metadata,
                    "generation_settings": settings_schema.model_dump(),
                    "latency_seconds": elapsed,
                    "response_length_tokens": 0,
                    "response_length_characters": 0,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            raise GenerationServiceError(request_id, str(exc)) from exc

        elapsed = round(clock() - start, 6)
        active_logger.write(
            {
                "request_id": request_id,
                "endpoint": endpoint,
                "status": "ok",
                "model_name": active_manager.model_name,
                "mock": active_manager.mock,
                "prompt": prompt,
                "instruction": item.instruction,
                "input": item.input,
                "metadata": item.metadata,
                "generation_settings": settings_schema.model_dump(),
                "latency_seconds": elapsed,
                "response_length_tokens": output.generated_tokens,
                "response_length_characters": len(output.text),
                "error": None,
            }
        )
        return GenerationResponse(
            request_id=request_id,
            model_name=active_manager.model_name,
            response=output.text,
            generated_tokens=output.generated_tokens,
            latency_seconds=elapsed,
            settings=settings_schema,
            mock=active_manager.mock,
        )

    @api.exception_handler(RequestValidationError)
    async def validation_error_handler(
        request: Request,
        exc: RequestValidationError,
    ) -> JSONResponse:
        request_id = request_id_factory()
        active_logger.write(
            {
                "request_id": request_id,
                "endpoint": request.url.path,
                "status": "error",
                "model_name": active_manager.model_name,
                "mock": active_manager.mock,
                "prompt": None,
                "instruction": None,
                "input": None,
                "metadata": {},
                "generation_settings": None,
                "latency_seconds": 0.0,
                "response_length_tokens": 0,
                "response_length_characters": 0,
                "error": "validation_error",
            }
        )
        payload = ErrorResponse(
            error="validation_error",
            request_id=request_id,
            detail=jsonable_encoder(exc.errors()),
        )
        return JSONResponse(status_code=422, content=payload.model_dump())

    @api.exception_handler(GenerationServiceError)
    async def generation_error_handler(
        request: Request,
        exc: GenerationServiceError,
    ) -> JSONResponse:
        del request
        payload = ErrorResponse(
            error="generation_failed",
            request_id=exc.request_id,
            detail=exc.detail,
        )
        return JSONResponse(status_code=500, content=payload.model_dump())

    @api.get("/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        return HealthResponse(
            status="ok",
            model_loaded=active_manager.loaded,
            mock=active_manager.mock,
        )

    @api.get("/model-info", response_model=ModelInfoResponse)
    async def model_info() -> ModelInfoResponse:
        return ModelInfoResponse(
            model_name=active_manager.model_name,
            model_loaded=active_manager.loaded,
            mock=active_manager.mock,
            device=active_manager.device,
        )

    @api.post(
        "/generate",
        response_model=GenerationResponse,
        responses={500: {"model": ErrorResponse}},
    )
    async def generate(request: GenerationRequest) -> GenerationResponse:
        request_id = request_id_factory()
        return perform_generation(
            GenerationInput(
                instruction=request.instruction,
                input=request.input,
                metadata=request.metadata,
            ),
            _settings_schema(request),
            request_id=request_id,
            endpoint="/generate",
        )

    @api.post(
        "/batch-generate",
        response_model=BatchGenerationResponse,
        responses={500: {"model": ErrorResponse}},
    )
    async def batch_generate(request: BatchGenerationRequest) -> BatchGenerationResponse:
        request_id = request_id_factory()
        settings = _settings_schema(request)
        start = clock()
        results = [
            perform_generation(
                item,
                settings,
                request_id=f"{request_id}:{index}",
                endpoint="/batch-generate",
            )
            for index, item in enumerate(request.items)
        ]
        return BatchGenerationResponse(
            request_id=request_id,
            results=results,
            total_latency_seconds=round(clock() - start, 6),
        )

    @api.post("/evaluate-generation", response_model=EvaluationResponse)
    async def evaluate_generation(request: EvaluationRequest) -> EvaluationResponse:
        request_id = request_id_factory()
        start = clock()
        example = EvaluationExample(
            id=request_id,
            split=cast(SplitName, "test"),
            instruction=request.instruction,
            input=request.input,
            reference_output=request.reference_output,
            required_facts=request.required_facts,
            forbidden_terms=request.forbidden_terms,
            task_type=cast(TaskType, request.task_type),
            metadata=request.metadata,
        )
        diagnostics = evaluate_response(example, request.generated_response)
        elapsed = round(clock() - start, 6)
        active_logger.write(
            {
                "request_id": request_id,
                "endpoint": "/evaluate-generation",
                "status": "ok",
                "model_name": active_manager.model_name,
                "mock": active_manager.mock,
                "prompt": format_instruction_prompt(request.instruction, request.input),
                "instruction": request.instruction,
                "input": request.input,
                "metadata": request.metadata,
                "generation_settings": None,
                "latency_seconds": elapsed,
                "response_length_tokens": diagnostics.response_length,
                "response_length_characters": len(request.generated_response),
                "error": None,
            }
        )
        return EvaluationResponse(
            request_id=request_id,
            diagnostics=EvaluationDiagnosticsResponse(**asdict(diagnostics)),
        )

    return api


app = create_app()
