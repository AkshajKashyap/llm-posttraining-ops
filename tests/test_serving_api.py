import asyncio
import itertools
import json
from pathlib import Path

import httpx
import pytest
from fastapi import FastAPI
from pydantic import ValidationError

from llm_posttraining_ops.inference.config import GenerationSettings
from llm_posttraining_ops.inference.huggingface import GenerationOutput
from llm_posttraining_ops.serving.app import create_app
from llm_posttraining_ops.serving.logging import InferenceLogger
from llm_posttraining_ops.serving.model_manager import MockGenerator, ModelManager
from llm_posttraining_ops.serving.schemas import GenerationRequest


def _app(tmp_path: Path) -> tuple[FastAPI, Path, ModelManager]:
    counter = itertools.count()
    log_path = tmp_path / "inference.jsonl"
    manager = ModelManager("mock/model", mock=True)
    app = create_app(
        manager=manager,
        logger=InferenceLogger(
            log_path,
            timestamp_factory=lambda: "2026-06-30T00:00:00+00:00",
        ),
        request_id_factory=lambda: f"request-{next(counter)}",
    )
    return app, log_path, manager


def _request(
    app: FastAPI,
    method: str,
    path: str,
    *,
    json_body: dict[str, object] | None = None,
) -> httpx.Response:
    async def send() -> httpx.Response:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as client:
            return await client.request(method, path, json=json_body)

    return asyncio.run(send())


def test_health_and_model_info_do_not_load_model(tmp_path: Path) -> None:
    app, _, manager = _app(tmp_path)

    assert _request(app, "GET", "/health").json() == {
        "status": "ok",
        "model_loaded": False,
        "mock": True,
    }
    assert _request(app, "GET", "/model-info").json() == {
        "model_name": "mock/model",
        "model_loaded": False,
        "mock": True,
        "device": "cpu",
    }
    assert manager.loaded is False


def test_generate_endpoint_with_mock_and_log(tmp_path: Path) -> None:
    app, log_path, manager = _app(tmp_path)

    response = _request(
        app,
        "POST",
        "/generate",
        json_body={
            "instruction": "Explain SFT in one sentence.",
            "input": "",
            "max_new_tokens": 12,
            "metadata": {"caller": "test"},
        },
    )
    payload = response.json()

    assert response.status_code == 200
    assert payload["request_id"] == "request-0"
    assert payload["response"] == "Mock response: Explain SFT in one sentence."
    assert payload["settings"]["max_new_tokens"] == 12
    assert payload["mock"] is True
    assert manager.loaded is True
    log = json.loads(log_path.read_text(encoding="utf-8"))
    assert log["status"] == "ok"
    assert log["request_id"] == "request-0"
    assert log["metadata"] == {"caller": "test"}
    assert log["response_length_tokens"] == payload["generated_tokens"]
    assert log["error"] is None


def test_batch_generate_endpoint_with_mock(tmp_path: Path) -> None:
    app, log_path, _ = _app(tmp_path)

    response = _request(
        app,
        "POST",
        "/batch-generate",
        json_body={
            "items": [
                {"instruction": "First instruction.", "input": ""},
                {"instruction": "Second instruction.", "input": "context"},
            ],
            "seed": 7,
        },
    )
    payload = response.json()

    assert response.status_code == 200
    assert payload["request_id"] == "request-0"
    assert [item["request_id"] for item in payload["results"]] == [
        "request-0:0",
        "request-0:1",
    ]
    assert [item["response"] for item in payload["results"]] == [
        "Mock response: First instruction.",
        "Mock response: Second instruction.",
    ]
    assert len(log_path.read_text(encoding="utf-8").splitlines()) == 2


def test_evaluate_generation_endpoint(tmp_path: Path) -> None:
    app, log_path, manager = _app(tmp_path)

    response = _request(
        app,
        "POST",
        "/evaluate-generation",
        json_body={
            "instruction": "Name the capital of France.",
            "reference_output": "Paris.",
            "generated_response": "London.",
            "required_facts": ["Paris"],
            "forbidden_terms": ["London"],
            "task_type": "short_answer",
            "metadata": {"max_tokens": 3},
        },
    )
    diagnostics = response.json()["diagnostics"]

    assert response.status_code == 200
    assert diagnostics["required_fact_coverage"] == 0.0
    assert diagnostics["forbidden_term_violation"] == 1.0
    assert diagnostics["unsupported_named_entities"] == ["London"]
    assert manager.loaded is False
    assert json.loads(log_path.read_text())["endpoint"] == "/evaluate-generation"


def test_request_schema_and_validation_error(tmp_path: Path) -> None:
    app, log_path, _ = _app(tmp_path)

    with pytest.raises(ValidationError):
        GenerationRequest(
            instruction="Valid",
            max_new_tokens=0,
        )
    response = _request(
        app,
        "POST",
        "/generate",
        json_body={"instruction": "", "unexpected": True},
    )

    assert response.status_code == 422
    assert response.json()["error"] == "validation_error"
    assert response.json()["request_id"] == "request-0"
    assert json.loads(log_path.read_text())["status"] == "error"


class FailingGenerator:
    model_name = "broken/model"

    def generate(
        self,
        prompt: str,
        settings: GenerationSettings,
    ) -> GenerationOutput:
        del prompt, settings
        raise RuntimeError("deliberate failure")


def test_generation_error_handling_and_logging(tmp_path: Path) -> None:
    log_path = tmp_path / "errors.jsonl"
    manager = ModelManager(
        "broken/model",
        generator_factory=FailingGenerator,
    )
    app = create_app(
        manager=manager,
        logger=InferenceLogger(log_path, timestamp_factory=lambda: "fixed"),
        request_id_factory=lambda: "error-request",
    )
    response = _request(
        app,
        "POST",
        "/generate",
        json_body={"instruction": "Fail now."},
    )

    assert response.status_code == 500
    assert response.json() == {
        "error": "generation_failed",
        "request_id": "error-request",
        "detail": "deliberate failure",
    }
    assert manager.loaded is False
    log = json.loads(log_path.read_text())
    assert log["status"] == "error"
    assert log["error"] == "RuntimeError: deliberate failure"


def test_model_manager_factory_is_lazy_and_reused() -> None:
    calls: list[str] = []

    def factory() -> MockGenerator:
        calls.append("loaded")
        return MockGenerator("mock/model")

    manager = ModelManager("mock/model", mock=True, generator_factory=factory)
    settings = GenerationSettings(model_name="mock/model")

    assert calls == []
    manager.generate("### Instruction:\nOne\n\n### Response:", settings)
    manager.generate("### Instruction:\nTwo\n\n### Response:", settings)
    assert calls == ["loaded"]
