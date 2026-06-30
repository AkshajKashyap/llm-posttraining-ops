FROM python:3.11-slim

LABEL org.opencontainers.image.title="llm-posttraining-ops"
LABEL org.opencontainers.image.version="0.1.0"
LABEL org.opencontainers.image.description="CPU-only deterministic mock inference API"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN useradd --create-home --uid 10001 appuser

COPY pyproject.toml README.md LICENSE ./
COPY src ./src

# Mock serving needs only the lightweight API/config dependencies. The package is
# installed without the model-training dependency stack to keep this image small.
RUN python -m pip install \
      "fastapi>=0.115,<0.130" \
      "pydantic>=2,<3" \
      "pyyaml>=6.0" \
      "typer>=0.12" \
      "uvicorn>=0.30,<1" \
    && python -m pip install --no-deps .

RUN mkdir -p /app/artifacts/logs && chown -R appuser:appuser /app

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=10s --timeout=3s --start-period=5s --retries=3 \
  CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2)"]

CMD ["python", "-m", "llm_posttraining_ops.cli", "serve", "--mock", "--host", "0.0.0.0", "--port", "8000"]
