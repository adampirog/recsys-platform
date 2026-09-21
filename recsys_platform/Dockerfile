# syntax=docker/dockerfile:1

FROM python:3.13-slim AS builder

WORKDIR /build

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

COPY pyproject.toml README.md LICENSE ./
COPY recsys_platform ./recsys_platform

RUN python -m pip wheel \
    --wheel-dir /wheels \
    ".[serving,popularity-inference]"


FROM python:3.13-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

COPY --from=builder /wheels /wheels

RUN python -m pip install \
        --no-cache-dir \
        --no-index \
        --find-links=/wheels \
        "recsys-platform[serving,popularity-inference]" \
    && rm -rf /wheels \
    && useradd --create-home --uid 10001 appuser \
    && mkdir -p /models \
    && chown appuser:appuser /models

USER appuser

EXPOSE 8000

CMD [
    "python",
    "-m",
    "recsys_platform.serving.app",
    "/models",
    "--host",
    "0.0.0.0",
    "--port",
    "8000"
]