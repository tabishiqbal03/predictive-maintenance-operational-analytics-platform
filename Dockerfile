FROM ghcr.io/astral-sh/uv:0.9.2 AS uv
FROM python:3.12-slim
COPY --from=uv /uv /usr/local/bin/uv
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 UV_LINK_MODE=copy MAINTENANCE_ROOT=/app
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
COPY pyproject.toml uv.lock ./
COPY src ./src
RUN uv sync --frozen --no-dev --no-editable
COPY api ./api
RUN useradd --uid 10001 --create-home service && mkdir -p /app/artifacts && chown service:service /app/artifacts
USER service
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s CMD ["/app/.venv/bin/python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=5)"]
CMD ["/app/.venv/bin/uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
