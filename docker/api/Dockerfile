FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    libglib2.0-0 libsm6 libxrender1 libxext6 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN pip install --no-cache-dir uv

COPY pyproject.toml ./
COPY README.md ./
COPY src/ ./src/

RUN uv sync --no-dev

ENV CONFIG_PATH=/config/config.yaml
ENV TRANSFORMERS_CACHE=/cache/hf

EXPOSE 8000
CMD ["uv", "run", "uvicorn", "mindlm.api.main:app", \
     "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
