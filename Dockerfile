# Python image shared by the `api` and `memory-mcp` services (spec §11 P4).
FROM python:3.11-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/opt/models

WORKDIR /app

# Build tools for wheels that lack a manylinux build; dropped from the image.
RUN apt-get update \
 && apt-get install -y --no-install-recommends build-essential curl \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install -r requirements.txt

# Bake bge-small into an image LAYER. Without this every container start pays a
# ~130 MB download before the first archival write, which also makes startup
# depend on Hugging Face being reachable. Cached here, it is already on disk.
RUN python -c "from sentence_transformers import SentenceTransformer; \
SentenceTransformer('BAAI/bge-small-en-v1.5')" \
 && chmod -R a+rX /opt/models

RUN apt-get purge -y build-essential && apt-get autoremove -y

COPY . .

# Non-root: nothing here needs to write outside /app/data and /app/workspace.
RUN useradd --create-home --uid 10001 memassist \
 && mkdir -p /app/data /app/workspace \
 && chown -R memassist:memassist /app
USER memassist

EXPOSE 8000 8090
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
