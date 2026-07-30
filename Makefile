# MemAssist — dev commands.
#
# `python` is not always on PATH (e.g. Windows with a per-user Python). Override
# the interpreter with:  make dev PY="C:/Users/you/miniconda3/python.exe"

PY ?= python

.PHONY: help install api web dev test bench mcp mcp-http up down clean

help:
	@echo "make install    - install runtime + dev dependencies (editable)"
	@echo "make up         - docker compose: web + api + memory-mcp + postgres"
	@echo "make down       - stop the stack (add -v to drop the pgdata volume)"
	@echo "make api        - FastAPI on :8000"
	@echo "make web        - Next.js on :3000"
	@echo "make dev        - Streamlit on :8501 (Phase 1 UI)"
	@echo "make test       - pytest (no keys, no network)"
	@echo "make bench      - benchmark harness (add LIVE=1 for a live smoke)"
	@echo "make mcp        - MCP memory server on stdio"
	@echo "make mcp-http   - MCP memory server on Streamable HTTP :8090"
	@echo "make clean      - remove caches and local data/"

install:
	$(PY) -m pip install -e ".[dev]"

up:
	docker compose up --build

down:
	docker compose down

api:
	$(PY) -m uvicorn api.main:app --reload --port 8000

web:
	cd web && npm run dev

dev:
	$(PY) -m streamlit run app/streamlit_app.py

test:
	$(PY) -m pytest

bench:
	$(PY) -m bench $(if $(LIVE),--live,)

mcp:
	$(PY) -m memory_server

mcp-http:
	$(PY) -m memory_server --http

clean:
	rm -rf .pytest_cache **/__pycache__ *.egg-info build dist data
