# MemAssist — dev commands.
#
# `python` is not always on PATH (e.g. Windows with a per-user Python). Override
# the interpreter with:  make dev PY="C:/Users/you/miniconda3/python.exe"
# On Windows without `make`, see the "Manual commands" section in README.md.

PY ?= python

.PHONY: help install dev test bench mcp clean

help:
	@echo "make install   - install runtime + dev dependencies (editable)"
	@echo "make dev        - run the Streamlit MVP"
	@echo "make test       - run the pytest suite"
	@echo "make bench      - run the benchmark harness (add LIVE=1 for a live smoke)"
	@echo "make mcp        - run the MCP memory server (Phase 2, stub)"
	@echo "make clean      - remove caches and local data/"

install:
	$(PY) -m pip install -e ".[dev]"

dev:
	$(PY) -m streamlit run app/streamlit_app.py

test:
	$(PY) -m pytest

bench:
	$(PY) -m bench $(if $(LIVE),--live,)

mcp:
	$(PY) -m memory_server

clean:
	rm -rf .pytest_cache **/__pycache__ *.egg-info build dist data
