.PHONY: help install install-local api ui dev web index test safety lint format check clean

help:
	@echo "SafeSpace"
	@echo ""
	@echo "  make install        Install runtime + dev dependencies (no torch)"
	@echo "  make install-local  Also install Ollama/Streamlit/sentence-transformers"
	@echo "  make api            Run the API on :8000"
	@echo "  make web            Open the static chat UI (API must be running)"
	@echo "  make ui             Run the Streamlit dev UI on :8501"
	@echo "  make index          Rebuild corpus embeddings"
	@echo "  make test           Run the test suite"
	@echo "  make safety         Run the safety eval set only"
	@echo "  make lint           Lint"
	@echo "  make format         Format"
	@echo "  make check          Lint + format check + tests (what CI runs)"

install:
	uv sync --extra dev

install-local:
	uv sync --extra dev --extra local

api:
	uv run uvicorn backend.main:app --reload --port 8000

web:
	@echo "The API serves the static UI at http://localhost:8000"

ui:
	uv run --extra local streamlit run frontend.py

index:
	uv run --extra local python -m scripts.build_index

test:
	uv run pytest -q

safety:
	uv run pytest tests/test_safety.py -v

lint:
	uv run ruff check .

format:
	uv run ruff format .

check: lint
	uv run ruff format --check .
	uv run pytest -q

clean:
	rm -rf .pytest_cache .ruff_cache __pycache__ */__pycache__ */*/__pycache__
