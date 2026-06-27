# grounded-rag-agent — developer workflow.
# Uses `uv` as the runner so no manual venv activation is needed:
#   https://docs.astral.sh/uv/   (install: `curl -LsSf https://astral.sh/uv/install.sh | sh`)
#
# Every target is offline-safe except `ask`/`ingest`/`eval` with a real key.

.DEFAULT_GOAL := help
RUN := uv run
DEV := uv run --extra dev

.PHONY: help setup test lint typecheck fmt check eval ingest ask clean

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

setup: ## Create the venv and install the package + dev tools (editable)
	uv venv
	uv pip install -e ".[dev]"

test: ## Run the full test suite (offline; no API key needed)
	$(DEV) pytest

lint: ## Lint with ruff
	$(DEV) ruff check src tests

typecheck: ## Static type check with mypy
	$(DEV) mypy src

fmt: ## Auto-format and auto-fix with ruff
	$(DEV) ruff format src tests
	$(DEV) ruff check --fix src tests

check: lint typecheck test ## Lint + types + tests (what CI runs)

ingest: ## Build the retrieval index from data/docs/ (needs CO_API_KEY)
	$(RUN) python -m grounded_rag.cli ingest

ask: ## Ask the agent a question, e.g. `make ask Q="what does the spec say about retries?"` (needs CO_API_KEY)
	$(RUN) python -m grounded_rag.cli ask "$(Q)"

eval: ## Run the evaluation harness and write a versioned report (needs CO_API_KEY)
	$(RUN) python -m grounded_rag.cli eval

clean: ## Remove caches and build artifacts
	rm -rf .pytest_cache .ruff_cache .mypy_cache htmlcov .coverage dist build
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
