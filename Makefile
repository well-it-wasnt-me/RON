# ============================================================================
# DeskBot developer Makefile
# ----------------------------------------------------------------------------
# Self-documenting: `make help`
# ============================================================================

UV          ?= uv
PY          ?= $(UV) run python
RUFF        ?= $(UV) run ruff
MYPY        ?= $(UV) run mypy
PYTEST      ?= $(UV) run pytest
COVERAGE    ?= $(UV) run coverage
PRE_COMMIT  ?= $(UV) run pre-commit
MKDOCS      ?= $(UV) run mkdocs

SRC   := src
TESTS := tests

.DEFAULT_GOAL := help

# ---------------------------------------------------------------------------
# All targets are phony (no files with these names are produced)
# ---------------------------------------------------------------------------
.PHONY: help install install-dev lock hooks \
        lint format typecheck check check-fast pre-commit \
        test test-fast test-ci coverage coverage-html \
        run run-real doctor eye-demo simulate display-test interactive \
        docs docs-serve docs-clean docs-deploy \
        clean nuke tree ci

help: ## Show this help.
	@awk 'BEGIN {FS = ":.*?## "; printf "\nDeskBot developer commands:\n\n"} \
		/^[a-zA-Z0-9_.-]+:.*?## / {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}' \
		$(MAKEFILE_LIST)

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
install: ## Install all dependencies (incl. dev, hardware, ai, audio).
	$(UV) sync --all-extras

install-dev: ## Install development dependencies only.
	$(UV) sync --group dev

lock: ## Refresh the dependency lockfile.
	$(UV) lock

hooks: ## Install pre-commit hooks.
	$(PRE_COMMIT) install

# ---------------------------------------------------------------------------
# Quality
# ---------------------------------------------------------------------------
lint: ## Run Ruff linter.
	$(RUFF) check $(SRC) $(TESTS)

format: ## Auto-format with Ruff.
	$(RUFF) format $(SRC) $(TESTS)
	$(RUFF) check --fix $(SRC) $(TESTS)

typecheck: ## Run MyPy in strict mode.
	$(MYPY) $(SRC) $(TESTS)

check: lint typecheck test ## Run every static check and the full test suite.
check-fast: lint typecheck test-fast ## Same as check, but skip slow/hardware tests.

pre-commit: ## Run every pre-commit hook on every file.
	$(PRE_COMMIT) run --all-files

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
test: ## Run the full test suite (same flags as CI).
	$(PYTEST) --tb=short -q

test-fast: ## Skip slow and hardware tests.
	$(PYTEST) -m "not slow and not hardware" -q

test-ci: ## Exactly what CI runs (strict markers, short tracebacks).
	$(PYTEST) --tb=short -q

coverage: ## Run tests with branch coverage and print the report.
	$(COVERAGE) run -m pytest --tb=short -q
	$(COVERAGE) report

coverage-html: coverage ## Generate and open the HTML coverage report.
	$(COVERAGE) html
	@echo "Coverage report -> htmlcov/index.html"

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
run: ## Run DeskBot locally with the mock stack.
	$(UV) run deskbot

run-real: ## Run DeskBot with real hardware (Pi only).
	DESKBOT_HARDWARE=real $(UV) run deskbot

doctor: ## Diagnose environment, hardware, and configuration.
	$(UV) run deskbot-doctor

eye-demo: ## Cycle through every eye animation against mock displays.
	$(UV) run deskbot-eye-demo

simulate: ## Run the full robot stack against the in-memory mock display.
	$(UV) run deskbot-simulate

display-test: ## Run the standalone GC9A01 wiring smoke test on the Pi.
	$(UV) run deskbot-display-test

interactive: ## Launch the interactive TUI (braille face + servo dashboard).
	$(UV) run deskbot-interactive

# ---------------------------------------------------------------------------
# Docs (MkDocs)
# ---------------------------------------------------------------------------
docs: ## Build the MkDocs site into ./site (strict mode).
	$(MKDOCS) build --strict

docs-serve: ## Serve the MkDocs site locally with live reload on :8000.
	$(MKDOCS) serve

docs-clean: ## Remove the MkDocs build directory.
	rm -rf site

docs-deploy: ## Deploy the docs to GitHub Pages.
	$(MKDOCS) gh-deploy --force

# ---------------------------------------------------------------------------
# CI mirror
# ---------------------------------------------------------------------------
ci: lint typecheck test-ci docs ## Run the full CI pipeline locally.

# ---------------------------------------------------------------------------
# Hygiene
# ---------------------------------------------------------------------------
clean: docs-clean ## Remove caches, build artefacts, and the docs site.
	rm -rf .ruff_cache .mypy_cache .pytest_cache .coverage htmlcov build dist
	find . -type d -name __pycache__ -prune -exec rm -rf {} +

nuke: clean ## Remove caches, docs site, AND the virtualenv.
	rm -rf .venv

tree: ## Show the project tree (excluding artefacts).
	@command -v tree >/dev/null 2>&1 && tree -I '__pycache__|.venv|.git|.mypy_cache|.ruff_cache|.pytest_cache|htmlcov|build|dist|site' || \
		find . -type d \( -name .git -o -name .venv -o -name __pycache__ -o -name .mypy_cache -o -name .ruff_cache -o -name .pytest_cache -o -name htmlcov -o -name build -o -name dist -o -name site \) -prune -o -print
