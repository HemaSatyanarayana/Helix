# Makefile for RAG-Production-Grade
# Python environment management

PYTHON := python3
VENV := .venv
VENV_BIN := $(VENV)/bin

.PHONY: help venv install clean freeze

help:  ## Show available targets
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

venv:  ## Create the virtual environment
	$(PYTHON) -m venv $(VENV)
	$(VENV_BIN)/pip install --upgrade pip setuptools wheel

install: venv  ## Create venv and install dependencies from requirements.txt
	@test -f requirements.txt && $(VENV_BIN)/pip install -r requirements.txt || echo "No requirements.txt found."

freeze:  ## Write current dependencies to requirements.txt
	$(VENV_BIN)/pip freeze > requirements.txt

clean:  ## Remove the virtual environment and caches
	rm -rf $(VENV)
	find . -type d -name __pycache__ -exec rm -rf {} +
