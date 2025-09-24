PYTHON ?= python
PYTEST ?= $(PYTHON) -m pytest

.PHONY: test test-fast test-e2e test-gui ci

## Run the full test suite (unit + integration + GUI smoke)
test:
	$(PYTEST)

## Skip GUI-specific tests for faster feedback
test-fast:
	$(PYTEST) -m "not gui"

## Execute end-to-end workflow tests only
test-e2e:
	$(PYTEST) tests/test_e2e_workflow.py

## Run GUI smoke tests explicitly
test-gui:
	$(PYTEST) -m gui

## Target intended for CI pipelines
test-ci: test

ci: test-ci
