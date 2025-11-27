PYTHON ?= python
PYTEST ?= $(PYTHON) -m pytest

.PHONY: test test-fast test-e2e test-gui ci

## Run the full test suite (unit + integration + GUI smoke)
test:
	$(PYTEST)

## Skip GUI-specific tests for faster feedback
test-fast:
	$(PYTEST) -m "not gui"

## Run a lightweight critical subset (result sink, model manager, core engines)
test-important:
	$(PYTEST) tests/test_result_handler.py \
		tests/test_model_manager_overrides.py \
		tests/test_detection_system.py \
		tests/test_yolo_inference_model.py

## Execute end-to-end workflow tests only
test-e2e:
	$(PYTEST) tests/test_e2e_workflow.py

## Run GUI smoke tests explicitly
test-gui:
	$(PYTEST) -m gui

## Target intended for CI pipelines
test-ci: test

ci: test-ci
