.PHONY: test test-ci lint type cli-smoke diff-check check ci-check

PYTHON ?= python
CHECK_PYTHON ?= $(PYTHON)
SMOKE_BRIEF_OUT ?= /tmp/trajectory-forge-smoke-brief.md

test:
	$(CHECK_PYTHON) -m pytest -q

test-ci:
	$(CHECK_PYTHON) -m pytest -q -m "not local_artifacts"

lint:
	$(CHECK_PYTHON) -m ruff check src tests

type:
	$(CHECK_PYTHON) -m mypy src

cli-smoke:
	rm -f "$(SMOKE_BRIEF_OUT)"
	PYTHONPATH=src $(CHECK_PYTHON) -m trajectory_forge.cli embodiment check configs/embodiments/lift2_r5a_bimanual.yaml
	PYTHONPATH=src $(CHECK_PYTHON) -m trajectory_forge.cli embodiment check configs/embodiments/template_bimanual_dexterous_tactile.yaml
	PYTHONPATH=src $(CHECK_PYTHON) -m trajectory_forge.cli controller preflight --controller examples/sample_privileged_controller.py --embodiment configs/embodiments/lift2_r5a_bimanual.yaml

diff-check:
	git diff --check

check: test lint cli-smoke diff-check

ci-check: test-ci lint cli-smoke diff-check
