# One-line entry points for every pipeline stage. Each target is a single
# `python -m ...` command, so it can also be copied and run directly (e.g. on Windows).
#
#   make setup && make data && make train-demo && make register && make evaluate \
#        && make promote && make infer && make mesh && make viewer && make trace

PYTHON  ?= .venv/bin/python
EXP     ?= EXP-001
VERSION ?= v1.0
# A test-split scan: never seen during training or model selection.
INPUT   ?= data/Task04_Hippocampus/imagesTr/hippocampus_017.nii.gz
CASE    ?= $(shell ls -t artifacts/inference 2>/dev/null | head -n 1)

.PHONY: setup lint typecheck test test-fast data synthetic train-smoke train-demo register \
        evaluate promote infer mesh viewer slices trace models demo

setup:
	python3 -m venv .venv
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install torch --index-url https://download.pytorch.org/whl/cpu
	$(PYTHON) -m pip install -e ".[dev]"

lint:
	$(PYTHON) -m ruff check src tests
	$(PYTHON) -m ruff format --check src tests

typecheck:
	$(PYTHON) -m mypy src

test:
	$(PYTHON) -m pytest -q

test-fast:          ## everything except the end-to-end reproducibility run
	$(PYTHON) -m pytest -q -m "not slow"

data:               ## download MSD hippocampus (~27 MB, checksum-verified) and build the manifest
	$(PYTHON) -m src.data.download --out data
	$(PYTHON) -m src.data.manifest --config configs/data.yaml

synthetic:
	$(PYTHON) -m src.data.synthetic --out data/synthetic --cases 12

train-smoke: synthetic
	$(PYTHON) -m src.training.train --config configs/smoke/train.yaml --artifacts-dir artifacts_smoke

train-demo:         ## ~55 min on a laptop CPU
	$(PYTHON) -m src.training.train --config configs/train.yaml

register:
	$(PYTHON) -m src.registry.model_registry register --experiment $(EXP) --version $(VERSION)

evaluate:
	$(PYTHON) -m src.evaluation.evaluate --model-version $(VERSION) --split test

promote:
	$(PYTHON) -m src.registry.model_registry promote --version $(VERSION) --to validated \
		--reason "test-split evaluation meets the release gate in configs/release.yaml"
	$(PYTHON) -m src.registry.model_registry promote --version $(VERSION) --to production \
		--reason "approved for the demo release"

models:
	$(PYTHON) -m src.registry.model_registry list

infer:
	$(PYTHON) -m src.inference.predict --input $(INPUT) --model-version production

mesh:
	$(PYTHON) -m src.reconstruction.mesh --inference-id $(CASE)

viewer:
	$(PYTHON) -m src.visualization.viewer --case $(CASE)

slices:
	$(PYTHON) -m src.visualization.slices --case $(CASE)

trace:
	$(PYTHON) -m src.lineage.trace --inference-id $(CASE)

demo:               ## one command: prepares everything on first run, then segment + 3D + trace + viewer
	$(PYTHON) -m src.demo --input $(INPUT)
