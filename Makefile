SHELL := /bin/bash

VENV ?= .venv
PYTHON := $(VENV)/bin/python
UV ?= uv
CONFIG ?= configs/h100x8.yaml

.PHONY: venv install install-metricx vllm pipeline sharded test clean

venv:
	$(UV) python install 3.11
	$(UV) venv --python 3.11 $(VENV)

install: venv
	$(UV) pip install --python $(PYTHON) -e .

install-metricx: venv
	$(UV) pip install --python $(PYTHON) -e .
	METRICX_PYTHON=$(PYTHON) ./scripts/install_metricx_official.sh

vllm:
	./scripts/launch_vllm_h100x8.sh

pipeline:
	$(VENV)/bin/synth_parallel run --config $(CONFIG) --stage sample_sources
	$(VENV)/bin/synth_parallel run --config $(CONFIG) --stage prefilter_score
	$(VENV)/bin/synth_parallel run --config $(CONFIG) --stage select_sources
	$(VENV)/bin/synth_parallel run --config $(CONFIG) --stage generate_128
	$(VENV)/bin/synth_parallel run --config $(CONFIG) --stage score_select_best
	$(VENV)/bin/synth_parallel run --config $(CONFIG) --stage format_filter
	$(VENV)/bin/synth_parallel run --config $(CONFIG) --stage export

sharded:
	CONFIG=$(CONFIG) SHARDS=8 ./scripts/run_sharded_pipeline.sh $(CONFIG)

test:
	$(UV) pip install --python $(PYTHON) -e ".[test]"
	$(VENV)/bin/pytest -q

clean:
	rm -rf $(VENV)
