SHELL := /bin/bash

VENV ?= .venv
PYTHON := $(VENV)/bin/python
PIP := $(VENV)/bin/pip
CONFIG ?= configs/h100x8.yaml

.PHONY: venv install install-metricx vllm pipeline sharded test clean

venv:
	python3 -m venv $(VENV)

install: venv
	$(PIP) install -U pip
	$(PIP) install -e .

install-metricx: venv
	$(PIP) install -U pip
	$(PIP) install -e .
	./scripts/install_metricx_official.sh

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
	$(PIP) install -e ".[test]"
	$(VENV)/bin/pytest -q

clean:
	rm -rf $(VENV)
