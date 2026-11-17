.PHONY: build run-cpu test push setup

VENV := .venv
PYTHON := $(VENV)/bin/python
PIP := $(VENV)/bin/pip

# Integration tests start a local mongod; use the cached 8.2.6 binary when present.
MONGOD_BIN ?= $(firstword $(wildcard $(HOME)/.cache/mongodb-binaries/mongod-*-8.2.6))
export MONGOD_BIN

# news-clustering-job image. One image; GPU via cuml.accel.
IMAGE ?= mera-news-clustering-job
TAG ?= latest

# --- clustering job ---------------------------------------------------------

# venv + dev/test deps (CPU deps; NO cuml — that is for the GPU image only).
setup:
	python3.14 -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements-dev.txt

# Build the GPU/CPU clustering image (~6-8 GB of cuML wheels).
build:
	docker build -f Dockerfile.clustering -t $(IMAGE):$(TAG) .

# Run the job locally in CPU mode against Mongo (set MONGODB_URI in the
# environment first).
run-cpu:
	LOG_LEVEL=info \
	$(PYTHON) -m src.job_main_mongo

# Unit + integration tests with coverage (gate in pyproject.toml).
test:
	$(PYTHON) -m pytest tests/

# Push the built image to the registry.
push:
	docker push $(IMAGE):$(TAG)
