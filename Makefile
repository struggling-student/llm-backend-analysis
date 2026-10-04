# Reproducible analysis pipeline:  raw data -> processed tables -> executed notebooks (+ exports)
#
#   make setup       install the locked environment (uv)
#   make data        rebuild data/processed/ from data/raw/
#   make notebooks   execute every notebook in place (regenerates results/figures and results/tables)
#   make validate    repository checks + unit tests
#   make all         data + notebooks + validate
#   make sync        copy new cluster results into data/raw/ (needs the cresco8 SSH alias)
#   make checksums   record newly synced raw files in data/metadata/raw_checksums.sha256

UV ?= uv
NOTEBOOKS := $(sort $(wildcard notebooks/*/[0-9][0-9]_*.ipynb))

.PHONY: all setup data notebooks validate test sync checksums clean-outputs

all: data notebooks validate

setup:
	$(UV) sync

data:
	$(UV) run python scripts/processing/build_processed_data.py

notebooks:
	$(UV) run python scripts/utilities/execute_notebooks.py $(NOTEBOOKS)

validate:
	$(UV) run python scripts/utilities/validate_repository.py
	$(UV) run pytest

test:
	$(UV) run pytest

sync:
	bash scripts/cluster/sync_results.sh

checksums:
	$(UV) run python scripts/utilities/raw_checksums.py update

clean-outputs:
	$(UV) run jupyter nbconvert --clear-output --inplace $(NOTEBOOKS)
