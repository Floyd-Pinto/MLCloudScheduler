# Convenience shortcuts — works with the local venv OR inside Docker.

VENV_PYTHON := .venv/bin/python

.PHONY: venv install run docker-build docker-run clean help

help:          ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

venv:          ## Create virtual environment
	python3 -m venv .venv
	@echo "Activate with: source .venv/bin/activate  (bash/zsh) or  source .venv/bin/activate.fish"

install:       ## Install Python dependencies into the venv
	$(VENV_PYTHON) -m pip install --upgrade pip
	$(VENV_PYTHON) -m pip install -r requirements.txt

run:           ## Run full simulation (local venv)
	$(VENV_PYTHON) main.py

docker-build:  ## Build the Docker image
	docker compose build

docker-run:    ## Run simulation inside Docker (outputs written to host)
	docker compose run --rm scheduler

clean:         ## Remove generated data, models, and plots
	rm -f data/*.csv models/*.pkl models/*.joblib outputs/*.png
