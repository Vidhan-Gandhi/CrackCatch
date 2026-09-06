# CrackCatch - common tasks.
#   make setup      one-time local environment
#   make demo       run the full pipeline on the sample clip
#   make up         start the whole stack in Docker
#   make test       run the test suite

PY      := .venv/bin/python
PIP     := .venv/bin/pip
PYTEST  := .venv/bin/pytest
export PYTHONPATH := backend:model

.PHONY: help setup venv deps frontend-deps sample dataset train evaluate benchmark \
        demo demo-local backend dashboard up down logs test lint clean reset-db

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
	 awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

venv:
	@test -d .venv || python3 -m venv .venv

deps: venv  ## Install Python dependencies
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

frontend-deps:  ## Install dashboard dependencies
	cd frontend && npm install

sample:  ## Generate the synthetic demo clip
	$(PY) data/scripts/make_demo_video.py

setup: deps frontend-deps sample  ## One-time local setup
	@echo ""
	@echo "Setup complete. Next:"
	@echo "  make demo        run the pipeline on the sample clip"
	@echo "  make up          start the full stack in Docker"

dataset:  ## Generate the offline synthetic training set
	$(PY) data/scripts/make_synthetic_dataset.py --train 500 --val 120

train:  ## Fine-tune YOLOv8 on the synthetic set (see README for RDD2022)
	$(PY) model/scripts/train.py --data data/synthetic_yolo/data.yaml \
	  --base model/weights/yolov8n.pt --epochs 30

evaluate:  ## mAP, per-class P/R, confusion matrix, achieved FPS
	$(PY) model/scripts/evaluate.py --data data/synthetic_yolo/data.yaml \
	  --weights model/weights/crackcatch.pt --also-cpu-fps

benchmark:  ## Speed/size ablation across checkpoints and devices
	$(PY) model/scripts/benchmark_fps.py \
	  --weights model/weights/crackcatch.pt model/weights/yolov8s.pt

demo:  ## Replay the sample clip end to end (uses the backend if it is up)
	$(PY) scripts/run_demo.py

demo-local:  ## Replay the clip with no backend or database
	$(PY) scripts/run_demo.py --local

backend:  ## Run the API with autoreload
	.venv/bin/uvicorn app.main:app --reload --app-dir backend --port 8000

dashboard:  ## Run the dashboard dev server
	cd frontend && npm run dev

up:  ## Start mongo + backend + dashboard in Docker
	docker compose up --build -d
	@echo ""
	@echo "  dashboard : http://localhost:5173"
	@echo "  API docs  : http://localhost:8000/docs"

down:  ## Stop the stack
	docker compose down

logs:  ## Tail container logs
	docker compose logs -f --tail=80

test:  ## Run the test suite
	$(PYTEST) backend/tests -q -c backend/pytest.ini

lint:  ## Lint the dashboard sources
	cd frontend && npm run lint || true

reset-db:  ## Drop all defect records (destructive)
	docker compose exec mongo mongosh crackcatch --quiet \
	  --eval 'db.defects.deleteMany({}); db.ingest_jobs.deleteMany({}); print("cleared")'

clean:  ## Remove generated artefacts (keeps .venv and node_modules)
	rm -rf storage/snapshots/* storage/uploads/* storage/repairs/* \
	       model/runs frontend/dist .pytest_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
