.PHONY: help install train backend frontend dev clean

VENV = source .venv/bin/activate &&
NODE = export NVM_DIR="$$HOME/.nvm" && . "$$NVM_DIR/nvm.sh" &&

help:
	@echo ""
	@echo "  ML-Based Adaptive Cloud Resource Scheduling (V3)"
	@echo "  ================================================"
	@echo ""
	@echo "  make install     Install all Python + Node dependencies"
	@echo "  make train       Train all 4 ML models (GBR, LSTM, ARIMA, Combined)"
	@echo "  make migrate     Run Django migrations"
	@echo "  make backend     Start Django backend (port 8000)"
	@echo "  make frontend    Start React frontend (port 5173)"
	@echo "  make dev         Start BOTH backend and frontend"
	@echo "  make test-api    Quick smoke-test all API endpoints"
	@echo "  make clean       Remove pycache and build artifacts"
	@echo ""

install:
	python -m venv .venv
	$(VENV) pip install --upgrade pip
	$(VENV) pip install django djangorestframework django-cors-headers python-dotenv \
	                    numpy pandas scikit-learn joblib statsmodels matplotlib seaborn
	$(VENV) pip install torch --index-url https://download.pytorch.org/whl/cpu
	$(NODE) cd frontend && npm install

train:
	$(VENV) python model/train_all.py

migrate:
	$(VENV) cd backend && python manage.py makemigrations && python manage.py migrate

backend:
	$(VENV) cd backend && python manage.py runserver 8000

frontend:
	$(NODE) cd frontend && npm run dev

dev:
	@echo "Starting backend on :8000 and frontend on :5173 ..."
	$(VENV) cd backend && python manage.py runserver 8000 &
	$(NODE) cd frontend && npm run dev

test-api:
	@echo "Testing all endpoints..."
	@curl -sf -X POST http://localhost:8000/api/simulation/generate/ \
	  -H "Content-Type: application/json" -d '{"pattern":"gradual","steps":50,"seed":1}' > /dev/null \
	  && echo "✓ simulation/generate"
	@curl -sf http://localhost:8000/api/simulation/runs/ > /dev/null && echo "✓ simulation/runs"
	@curl -sf -X POST http://localhost:8000/api/scheduler/compare/ \
	  -H "Content-Type: application/json" -d '{"pattern":"spike","steps":100,"seed":42}' > /dev/null \
	  && echo "✓ scheduler/compare"
	@curl -sf http://localhost:8000/api/ml/status/ > /dev/null && echo "✓ ml/status"
	@curl -sf http://localhost:8000/api/ml/history/ > /dev/null && echo "✓ ml/history"
	@curl -sf -X POST http://localhost:8000/api/ml/predict-all/ \
	  -H "Content-Type: application/json" -d '{"history":[10,20,30,40,50,60,70,80,90,100]}' > /dev/null \
	  && echo "✓ ml/predict-all"
	@curl -sf http://localhost:8000/api/ml/compare-models/ > /dev/null && echo "✓ ml/compare-models"
	@curl -sf http://localhost:8000/api/metrics/summary/ > /dev/null && echo "✓ metrics/summary"
	@echo "All API checks passed ✅"

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	rm -rf frontend/dist frontend/.vite 2>/dev/null || true
	@echo "Cleaned."
