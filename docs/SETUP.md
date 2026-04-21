# Setup Guide

## Prerequisites

| Requirement | Version | Notes |
|---|---|---|
| Python | 3.11+ | 3.14 confirmed working |
| Node.js | 20+ (LTS) | Install via nvm if needed |
| pip | Latest | |
| npm | Bundled with Node.js | |

---

## Step 1 — Clone and create Python environment

```bash
git clone <repository-url>
cd ml-cloud-scheduler

python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

pip install django djangorestframework django-cors-headers python-dotenv \
            numpy pandas scikit-learn joblib statsmodels matplotlib seaborn
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

---

## Step 2 — Install Node.js (if not installed)

```bash
# Install nvm
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.3/install.sh | bash
source ~/.bashrc   # or restart terminal

# Install Node.js LTS
nvm install --lts
nvm use --lts
```

---

## Step 3 — Train the ML models

```bash
# From project root (ml-cloud-scheduler/)
python model/train_all.py
```

Expected output:
```
  ML CLOUD SCHEDULER — Training All Models

[1/4] GradientBoosting Regressor (GBR)
...
[2/4] Long Short-Term Memory (LSTM) — PyTorch
...
[3/4] ARIMA (Auto-Regressive Integrated Moving Average)
...
[4/4] Combined Hybrid (LSTM + ARIMA)
...
```

> The model artifacts are already committed (`model/saved_models/gbr_model.pkl`), so this step is optional on first run.

---

## Step 4 — Set up and run the backend

```bash
cd backend
python manage.py migrate   # sets up SQLite database
python manage.py runserver
```

Backend runs at: **http://localhost:8000**

---

## Step 5 — Set up and run the frontend

```bash
# In a new terminal
cd frontend
npm install
npm run dev
```

Frontend runs at: **http://localhost:5173**

---

## Step 6 — Demo Workflow

1. Open **http://localhost:5173**
2. **Simulation** → Generate workload (pattern: combined, steps: 200)
3. **ML Training** → Click "Start Training" (takes ~15s)
4. **Comparison** → Click "Run Comparison" → see overload reduction %
5. **Metrics** → View aggregated KPIs
6. **Run Logs** → Browse step-by-step scheduler actions

---

## Environment Variables

Copy `.env.example` to `.env` and adjust:

```bash
DJANGO_SECRET_KEY=your-random-secret-key
DJANGO_DEBUG=True
ALLOWED_HOSTS=localhost,127.0.0.1
```

Frontend `.env` (frontend/.env):
```
VITE_API_URL=http://localhost:8000/api
```
