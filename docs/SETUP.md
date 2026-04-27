# Setup Guide

Complete installation and setup instructions for the ML Cloud Scheduler.

---

## Prerequisites

| Requirement | Version | Check Command |
|---|---|---|
| Python | 3.11+ (3.14 confirmed) | `python --version` |
| Node.js | 20+ (LTS) | `node --version` |
| pip | Latest | `pip --version` |
| npm | Bundled with Node.js | `npm --version` |

---

## Step 1 — Clone and Create Python Environment

```bash
git clone <repository-url>
cd ml-cloud-scheduler

# Create virtual environment
python -m venv .venv
source .venv/bin/activate   # Linux/macOS
# .venv\Scripts\activate    # Windows

# Install backend dependencies
pip install -r backend/requirements.txt

# Install PyTorch (CPU-only, sufficient for this project)
pip install torch --index-url https://download.pytorch.org/whl/cpu

# Install ML dependencies
pip install numpy pandas scikit-learn joblib statsmodels matplotlib seaborn
```

---

## Step 2 — Install Node.js (if not installed)

```bash
# Option A: Install nvm (recommended)
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.3/install.sh | bash
source ~/.bashrc   # or restart terminal

nvm install --lts
nvm use --lts

# Option B: Direct install
# Download from https://nodejs.org
```

---

## Step 3 — Train the ML Models

```bash
# From project root (ml-cloud-scheduler/)
python model/train_all.py
```

Expected output:
```
  ML CLOUD SCHEDULER — Training All Models

[1/4] GradientBoosting Regressor (GBR)
  → Saved to model/saved_models/gbr_model.pkl
  → R²=0.9709, RMSE=4.85

[2/4] Long Short-Term Memory (LSTM) — PyTorch
  → 150 epochs, hidden=128, window=20
  → Saved to model/saved_models/lstm_model.pt
  → R²=0.9696, RMSE=5.11

[3/4] ARIMA (Auto-Regressive Integrated Moving Average)
  → Order: (2, 1, 2), AIC: ~1200
  → Saved to model/saved_models/arima_meta.json
  → R²=0.6351, RMSE=1.85

[4/4] Combined Hybrid (LSTM + ARIMA)
  → Weights: w_lstm=0.434, w_arima=0.566
  → Saved to model/saved_models/combined_meta.json
  → R²=0.7952, RMSE=14.53
```

> **Note**: Pre-trained models are committed to `model/saved_models/`, so this step is optional on first run. However, retraining ensures models match the current training data.

**Where are models saved?**

| Model | File | Size |
|---|---|---|
| GBR | `model/saved_models/gbr_model.pkl` | ~478 KB |
| GBR Scaler | `model/saved_models/scaler.pkl` | ~1 KB |
| LSTM | `model/saved_models/lstm_model.pt` | ~827 KB |
| LSTM Scaler | `model/saved_models/lstm_scaler.pkl` | ~1 KB |
| LSTM Config | `model/saved_models/lstm_meta.json` | JSON |
| ARIMA Config | `model/saved_models/arima_meta.json` | JSON |
| Combined Config | `model/saved_models/combined_meta.json` | JSON |

---

## Step 4 — Set Up and Run the Backend

```bash
cd backend
python manage.py migrate   # creates/updates SQLite database
python manage.py runserver
```

Backend runs at: **http://localhost:8000**

Verify it's working:
```bash
curl http://localhost:8000/api/ml/status/
# Should return JSON with model statuses
```

---

## Step 5 — Set Up and Run the Frontend

```bash
# In a NEW terminal
cd frontend
npm install      # install Node dependencies (first time only)
npm run dev
```

Frontend runs at: **http://localhost:5173**

---

## Step 6 — Demo Workflow

1. Open **http://localhost:5173** in your browser
2. **Research Overview** → Check that all 3 models show "✓ Trained"
3. **Workload Simulation** → Generate a workload (pattern: combined, steps: 200)
4. **Model Training** → Click "Train All Models" (~2 minutes)
5. **Findings** → Scheduler Comparison tab → Click "Run Comparison" → see overload reduction %
6. **Findings** → Model Accuracy tab → Click "Evaluate Models" → see R² comparison
7. **Metrics** → View aggregated reactive vs predictive stats
8. **Run Logs** → Click any run to see step-by-step scheduler decisions

---

## Environment Variables

### Backend (.env in project root)

```bash
DJANGO_SECRET_KEY=your-random-secret-key
DJANGO_DEBUG=True
ALLOWED_HOSTS=localhost,127.0.0.1
```

### Frontend (frontend/.env)

```bash
VITE_API_URL=http://localhost:8000/api
```

---

## Troubleshooting

### "ModuleNotFoundError: No module named 'model'"
→ Run Python commands from the **project root** (`ml-cloud-scheduler/`), not from `model/` or `backend/`.

### "CORS error in browser console"
→ Ensure `django-cors-headers` is installed and `CORS_ALLOW_ALL_ORIGINS = True` is set in `backend/config/settings.py`.

### "Model not ready" on dashboard
→ Train models: `python model/train_all.py` or use the "Train All Models" button on the Training page.

### "Port 8000/5173 already in use"
→ Kill existing processes: `lsof -i :8000` then `kill <PID>`, or use `python manage.py runserver 0.0.0.0:8001`.

### "torch not found"
→ Install CPU-only PyTorch: `pip install torch --index-url https://download.pytorch.org/whl/cpu`

---

## Docker (Optional)

```bash
# Build and run with Docker Compose
docker-compose up --build

# Or build individually
docker build -t ml-scheduler .
docker run -p 8000:8000 ml-scheduler
```
