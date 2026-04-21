// src/services/api.js — updated with new endpoints
import axios from 'axios';

const BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000/api';

const api = axios.create({
  baseURL: BASE_URL,
  headers: { 'Content-Type': 'application/json' },
  timeout: 300000,  // 5 min for train-all
});

// ── Simulation ──────────────────────────────────────────────
export const simulationAPI = {
  generate: (data) => api.post('/simulation/generate/', data),
  listRuns: () => api.get('/simulation/runs/'),
  getRun:   (id) => api.get(`/simulation/runs/${id}/`),
  deleteRun:(id) => api.delete(`/simulation/runs/${id}/`),
};

// ── Scheduler ───────────────────────────────────────────────
export const schedulerAPI = {
  runReactive:   (data) => api.post('/scheduler/reactive/', data),
  runPredictive: (data) => api.post('/scheduler/predictive/', data),
  compare:       (data) => api.post('/scheduler/compare/', data),
  listRuns:      () => api.get('/scheduler/runs/'),
  getRun:        (id) => api.get(`/scheduler/runs/${id}/`),
};

// ── ML Model ────────────────────────────────────────────────
export const mlAPI = {
  train:          (data) => api.post('/ml/train/', data),
  trainAll:       ()     => api.post('/ml/train/', { model_type: 'all' }),
  status:         ()     => api.get('/ml/status/'),
  history:        (mt)   => api.get('/ml/history/', { params: mt ? { model_type: mt } : {} }),
  predict:        (data) => api.post('/ml/predict/', data),
  predictAll:     (data) => api.post('/ml/predict-all/', data),
  compareModels:  (data) => api.post('/ml/compare-models/', data),
  comparisonList: ()     => api.get('/ml/compare-models/'),
};

// ── Metrics ─────────────────────────────────────────────────
export const metricsAPI = {
  list:    (params) => api.get('/metrics/', { params }),
  summary: () => api.get('/metrics/summary/'),
};

// ── Evaluation ──────────────────────────────────────────────
export const evaluationAPI = {
  run:        (data)    => api.post('/evaluation/run/', data),
  list:       ()        => api.get('/evaluation/'),
  comparison: (pattern) => api.get('/evaluation/comparison/', { params: pattern ? { pattern } : {} }),
};

export default api;
