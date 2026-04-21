// src/charts/ForecastChart.jsx — Multi-model actual vs predicted
import {
  Chart as ChartJS, CategoryScale, LinearScale,
  PointElement, LineElement, Filler, Legend, Tooltip,
} from 'chart.js';
import { Line } from 'react-chartjs-2';

ChartJS.register(CategoryScale, LinearScale, PointElement, LineElement, Filler, Legend, Tooltip);

const MODEL_COLORS = {
  actual:   { border: '#94a3b8', bg: 'rgba(148,163,184,0.06)' },
  gbr:      { border: '#3b82f6', bg: 'rgba(59,130,246,0.06)'  },
  lstm:     { border: '#8b5cf6', bg: 'rgba(139,92,246,0.06)'  },
  arima:    { border: '#f59e0b', bg: 'rgba(245,158,11,0.06)'  },
  combined: { border: '#10b981', bg: 'rgba(16,185,129,0.06)'  },
};

const MODEL_LABELS = {
  actual:   'Actual Workload',
  gbr:      'GBR Forecast',
  lstm:     'LSTM Forecast',
  arima:    'ARIMA Forecast',
  combined: 'Combined (LSTM+ARIMA)',
};

const OPTIONS = {
  responsive: true,
  maintainAspectRatio: false,
  interaction: { mode: 'index', intersect: false },
  plugins: {
    legend: {
      labels: { color: '#94a3b8', font: { size: 11, family: 'Inter' }, boxWidth: 10, padding: 14 },
    },
    tooltip: {
      backgroundColor: '#1a2236',
      borderColor: '#1e3a5f',
      borderWidth: 1,
      titleColor: '#f1f5f9',
      bodyColor: '#94a3b8',
      callbacks: {
        label: ctx => `${ctx.dataset.label}: ${ctx.parsed.y?.toFixed(2)}`,
      },
    },
  },
  scales: {
    x: {
      grid: { color: '#162032' },
      ticks: { color: '#475569', font: { size: 10 }, maxTicksLimit: 15 },
      title: { display: true, text: 'Time Step', color: '#475569', font: { size: 11 } },
    },
    y: {
      grid: { color: '#162032' },
      ticks: { color: '#475569', font: { size: 10 } },
      title: { display: true, text: 'Workload', color: '#475569', font: { size: 11 } },
    },
  },
};

export default function ForecastChart({ chartData, visibleModels, title }) {
  if (!chartData?.timestamps?.length) return null;

  const labels   = chartData.timestamps;
  const datasets = [];
  const keys     = ['actual', 'gbr', 'lstm', 'arima', 'combined'];

  for (const key of keys) {
    if (!chartData[key] || (visibleModels && !visibleModels.includes(key))) continue;
    const col = MODEL_COLORS[key];
    datasets.push({
      label:           MODEL_LABELS[key] || key,
      data:            chartData[key],
      borderColor:     col.border,
      backgroundColor: col.bg,
      borderWidth:     key === 'actual' ? 2 : 1.5,
      pointRadius:     0,
      fill:            false,
      tension:         0.3,
      borderDash:      key === 'actual' ? [] : [],
    });
  }

  return (
    <div>
      {title && <div className="section-title">{title}</div>}
      <div style={{ height: 320 }}>
        <Line data={{ labels, datasets }} options={OPTIONS} />
      </div>
    </div>
  );
}
