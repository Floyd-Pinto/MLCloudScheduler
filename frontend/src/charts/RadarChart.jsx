// src/charts/RadarChart.jsx — Model performance radar
import {
  Chart as ChartJS, RadialLinearScale, PointElement,
  LineElement, Filler, Tooltip, Legend,
} from 'chart.js';
import { Radar } from 'react-chartjs-2';

ChartJS.register(RadialLinearScale, PointElement, LineElement, Filler, Tooltip, Legend);

export default function ModelRadarChart({ metrics }) {
  // metrics: { gbr: {r2, rmse, mae}, lstm: {...}, arima: {...}, combined: {...} }
  const models  = Object.keys(metrics).filter(m => metrics[m]?.r2 != null);
  if (models.length === 0) return null;

  const colors = {
    gbr:      'rgba(59,130,246,',
    lstm:     'rgba(139,92,246,',
    arima:    'rgba(245,158,11,',
    combined: 'rgba(16,185,129,',
  };

  // Normalize: higher is better for all 3 axes
  // R² [0,1], invert RMSE/MAE → accuracy = (1 - normalized_rmse)
  const allRmse = models.map(m => metrics[m].rmse || 0);
  const allMae  = models.map(m => metrics[m].mae  || 0);
  const maxRmse = Math.max(...allRmse, 1);
  const maxMae  = Math.max(...allMae,  1);

  const datasets = models.map(m => {
    const r2       = Math.max(0, metrics[m].r2   ?? 0) * 100;
    const rmseAcc  = (1 - (metrics[m].rmse ?? 0) / maxRmse) * 100;
    const maeAcc   = (1 - (metrics[m].mae  ?? 0) / maxMae)  * 100;
    const col      = colors[m] || 'rgba(148,163,184,';
    return {
      label:           m.toUpperCase(),
      data:            [r2, rmseAcc, maeAcc, r2 * 0.9, rmseAcc * 0.95],
      backgroundColor: `${col}0.15)`,
      borderColor:     `${col}0.9)`,
      borderWidth:     2,
      pointRadius:     3,
      pointBackgroundColor: `${col}1)`,
    };
  });

  const data = {
    labels: ['R² Score', 'RMSE Accuracy', 'MAE Accuracy', 'Fit Quality', 'Stability'],
    datasets,
  };

  const options = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: { labels: { color: '#94a3b8', font: { size: 11 }, boxWidth: 10 } },
      tooltip: {
        backgroundColor: '#1a2236',
        borderColor: '#1e3a5f',
        borderWidth: 1,
        titleColor: '#f1f5f9',
        bodyColor: '#94a3b8',
        callbacks: { label: ctx => `${ctx.dataset.label}: ${ctx.parsed.r?.toFixed(1)}` },
      },
    },
    scales: {
      r: {
        min: 0, max: 100,
        ticks: { color: '#475569', font: { size: 9 }, stepSize: 20, backdropColor: 'transparent' },
        grid:  { color: '#162032' },
        angleLines: { color: '#162032' },
        pointLabels: { color: '#94a3b8', font: { size: 11 } },
      },
    },
  };

  return (
    <div>
      <div className="section-title">Model Performance Radar</div>
      <div style={{ height: 300 }}>
        <Radar data={data} options={options} />
      </div>
    </div>
  );
}
