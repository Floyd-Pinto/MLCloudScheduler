// src/charts/BarCompareChart.jsx
import {
  Chart as ChartJS, CategoryScale, LinearScale,
  BarElement, Legend, Tooltip,
} from 'chart.js';
import { Bar } from 'react-chartjs-2';

ChartJS.register(CategoryScale, LinearScale, BarElement, Legend, Tooltip);

export default function BarCompareChart({ reactive, predictive, metrics, title }) {
  if (!reactive || !predictive) return null;

  const data = {
    labels: metrics.map(m => m.label),
    datasets: [
      {
        label: 'Reactive',
        data: metrics.map(m => reactive[m.key] ?? 0),
        backgroundColor: 'rgba(239,68,68,0.75)',
        borderRadius: 6,
      },
      {
        label: 'Predictive',
        data: metrics.map(m => predictive[m.key] ?? 0),
        backgroundColor: 'rgba(16,185,129,0.75)',
        borderRadius: 6,
      },
    ],
  };

  const options = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        labels: { color: '#94a3b8', font: { size: 12, family: 'Inter' }, boxWidth: 12 },
      },
      tooltip: {
        backgroundColor: '#1a2236',
        borderColor: '#1e3a5f',
        borderWidth: 1,
        titleColor: '#f1f5f9',
        bodyColor: '#94a3b8',
      },
    },
    scales: {
      x: { grid: { display: false }, ticks: { color: '#94a3b8', font: { size: 12 } } },
      y: { grid: { color: '#162032' }, ticks: { color: '#475569', font: { size: 11 } }, beginAtZero: true },
    },
  };

  return (
    <div>
      {title && <div className="section-title">{title}</div>}
      <div style={{ height: 260 }}>
        <Bar data={data} options={options} />
      </div>
    </div>
  );
}
