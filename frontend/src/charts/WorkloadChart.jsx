// src/charts/WorkloadChart.jsx
import {
  Chart as ChartJS, CategoryScale, LinearScale,
  PointElement, LineElement, Filler, Legend, Tooltip,
} from 'chart.js';
import { Line } from 'react-chartjs-2';

ChartJS.register(CategoryScale, LinearScale, PointElement, LineElement, Filler, Legend, Tooltip);

const OPTIONS = {
  responsive: true,
  maintainAspectRatio: false,
  interaction: { mode: 'index', intersect: false },
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
    x: {
      grid: { color: '#162032' },
      ticks: { color: '#475569', font: { size: 11 }, maxTicksLimit: 15 },
    },
    y: {
      grid: { color: '#162032' },
      ticks: { color: '#475569', font: { size: 11 } },
      suggestedMin: 0,
    },
  },
};

export default function WorkloadChart({ records, title = 'Workload Over Time' }) {
  if (!records || records.length === 0) return null;

  const labels   = records.map(r => r.time_step);
  const workload = records.map(r => r.workload ?? r.workload);

  const data = {
    labels,
    datasets: [{
      label: 'Workload',
      data: workload,
      borderColor: '#3b82f6',
      backgroundColor: 'rgba(59,130,246,0.08)',
      borderWidth: 2,
      pointRadius: 0,
      fill: true,
      tension: 0.4,
    }],
  };

  return (
    <div>
      <div className="section-title">{title}</div>
      <div style={{ height: 220 }}>
        <Line data={data} options={OPTIONS} />
      </div>
    </div>
  );
}
