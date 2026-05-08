// src/charts/WorkloadChart.jsx — Multi-resource workload chart
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
      labels: { color: '#9a9a9a', font: { size: 11, family: 'Inter' }, boxWidth: 10, usePointStyle: true, pointStyle: 'line' },
    },
    tooltip: {
      backgroundColor: '#111', borderColor: '#2a2a2a', borderWidth: 1,
      titleColor: '#e5e5e5', bodyColor: '#9a9a9a',
      titleFont: { family: 'JetBrains Mono', size: 11 },
      bodyFont: { family: 'JetBrains Mono', size: 11 },
    },
  },
  scales: {
    x: {
      grid: { color: '#1a1a1a' },
      ticks: { color: '#555', font: { size: 10 }, maxTicksLimit: 15 },
    },
    y: {
      grid: { color: '#1a1a1a' },
      ticks: { color: '#555', font: { size: 11, family: 'JetBrains Mono' } },
      suggestedMin: 0,
    },
  },
};

export default function WorkloadChart({ records, title = 'Workload Over Time' }) {
  if (!records || records.length === 0) return null;

  const labels = records.map(r => r.time_step);
  const datasets = [
    {
      label: 'CPU',
      data: records.map(r => r.workload ?? r.cpu_usage),
      borderColor: '#e5e5e5',
      backgroundColor: 'rgba(229,229,229,0.05)',
      borderWidth: 2,
      pointRadius: 0,
      fill: true,
      tension: 0.4,
    },
  ];

  // Add memory and network if available
  if (records[0]?.memory_usage != null) {
    datasets.push({
      label: 'Memory',
      data: records.map(r => r.memory_usage),
      borderColor: '#888888',
      backgroundColor: 'rgba(136,136,136,0.05)',
      borderWidth: 1.5,
      pointRadius: 0,
      fill: false,
      tension: 0.4,
      borderDash: [6, 3],
    });
  }

  if (records[0]?.network_io != null) {
    datasets.push({
      label: 'Network',
      data: records.map(r => r.network_io),
      borderColor: '#555555',
      backgroundColor: 'rgba(85,85,85,0.05)',
      borderWidth: 1.5,
      pointRadius: 0,
      fill: false,
      tension: 0.4,
      borderDash: [2, 2],
    });
  }

  return (
    <div>
      <div className="section-title">{title}</div>
      <div style={{ height: 250 }}>
        <Line data={{ labels, datasets }} options={OPTIONS} />
      </div>
    </div>
  );
}
