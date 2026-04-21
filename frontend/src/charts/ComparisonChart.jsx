// src/charts/ComparisonChart.jsx
import {
  Chart as ChartJS, CategoryScale, LinearScale,
  PointElement, LineElement, Filler, Legend, Tooltip,
} from 'chart.js';
import { Line } from 'react-chartjs-2';

ChartJS.register(CategoryScale, LinearScale, PointElement, LineElement, Filler, Legend, Tooltip);

const OPTIONS = (yLabel = '') => ({
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
      ticks: { color: '#475569', font: { size: 10 }, maxTicksLimit: 12 },
    },
    y: {
      grid: { color: '#162032' },
      ticks: { color: '#475569', font: { size: 11 } },
      title: { display: !!yLabel, text: yLabel, color: '#475569', font: { size: 11 } },
      suggestedMin: 0,
    },
  },
});

export function CpuComparisonChart({ reactiveRecords, predictiveRecords }) {
  if (!reactiveRecords?.length || !predictiveRecords?.length) return null;

  const labels = reactiveRecords.map(r => r.time_step);
  const data = {
    labels,
    datasets: [
      {
        label: 'Reactive CPU %',
        data: reactiveRecords.map(r => r.cpu_usage),
        borderColor: '#ef4444',
        backgroundColor: 'rgba(239,68,68,0.07)',
        borderWidth: 2,
        pointRadius: 0,
        fill: true,
        tension: 0.3,
      },
      {
        label: 'Predictive CPU %',
        data: predictiveRecords.map(r => r.cpu_usage),
        borderColor: '#10b981',
        backgroundColor: 'rgba(16,185,129,0.07)',
        borderWidth: 2,
        pointRadius: 0,
        fill: true,
        tension: 0.3,
      },
    ],
  };
  return (
    <div>
      <div className="section-title">CPU Utilization — Reactive vs Predictive</div>
      <div style={{ height: 240 }}>
        <Line data={data} options={OPTIONS('CPU %')} />
      </div>
    </div>
  );
}

export function CapacityComparisonChart({ reactiveRecords, predictiveRecords }) {
  if (!reactiveRecords?.length || !predictiveRecords?.length) return null;

  const labels = reactiveRecords.map(r => r.time_step);
  const data = {
    labels,
    datasets: [
      {
        label: 'Reactive Capacity',
        data: reactiveRecords.map(r => r.capacity),
        borderColor: '#f59e0b',
        backgroundColor: 'rgba(245,158,11,0.07)',
        borderWidth: 2,
        pointRadius: 0,
        stepped: true,
      },
      {
        label: 'Predictive Capacity',
        data: predictiveRecords.map(r => r.capacity),
        borderColor: '#8b5cf6',
        backgroundColor: 'rgba(139,92,246,0.07)',
        borderWidth: 2,
        pointRadius: 0,
        stepped: true,
      },
    ],
  };
  return (
    <div>
      <div className="section-title">Allocated Capacity — Reactive vs Predictive</div>
      <div style={{ height: 240 }}>
        <Line data={data} options={OPTIONS('Resource Units')} />
      </div>
    </div>
  );
}
