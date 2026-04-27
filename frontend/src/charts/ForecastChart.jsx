// src/charts/ForecastChart.jsx — Monochrome multi-line forecast chart
import { Line } from 'react-chartjs-2';
import {
  Chart as ChartJS, CategoryScale, LinearScale, PointElement,
  LineElement, Tooltip, Legend,
} from 'chart.js';

ChartJS.register(CategoryScale, LinearScale, PointElement, LineElement, Tooltip, Legend);

export default function ForecastChart({ data }) {
  if (!data?.timestamps) return null;

  const datasets = [
    { label: 'Actual', data: data.actual,   borderColor: '#ffffff', backgroundColor: '#ffffff',
      borderWidth: 2, pointRadius: 0, borderDash: [] },
  ];

  // Only show LSTM, ARIMA, Combined (skip GBR)
  if (data.lstm)     datasets.push({ label: 'LSTM',     data: data.lstm,     borderColor: '#888888', backgroundColor: '#888888',
                                      borderWidth: 1.5, pointRadius: 0, borderDash: [6, 3] });
  if (data.arima)    datasets.push({ label: 'ARIMA',    data: data.arima,    borderColor: '#555555', backgroundColor: '#555555',
                                      borderWidth: 1.5, pointRadius: 0, borderDash: [2, 2] });
  if (data.combined) datasets.push({ label: 'Combined', data: data.combined, borderColor: '#bbbbbb', backgroundColor: '#bbbbbb',
                                      borderWidth: 2, pointRadius: 0, borderDash: [10, 4] });

  return (
    <div style={{ height: 350 }}>
      <Line
        data={{ labels: data.timestamps, datasets }}
        options={{
          responsive: true,
          maintainAspectRatio: false,
          plugins: {
            legend: {
              labels: { color: '#9a9a9a', font: { family: 'Inter', size: 11, weight: 600 },
                        usePointStyle: true, pointStyle: 'line', padding: 16 },
            },
            tooltip: {
              backgroundColor: '#111', titleColor: '#e5e5e5', bodyColor: '#9a9a9a',
              borderColor: '#2a2a2a', borderWidth: 1,
              titleFont: { family: 'JetBrains Mono', size: 11 },
              bodyFont: { family: 'JetBrains Mono', size: 11 },
            },
          },
          scales: {
            x: { ticks: { color: '#555', font: { size: 10 } }, grid: { color: '#1a1a1a' } },
            y: { ticks: { color: '#555', font: { size: 10, family: 'JetBrains Mono' } }, grid: { color: '#1a1a1a' } },
          },
          interaction: { intersect: false, mode: 'index' },
        }}
      />
    </div>
  );
}
