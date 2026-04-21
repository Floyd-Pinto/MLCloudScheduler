// src/App.jsx
import { BrowserRouter, Routes, Route } from 'react-router-dom';
import { Toaster } from 'react-hot-toast';
import Sidebar              from './components/Sidebar';
import DashboardPage        from './pages/DashboardPage';
import SimulationPage       from './pages/SimulationPage';
import TrainingPage         from './pages/TrainingPage';
import ModelComparisonPage  from './pages/ModelComparisonPage';
import ComparisonPage       from './pages/ComparisonPage';
import MetricsPage          from './pages/MetricsPage';
import LogsPage             from './pages/LogsPage';
import './styles/global.css';

export default function App() {
  return (
    <BrowserRouter>
      <div className="app-shell">
        <Sidebar />
        <main className="main-content">
          <Routes>
            <Route path="/"              element={<DashboardPage        />} />
            <Route path="/simulation"    element={<SimulationPage       />} />
            <Route path="/training"      element={<TrainingPage         />} />
            <Route path="/models"        element={<ModelComparisonPage  />} />
            <Route path="/comparison"    element={<ComparisonPage       />} />
            <Route path="/metrics"       element={<MetricsPage          />} />
            <Route path="/logs"          element={<LogsPage             />} />
          </Routes>
        </main>
      </div>
      <Toaster
        position="top-right"
        toastOptions={{
          style: {
            background: 'var(--bg-card)',
            color: 'var(--text-primary)',
            border: '1px solid var(--border)',
            fontSize: '13px',
          },
        }}
      />
    </BrowserRouter>
  );
}
