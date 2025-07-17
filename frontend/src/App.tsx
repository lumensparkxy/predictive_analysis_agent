import React, { useEffect, useState } from 'react';
import { Routes, Route, Navigate } from 'react-router-dom';
import { Activity, AlertTriangle, BarChart3, Settings, TrendingUp, Users, Zap } from 'lucide-react';
import DashboardLayout from './components/dashboard/DashboardLayout';
import Dashboard from './pages/Dashboard';
import Analytics from './pages/Analytics';
import Alerts from './pages/Alerts';
import Anomalies from './pages/Anomalies';
import Predictions from './pages/Predictions';
import Settings from './pages/Settings';
import { realTimeService } from './services/websocket';
import { ENV_CONFIG } from './config/config';

// Navigation configuration
const navigation = [
  { name: 'Dashboard', href: '/', icon: BarChart3 },
  { name: 'Analytics', href: '/analytics', icon: TrendingUp },
  { name: 'Alerts', href: '/alerts', icon: AlertTriangle },
  { name: 'Anomalies', href: '/anomalies', icon: Activity },
  { name: 'Predictions', href: '/predictions', icon: Zap },
  { name: 'Settings', href: '/settings', icon: Settings },
];

function App() {
  const [isRealTimeActive, setIsRealTimeActive] = useState(false);
  const [connectionStatus, setConnectionStatus] = useState<{
    connected: boolean;
    reconnectAttempts: number;
  }>({ connected: false, reconnectAttempts: 0 });

  useEffect(() => {
    // Initialize real-time service
    realTimeService.start();
    setIsRealTimeActive(true);

    // Subscribe to connection status
    const connectionSub = realTimeService.subscribe('connection', (status) => {
      setConnectionStatus(status);
    });

    // Cleanup on unmount
    return () => {
      realTimeService.unsubscribe(connectionSub);
      realTimeService.stop();
    };
  }, []);

  const handleToggleRealTime = () => {
    if (isRealTimeActive) {
      realTimeService.stop();
      setIsRealTimeActive(false);
    } else {
      realTimeService.start();
      setIsRealTimeActive(true);
    }
  };

  return (
    <div className="min-h-screen bg-gray-50">
      <DashboardLayout
        navigation={navigation}
        isRealTimeActive={isRealTimeActive}
        connectionStatus={connectionStatus}
        onToggleRealTime={handleToggleRealTime}
      >
        <Routes>
          <Route path="/" element={<Dashboard />} />
          <Route path="/analytics" element={<Analytics />} />
          <Route path="/alerts" element={<Alerts />} />
          <Route path="/anomalies" element={<Anomalies />} />
          <Route path="/predictions" element={<Predictions />} />
          <Route path="/settings" element={<Settings />} />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </DashboardLayout>
    </div>
  );
}

export default App;