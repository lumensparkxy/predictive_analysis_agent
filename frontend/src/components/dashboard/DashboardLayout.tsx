import React, { useState, useEffect } from 'react';
import { Link, useLocation } from 'react-router-dom';
import { 
  Menu, 
  X, 
  Wifi, 
  WifiOff, 
  Play, 
  Pause, 
  Bell, 
  Settings as SettingsIcon,
  Activity,
  Database,
  Sun,
  Moon
} from 'lucide-react';
import { DASHBOARD_CONFIG } from '../../config/config';

interface NavigationItem {
  name: string;
  href: string;
  icon: React.ComponentType<{ className?: string }>;
}

interface DashboardLayoutProps {
  children: React.ReactNode;
  navigation: NavigationItem[];
  isRealTimeActive: boolean;
  connectionStatus: {
    connected: boolean;
    reconnectAttempts: number;
  };
  onToggleRealTime: () => void;
}

export default function DashboardLayout({
  children,
  navigation,
  isRealTimeActive,
  connectionStatus,
  onToggleRealTime,
}: DashboardLayoutProps) {
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [darkMode, setDarkMode] = useState(false);
  const [lastUpdateTime, setLastUpdateTime] = useState<string>('');
  const location = useLocation();

  useEffect(() => {
    // Update last update time
    const interval = setInterval(() => {
      setLastUpdateTime(new Date().toLocaleTimeString());
    }, 1000);

    return () => clearInterval(interval);
  }, []);

  const toggleDarkMode = () => {
    setDarkMode(!darkMode);
    // In a real app, you'd persist this preference
  };

  return (
    <div className={`h-screen flex ${darkMode ? 'dark' : ''}`}>
      {/* Sidebar */}
      <div className={`${sidebarOpen ? 'translate-x-0' : '-translate-x-full'} fixed inset-y-0 left-0 z-50 w-64 bg-white shadow-lg transform transition-transform duration-200 ease-in-out lg:translate-x-0 lg:static lg:inset-0`}>
        <div className="flex items-center justify-between h-16 px-4 bg-primary-600 text-white">
          <div className="flex items-center">
            <Database className="h-8 w-8 mr-3" />
            <h1 className="text-lg font-semibold">Snowflake Analytics</h1>
          </div>
          <button
            onClick={() => setSidebarOpen(false)}
            className="lg:hidden"
          >
            <X className="h-6 w-6" />
          </button>
        </div>

        <nav className="mt-5 px-2">
          {navigation.map((item) => {
            const Icon = item.icon;
            const isActive = location.pathname === item.href;
            
            return (
              <Link
                key={item.name}
                to={item.href}
                className={`group flex items-center px-2 py-2 text-base font-medium rounded-md transition-colors ${
                  isActive
                    ? 'bg-primary-100 text-primary-700'
                    : 'text-gray-600 hover:bg-gray-50 hover:text-gray-900'
                }`}
                onClick={() => setSidebarOpen(false)}
              >
                <Icon className={`mr-4 h-5 w-5 ${isActive ? 'text-primary-500' : 'text-gray-400 group-hover:text-gray-500'}`} />
                {item.name}
              </Link>
            );
          })}
        </nav>

        {/* Sidebar Footer */}
        <div className="absolute bottom-0 left-0 right-0 p-4 bg-gray-50 border-t">
          <div className="flex items-center justify-between mb-2">
            <span className="text-sm text-gray-600">Real-time Updates</span>
            <button
              onClick={onToggleRealTime}
              className={`p-1 rounded ${isRealTimeActive ? 'text-green-600 hover:bg-green-50' : 'text-gray-400 hover:bg-gray-100'}`}
            >
              {isRealTimeActive ? <Play className="h-4 w-4" /> : <Pause className="h-4 w-4" />}
            </button>
          </div>
          <div className="flex items-center text-sm">
            {connectionStatus.connected ? (
              <>
                <Wifi className="h-4 w-4 text-green-500 mr-2" />
                <span className="text-green-600">Connected</span>
              </>
            ) : (
              <>
                <WifiOff className="h-4 w-4 text-red-500 mr-2" />
                <span className="text-red-600">Disconnected</span>
              </>
            )}
          </div>
          {connectionStatus.reconnectAttempts > 0 && (
            <div className="text-xs text-gray-500 mt-1">
              Reconnect attempts: {connectionStatus.reconnectAttempts}
            </div>
          )}
        </div>
      </div>

      {/* Main content */}
      <div className="flex-1 lg:ml-0">
        {/* Top navigation */}
        <div className="bg-white shadow-sm border-b border-gray-200">
          <div className="px-4 sm:px-6 lg:px-8">
            <div className="flex justify-between h-16">
              <div className="flex items-center">
                <button
                  onClick={() => setSidebarOpen(true)}
                  className="lg:hidden -ml-0.5 -mt-0.5 h-12 w-12 inline-flex items-center justify-center rounded-md text-gray-500 hover:text-gray-900 focus:outline-none focus:ring-2 focus:ring-inset focus:ring-primary-500"
                >
                  <Menu className="h-6 w-6" />
                </button>
                <div className="ml-4 lg:ml-0">
                  <h1 className="text-2xl font-bold text-gray-900">
                    {navigation.find(item => item.href === location.pathname)?.name || 'Dashboard'}
                  </h1>
                </div>
              </div>

              <div className="flex items-center space-x-4">
                {/* Status indicators */}
                <div className="flex items-center space-x-2 text-sm text-gray-600">
                  <Activity className="h-4 w-4" />
                  <span>Last updated: {lastUpdateTime}</span>
                </div>

                {/* Real-time status */}
                <div className={`flex items-center space-x-2 px-3 py-1 rounded-full text-sm ${
                  isRealTimeActive && connectionStatus.connected
                    ? 'bg-green-100 text-green-800'
                    : 'bg-gray-100 text-gray-600'
                }`}>
                  <div className={`w-2 h-2 rounded-full ${
                    isRealTimeActive && connectionStatus.connected
                      ? 'bg-green-500 animate-pulse'
                      : 'bg-gray-400'
                  }`} />
                  <span>
                    {isRealTimeActive && connectionStatus.connected ? 'Live' : 'Paused'}
                  </span>
                </div>

                {/* Notifications */}
                <button className="p-2 text-gray-400 hover:text-gray-500">
                  <Bell className="h-5 w-5" />
                </button>

                {/* Dark mode toggle */}
                <button
                  onClick={toggleDarkMode}
                  className="p-2 text-gray-400 hover:text-gray-500"
                >
                  {darkMode ? <Sun className="h-5 w-5" /> : <Moon className="h-5 w-5" />}
                </button>

                {/* Settings */}
                <Link
                  to="/settings"
                  className="p-2 text-gray-400 hover:text-gray-500"
                >
                  <SettingsIcon className="h-5 w-5" />
                </Link>
              </div>
            </div>
          </div>
        </div>

        {/* Page content */}
        <main className="flex-1 overflow-auto">
          <div className="px-4 sm:px-6 lg:px-8 py-6">
            {children}
          </div>
        </main>
      </div>

      {/* Sidebar overlay */}
      {sidebarOpen && (
        <div 
          className="fixed inset-0 bg-black bg-opacity-50 z-40 lg:hidden"
          onClick={() => setSidebarOpen(false)}
        />
      )}
    </div>
  );
}