import React, { useState } from 'react';
import { 
  Settings as SettingsIcon, 
  Bell, 
  Shield, 
  Database, 
  Zap, 
  Save,
  RefreshCw,
  AlertTriangle
} from 'lucide-react';

export default function Settings() {
  const [settings, setSettings] = useState({
    notifications: {
      emailAlerts: true,
      slackNotifications: true,
      pushNotifications: false,
      alertThreshold: 'medium',
    },
    dashboard: {
      refreshInterval: 30,
      autoRefresh: true,
      darkMode: false,
      compactView: false,
    },
    monitoring: {
      costThreshold: 1000,
      usageThreshold: 90,
      anomalyThreshold: 0.8,
      alertRetention: 30,
    },
    system: {
      apiKey: 'demo_key_001',
      maxRetries: 3,
      timeout: 10000,
      debugMode: false,
    },
  });

  const [activeTab, setActiveTab] = useState('notifications');

  const handleSave = () => {
    // Save settings logic here
    console.log('Settings saved:', settings);
  };

  const handleReset = () => {
    // Reset to defaults
    setSettings({
      notifications: {
        emailAlerts: true,
        slackNotifications: true,
        pushNotifications: false,
        alertThreshold: 'medium',
      },
      dashboard: {
        refreshInterval: 30,
        autoRefresh: true,
        darkMode: false,
        compactView: false,
      },
      monitoring: {
        costThreshold: 1000,
        usageThreshold: 90,
        anomalyThreshold: 0.8,
        alertRetention: 30,
      },
      system: {
        apiKey: 'demo_key_001',
        maxRetries: 3,
        timeout: 10000,
        debugMode: false,
      },
    });
  };

  const tabs = [
    { id: 'notifications', label: 'Notifications', icon: Bell },
    { id: 'dashboard', label: 'Dashboard', icon: SettingsIcon },
    { id: 'monitoring', label: 'Monitoring', icon: Database },
    { id: 'system', label: 'System', icon: Shield },
  ];

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold text-gray-900">Settings</h1>
        <p className="text-gray-600 mt-2">
          Configure your dashboard preferences and system settings
        </p>
      </div>

      <div className="bg-white rounded-lg shadow">
        {/* Tabs */}
        <div className="border-b border-gray-200">
          <nav className="flex space-x-8 px-6">
            {tabs.map((tab) => {
              const Icon = tab.icon;
              return (
                <button
                  key={tab.id}
                  onClick={() => setActiveTab(tab.id)}
                  className={`flex items-center py-4 px-1 border-b-2 font-medium text-sm ${
                    activeTab === tab.id
                      ? 'border-blue-500 text-blue-600'
                      : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                  }`}
                >
                  <Icon className="h-5 w-5 mr-2" />
                  {tab.label}
                </button>
              );
            })}
          </nav>
        </div>

        {/* Tab content */}
        <div className="p-6">
          {activeTab === 'notifications' && (
            <div className="space-y-6">
              <h3 className="text-lg font-medium text-gray-900">Notification Settings</h3>
              
              <div className="space-y-4">
                <div className="flex items-center justify-between">
                  <div>
                    <label className="text-sm font-medium text-gray-700">Email Alerts</label>
                    <p className="text-sm text-gray-500">Receive alert notifications via email</p>
                  </div>
                  <input
                    type="checkbox"
                    checked={settings.notifications.emailAlerts}
                    onChange={(e) => setSettings({
                      ...settings,
                      notifications: { ...settings.notifications, emailAlerts: e.target.checked }
                    })}
                    className="h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300 rounded"
                  />
                </div>
                
                <div className="flex items-center justify-between">
                  <div>
                    <label className="text-sm font-medium text-gray-700">Slack Notifications</label>
                    <p className="text-sm text-gray-500">Send notifications to Slack channels</p>
                  </div>
                  <input
                    type="checkbox"
                    checked={settings.notifications.slackNotifications}
                    onChange={(e) => setSettings({
                      ...settings,
                      notifications: { ...settings.notifications, slackNotifications: e.target.checked }
                    })}
                    className="h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300 rounded"
                  />
                </div>
                
                <div className="flex items-center justify-between">
                  <div>
                    <label className="text-sm font-medium text-gray-700">Push Notifications</label>
                    <p className="text-sm text-gray-500">Browser push notifications</p>
                  </div>
                  <input
                    type="checkbox"
                    checked={settings.notifications.pushNotifications}
                    onChange={(e) => setSettings({
                      ...settings,
                      notifications: { ...settings.notifications, pushNotifications: e.target.checked }
                    })}
                    className="h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300 rounded"
                  />
                </div>
                
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">Alert Threshold</label>
                  <select
                    value={settings.notifications.alertThreshold}
                    onChange={(e) => setSettings({
                      ...settings,
                      notifications: { ...settings.notifications, alertThreshold: e.target.value }
                    })}
                    className="block w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500"
                  >
                    <option value="low">Low</option>
                    <option value="medium">Medium</option>
                    <option value="high">High</option>
                    <option value="critical">Critical</option>
                  </select>
                </div>
              </div>
            </div>
          )}

          {activeTab === 'dashboard' && (
            <div className="space-y-6">
              <h3 className="text-lg font-medium text-gray-900">Dashboard Settings</h3>
              
              <div className="space-y-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Refresh Interval (seconds)
                  </label>
                  <input
                    type="number"
                    value={settings.dashboard.refreshInterval}
                    onChange={(e) => setSettings({
                      ...settings,
                      dashboard: { ...settings.dashboard, refreshInterval: parseInt(e.target.value) }
                    })}
                    className="block w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500"
                    min="10"
                    max="300"
                  />
                </div>
                
                <div className="flex items-center justify-between">
                  <div>
                    <label className="text-sm font-medium text-gray-700">Auto Refresh</label>
                    <p className="text-sm text-gray-500">Automatically refresh dashboard data</p>
                  </div>
                  <input
                    type="checkbox"
                    checked={settings.dashboard.autoRefresh}
                    onChange={(e) => setSettings({
                      ...settings,
                      dashboard: { ...settings.dashboard, autoRefresh: e.target.checked }
                    })}
                    className="h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300 rounded"
                  />
                </div>
                
                <div className="flex items-center justify-between">
                  <div>
                    <label className="text-sm font-medium text-gray-700">Dark Mode</label>
                    <p className="text-sm text-gray-500">Enable dark theme</p>
                  </div>
                  <input
                    type="checkbox"
                    checked={settings.dashboard.darkMode}
                    onChange={(e) => setSettings({
                      ...settings,
                      dashboard: { ...settings.dashboard, darkMode: e.target.checked }
                    })}
                    className="h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300 rounded"
                  />
                </div>
                
                <div className="flex items-center justify-between">
                  <div>
                    <label className="text-sm font-medium text-gray-700">Compact View</label>
                    <p className="text-sm text-gray-500">Use compact layout for widgets</p>
                  </div>
                  <input
                    type="checkbox"
                    checked={settings.dashboard.compactView}
                    onChange={(e) => setSettings({
                      ...settings,
                      dashboard: { ...settings.dashboard, compactView: e.target.checked }
                    })}
                    className="h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300 rounded"
                  />
                </div>
              </div>
            </div>
          )}

          {activeTab === 'monitoring' && (
            <div className="space-y-6">
              <h3 className="text-lg font-medium text-gray-900">Monitoring Settings</h3>
              
              <div className="space-y-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Cost Threshold ($)
                  </label>
                  <input
                    type="number"
                    value={settings.monitoring.costThreshold}
                    onChange={(e) => setSettings({
                      ...settings,
                      monitoring: { ...settings.monitoring, costThreshold: parseFloat(e.target.value) }
                    })}
                    className="block w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500"
                    min="0"
                    step="100"
                  />
                </div>
                
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Usage Threshold (%)
                  </label>
                  <input
                    type="number"
                    value={settings.monitoring.usageThreshold}
                    onChange={(e) => setSettings({
                      ...settings,
                      monitoring: { ...settings.monitoring, usageThreshold: parseFloat(e.target.value) }
                    })}
                    className="block w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500"
                    min="0"
                    max="100"
                  />
                </div>
                
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Anomaly Threshold (0-1)
                  </label>
                  <input
                    type="number"
                    value={settings.monitoring.anomalyThreshold}
                    onChange={(e) => setSettings({
                      ...settings,
                      monitoring: { ...settings.monitoring, anomalyThreshold: parseFloat(e.target.value) }
                    })}
                    className="block w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500"
                    min="0"
                    max="1"
                    step="0.1"
                  />
                </div>
                
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Alert Retention (days)
                  </label>
                  <input
                    type="number"
                    value={settings.monitoring.alertRetention}
                    onChange={(e) => setSettings({
                      ...settings,
                      monitoring: { ...settings.monitoring, alertRetention: parseInt(e.target.value) }
                    })}
                    className="block w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500"
                    min="1"
                    max="365"
                  />
                </div>
              </div>
            </div>
          )}

          {activeTab === 'system' && (
            <div className="space-y-6">
              <h3 className="text-lg font-medium text-gray-900">System Settings</h3>
              
              <div className="space-y-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    API Key
                  </label>
                  <input
                    type="text"
                    value={settings.system.apiKey}
                    onChange={(e) => setSettings({
                      ...settings,
                      system: { ...settings.system, apiKey: e.target.value }
                    })}
                    className="block w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500"
                    placeholder="Enter API key"
                  />
                </div>
                
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Max Retries
                  </label>
                  <input
                    type="number"
                    value={settings.system.maxRetries}
                    onChange={(e) => setSettings({
                      ...settings,
                      system: { ...settings.system, maxRetries: parseInt(e.target.value) }
                    })}
                    className="block w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500"
                    min="0"
                    max="10"
                  />
                </div>
                
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Timeout (ms)
                  </label>
                  <input
                    type="number"
                    value={settings.system.timeout}
                    onChange={(e) => setSettings({
                      ...settings,
                      system: { ...settings.system, timeout: parseInt(e.target.value) }
                    })}
                    className="block w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500"
                    min="1000"
                    max="60000"
                    step="1000"
                  />
                </div>
                
                <div className="flex items-center justify-between">
                  <div>
                    <label className="text-sm font-medium text-gray-700">Debug Mode</label>
                    <p className="text-sm text-gray-500">Enable debug logging</p>
                  </div>
                  <input
                    type="checkbox"
                    checked={settings.system.debugMode}
                    onChange={(e) => setSettings({
                      ...settings,
                      system: { ...settings.system, debugMode: e.target.checked }
                    })}
                    className="h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300 rounded"
                  />
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Action buttons */}
        <div className="px-6 py-4 bg-gray-50 border-t border-gray-200 rounded-b-lg">
          <div className="flex justify-end space-x-3">
            <button
              onClick={handleReset}
              className="px-4 py-2 text-sm font-medium text-gray-700 bg-white border border-gray-300 rounded-md hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500"
            >
              <RefreshCw className="h-4 w-4 mr-2 inline" />
              Reset
            </button>
            <button
              onClick={handleSave}
              className="px-4 py-2 text-sm font-medium text-white bg-blue-600 border border-transparent rounded-md hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500"
            >
              <Save className="h-4 w-4 mr-2 inline" />
              Save Settings
            </button>
          </div>
        </div>
      </div>

      {/* Warning note */}
      <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-4">
        <div className="flex">
          <AlertTriangle className="h-5 w-5 text-yellow-400 mr-2" />
          <div>
            <h3 className="text-sm font-medium text-yellow-800">Configuration Note</h3>
            <p className="text-sm text-yellow-700 mt-1">
              Settings are currently stored locally in your browser. In a production environment, 
              these would be persisted to a secure backend service.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}