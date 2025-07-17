import React from 'react';
import { useQuery } from 'react-query';
import { AlertTriangle, CheckCircle, Clock, User } from 'lucide-react';
import { apiService } from '../services/api';
import { Alert } from '../types';
import LoadingSpinner from '../components/common/LoadingSpinner';
import ErrorMessage from '../components/common/ErrorMessage';

const SeverityBadge: React.FC<{ severity: string }> = ({ severity }) => {
  const getColorClass = (severity: string) => {
    switch (severity) {
      case 'critical': return 'bg-red-100 text-red-800';
      case 'high': return 'bg-orange-100 text-orange-800';
      case 'medium': return 'bg-yellow-100 text-yellow-800';
      case 'low': return 'bg-green-100 text-green-800';
      default: return 'bg-gray-100 text-gray-800';
    }
  };

  return (
    <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${getColorClass(severity)}`}>
      {severity}
    </span>
  );
};

const AlertCard: React.FC<{ alert: Alert }> = ({ alert }) => {
  const handleAcknowledge = async () => {
    try {
      await apiService.acknowledgeAlert(alert.id);
    } catch (error) {
      console.error('Failed to acknowledge alert:', error);
    }
  };

  const handleResolve = async () => {
    try {
      await apiService.resolveAlert(alert.id);
    } catch (error) {
      console.error('Failed to resolve alert:', error);
    }
  };

  return (
    <div className="bg-white rounded-lg shadow p-6 border-l-4 border-l-red-500">
      <div className="flex items-start justify-between">
        <div className="flex-1">
          <div className="flex items-center space-x-2 mb-2">
            <AlertTriangle className="h-5 w-5 text-red-500" />
            <h3 className="text-lg font-medium text-gray-900">{alert.title}</h3>
            <SeverityBadge severity={alert.severity} />
          </div>
          <p className="text-gray-600 mb-3">{alert.description}</p>
          <div className="flex items-center space-x-4 text-sm text-gray-500">
            <div className="flex items-center">
              <Clock className="h-4 w-4 mr-1" />
              {new Date(alert.created_at).toLocaleString()}
            </div>
            <div className="flex items-center">
              <User className="h-4 w-4 mr-1" />
              {alert.assigned_to}
            </div>
            {alert.warehouse && (
              <div className="flex items-center">
                <span className="font-medium">Warehouse: {alert.warehouse}</span>
              </div>
            )}
          </div>
        </div>
        <div className="flex flex-col space-y-2">
          {!alert.acknowledged && (
            <button
              onClick={handleAcknowledge}
              className="px-3 py-1 text-sm bg-yellow-100 text-yellow-800 rounded hover:bg-yellow-200"
            >
              Acknowledge
            </button>
          )}
          {alert.status === 'active' && (
            <button
              onClick={handleResolve}
              className="px-3 py-1 text-sm bg-green-100 text-green-800 rounded hover:bg-green-200"
            >
              Resolve
            </button>
          )}
          {alert.acknowledged && (
            <div className="flex items-center text-sm text-green-600">
              <CheckCircle className="h-4 w-4 mr-1" />
              Acknowledged
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default function Alerts() {
  const { data: alertsData, isLoading, error, refetch } = useQuery(
    'active-alerts',
    () => apiService.getActiveAlerts(),
    {
      refetchInterval: 30000,
    }
  );

  if (isLoading) {
    return <LoadingSpinner />;
  }

  if (error) {
    return <ErrorMessage message="Failed to load alerts" onRetry={refetch} />;
  }

  const alerts = alertsData?.data?.alerts || [];
  const summary = alertsData?.data?.summary || {};

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold text-gray-900">Alerts</h1>
        <p className="text-gray-600 mt-2">
          Monitor and manage active alerts across your Snowflake environment
        </p>
      </div>

      {/* Alert summary */}
      <div className="bg-white rounded-lg shadow p-6">
        <h2 className="text-xl font-semibold text-gray-900 mb-4">Alert Summary</h2>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <div className="text-center">
            <div className="text-2xl font-bold text-gray-900">{summary.total_alerts || 0}</div>
            <div className="text-sm text-gray-500">Total Alerts</div>
          </div>
          <div className="text-center">
            <div className="text-2xl font-bold text-red-600">{summary.critical || 0}</div>
            <div className="text-sm text-gray-500">Critical</div>
          </div>
          <div className="text-center">
            <div className="text-2xl font-bold text-orange-600">{summary.high || 0}</div>
            <div className="text-sm text-gray-500">High</div>
          </div>
          <div className="text-center">
            <div className="text-2xl font-bold text-green-600">{summary.acknowledged || 0}</div>
            <div className="text-sm text-gray-500">Acknowledged</div>
          </div>
        </div>
      </div>

      {/* Active alerts */}
      <div className="space-y-4">
        <h2 className="text-xl font-semibold text-gray-900">Active Alerts</h2>
        {alerts.length === 0 ? (
          <div className="bg-white rounded-lg shadow p-6 text-center">
            <CheckCircle className="h-12 w-12 text-green-500 mx-auto mb-4" />
            <h3 className="text-lg font-medium text-gray-900 mb-2">No Active Alerts</h3>
            <p className="text-gray-600">All systems are running normally.</p>
          </div>
        ) : (
          <div className="space-y-4">
            {alerts.map((alert: Alert) => (
              <AlertCard key={alert.id} alert={alert} />
            ))}
          </div>
        )}
      </div>
    </div>
  );
}