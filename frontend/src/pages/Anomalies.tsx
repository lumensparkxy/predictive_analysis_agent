import React from 'react';
import { useQuery } from 'react-query';
import { Activity, AlertTriangle, Clock, TrendingUp } from 'lucide-react';
import { apiService } from '../services/api';
import { Anomaly } from '../types';
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

const AnomalyCard: React.FC<{ anomaly: Anomaly }> = ({ anomaly }) => {
  const deviationColor = anomaly.deviation > 100 ? 'text-red-600' : 
                         anomaly.deviation > 50 ? 'text-orange-600' : 
                         'text-yellow-600';

  return (
    <div className="bg-white rounded-lg shadow p-6 border-l-4 border-l-orange-500">
      <div className="flex items-start justify-between">
        <div className="flex-1">
          <div className="flex items-center space-x-2 mb-2">
            <Activity className="h-5 w-5 text-orange-500" />
            <h3 className="text-lg font-medium text-gray-900">{anomaly.title}</h3>
            <SeverityBadge severity={anomaly.severity} />
          </div>
          <p className="text-gray-600 mb-3">{anomaly.description}</p>
          
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-4">
            <div>
              <div className="text-sm text-gray-500">Current Value</div>
              <div className="font-medium">{anomaly.value.toLocaleString()}</div>
            </div>
            <div>
              <div className="text-sm text-gray-500">Baseline</div>
              <div className="font-medium">{anomaly.baseline.toLocaleString()}</div>
            </div>
            <div>
              <div className="text-sm text-gray-500">Deviation</div>
              <div className={`font-medium ${deviationColor}`}>
                {anomaly.deviation.toFixed(1)}%
              </div>
            </div>
            <div>
              <div className="text-sm text-gray-500">Confidence</div>
              <div className="font-medium">{(anomaly.confidence * 100).toFixed(1)}%</div>
            </div>
          </div>
          
          <div className="flex items-center space-x-4 text-sm text-gray-500">
            <div className="flex items-center">
              <Clock className="h-4 w-4 mr-1" />
              {new Date(anomaly.detected_at).toLocaleString()}
            </div>
            {anomaly.warehouse && (
              <div className="flex items-center">
                <span className="font-medium">Warehouse: {anomaly.warehouse}</span>
              </div>
            )}
            {anomaly.user && (
              <div className="flex items-center">
                <span className="font-medium">User: {anomaly.user}</span>
              </div>
            )}
          </div>
        </div>
        <div className="flex flex-col items-end">
          <div className={`px-2 py-1 rounded text-xs font-medium ${
            anomaly.status === 'active' ? 'bg-red-100 text-red-800' :
            anomaly.status === 'investigating' ? 'bg-yellow-100 text-yellow-800' :
            'bg-green-100 text-green-800'
          }`}>
            {anomaly.status}
          </div>
        </div>
      </div>
    </div>
  );
};

export default function Anomalies() {
  const { data: anomaliesData, isLoading, error, refetch } = useQuery(
    'current-anomalies',
    () => apiService.getCurrentAnomalies(),
    {
      refetchInterval: 30000,
    }
  );

  const { data: statisticsData } = useQuery(
    'anomaly-statistics',
    () => apiService.getAnomalyStatistics(),
    {
      refetchInterval: 60000,
    }
  );

  if (isLoading) {
    return <LoadingSpinner />;
  }

  if (error) {
    return <ErrorMessage message="Failed to load anomalies" onRetry={refetch} />;
  }

  const anomalies = anomaliesData?.data?.anomalies || [];
  const summary = anomaliesData?.data?.summary || {};
  const stats = statisticsData?.data || {};

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold text-gray-900">Anomalies</h1>
        <p className="text-gray-600 mt-2">
          Detect and investigate unusual patterns in your Snowflake environment
        </p>
      </div>

      {/* Anomaly summary */}
      <div className="bg-white rounded-lg shadow p-6">
        <h2 className="text-xl font-semibold text-gray-900 mb-4">Anomaly Summary</h2>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <div className="text-center">
            <div className="text-2xl font-bold text-gray-900">{summary.total_anomalies || 0}</div>
            <div className="text-sm text-gray-500">Total Anomalies</div>
          </div>
          <div className="text-center">
            <div className="text-2xl font-bold text-red-600">{summary.critical || 0}</div>
            <div className="text-sm text-gray-500">Critical</div>
          </div>
          <div className="text-center">
            <div className="text-2xl font-bold text-orange-600">{summary.active || 0}</div>
            <div className="text-sm text-gray-500">Active</div>
          </div>
          <div className="text-center">
            <div className="text-2xl font-bold text-yellow-600">{summary.investigating || 0}</div>
            <div className="text-sm text-gray-500">Investigating</div>
          </div>
        </div>
      </div>

      {/* Detection performance */}
      {stats.detection_performance && (
        <div className="bg-white rounded-lg shadow p-6">
          <h2 className="text-xl font-semibold text-gray-900 mb-4">Detection Performance</h2>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div className="text-center">
              <div className="text-2xl font-bold text-blue-600">
                {(stats.detection_performance.precision * 100).toFixed(1)}%
              </div>
              <div className="text-sm text-gray-500">Precision</div>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold text-green-600">
                {(stats.detection_performance.recall * 100).toFixed(1)}%
              </div>
              <div className="text-sm text-gray-500">Recall</div>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold text-purple-600">
                {(stats.detection_performance.f1_score * 100).toFixed(1)}%
              </div>
              <div className="text-sm text-gray-500">F1 Score</div>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold text-gray-900">
                {stats.response_metrics?.avg_detection_time?.toFixed(1) || 0}min
              </div>
              <div className="text-sm text-gray-500">Avg Detection Time</div>
            </div>
          </div>
        </div>
      )}

      {/* Current anomalies */}
      <div className="space-y-4">
        <h2 className="text-xl font-semibold text-gray-900">Current Anomalies</h2>
        {anomalies.length === 0 ? (
          <div className="bg-white rounded-lg shadow p-6 text-center">
            <TrendingUp className="h-12 w-12 text-green-500 mx-auto mb-4" />
            <h3 className="text-lg font-medium text-gray-900 mb-2">No Anomalies Detected</h3>
            <p className="text-gray-600">All metrics are within normal ranges.</p>
          </div>
        ) : (
          <div className="space-y-4">
            {anomalies.map((anomaly: Anomaly) => (
              <AnomalyCard key={anomaly.id} anomaly={anomaly} />
            ))}
          </div>
        )}
      </div>
    </div>
  );
}