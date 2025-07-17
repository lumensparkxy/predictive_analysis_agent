import React, { useEffect, useState } from 'react';
import { useQuery } from 'react-query';
import { 
  DollarSign, 
  Activity, 
  AlertTriangle, 
  TrendingUp, 
  Users, 
  Database,
  Zap,
  Eye,
  ArrowUp,
  ArrowDown,
  Clock,
  CheckCircle,
  XCircle
} from 'lucide-react';
import { apiService } from '../services/api';
import { realTimeService } from '../services/websocket';
import { CostSummary, UsageMetrics, RealTimeData } from '../types';
import LoadingSpinner from '../components/common/LoadingSpinner';
import ErrorMessage from '../components/common/ErrorMessage';

interface MetricCardProps {
  title: string;
  value: string | number;
  change?: number;
  icon: React.ComponentType<{ className?: string }>;
  trend?: 'up' | 'down' | 'stable';
  subtitle?: string;
  color?: 'blue' | 'green' | 'red' | 'yellow' | 'purple';
}

const MetricCard: React.FC<MetricCardProps> = ({ 
  title, 
  value, 
  change, 
  icon: Icon, 
  trend, 
  subtitle,
  color = 'blue' 
}) => {
  const colorClasses = {
    blue: 'bg-blue-50 text-blue-700',
    green: 'bg-green-50 text-green-700',
    red: 'bg-red-50 text-red-700',
    yellow: 'bg-yellow-50 text-yellow-700',
    purple: 'bg-purple-50 text-purple-700',
  };

  return (
    <div className="bg-white rounded-lg shadow p-6 hover:shadow-lg transition-shadow">
      <div className="flex items-center justify-between">
        <div className="flex items-center">
          <div className={`p-2 rounded-lg ${colorClasses[color]}`}>
            <Icon className="h-6 w-6" />
          </div>
          <div className="ml-4">
            <h3 className="text-sm font-medium text-gray-900">{title}</h3>
            <p className="text-2xl font-bold text-gray-900">{value}</p>
            {subtitle && <p className="text-sm text-gray-500">{subtitle}</p>}
          </div>
        </div>
        {change !== undefined && (
          <div className={`flex items-center ${
            change > 0 ? 'text-green-600' : change < 0 ? 'text-red-600' : 'text-gray-500'
          }`}>
            {change > 0 ? (
              <ArrowUp className="h-4 w-4 mr-1" />
            ) : change < 0 ? (
              <ArrowDown className="h-4 w-4 mr-1" />
            ) : null}
            <span className="text-sm font-medium">{Math.abs(change)}%</span>
          </div>
        )}
      </div>
    </div>
  );
};

interface QuickStatsProps {
  stats: {
    activeWarehouses: number;
    activeUsers: number;
    activeAlerts: number;
    activeAnomalies: number;
    systemHealth: 'healthy' | 'warning' | 'error';
    lastUpdate: string;
  };
}

const QuickStats: React.FC<QuickStatsProps> = ({ stats }) => {
  const getHealthColor = (health: string) => {
    switch (health) {
      case 'healthy': return 'text-green-600';
      case 'warning': return 'text-yellow-600';
      case 'error': return 'text-red-600';
      default: return 'text-gray-600';
    }
  };

  const getHealthIcon = (health: string) => {
    switch (health) {
      case 'healthy': return CheckCircle;
      case 'warning': return AlertTriangle;
      case 'error': return XCircle;
      default: return Activity;
    }
  };

  const HealthIcon = getHealthIcon(stats.systemHealth);

  return (
    <div className="bg-white rounded-lg shadow p-6">
      <h3 className="text-lg font-semibold text-gray-900 mb-4">System Overview</h3>
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <div className="text-center">
          <div className="text-2xl font-bold text-blue-600">{stats.activeWarehouses}</div>
          <div className="text-sm text-gray-500">Active Warehouses</div>
        </div>
        <div className="text-center">
          <div className="text-2xl font-bold text-green-600">{stats.activeUsers}</div>
          <div className="text-sm text-gray-500">Active Users</div>
        </div>
        <div className="text-center">
          <div className="text-2xl font-bold text-red-600">{stats.activeAlerts}</div>
          <div className="text-sm text-gray-500">Active Alerts</div>
        </div>
        <div className="text-center">
          <div className="text-2xl font-bold text-purple-600">{stats.activeAnomalies}</div>
          <div className="text-sm text-gray-500">Anomalies</div>
        </div>
      </div>
      <div className="mt-6 pt-4 border-t border-gray-200">
        <div className="flex items-center justify-between">
          <div className="flex items-center">
            <HealthIcon className={`h-5 w-5 mr-2 ${getHealthColor(stats.systemHealth)}`} />
            <span className={`font-medium ${getHealthColor(stats.systemHealth)}`}>
              System {stats.systemHealth}
            </span>
          </div>
          <div className="flex items-center text-sm text-gray-500">
            <Clock className="h-4 w-4 mr-1" />
            Updated: {stats.lastUpdate}
          </div>
        </div>
      </div>
    </div>
  );
};

export default function Dashboard() {
  const [realTimeData, setRealTimeData] = useState<RealTimeData | null>(null);
  const [quickStats, setQuickStats] = useState({
    activeWarehouses: 0,
    activeUsers: 0,
    activeAlerts: 0,
    activeAnomalies: 0,
    systemHealth: 'healthy' as const,
    lastUpdate: new Date().toLocaleTimeString(),
  });

  // Fetch cost summary
  const { data: costData, isLoading: costLoading, error: costError } = useQuery(
    'cost-summary',
    () => apiService.getCostSummary(),
    {
      refetchInterval: 30000, // Refresh every 30 seconds
    }
  );

  // Fetch usage metrics
  const { data: usageData, isLoading: usageLoading, error: usageError } = useQuery(
    'usage-metrics',
    () => apiService.getUsageMetrics(),
    {
      refetchInterval: 30000,
    }
  );

  // Fetch active alerts
  const { data: alertsData, isLoading: alertsLoading } = useQuery(
    'active-alerts',
    () => apiService.getActiveAlerts(),
    {
      refetchInterval: 15000, // Refresh every 15 seconds
    }
  );

  // Fetch current anomalies
  const { data: anomaliesData, isLoading: anomaliesLoading } = useQuery(
    'current-anomalies',
    () => apiService.getCurrentAnomalies(),
    {
      refetchInterval: 15000,
    }
  );

  // Fetch system health
  const { data: healthData } = useQuery(
    'system-health',
    () => apiService.getSystemHealth(),
    {
      refetchInterval: 10000, // Refresh every 10 seconds
    }
  );

  // Set up real-time data subscription
  useEffect(() => {
    const subscription = realTimeService.subscribe('metrics', (data: RealTimeData) => {
      setRealTimeData(data);
      
      // Update quick stats with real-time data
      setQuickStats(prev => ({
        ...prev,
        activeWarehouses: data.data.warehouses.filter(w => w.status === 'running').length,
        activeUsers: data.data.metrics.active_users,
        activeAlerts: data.data.alerts.active_count,
        activeAnomalies: data.data.anomalies.detected_count,
        lastUpdate: new Date().toLocaleTimeString(),
      }));
    });

    return () => {
      realTimeService.unsubscribe(subscription);
    };
  }, []);

  // Update quick stats when data changes
  useEffect(() => {
    if (usageData?.data && alertsData?.data && anomaliesData?.data && healthData?.data) {
      setQuickStats(prev => ({
        ...prev,
        activeWarehouses: usageData.data.active_warehouses,
        activeUsers: usageData.data.active_users,
        activeAlerts: alertsData.data.summary.total_alerts,
        activeAnomalies: anomaliesData.data.summary.total_anomalies,
        systemHealth: healthData.data.status === 'healthy' ? 'healthy' : 'warning',
      }));
    }
  }, [usageData, alertsData, anomaliesData, healthData]);

  if (costLoading || usageLoading) {
    return <LoadingSpinner />;
  }

  if (costError || usageError) {
    return <ErrorMessage message="Failed to load dashboard data" />;
  }

  const cost = costData?.data as CostSummary;
  const usage = usageData?.data as UsageMetrics;

  return (
    <div className="space-y-6">
      {/* Page header */}
      <div>
        <h1 className="text-3xl font-bold text-gray-900">Dashboard</h1>
        <p className="text-gray-600 mt-2">
          Real-time analytics and monitoring for your Snowflake environment
        </p>
      </div>

      {/* Key metrics */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <MetricCard
          title="Total Cost"
          value={cost ? `$${cost.total_cost.toLocaleString()}` : '—'}
          change={cost?.cost_change_percent}
          icon={DollarSign}
          trend={cost?.cost_trend === 'increasing' ? 'up' : cost?.cost_trend === 'decreasing' ? 'down' : 'stable'}
          subtitle={cost?.time_range}
          color="blue"
        />
        
        <MetricCard
          title="Active Queries"
          value={realTimeData?.data.metrics.active_queries || usage?.total_queries || 0}
          icon={Activity}
          subtitle="Currently running"
          color="green"
        />
        
        <MetricCard
          title="Active Alerts"
          value={realTimeData?.data.alerts.active_count || quickStats.activeAlerts}
          icon={AlertTriangle}
          subtitle={`${realTimeData?.data.alerts.critical_count || 0} critical`}
          color="red"
        />
        
        <MetricCard
          title="Query Success Rate"
          value={usage ? `${usage.query_success_rate.toFixed(1)}%` : '—'}
          icon={TrendingUp}
          subtitle="Last 24 hours"
          color="green"
        />
      </div>

      {/* Quick stats */}
      <QuickStats stats={quickStats} />

      {/* Real-time metrics */}
      {realTimeData && (
        <div className="bg-white rounded-lg shadow p-6">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">Real-time Metrics</h3>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            <div>
              <h4 className="text-sm font-medium text-gray-700 mb-2">Current Usage</h4>
              <div className="space-y-2">
                <div className="flex justify-between">
                  <span className="text-sm text-gray-600">Active Queries</span>
                  <span className="text-sm font-medium">{realTimeData.data.metrics.active_queries}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-sm text-gray-600">Active Users</span>
                  <span className="text-sm font-medium">{realTimeData.data.metrics.active_users}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-sm text-gray-600">Current Cost</span>
                  <span className="text-sm font-medium">${realTimeData.data.metrics.current_cost.toFixed(2)}</span>
                </div>
              </div>
            </div>
            
            <div>
              <h4 className="text-sm font-medium text-gray-700 mb-2">Performance</h4>
              <div className="space-y-2">
                <div className="flex justify-between">
                  <span className="text-sm text-gray-600">Queries/Second</span>
                  <span className="text-sm font-medium">{realTimeData.data.metrics.queries_per_second.toFixed(1)}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-sm text-gray-600">Avg Query Time</span>
                  <span className="text-sm font-medium">{realTimeData.data.metrics.avg_query_time.toFixed(1)}s</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-sm text-gray-600">Data Processed</span>
                  <span className="text-sm font-medium">{realTimeData.data.metrics.data_processed_mb.toFixed(1)} MB</span>
                </div>
              </div>
            </div>
            
            <div>
              <h4 className="text-sm font-medium text-gray-700 mb-2">Warehouse Status</h4>
              <div className="space-y-2">
                {realTimeData.data.warehouses.map((warehouse, index) => (
                  <div key={index} className="flex items-center justify-between">
                    <span className="text-sm text-gray-600">{warehouse.name}</span>
                    <div className="flex items-center">
                      <div className={`w-2 h-2 rounded-full mr-2 ${
                        warehouse.status === 'running' ? 'bg-green-500' : 'bg-gray-400'
                      }`} />
                      <span className="text-sm font-medium">{warehouse.utilization.toFixed(1)}%</span>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Recent activity */}
      <div className="bg-white rounded-lg shadow p-6">
        <h3 className="text-lg font-semibold text-gray-900 mb-4">Recent Activity</h3>
        <div className="space-y-4">
          <div className="flex items-center">
            <div className="w-2 h-2 bg-green-500 rounded-full mr-3" />
            <div>
              <p className="text-sm text-gray-900">Dashboard loaded successfully</p>
              <p className="text-xs text-gray-500">Just now</p>
            </div>
          </div>
          <div className="flex items-center">
            <div className="w-2 h-2 bg-blue-500 rounded-full mr-3" />
            <div>
              <p className="text-sm text-gray-900">Real-time monitoring active</p>
              <p className="text-xs text-gray-500">30 seconds ago</p>
            </div>
          </div>
          <div className="flex items-center">
            <div className="w-2 h-2 bg-yellow-500 rounded-full mr-3" />
            <div>
              <p className="text-sm text-gray-900">Data collection in progress</p>
              <p className="text-xs text-gray-500">1 minute ago</p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}