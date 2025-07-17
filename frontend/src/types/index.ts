// API Response Types
export interface APIResponse<T = any> {
  status: 'success' | 'error';
  data?: T;
  error?: string;
  timestamp: string;
}

// Cost Analytics Types
export interface CostSummary {
  total_cost: number;
  cost_change_percent: number;
  cost_trend: 'increasing' | 'decreasing' | 'stable';
  breakdown: CostBreakdown[];
  daily_costs: DailyCost[];
  warehouse_costs: WarehouseCost[];
  time_range: string;
  currency: string;
  last_updated: string;
}

export interface CostBreakdown {
  category: string;
  cost: number;
  percentage: number;
}

export interface DailyCost {
  date: string;
  cost: number;
}

export interface WarehouseCost {
  warehouse: string;
  cost: number;
  percentage: number;
}

// Usage Metrics Types
export interface UsageMetrics {
  total_queries: number;
  active_users: number;
  active_warehouses: number;
  data_processed_gb: number;
  avg_query_duration: number;
  query_success_rate: number;
  peak_concurrent_queries: number;
  query_types: QueryType[];
  user_activity: UserActivity[];
  hourly_usage: HourlyUsage[];
  time_range: string;
  last_updated: string;
}

export interface QueryType {
  type: string;
  count: number;
  percentage: number;
}

export interface UserActivity {
  user: string;
  queries: number;
  data_gb: number;
}

export interface HourlyUsage {
  hour: string;
  queries: number;
  users: number;
}

export interface QueryPerformance {
  query_id: string;
  duration_seconds: number;
  rows_returned: number;
  bytes_scanned: number;
  warehouse: string;
  user: string;
  status: 'success' | 'failed';
  timestamp: string;
}

export interface WarehouseUtilization {
  warehouse: string;
  status: 'running' | 'suspended';
  utilization: number;
  queue_depth: number;
  avg_queue_time: number;
  active_queries: number;
  credits_used: number;
  credits_remaining: number;
  auto_suspend_time: number;
  size: string;
  scaling_policy: string;
}

// Prediction Types
export interface CostForecast {
  date: string;
  predicted_cost: number;
  confidence_lower: number;
  confidence_upper: number;
  confidence_level: number;
}

export interface UsageForecast {
  date: string;
  predicted_queries: number;
  predicted_users: number;
  predicted_data_gb: number;
  confidence_level: number;
}

export interface ModelInfo {
  model_type: string;
  accuracy: number;
  last_trained: string;
  features_used: string[];
}

export interface OptimizationRecommendation {
  id: string;
  type: 'cost_optimization' | 'performance_optimization' | 'capacity_planning' | 'storage_optimization';
  priority: 'low' | 'medium' | 'high';
  title: string;
  description: string;
  potential_savings: number;
  impact: 'low' | 'medium' | 'high';
  implementation_effort: 'low' | 'medium' | 'high';
  category: string;
}

// Anomaly Types
export interface Anomaly {
  id: string;
  type: 'cost_spike' | 'query_performance' | 'usage_pattern' | 'data_volume' | 'connection_anomaly';
  severity: 'low' | 'medium' | 'high' | 'critical';
  title: string;
  description: string;
  value: number;
  baseline: number;
  threshold: number;
  deviation: number;
  warehouse?: string;
  user?: string;
  detected_at: string;
  status: 'active' | 'investigating' | 'resolved';
  confidence: number;
}

export interface AnomalyDetails extends Anomaly {
  impact: {
    cost_impact: number;
    performance_impact: string;
    affected_users: number;
    affected_queries: number;
  };
  root_cause: {
    primary_cause: string;
    contributing_factors: string[];
  };
  timeline: AnomalyTimelineEvent[];
  recommendations: AnomalyRecommendation[];
}

export interface AnomalyTimelineEvent {
  timestamp: string;
  event: string;
  description: string;
}

export interface AnomalyRecommendation {
  action: string;
  priority: 'low' | 'medium' | 'high';
  estimated_impact: string;
}

// Alert Types
export interface Alert {
  id: string;
  type: 'cost_threshold' | 'query_performance' | 'warehouse_utilization' | 'data_volume' | 'connection_failures';
  severity: 'low' | 'medium' | 'high' | 'critical';
  title: string;
  description: string;
  threshold: number;
  current_value: number;
  warehouse?: string;
  created_at: string;
  status: 'active' | 'resolved';
  acknowledged: boolean;
  escalated: boolean;
  assigned_to: string;
  rule_id: string;
}

export interface AlertRule {
  id: string;
  name: string;
  type: string;
  description: string;
  threshold: number;
  operator: 'greater_than' | 'less_than' | 'equals';
  timeframe: string;
  severity: 'low' | 'medium' | 'high' | 'critical';
  enabled: boolean;
  warehouse?: string;
  notification_channels: string[];
  escalation_timeout: number;
  created_at: string;
  last_triggered?: string;
}

// Real-time Data Types
export interface RealTimeMetrics {
  active_queries: number;
  active_users: number;
  current_cost: number;
  queries_per_second: number;
  data_processed_mb: number;
  avg_query_time: number;
}

export interface RealTimeWarehouse {
  name: string;
  status: 'running' | 'suspended';
  utilization: number;
  queue_depth: number;
  active_queries: number;
}

export interface RealTimeAlerts {
  active_count: number;
  critical_count: number;
  last_alert: string;
}

export interface RealTimeAnomalies {
  detected_count: number;
  investigating_count: number;
  confidence_avg: number;
}

export interface RealTimeData {
  timestamp: string;
  type: 'real_time_update' | 'alert_notification' | 'anomaly_notification';
  data: {
    metrics: RealTimeMetrics;
    warehouses: RealTimeWarehouse[];
    alerts: RealTimeAlerts;
    anomalies: RealTimeAnomalies;
  };
}

// System Types
export interface SystemHealth {
  status: 'healthy' | 'unhealthy' | 'error';
  timestamp: string;
  version: string;
  services: Record<string, string>;
}

export interface SystemStatus {
  timestamp: string;
  version: string;
  uptime: string;
  components: Record<string, string>;
  websocket: {
    active_connections: number;
    data_stream_active: boolean;
  };
  endpoints: {
    total: number;
    active: number;
    v1_endpoints: number;
    legacy_endpoints: number;
  };
}

export interface SystemMetrics {
  timestamp: string;
  system: {
    cpu_percent: number;
    memory_percent: number;
    memory_available_gb: number;
    disk_percent: number;
    disk_free_gb: number;
  };
  process: {
    pid: number;
    threads: number;
  };
  api: {
    endpoints_active: number;
    websocket_connections: number;
  };
}

// Chart Data Types
export interface ChartDataPoint {
  x: string | number;
  y: number;
  label?: string;
}

export interface ChartSeries {
  name: string;
  data: ChartDataPoint[];
  color?: string;
}

export interface ChartConfig {
  type: 'line' | 'bar' | 'pie' | 'area' | 'scatter' | 'gauge';
  title: string;
  subtitle?: string;
  xAxis?: {
    title: string;
    type: 'category' | 'datetime' | 'numeric';
  };
  yAxis?: {
    title: string;
    format?: string;
  };
  series: ChartSeries[];
  height?: number;
  width?: number;
}

// Filter Types
export interface FilterOption {
  label: string;
  value: string;
  selected?: boolean;
}

export interface DateRange {
  start: Date;
  end: Date;
}

export interface DashboardFilters {
  timeRange: string;
  warehouse?: string;
  user?: string;
  severity?: string;
  dateRange?: DateRange;
}

// UI State Types
export interface LoadingState {
  isLoading: boolean;
  error?: string;
  lastUpdated?: string;
}

export interface NotificationState {
  id: string;
  type: 'success' | 'error' | 'warning' | 'info';
  title: string;
  message: string;
  timestamp: string;
  duration?: number;
  action?: {
    label: string;
    handler: () => void;
  };
}