// API Configuration
export const API_CONFIG = {
  BASE_URL: import.meta.env.VITE_API_URL || 'http://localhost:8000',
  WEBSOCKET_URL: import.meta.env.VITE_WS_URL || 'ws://localhost:8000',
  API_VERSION: 'v1',
  TIMEOUT: 10000,
  RETRY_ATTEMPTS: 3,
  RETRY_DELAY: 1000,
};

// API Endpoints
export const API_ENDPOINTS = {
  // Cost Analytics
  COST_SUMMARY: '/api/v1/costs/summary',
  COST_TRENDS: '/api/v1/costs/trends',
  WAREHOUSE_COSTS: '/api/v1/costs/warehouses',
  
  // Usage Metrics
  USAGE_METRICS: '/api/v1/usage/metrics',
  QUERY_PERFORMANCE: '/api/v1/usage/performance',
  WAREHOUSE_UTILIZATION: '/api/v1/usage/warehouses',
  
  // Predictions
  COST_FORECAST: '/api/v1/predictions/forecast',
  USAGE_FORECAST: '/api/v1/predictions/usage',
  RECOMMENDATIONS: '/api/v1/predictions/recommendations',
  MODEL_PERFORMANCE: '/api/v1/predictions/models',
  
  // Anomalies
  CURRENT_ANOMALIES: '/api/v1/anomalies/current',
  ANOMALY_HISTORY: '/api/v1/anomalies/history',
  ANOMALY_DETAILS: '/api/v1/anomalies',
  ANOMALY_STATISTICS: '/api/v1/anomalies/statistics',
  
  // Alerts
  ACTIVE_ALERTS: '/api/v1/alerts/active',
  ALERT_RULES: '/api/v1/alerts/rules',
  ALERT_HISTORY: '/api/v1/alerts/history',
  ACKNOWLEDGE_ALERT: '/api/v1/alerts',
  RESOLVE_ALERT: '/api/v1/alerts',
  
  // System
  HEALTH: '/api/health',
  STATUS: '/api/status',
  METRICS: '/api/metrics',
  WEBSOCKET_STATS: '/api/v1/websocket/stats',
};

// WebSocket Endpoints
export const WS_ENDPOINTS = {
  REAL_TIME: '/ws/real-time',
  ALERTS: '/ws/alerts',
};

// Dashboard Configuration
export const DASHBOARD_CONFIG = {
  REFRESH_INTERVAL: 30000, // 30 seconds
  CHART_UPDATE_INTERVAL: 5000, // 5 seconds
  WEBSOCKET_RECONNECT_DELAY: 2000, // 2 seconds
  MAX_RECONNECT_ATTEMPTS: 5,
  PAGINATION_SIZE: 50,
  CHART_COLORS: {
    PRIMARY: '#0ea5e9',
    SECONDARY: '#d946ef',
    SUCCESS: '#22c55e',
    WARNING: '#f59e0b',
    ERROR: '#ef4444',
    INFO: '#3b82f6',
  },
  SEVERITY_COLORS: {
    LOW: '#22c55e',
    MEDIUM: '#f59e0b',
    HIGH: '#ef4444',
    CRITICAL: '#dc2626',
  },
};

// Time Range Options
export const TIME_RANGES = {
  '1h': { label: '1 Hour', value: '1h' },
  '24h': { label: '24 Hours', value: '24h' },
  '7d': { label: '7 Days', value: '7d' },
  '30d': { label: '30 Days', value: '30d' },
  '90d': { label: '90 Days', value: '90d' },
  '1y': { label: '1 Year', value: '1y' },
};

// Chart Configuration
export const CHART_CONFIG = {
  ANIMATION_DURATION: 300,
  HOVER_ANIMATION_DURATION: 200,
  RESPONSIVE: true,
  MAINTAIN_ASPECT_RATIO: false,
  PLUGINS: {
    LEGEND: {
      position: 'top' as const,
      labels: {
        usePointStyle: true,
        padding: 20,
      },
    },
    TOOLTIP: {
      backgroundColor: 'rgba(0, 0, 0, 0.8)',
      titleColor: 'white',
      bodyColor: 'white',
      borderColor: 'rgba(255, 255, 255, 0.1)',
      borderWidth: 1,
      cornerRadius: 8,
      displayColors: false,
    },
  },
  SCALES: {
    X: {
      grid: {
        display: false,
      },
      ticks: {
        maxRotation: 45,
        minRotation: 0,
      },
    },
    Y: {
      beginAtZero: true,
      grid: {
        color: 'rgba(0, 0, 0, 0.1)',
      },
    },
  },
};

// Feature Flags
export const FEATURE_FLAGS = {
  REAL_TIME_UPDATES: true,
  DARK_MODE: false,
  EXPORT_DATA: true,
  ADVANCED_CHARTS: true,
  NOTIFICATIONS: true,
  MOBILE_RESPONSIVE: true,
};

// Environment Configuration
export const ENV_CONFIG = {
  NODE_ENV: import.meta.env.MODE,
  IS_DEVELOPMENT: import.meta.env.DEV,
  IS_PRODUCTION: import.meta.env.PROD,
  API_KEY: import.meta.env.VITE_API_KEY,
  DEBUG: import.meta.env.VITE_DEBUG === 'true',
};