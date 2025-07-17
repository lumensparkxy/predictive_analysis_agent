import axios, { AxiosInstance, AxiosRequestConfig, AxiosResponse } from 'axios';
import { API_CONFIG, API_ENDPOINTS } from '../config/config';
import { APIResponse } from '../types';

class ApiService {
  private api: AxiosInstance;

  constructor() {
    this.api = axios.create({
      baseURL: API_CONFIG.BASE_URL,
      timeout: API_CONFIG.TIMEOUT,
      headers: {
        'Content-Type': 'application/json',
      },
    });

    // Request interceptor
    this.api.interceptors.request.use(
      (config) => {
        // Add API key if available
        const apiKey = localStorage.getItem('api_key');
        if (apiKey) {
          config.headers['X-API-Key'] = apiKey;
        }

        // Add timestamp to prevent caching
        if (config.method === 'get') {
          config.params = {
            ...config.params,
            _t: Date.now(),
          };
        }

        return config;
      },
      (error) => Promise.reject(error)
    );

    // Response interceptor
    this.api.interceptors.response.use(
      (response: AxiosResponse) => response,
      async (error) => {
        if (error.response?.status === 401) {
          // Handle authentication error
          localStorage.removeItem('api_key');
          window.location.href = '/login';
        }

        if (error.response?.status === 429) {
          // Handle rate limiting
          const retryAfter = error.response.headers['retry-after'] || 60;
          throw new Error(`Rate limited. Please try again in ${retryAfter} seconds.`);
        }

        return Promise.reject(error);
      }
    );
  }

  private async request<T>(config: AxiosRequestConfig): Promise<APIResponse<T>> {
    try {
      const response = await this.api.request(config);
      return response.data;
    } catch (error: any) {
      console.error('API request failed:', error);
      
      if (error.response?.data) {
        return error.response.data;
      }
      
      return {
        status: 'error',
        error: error.message || 'An unexpected error occurred',
        timestamp: new Date().toISOString(),
      };
    }
  }

  // Cost Analytics
  async getCostSummary(timeRange: string = '30d') {
    return this.request({
      method: 'GET',
      url: API_ENDPOINTS.COST_SUMMARY,
      params: { time_range: timeRange },
    });
  }

  async getCostTrends(days: number = 30) {
    return this.request({
      method: 'GET',
      url: API_ENDPOINTS.COST_TRENDS,
      params: { days },
    });
  }

  async getWarehouseCosts(warehouse?: string) {
    return this.request({
      method: 'GET',
      url: API_ENDPOINTS.WAREHOUSE_COSTS,
      params: warehouse ? { warehouse } : undefined,
    });
  }

  // Usage Metrics
  async getUsageMetrics(timeRange: string = '24h') {
    return this.request({
      method: 'GET',
      url: API_ENDPOINTS.USAGE_METRICS,
      params: { time_range: timeRange },
    });
  }

  async getQueryPerformance(limit: number = 100) {
    return this.request({
      method: 'GET',
      url: API_ENDPOINTS.QUERY_PERFORMANCE,
      params: { limit },
    });
  }

  async getWarehouseUtilization() {
    return this.request({
      method: 'GET',
      url: API_ENDPOINTS.WAREHOUSE_UTILIZATION,
    });
  }

  // Predictions
  async getCostForecast(days: number = 30) {
    return this.request({
      method: 'GET',
      url: API_ENDPOINTS.COST_FORECAST,
      params: { days },
    });
  }

  async getUsageForecast(days: number = 30) {
    return this.request({
      method: 'GET',
      url: API_ENDPOINTS.USAGE_FORECAST,
      params: { days },
    });
  }

  async getOptimizationRecommendations() {
    return this.request({
      method: 'GET',
      url: API_ENDPOINTS.RECOMMENDATIONS,
    });
  }

  async getModelPerformance() {
    return this.request({
      method: 'GET',
      url: API_ENDPOINTS.MODEL_PERFORMANCE,
    });
  }

  // Anomalies
  async getCurrentAnomalies(severity?: string) {
    return this.request({
      method: 'GET',
      url: API_ENDPOINTS.CURRENT_ANOMALIES,
      params: severity ? { severity } : undefined,
    });
  }

  async getAnomalyHistory(days: number = 7) {
    return this.request({
      method: 'GET',
      url: API_ENDPOINTS.ANOMALY_HISTORY,
      params: { days },
    });
  }

  async getAnomalyDetails(anomalyId: string) {
    return this.request({
      method: 'GET',
      url: `${API_ENDPOINTS.ANOMALY_DETAILS}/${anomalyId}`,
    });
  }

  async getAnomalyStatistics() {
    return this.request({
      method: 'GET',
      url: API_ENDPOINTS.ANOMALY_STATISTICS,
    });
  }

  // Alerts
  async getActiveAlerts(severity?: string) {
    return this.request({
      method: 'GET',
      url: API_ENDPOINTS.ACTIVE_ALERTS,
      params: severity ? { severity } : undefined,
    });
  }

  async getAlertRules() {
    return this.request({
      method: 'GET',
      url: API_ENDPOINTS.ALERT_RULES,
    });
  }

  async getAlertHistory(days: number = 7) {
    return this.request({
      method: 'GET',
      url: API_ENDPOINTS.ALERT_HISTORY,
      params: { days },
    });
  }

  async acknowledgeAlert(alertId: string) {
    return this.request({
      method: 'POST',
      url: `${API_ENDPOINTS.ACKNOWLEDGE_ALERT}/${alertId}/acknowledge`,
    });
  }

  async resolveAlert(alertId: string, resolutionNote?: string) {
    return this.request({
      method: 'POST',
      url: `${API_ENDPOINTS.RESOLVE_ALERT}/${alertId}/resolve`,
      data: { note: resolutionNote },
    });
  }

  // System
  async getSystemHealth() {
    return this.request({
      method: 'GET',
      url: API_ENDPOINTS.HEALTH,
    });
  }

  async getSystemStatus() {
    return this.request({
      method: 'GET',
      url: API_ENDPOINTS.STATUS,
    });
  }

  async getSystemMetrics() {
    return this.request({
      method: 'GET',
      url: API_ENDPOINTS.METRICS,
    });
  }

  async getWebSocketStats() {
    return this.request({
      method: 'GET',
      url: API_ENDPOINTS.WEBSOCKET_STATS,
    });
  }

  // Utility methods
  async retryRequest<T>(
    requestFn: () => Promise<APIResponse<T>>,
    maxRetries: number = API_CONFIG.RETRY_ATTEMPTS
  ): Promise<APIResponse<T>> {
    let lastError: any;
    
    for (let i = 0; i < maxRetries; i++) {
      try {
        const result = await requestFn();
        if (result.status === 'success') {
          return result;
        }
        lastError = result.error;
      } catch (error) {
        lastError = error;
      }
      
      if (i < maxRetries - 1) {
        await new Promise(resolve => 
          setTimeout(resolve, API_CONFIG.RETRY_DELAY * (i + 1))
        );
      }
    }
    
    return {
      status: 'error',
      error: lastError?.message || 'Request failed after retries',
      timestamp: new Date().toISOString(),
    };
  }

  // Batch requests
  async batchRequest<T>(requests: Array<() => Promise<APIResponse<T>>>) {
    const results = await Promise.allSettled(
      requests.map(request => request())
    );
    
    return results.map(result => 
      result.status === 'fulfilled' ? result.value : {
        status: 'error' as const,
        error: result.reason?.message || 'Request failed',
        timestamp: new Date().toISOString(),
      }
    );
  }
}

export const apiService = new ApiService();
export default apiService;