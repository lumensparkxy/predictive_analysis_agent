"""
Response schemas for API endpoints.
"""

from typing import Dict, Any, List, Optional, Union
from datetime import datetime

class APIResponse:
    """Base API response schema."""
    
    def __init__(self, status: str, data: Any = None, error: str = None, timestamp: str = None):
        self.status = status
        self.data = data
        self.error = error
        self.timestamp = timestamp or datetime.now().isoformat()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = {
            "status": self.status,
            "timestamp": self.timestamp
        }
        if self.data is not None:
            result["data"] = self.data
        if self.error is not None:
            result["error"] = self.error
        return result

class CostSummaryResponse(APIResponse):
    """Response schema for cost summary endpoint."""
    
    def __init__(self, total_cost: float, cost_change_percent: float, 
                 cost_trend: str, breakdown: List[Dict], daily_costs: List[Dict],
                 warehouse_costs: List[Dict], time_range: str, currency: str = "USD"):
        data = {
            "total_cost": total_cost,
            "cost_change_percent": cost_change_percent,
            "cost_trend": cost_trend,
            "breakdown": breakdown,
            "daily_costs": daily_costs,
            "warehouse_costs": warehouse_costs,
            "time_range": time_range,
            "currency": currency,
            "last_updated": datetime.now().isoformat()
        }
        super().__init__("success", data)

class UsageMetricsResponse(APIResponse):
    """Response schema for usage metrics endpoint."""
    
    def __init__(self, total_queries: int, active_users: int, active_warehouses: int,
                 data_processed_gb: float, avg_query_duration: float, 
                 query_success_rate: float, peak_concurrent_queries: int,
                 query_types: List[Dict], user_activity: List[Dict],
                 hourly_usage: List[Dict], time_range: str):
        data = {
            "total_queries": total_queries,
            "active_users": active_users,
            "active_warehouses": active_warehouses,
            "data_processed_gb": data_processed_gb,
            "avg_query_duration": avg_query_duration,
            "query_success_rate": query_success_rate,
            "peak_concurrent_queries": peak_concurrent_queries,
            "query_types": query_types,
            "user_activity": user_activity,
            "hourly_usage": hourly_usage,
            "time_range": time_range,
            "last_updated": datetime.now().isoformat()
        }
        super().__init__("success", data)

class PredictionResponse(APIResponse):
    """Response schema for prediction endpoints."""
    
    def __init__(self, forecasts: List[Dict], model_info: Dict, summary: Dict):
        data = {
            "forecasts": forecasts,
            "model_info": model_info,
            "summary": summary
        }
        super().__init__("success", data)

class AnomalyResponse(APIResponse):
    """Response schema for anomaly endpoints."""
    
    def __init__(self, anomalies: List[Dict], summary: Dict):
        data = {
            "anomalies": anomalies,
            "summary": summary
        }
        super().__init__("success", data)

class AlertResponse(APIResponse):
    """Response schema for alert endpoints."""
    
    def __init__(self, alerts: List[Dict], summary: Dict):
        data = {
            "alerts": alerts,
            "summary": summary
        }
        super().__init__("success", data)

class RealTimeDataResponse(APIResponse):
    """Response schema for real-time data."""
    
    def __init__(self, metrics: Dict, warehouses: List[Dict], 
                 alerts: Dict, anomalies: Dict):
        data = {
            "metrics": metrics,
            "warehouses": warehouses,
            "alerts": alerts,
            "anomalies": anomalies
        }
        super().__init__("success", data)

class ErrorResponse(APIResponse):
    """Response schema for error responses."""
    
    def __init__(self, error_message: str, error_code: str = None):
        super().__init__("error", error=error_message)
        if error_code:
            self.data = {"error_code": error_code}

# Response type mappings
RESPONSE_SCHEMAS = {
    "cost_summary": CostSummaryResponse,
    "usage_metrics": UsageMetricsResponse,
    "prediction": PredictionResponse,
    "anomaly": AnomalyResponse,
    "alert": AlertResponse,
    "real_time": RealTimeDataResponse,
    "error": ErrorResponse
}