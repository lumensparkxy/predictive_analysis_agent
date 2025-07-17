"""
Request schemas for API endpoints.
"""

from typing import Dict, Any, List, Optional, Union
from datetime import datetime

class APIRequest:
    """Base API request schema."""
    
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
    
    def validate(self) -> bool:
        """Validate the request parameters."""
        return True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}

class CostSummaryRequest(APIRequest):
    """Request schema for cost summary endpoint."""
    
    def __init__(self, time_range: str = "30d", warehouse: Optional[str] = None):
        self.time_range = time_range
        self.warehouse = warehouse
    
    def validate(self) -> bool:
        """Validate cost summary request parameters."""
        valid_ranges = ["1d", "7d", "30d", "90d", "1y"]
        return self.time_range in valid_ranges

class UsageMetricsRequest(APIRequest):
    """Request schema for usage metrics endpoint."""
    
    def __init__(self, time_range: str = "24h", warehouse: Optional[str] = None,
                 user: Optional[str] = None):
        self.time_range = time_range
        self.warehouse = warehouse
        self.user = user
    
    def validate(self) -> bool:
        """Validate usage metrics request parameters."""
        valid_ranges = ["1h", "24h", "7d", "30d"]
        return self.time_range in valid_ranges

class PredictionRequest(APIRequest):
    """Request schema for prediction endpoints."""
    
    def __init__(self, days: int = 30, model_type: Optional[str] = None,
                 confidence_level: float = 0.95):
        self.days = days
        self.model_type = model_type
        self.confidence_level = confidence_level
    
    def validate(self) -> bool:
        """Validate prediction request parameters."""
        return (1 <= self.days <= 365 and 
                0.5 <= self.confidence_level <= 0.99)

class AnomalyRequest(APIRequest):
    """Request schema for anomaly endpoints."""
    
    def __init__(self, severity: Optional[str] = None, 
                 status: Optional[str] = None,
                 time_range: str = "24h"):
        self.severity = severity
        self.status = status
        self.time_range = time_range
    
    def validate(self) -> bool:
        """Validate anomaly request parameters."""
        valid_severities = ["low", "medium", "high", "critical"]
        valid_statuses = ["active", "investigating", "resolved"]
        valid_ranges = ["1h", "24h", "7d", "30d"]
        
        return (
            (self.severity is None or self.severity in valid_severities) and
            (self.status is None or self.status in valid_statuses) and
            self.time_range in valid_ranges
        )

class AlertRequest(APIRequest):
    """Request schema for alert endpoints."""
    
    def __init__(self, severity: Optional[str] = None,
                 status: Optional[str] = None,
                 acknowledged: Optional[bool] = None):
        self.severity = severity
        self.status = status
        self.acknowledged = acknowledged
    
    def validate(self) -> bool:
        """Validate alert request parameters."""
        valid_severities = ["low", "medium", "high", "critical"]
        valid_statuses = ["active", "resolved"]
        
        return (
            (self.severity is None or self.severity in valid_severities) and
            (self.status is None or self.status in valid_statuses)
        )

class AlertActionRequest(APIRequest):
    """Request schema for alert action endpoints."""
    
    def __init__(self, alert_id: str, action: str, user: str, 
                 note: Optional[str] = None):
        self.alert_id = alert_id
        self.action = action
        self.user = user
        self.note = note
    
    def validate(self) -> bool:
        """Validate alert action request parameters."""
        valid_actions = ["acknowledge", "resolve", "escalate"]
        return (
            self.alert_id and
            self.action in valid_actions and
            self.user
        )

class QueryPerformanceRequest(APIRequest):
    """Request schema for query performance endpoint."""
    
    def __init__(self, limit: int = 100, warehouse: Optional[str] = None,
                 min_duration: Optional[float] = None,
                 status: Optional[str] = None):
        self.limit = limit
        self.warehouse = warehouse
        self.min_duration = min_duration
        self.status = status
    
    def validate(self) -> bool:
        """Validate query performance request parameters."""
        valid_statuses = ["success", "failed", "cancelled"]
        return (
            1 <= self.limit <= 1000 and
            (self.min_duration is None or self.min_duration >= 0) and
            (self.status is None or self.status in valid_statuses)
        )

class WarehouseUtilizationRequest(APIRequest):
    """Request schema for warehouse utilization endpoint."""
    
    def __init__(self, warehouse: Optional[str] = None,
                 time_range: str = "1h"):
        self.warehouse = warehouse
        self.time_range = time_range
    
    def validate(self) -> bool:
        """Validate warehouse utilization request parameters."""
        valid_ranges = ["1h", "24h", "7d"]
        return self.time_range in valid_ranges

class OptimizationRequest(APIRequest):
    """Request schema for optimization recommendations endpoint."""
    
    def __init__(self, category: Optional[str] = None,
                 priority: Optional[str] = None,
                 warehouse: Optional[str] = None):
        self.category = category
        self.priority = priority
        self.warehouse = warehouse
    
    def validate(self) -> bool:
        """Validate optimization request parameters."""
        valid_categories = ["cost_optimization", "performance_optimization", 
                           "capacity_planning", "storage_optimization"]
        valid_priorities = ["low", "medium", "high"]
        
        return (
            (self.category is None or self.category in valid_categories) and
            (self.priority is None or self.priority in valid_priorities)
        )

# Request type mappings
REQUEST_SCHEMAS = {
    "cost_summary": CostSummaryRequest,
    "usage_metrics": UsageMetricsRequest,
    "prediction": PredictionRequest,
    "anomaly": AnomalyRequest,
    "alert": AlertRequest,
    "alert_action": AlertActionRequest,
    "query_performance": QueryPerformanceRequest,
    "warehouse_utilization": WarehouseUtilizationRequest,
    "optimization": OptimizationRequest
}

def validate_request(request_type: str, **kwargs) -> bool:
    """
    Validate request parameters for a given endpoint type.
    
    Args:
        request_type: Type of request to validate
        **kwargs: Request parameters
        
    Returns:
        True if valid, False otherwise
    """
    if request_type not in REQUEST_SCHEMAS:
        return False
    
    try:
        request_schema = REQUEST_SCHEMAS[request_type](**kwargs)
        return request_schema.validate()
    except Exception:
        return False