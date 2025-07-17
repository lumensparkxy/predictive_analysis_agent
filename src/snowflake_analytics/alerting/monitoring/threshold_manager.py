"""
Threshold Management System

Dynamic threshold management for monitoring systems with adaptive thresholds,
threshold validation, and intelligent threshold adjustment capabilities.
"""

import time
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import statistics
import logging
import json


class ThresholdType(Enum):
    """Types of thresholds"""
    STATIC = "static"
    DYNAMIC = "dynamic"
    ADAPTIVE = "adaptive"
    PERCENTILE = "percentile"
    STATISTICAL = "statistical"


class ThresholdCondition(Enum):
    """Threshold condition operators"""
    GREATER_THAN = ">"
    LESS_THAN = "<"
    GREATER_EQUAL = ">="
    LESS_EQUAL = "<="
    EQUAL = "=="
    NOT_EQUAL = "!="


@dataclass
class Threshold:
    """Represents a monitoring threshold"""
    id: str
    name: str
    metric_name: str
    threshold_type: ThresholdType
    condition: ThresholdCondition
    value: float
    enabled: bool = True
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    tags: Dict[str, Any] = field(default_factory=dict)
    
    # Dynamic threshold parameters
    adaptation_period: int = 3600  # 1 hour in seconds
    sensitivity: float = 0.1  # 10% sensitivity
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    
    # Statistical parameters
    percentile: Optional[float] = None  # For percentile-based thresholds
    std_dev_multiplier: Optional[float] = None  # For statistical thresholds
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'name': self.name,
            'metric_name': self.metric_name,
            'threshold_type': self.threshold_type.value,
            'condition': self.condition.value,
            'value': self.value,
            'enabled': self.enabled,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'tags': self.tags,
            'adaptation_period': self.adaptation_period,
            'sensitivity': self.sensitivity,
            'min_value': self.min_value,
            'max_value': self.max_value,
            'percentile': self.percentile,
            'std_dev_multiplier': self.std_dev_multiplier
        }


@dataclass
class ThresholdViolation:
    """Represents a threshold violation"""
    threshold_id: str
    metric_name: str
    actual_value: float
    threshold_value: float
    condition: ThresholdCondition
    timestamp: datetime
    severity: str
    tags: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'threshold_id': self.threshold_id,
            'metric_name': self.metric_name,
            'actual_value': self.actual_value,
            'threshold_value': self.threshold_value,
            'condition': self.condition.value,
            'timestamp': self.timestamp.isoformat(),
            'severity': self.severity,
            'tags': self.tags
        }


class ThresholdManager:
    """
    Dynamic threshold management system
    
    Manages static, dynamic, and adaptive thresholds with intelligent
    threshold adjustment and validation capabilities.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Threshold storage
        self.thresholds = {}  # threshold_id -> Threshold
        self.metric_history = {}  # metric_name -> List[values]
        self.threshold_violations = []  # List of recent violations
        
        # Configuration
        self.history_window = self.config.get('history_window', 86400)  # 24 hours
        self.max_history_points = self.config.get('max_history_points', 1000)
        self.adaptation_interval = self.config.get('adaptation_interval', 300)  # 5 minutes
        
        # Callbacks
        self.violation_callbacks = []  # List of violation callback functions
        
        # Last adaptation check
        self.last_adaptation_check = datetime.now()
        
    def create_threshold(self, threshold: Threshold) -> str:
        """Create a new threshold"""
        if threshold.id in self.thresholds:
            raise ValueError(f"Threshold with ID {threshold.id} already exists")
        
        # Validate threshold
        self._validate_threshold(threshold)
        
        # Store threshold
        self.thresholds[threshold.id] = threshold
        
        self.logger.info(f"Created threshold: {threshold.name} ({threshold.id})")
        return threshold.id
    
    def update_threshold(self, threshold_id: str, updates: Dict[str, Any]) -> bool:
        """Update an existing threshold"""
        if threshold_id not in self.thresholds:
            return False
        
        threshold = self.thresholds[threshold_id]
        
        # Update fields
        for key, value in updates.items():
            if hasattr(threshold, key):
                setattr(threshold, key, value)
        
        threshold.updated_at = datetime.now()
        
        # Re-validate threshold
        self._validate_threshold(threshold)
        
        self.logger.info(f"Updated threshold: {threshold.name} ({threshold_id})")
        return True
    
    def delete_threshold(self, threshold_id: str) -> bool:
        """Delete a threshold"""
        if threshold_id not in self.thresholds:
            return False
        
        threshold = self.thresholds.pop(threshold_id)
        self.logger.info(f"Deleted threshold: {threshold.name} ({threshold_id})")
        return True
    
    def get_threshold(self, threshold_id: str) -> Optional[Threshold]:
        """Get a threshold by ID"""
        return self.thresholds.get(threshold_id)
    
    def get_thresholds_for_metric(self, metric_name: str) -> List[Threshold]:
        """Get all thresholds for a specific metric"""
        return [t for t in self.thresholds.values() if t.metric_name == metric_name]
    
    def add_metric_value(self, metric_name: str, value: float, timestamp: Optional[datetime] = None):
        """Add a metric value for threshold evaluation"""
        if timestamp is None:
            timestamp = datetime.now()
        
        # Store in history
        if metric_name not in self.metric_history:
            self.metric_history[metric_name] = []
        
        self.metric_history[metric_name].append({
            'value': value,
            'timestamp': timestamp
        })
        
        # Maintain history window
        self._cleanup_metric_history(metric_name)
        
        # Evaluate thresholds
        self._evaluate_thresholds(metric_name, value, timestamp)
        
        # Check for threshold adaptation
        self._check_threshold_adaptation()
    
    def _cleanup_metric_history(self, metric_name: str):
        """Clean up old metric history"""
        if metric_name not in self.metric_history:
            return
        
        history = self.metric_history[metric_name]
        cutoff_time = datetime.now() - timedelta(seconds=self.history_window)
        
        # Remove old entries
        self.metric_history[metric_name] = [
            entry for entry in history 
            if entry['timestamp'] >= cutoff_time
        ]
        
        # Limit number of points
        if len(self.metric_history[metric_name]) > self.max_history_points:
            self.metric_history[metric_name] = self.metric_history[metric_name][-self.max_history_points:]
    
    def _evaluate_thresholds(self, metric_name: str, value: float, timestamp: datetime):
        """Evaluate all thresholds for a metric"""
        thresholds = self.get_thresholds_for_metric(metric_name)
        
        for threshold in thresholds:
            if not threshold.enabled:
                continue
            
            # Get current threshold value (may be dynamic)
            current_threshold_value = self._get_current_threshold_value(threshold)
            
            # Check threshold violation
            if self._check_threshold_violation(value, current_threshold_value, threshold.condition):
                violation = ThresholdViolation(
                    threshold_id=threshold.id,
                    metric_name=metric_name,
                    actual_value=value,
                    threshold_value=current_threshold_value,
                    condition=threshold.condition,
                    timestamp=timestamp,
                    severity=self._calculate_violation_severity(value, current_threshold_value, threshold),
                    tags=threshold.tags.copy()
                )
                
                self._record_violation(violation)
    
    def _get_current_threshold_value(self, threshold: Threshold) -> float:
        """Get current threshold value (static or dynamic)"""
        if threshold.threshold_type == ThresholdType.STATIC:
            return threshold.value
        elif threshold.threshold_type == ThresholdType.DYNAMIC:
            return self._calculate_dynamic_threshold(threshold)
        elif threshold.threshold_type == ThresholdType.ADAPTIVE:
            return self._calculate_adaptive_threshold(threshold)
        elif threshold.threshold_type == ThresholdType.PERCENTILE:
            return self._calculate_percentile_threshold(threshold)
        elif threshold.threshold_type == ThresholdType.STATISTICAL:
            return self._calculate_statistical_threshold(threshold)
        else:
            return threshold.value
    
    def _calculate_dynamic_threshold(self, threshold: Threshold) -> float:
        """Calculate dynamic threshold based on recent metrics"""
        if threshold.metric_name not in self.metric_history:
            return threshold.value
        
        history = self.metric_history[threshold.metric_name]
        if not history:
            return threshold.value
        
        # Get recent values within adaptation period
        cutoff_time = datetime.now() - timedelta(seconds=threshold.adaptation_period)
        recent_values = [
            entry['value'] for entry in history 
            if entry['timestamp'] >= cutoff_time
        ]
        
        if not recent_values:
            return threshold.value
        
        # Calculate threshold based on recent average
        avg_value = statistics.mean(recent_values)
        
        # Apply sensitivity adjustment
        if threshold.condition in [ThresholdCondition.GREATER_THAN, ThresholdCondition.GREATER_EQUAL]:
            dynamic_value = avg_value * (1 + threshold.sensitivity)
        else:
            dynamic_value = avg_value * (1 - threshold.sensitivity)
        
        # Apply min/max constraints
        if threshold.min_value is not None:
            dynamic_value = max(dynamic_value, threshold.min_value)
        if threshold.max_value is not None:
            dynamic_value = min(dynamic_value, threshold.max_value)
        
        return dynamic_value
    
    def _calculate_adaptive_threshold(self, threshold: Threshold) -> float:
        """Calculate adaptive threshold using statistical methods"""
        if threshold.metric_name not in self.metric_history:
            return threshold.value
        
        history = self.metric_history[threshold.metric_name]
        if not history:
            return threshold.value
        
        # Get recent values
        cutoff_time = datetime.now() - timedelta(seconds=threshold.adaptation_period)
        recent_values = [
            entry['value'] for entry in history 
            if entry['timestamp'] >= cutoff_time
        ]
        
        if len(recent_values) < 10:  # Need sufficient data
            return threshold.value
        
        # Calculate statistical measures
        mean_val = statistics.mean(recent_values)
        std_dev = statistics.stdev(recent_values)
        
        # Adaptive threshold using standard deviation
        std_multiplier = threshold.std_dev_multiplier or 2.0
        
        if threshold.condition in [ThresholdCondition.GREATER_THAN, ThresholdCondition.GREATER_EQUAL]:
            adaptive_value = mean_val + (std_multiplier * std_dev)
        else:
            adaptive_value = mean_val - (std_multiplier * std_dev)
        
        # Apply constraints
        if threshold.min_value is not None:
            adaptive_value = max(adaptive_value, threshold.min_value)
        if threshold.max_value is not None:
            adaptive_value = min(adaptive_value, threshold.max_value)
        
        return adaptive_value
    
    def _calculate_percentile_threshold(self, threshold: Threshold) -> float:
        """Calculate percentile-based threshold"""
        if threshold.metric_name not in self.metric_history:
            return threshold.value
        
        history = self.metric_history[threshold.metric_name]
        if not history:
            return threshold.value
        
        # Get recent values
        cutoff_time = datetime.now() - timedelta(seconds=threshold.adaptation_period)
        recent_values = [
            entry['value'] for entry in history 
            if entry['timestamp'] >= cutoff_time
        ]
        
        if len(recent_values) < 10:
            return threshold.value
        
        # Calculate percentile
        percentile = threshold.percentile or 95.0
        percentile_value = statistics.quantiles(recent_values, n=100)[int(percentile) - 1]
        
        return percentile_value
    
    def _calculate_statistical_threshold(self, threshold: Threshold) -> float:
        """Calculate statistical threshold using z-score"""
        if threshold.metric_name not in self.metric_history:
            return threshold.value
        
        history = self.metric_history[threshold.metric_name]
        if not history:
            return threshold.value
        
        # Get recent values
        cutoff_time = datetime.now() - timedelta(seconds=threshold.adaptation_period)
        recent_values = [
            entry['value'] for entry in history 
            if entry['timestamp'] >= cutoff_time
        ]
        
        if len(recent_values) < 10:
            return threshold.value
        
        # Calculate z-score based threshold
        mean_val = statistics.mean(recent_values)
        std_dev = statistics.stdev(recent_values)
        
        z_score = threshold.std_dev_multiplier or 2.0
        
        if threshold.condition in [ThresholdCondition.GREATER_THAN, ThresholdCondition.GREATER_EQUAL]:
            statistical_value = mean_val + (z_score * std_dev)
        else:
            statistical_value = mean_val - (z_score * std_dev)
        
        return statistical_value
    
    def _check_threshold_violation(self, value: float, threshold_value: float, condition: ThresholdCondition) -> bool:
        """Check if a value violates a threshold"""
        if condition == ThresholdCondition.GREATER_THAN:
            return value > threshold_value
        elif condition == ThresholdCondition.LESS_THAN:
            return value < threshold_value
        elif condition == ThresholdCondition.GREATER_EQUAL:
            return value >= threshold_value
        elif condition == ThresholdCondition.LESS_EQUAL:
            return value <= threshold_value
        elif condition == ThresholdCondition.EQUAL:
            return value == threshold_value
        elif condition == ThresholdCondition.NOT_EQUAL:
            return value != threshold_value
        
        return False
    
    def _calculate_violation_severity(self, value: float, threshold_value: float, threshold: Threshold) -> str:
        """Calculate severity of threshold violation"""
        if threshold_value == 0:
            return "medium"
        
        # Calculate relative violation magnitude
        if threshold.condition in [ThresholdCondition.GREATER_THAN, ThresholdCondition.GREATER_EQUAL]:
            violation_ratio = (value - threshold_value) / threshold_value
        else:
            violation_ratio = (threshold_value - value) / threshold_value
        
        # Determine severity based on violation magnitude
        if violation_ratio >= 1.0:  # 100%+ violation
            return "critical"
        elif violation_ratio >= 0.5:  # 50%+ violation
            return "high"
        elif violation_ratio >= 0.2:  # 20%+ violation
            return "medium"
        else:
            return "low"
    
    def _record_violation(self, violation: ThresholdViolation):
        """Record a threshold violation"""
        self.threshold_violations.append(violation)
        
        # Maintain violation history (last 24 hours)
        cutoff_time = datetime.now() - timedelta(hours=24)
        self.threshold_violations = [
            v for v in self.threshold_violations 
            if v.timestamp >= cutoff_time
        ]
        
        # Trigger violation callbacks
        for callback in self.violation_callbacks:
            try:
                callback(violation)
            except Exception as e:
                self.logger.error(f"Error in violation callback: {e}")
        
        self.logger.warning(f"Threshold violation: {violation.metric_name} = {violation.actual_value:.2f} "
                           f"{violation.condition.value} {violation.threshold_value:.2f}")
    
    def _check_threshold_adaptation(self):
        """Check if thresholds need adaptation"""
        current_time = datetime.now()
        
        if (current_time - self.last_adaptation_check).total_seconds() >= self.adaptation_interval:
            self._adapt_thresholds()
            self.last_adaptation_check = current_time
    
    def _adapt_thresholds(self):
        """Adapt dynamic and adaptive thresholds"""
        for threshold in self.thresholds.values():
            if threshold.threshold_type in [ThresholdType.ADAPTIVE, ThresholdType.DYNAMIC]:
                new_value = self._get_current_threshold_value(threshold)
                
                if abs(new_value - threshold.value) > (threshold.value * 0.1):  # 10% change
                    old_value = threshold.value
                    threshold.value = new_value
                    threshold.updated_at = datetime.now()
                    
                    self.logger.info(f"Adapted threshold {threshold.name}: {old_value:.2f} -> {new_value:.2f}")
    
    def add_violation_callback(self, callback: Callable[[ThresholdViolation], None]):
        """Add callback for threshold violations"""
        self.violation_callbacks.append(callback)
    
    def _validate_threshold(self, threshold: Threshold):
        """Validate threshold configuration"""
        if not threshold.name:
            raise ValueError("Threshold name is required")
        
        if not threshold.metric_name:
            raise ValueError("Threshold metric name is required")
        
        if threshold.threshold_type == ThresholdType.PERCENTILE:
            if threshold.percentile is None or not (0 < threshold.percentile <= 100):
                raise ValueError("Percentile threshold requires valid percentile value (0-100)")
        
        if threshold.threshold_type == ThresholdType.STATISTICAL:
            if threshold.std_dev_multiplier is None or threshold.std_dev_multiplier <= 0:
                raise ValueError("Statistical threshold requires valid std_dev_multiplier")
        
        if threshold.min_value is not None and threshold.max_value is not None:
            if threshold.min_value >= threshold.max_value:
                raise ValueError("min_value must be less than max_value")
    
    def get_threshold_summary(self) -> Dict[str, Any]:
        """Get threshold management summary"""
        active_thresholds = [t for t in self.thresholds.values() if t.enabled]
        
        return {
            'total_thresholds': len(self.thresholds),
            'active_thresholds': len(active_thresholds),
            'threshold_types': {
                threshold_type.value: len([t for t in active_thresholds if t.threshold_type == threshold_type])
                for threshold_type in ThresholdType
            },
            'recent_violations': len(self.threshold_violations),
            'metrics_monitored': len(self.metric_history),
            'last_adaptation': self.last_adaptation_check.isoformat()
        }
    
    def get_violations_summary(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Get recent violations summary"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        recent_violations = [
            v for v in self.threshold_violations 
            if v.timestamp >= cutoff_time
        ]
        
        return [v.to_dict() for v in recent_violations]
    
    def get_metric_statistics(self, metric_name: str, hours: int = 24) -> Dict[str, Any]:
        """Get statistics for a specific metric"""
        if metric_name not in self.metric_history:
            return {}
        
        cutoff_time = datetime.now() - timedelta(hours=hours)
        recent_values = [
            entry['value'] for entry in self.metric_history[metric_name] 
            if entry['timestamp'] >= cutoff_time
        ]
        
        if not recent_values:
            return {}
        
        return {
            'count': len(recent_values),
            'mean': statistics.mean(recent_values),
            'median': statistics.median(recent_values),
            'min': min(recent_values),
            'max': max(recent_values),
            'std_dev': statistics.stdev(recent_values) if len(recent_values) > 1 else 0,
            'percentiles': {
                'p50': statistics.median(recent_values),
                'p95': statistics.quantiles(recent_values, n=20)[18] if len(recent_values) >= 20 else max(recent_values),
                'p99': statistics.quantiles(recent_values, n=100)[98] if len(recent_values) >= 100 else max(recent_values)
            }
        }
    
    def export_thresholds(self) -> Dict[str, Any]:
        """Export threshold configuration"""
        return {
            'thresholds': [t.to_dict() for t in self.thresholds.values()],
            'export_timestamp': datetime.now().isoformat()
        }
    
    def import_thresholds(self, threshold_data: Dict[str, Any]) -> int:
        """Import threshold configuration"""
        imported_count = 0
        
        for threshold_dict in threshold_data.get('thresholds', []):
            try:
                threshold = Threshold(
                    id=threshold_dict['id'],
                    name=threshold_dict['name'],
                    metric_name=threshold_dict['metric_name'],
                    threshold_type=ThresholdType(threshold_dict['threshold_type']),
                    condition=ThresholdCondition(threshold_dict['condition']),
                    value=threshold_dict['value'],
                    enabled=threshold_dict.get('enabled', True),
                    created_at=datetime.fromisoformat(threshold_dict.get('created_at', datetime.now().isoformat())),
                    updated_at=datetime.fromisoformat(threshold_dict.get('updated_at', datetime.now().isoformat())),
                    tags=threshold_dict.get('tags', {}),
                    adaptation_period=threshold_dict.get('adaptation_period', 3600),
                    sensitivity=threshold_dict.get('sensitivity', 0.1),
                    min_value=threshold_dict.get('min_value'),
                    max_value=threshold_dict.get('max_value'),
                    percentile=threshold_dict.get('percentile'),
                    std_dev_multiplier=threshold_dict.get('std_dev_multiplier')
                )
                
                self.thresholds[threshold.id] = threshold
                imported_count += 1
                
            except Exception as e:
                self.logger.error(f"Error importing threshold {threshold_dict.get('id', 'unknown')}: {e}")
        
        return imported_count