"""
Usage Pattern Monitoring System

Specialized monitoring for Snowflake usage patterns including warehouse utilization,
user activity anomalies, and resource exhaustion detection.
"""

import time
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import logging

from .real_time_monitor import RealTimeMonitor, MonitoringMetric, Alert, AlertSeverity


class UsageMetricType(Enum):
    """Types of usage metrics"""
    WAREHOUSE_UTILIZATION = "warehouse_utilization"
    QUERY_VOLUME = "query_volume"
    CONCURRENT_USERS = "concurrent_users"
    STORAGE_USAGE = "storage_usage"
    COMPUTE_USAGE = "compute_usage"
    ACTIVE_CONNECTIONS = "active_connections"
    FAILED_QUERIES = "failed_queries"
    LONG_RUNNING_QUERIES = "long_running_queries"


@dataclass
class UsageThreshold:
    """Represents a usage threshold configuration"""
    metric_type: UsageMetricType
    threshold_value: float
    comparison_operator: str = ">"  # ">", "<", ">=", "<=", "=="
    enabled: bool = True
    severity: AlertSeverity = AlertSeverity.MEDIUM
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'metric_type': self.metric_type.value,
            'threshold_value': self.threshold_value,
            'comparison_operator': self.comparison_operator,
            'enabled': self.enabled,
            'severity': self.severity.value
        }


@dataclass
class UsagePattern:
    """Represents a detected usage pattern"""
    pattern_type: str
    metrics_involved: List[str]
    pattern_strength: float
    timestamp: datetime
    description: str
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'pattern_type': self.pattern_type,
            'metrics_involved': self.metrics_involved,
            'pattern_strength': self.pattern_strength,
            'timestamp': self.timestamp.isoformat(),
            'description': self.description
        }


class UsageMonitor:
    """
    Usage pattern monitoring system for Snowflake
    
    Monitors warehouse utilization, query performance, user activity anomalies,
    and resource exhaustion with intelligent pattern detection.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Initialize base monitor
        self.monitor = RealTimeMonitor(config)
        
        # Usage-specific configuration
        self.usage_thresholds = {}  # metric_type -> UsageThreshold
        self.usage_patterns = []  # List of detected patterns
        self.baseline_metrics = {}  # metric_type -> baseline_value
        
        # Pattern detection parameters
        self.pattern_detection_window = self.config.get('pattern_detection_window', 3600)  # 1 hour
        self.pattern_strength_threshold = self.config.get('pattern_strength_threshold', 0.7)
        
        # Load configuration
        self._load_usage_config()
        
        # Register alert callback
        self.monitor.add_alert_callback(self._handle_usage_alert)
        
    def _load_usage_config(self):
        """Load usage monitoring configuration"""
        # Load usage thresholds
        thresholds_config = self.config.get('usage_thresholds', {})
        for metric_type, threshold_data in thresholds_config.items():
            if hasattr(UsageMetricType, metric_type.upper()):
                self.usage_thresholds[metric_type] = UsageThreshold(
                    metric_type=UsageMetricType(metric_type.lower()),
                    threshold_value=threshold_data.get('threshold_value', 80.0),
                    comparison_operator=threshold_data.get('comparison_operator', '>'),
                    enabled=threshold_data.get('enabled', True),
                    severity=AlertSeverity(threshold_data.get('severity', 'medium'))
                )
        
        # Load baseline metrics
        baselines_config = self.config.get('baseline_metrics', {})
        for metric_type, baseline_value in baselines_config.items():
            self.baseline_metrics[metric_type] = baseline_value
    
    def start_monitoring(self):
        """Start usage monitoring"""
        self.monitor.start_monitoring()
        self.logger.info("Usage monitoring started")
    
    def stop_monitoring(self):
        """Stop usage monitoring"""
        self.monitor.stop_monitoring()
        self.logger.info("Usage monitoring stopped")
    
    def add_usage_metric(self, metric_type: UsageMetricType, value: float,
                        source: str = "snowflake", tags: Optional[Dict[str, Any]] = None):
        """Add a usage metric for monitoring"""
        metric = MonitoringMetric(
            name=f"usage_{metric_type.value}",
            value=value,
            timestamp=datetime.now(),
            source=source,
            tags=tags or {}
        )
        
        self.monitor.add_metric(metric)
        
        # Check usage thresholds
        self._check_usage_thresholds(metric_type, value)
        
        # Update baseline if needed
        self._update_baseline(metric_type, value)
        
        # Check for usage patterns
        self._check_usage_patterns(metric_type, value)
    
    def _check_usage_thresholds(self, metric_type: UsageMetricType, value: float):
        """Check if usage exceeds defined thresholds"""
        if metric_type.value in self.usage_thresholds:
            threshold = self.usage_thresholds[metric_type.value]
            
            if threshold.enabled and self._evaluate_threshold(value, threshold):
                self._create_usage_threshold_alert(metric_type, value, threshold)
    
    def _evaluate_threshold(self, value: float, threshold: UsageThreshold) -> bool:
        """Evaluate if a value breaches a threshold"""
        if threshold.comparison_operator == ">":
            return value > threshold.threshold_value
        elif threshold.comparison_operator == "<":
            return value < threshold.threshold_value
        elif threshold.comparison_operator == ">=":
            return value >= threshold.threshold_value
        elif threshold.comparison_operator == "<=":
            return value <= threshold.threshold_value
        elif threshold.comparison_operator == "==":
            return value == threshold.threshold_value
        
        return False
    
    def _update_baseline(self, metric_type: UsageMetricType, value: float):
        """Update baseline metrics using exponential smoothing"""
        if metric_type.value not in self.baseline_metrics:
            self.baseline_metrics[metric_type.value] = value
        else:
            # Use exponential smoothing with alpha = 0.1
            alpha = 0.1
            current_baseline = self.baseline_metrics[metric_type.value]
            self.baseline_metrics[metric_type.value] = (alpha * value) + ((1 - alpha) * current_baseline)
    
    def _check_usage_patterns(self, metric_type: UsageMetricType, value: float):
        """Check for usage patterns and anomalies"""
        # Get recent metrics for pattern analysis
        recent_metrics = self._get_recent_metrics(metric_type, hours=1)
        
        if len(recent_metrics) < 10:  # Need sufficient data
            return
        
        # Detect patterns
        self._detect_spike_pattern(metric_type, recent_metrics)
        self._detect_gradual_increase_pattern(metric_type, recent_metrics)
        self._detect_oscillation_pattern(metric_type, recent_metrics)
        self._detect_resource_exhaustion_pattern(metric_type, recent_metrics)
    
    def _get_recent_metrics(self, metric_type: UsageMetricType, hours: int = 1) -> List[float]:
        """Get recent metric values for pattern analysis"""
        metric_name = f"usage_{metric_type.value}"
        history = self.monitor.get_metric_history(metric_name, limit=200)
        
        # Filter to recent hours
        cutoff_time = datetime.now() - timedelta(hours=hours)
        recent_values = []
        
        for record in history:
            timestamp = datetime.fromisoformat(record['timestamp'])
            if timestamp >= cutoff_time:
                recent_values.append(record['value'])
        
        return recent_values
    
    def _detect_spike_pattern(self, metric_type: UsageMetricType, values: List[float]):
        """Detect sudden spikes in usage"""
        if len(values) < 5:
            return
        
        # Calculate moving average
        window_size = min(5, len(values) // 2)
        moving_avg = []
        
        for i in range(window_size, len(values)):
            avg = sum(values[i-window_size:i]) / window_size
            moving_avg.append(avg)
        
        if not moving_avg:
            return
        
        # Check for spikes (value > 2 * moving average)
        latest_value = values[-1]
        latest_avg = moving_avg[-1]
        
        if latest_value > (2 * latest_avg) and latest_avg > 0:
            pattern = UsagePattern(
                pattern_type="spike",
                metrics_involved=[metric_type.value],
                pattern_strength=latest_value / latest_avg,
                timestamp=datetime.now(),
                description=f"Spike detected in {metric_type.value}: {latest_value:.2f} (avg: {latest_avg:.2f})"
            )
            
            self._record_pattern(pattern)
            self._create_pattern_alert(pattern)
    
    def _detect_gradual_increase_pattern(self, metric_type: UsageMetricType, values: List[float]):
        """Detect gradual increase in usage"""
        if len(values) < 10:
            return
        
        # Calculate trend using simple linear regression
        n = len(values)
        x = list(range(n))
        y = values
        
        # Calculate slope
        sum_x = sum(x)
        sum_y = sum(y)
        sum_xy = sum(x[i] * y[i] for i in range(n))
        sum_x2 = sum(xi * xi for xi in x)
        
        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
        
        # Check for significant positive trend
        avg_value = sum_y / n
        trend_strength = abs(slope) / (avg_value + 1e-10)  # Avoid division by zero
        
        if slope > 0 and trend_strength > 0.1:  # Significant positive trend
            pattern = UsagePattern(
                pattern_type="gradual_increase",
                metrics_involved=[metric_type.value],
                pattern_strength=trend_strength,
                timestamp=datetime.now(),
                description=f"Gradual increase detected in {metric_type.value}: slope={slope:.4f}"
            )
            
            self._record_pattern(pattern)
            
            if trend_strength > 0.3:  # Strong trend
                self._create_pattern_alert(pattern)
    
    def _detect_oscillation_pattern(self, metric_type: UsageMetricType, values: List[float]):
        """Detect oscillating usage patterns"""
        if len(values) < 15:
            return
        
        # Calculate coefficient of variation
        mean_val = sum(values) / len(values)
        variance = sum((x - mean_val) ** 2 for x in values) / len(values)
        std_dev = variance ** 0.5
        
        if mean_val > 0:
            coefficient_of_variation = std_dev / mean_val
            
            # Check for high variability (oscillation)
            if coefficient_of_variation > 0.5:
                pattern = UsagePattern(
                    pattern_type="oscillation",
                    metrics_involved=[metric_type.value],
                    pattern_strength=coefficient_of_variation,
                    timestamp=datetime.now(),
                    description=f"Oscillating pattern detected in {metric_type.value}: CV={coefficient_of_variation:.3f}"
                )
                
                self._record_pattern(pattern)
                
                if coefficient_of_variation > 1.0:  # High oscillation
                    self._create_pattern_alert(pattern)
    
    def _detect_resource_exhaustion_pattern(self, metric_type: UsageMetricType, values: List[float]):
        """Detect resource exhaustion patterns"""
        if len(values) < 5:
            return
        
        # Check for sustained high values
        if metric_type in [UsageMetricType.WAREHOUSE_UTILIZATION, UsageMetricType.COMPUTE_USAGE]:
            high_threshold = 90.0  # 90% utilization
            recent_values = values[-5:]  # Last 5 values
            
            if all(v > high_threshold for v in recent_values):
                pattern = UsagePattern(
                    pattern_type="resource_exhaustion",
                    metrics_involved=[metric_type.value],
                    pattern_strength=min(recent_values) / 100.0,
                    timestamp=datetime.now(),
                    description=f"Resource exhaustion detected in {metric_type.value}: sustained high usage"
                )
                
                self._record_pattern(pattern)
                self._create_pattern_alert(pattern)
    
    def _record_pattern(self, pattern: UsagePattern):
        """Record detected pattern"""
        self.usage_patterns.append(pattern)
        
        # Keep only recent patterns (last 24 hours)
        cutoff_time = datetime.now() - timedelta(hours=24)
        self.usage_patterns = [p for p in self.usage_patterns if p.timestamp >= cutoff_time]
        
        self.logger.info(f"Pattern detected: {pattern.pattern_type} - {pattern.description}")
    
    def _create_usage_threshold_alert(self, metric_type: UsageMetricType, value: float, threshold: UsageThreshold):
        """Create usage threshold alert"""
        alert_id = f"usage_threshold_{metric_type.value}_{int(time.time())}"
        
        metric = MonitoringMetric(
            name=f"usage_{metric_type.value}",
            value=value,
            timestamp=datetime.now(),
            source="usage_monitor",
            tags={
                'metric_type': metric_type.value,
                'threshold_value': threshold.threshold_value,
                'comparison_operator': threshold.comparison_operator
            }
        )
        
        alert = Alert(
            id=alert_id,
            title=f"Usage Threshold Alert: {metric_type.value}",
            description=f"Usage metric '{metric_type.value}' breached threshold: "
                       f"{value:.2f} {threshold.comparison_operator} {threshold.threshold_value:.2f}",
            severity=threshold.severity,
            metric=metric,
            threshold=threshold.threshold_value,
            timestamp=datetime.now(),
            tags={
                'type': 'usage_threshold',
                'metric_type': metric_type.value
            }
        )
        
        self._send_alert(alert)
    
    def _create_pattern_alert(self, pattern: UsagePattern):
        """Create pattern-based alert"""
        alert_id = f"usage_pattern_{pattern.pattern_type}_{int(time.time())}"
        
        # Determine severity based on pattern strength
        if pattern.pattern_strength > 2.0:
            severity = AlertSeverity.CRITICAL
        elif pattern.pattern_strength > 1.5:
            severity = AlertSeverity.HIGH
        elif pattern.pattern_strength > 1.0:
            severity = AlertSeverity.MEDIUM
        else:
            severity = AlertSeverity.LOW
        
        metric = MonitoringMetric(
            name=f"usage_pattern_{pattern.pattern_type}",
            value=pattern.pattern_strength,
            timestamp=pattern.timestamp,
            source="usage_monitor",
            tags={
                'pattern_type': pattern.pattern_type,
                'metrics_involved': pattern.metrics_involved
            }
        )
        
        alert = Alert(
            id=alert_id,
            title=f"Usage Pattern Alert: {pattern.pattern_type}",
            description=pattern.description,
            severity=severity,
            metric=metric,
            threshold=self.pattern_strength_threshold,
            timestamp=datetime.now(),
            tags={
                'type': 'usage_pattern',
                'pattern_type': pattern.pattern_type,
                'pattern_strength': pattern.pattern_strength
            }
        )
        
        self._send_alert(alert)
    
    def _handle_usage_alert(self, alert: Alert):
        """Handle usage-specific alerts"""
        self.logger.info(f"Usage alert received: {alert.title}")
        
        # Add usage-specific processing
        if alert.tags.get('type') == 'usage_threshold':
            self._process_threshold_alert(alert)
        elif alert.tags.get('type') == 'usage_pattern':
            self._process_pattern_alert(alert)
    
    def _process_threshold_alert(self, alert: Alert):
        """Process threshold-specific alerts"""
        metric_type = alert.tags.get('metric_type')
        self.logger.info(f"Processing threshold alert for metric: {metric_type}")
        
        # Add automated responses based on metric type
        if metric_type == UsageMetricType.WAREHOUSE_UTILIZATION.value:
            self._handle_warehouse_utilization_alert(alert)
        elif metric_type == UsageMetricType.FAILED_QUERIES.value:
            self._handle_failed_queries_alert(alert)
    
    def _process_pattern_alert(self, alert: Alert):
        """Process pattern-specific alerts"""
        pattern_type = alert.tags.get('pattern_type')
        self.logger.info(f"Processing pattern alert for pattern: {pattern_type}")
    
    def _handle_warehouse_utilization_alert(self, alert: Alert):
        """Handle warehouse utilization alerts"""
        self.logger.warning(f"High warehouse utilization detected: {alert.metric.value:.2f}%")
        # Could trigger auto-scaling or notifications
    
    def _handle_failed_queries_alert(self, alert: Alert):
        """Handle failed queries alerts"""
        self.logger.warning(f"High failed query rate detected: {alert.metric.value:.2f}")
        # Could trigger investigations or automated remediation
    
    def _send_alert(self, alert: Alert):
        """Send alert through the monitoring system"""
        # Delegate to the base monitor's alert system
        for callback in self.monitor.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                self.logger.error(f"Error in alert callback: {e}")
    
    def get_usage_summary(self) -> Dict[str, Any]:
        """Get usage monitoring summary"""
        summary = {
            'current_metrics': {},
            'thresholds': {},
            'patterns': [],
            'baselines': self.baseline_metrics
        }
        
        # Current metrics summary
        metrics_summary = self.monitor.get_metrics_summary()
        for metric_name, metric_data in metrics_summary.items():
            if metric_name.startswith('usage_'):
                summary['current_metrics'][metric_name] = metric_data
        
        # Thresholds summary
        for metric_type, threshold in self.usage_thresholds.items():
            summary['thresholds'][metric_type] = threshold.to_dict()
        
        # Recent patterns
        summary['patterns'] = [p.to_dict() for p in self.usage_patterns[-10:]]
        
        return summary
    
    def get_usage_trend(self, metric_type: UsageMetricType, hours: int = 24) -> List[Dict[str, Any]]:
        """Get usage trend for a specific metric"""
        metric_name = f"usage_{metric_type.value}"
        history = self.monitor.get_metric_history(metric_name, limit=1000)
        
        # Filter to specified time range
        cutoff_time = datetime.now() - timedelta(hours=hours)
        trend_data = []
        
        for record in history:
            timestamp = datetime.fromisoformat(record['timestamp'])
            if timestamp >= cutoff_time:
                trend_data.append({
                    'timestamp': record['timestamp'],
                    'value': record['value'],
                    'tags': record.get('tags', {})
                })
        
        return trend_data
    
    def get_patterns_by_type(self, pattern_type: str) -> List[Dict[str, Any]]:
        """Get patterns filtered by type"""
        filtered_patterns = [p for p in self.usage_patterns if p.pattern_type == pattern_type]
        return [p.to_dict() for p in filtered_patterns]