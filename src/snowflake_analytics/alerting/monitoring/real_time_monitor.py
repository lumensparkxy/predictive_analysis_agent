"""
Real-time Monitoring Engine

Provides real-time data stream monitoring with sliding window anomaly detection
and threshold-based alerting capabilities.
"""

import time
import threading
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime, timedelta
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
import json
import logging


class AlertSeverity(Enum):
    """Alert severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class MonitoringMetric:
    """Represents a monitoring metric with metadata"""
    name: str
    value: float
    timestamp: datetime
    source: str
    tags: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'value': self.value,
            'timestamp': self.timestamp.isoformat(),
            'source': self.source,
            'tags': self.tags
        }


@dataclass
class Alert:
    """Represents an alert with severity and context"""
    id: str
    title: str
    description: str
    severity: AlertSeverity
    metric: MonitoringMetric
    threshold: float
    timestamp: datetime
    tags: Dict[str, Any] = field(default_factory=dict)
    acknowledged: bool = False
    resolved: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'title': self.title,
            'description': self.description,
            'severity': self.severity.value,
            'metric': self.metric.to_dict(),
            'threshold': self.threshold,
            'timestamp': self.timestamp.isoformat(),
            'tags': self.tags,
            'acknowledged': self.acknowledged,
            'resolved': self.resolved
        }


class SlidingWindow:
    """Sliding window for time series data analysis"""
    
    def __init__(self, window_size: int = 60, max_points: int = 1000):
        self.window_size = window_size  # seconds
        self.max_points = max_points
        self.data = deque(maxlen=max_points)
        self._lock = threading.Lock()
    
    def add_point(self, metric: MonitoringMetric):
        """Add a new data point to the window"""
        with self._lock:
            self.data.append(metric)
            self._cleanup_old_data()
    
    def _cleanup_old_data(self):
        """Remove data points older than window_size"""
        cutoff_time = datetime.now() - timedelta(seconds=self.window_size)
        while self.data and self.data[0].timestamp < cutoff_time:
            self.data.popleft()
    
    def get_current_window(self) -> List[MonitoringMetric]:
        """Get current window data"""
        with self._lock:
            self._cleanup_old_data()
            return list(self.data)
    
    def get_statistics(self) -> Dict[str, float]:
        """Calculate statistics for current window"""
        with self._lock:
            self._cleanup_old_data()
            if not self.data:
                return {}
            
            values = [point.value for point in self.data]
            return {
                'count': len(values),
                'mean': sum(values) / len(values),
                'min': min(values),
                'max': max(values),
                'latest': values[-1] if values else 0.0
            }


class RealTimeMonitor:
    """
    Real-time monitoring engine for Snowflake metrics
    
    Provides sliding window anomaly detection and threshold-based alerting
    with configurable monitoring parameters.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Monitoring state
        self.is_monitoring = False
        self.monitoring_thread = None
        self.metrics_windows = {}  # metric_name -> SlidingWindow
        self.alert_callbacks = []  # List of alert callback functions
        
        # Configuration
        self.monitoring_interval = self.config.get('monitoring_interval', 10)  # seconds
        self.window_size = self.config.get('window_size', 300)  # 5 minutes
        self.anomaly_threshold = self.config.get('anomaly_threshold', 2.0)  # std deviations
        
        # Threading
        self._lock = threading.Lock()
        
    def start_monitoring(self):
        """Start real-time monitoring"""
        if self.is_monitoring:
            return
            
        self.is_monitoring = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop)
        self.monitoring_thread.daemon = True
        self.monitoring_thread.start()
        
        self.logger.info("Real-time monitoring started")
    
    def stop_monitoring(self):
        """Stop real-time monitoring"""
        self.is_monitoring = False
        if self.monitoring_thread:
            self.monitoring_thread.join()
        
        self.logger.info("Real-time monitoring stopped")
    
    def add_metric(self, metric: MonitoringMetric):
        """Add a new metric for monitoring"""
        with self._lock:
            if metric.name not in self.metrics_windows:
                self.metrics_windows[metric.name] = SlidingWindow(
                    window_size=self.window_size
                )
            
            self.metrics_windows[metric.name].add_point(metric)
            
            # Check for anomalies and threshold breaches
            self._check_anomalies(metric)
    
    def add_alert_callback(self, callback: Callable[[Alert], None]):
        """Add callback function for alert notifications"""
        self.alert_callbacks.append(callback)
    
    def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.is_monitoring:
            try:
                # Perform periodic monitoring tasks
                self._check_all_metrics()
                time.sleep(self.monitoring_interval)
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                time.sleep(self.monitoring_interval)
    
    def _check_all_metrics(self):
        """Check all metrics for anomalies and threshold breaches"""
        with self._lock:
            for metric_name, window in self.metrics_windows.items():
                data = window.get_current_window()
                if len(data) > 0:
                    self._analyze_metric_window(metric_name, data)
    
    def _analyze_metric_window(self, metric_name: str, data: List[MonitoringMetric]):
        """Analyze a metric window for patterns and anomalies"""
        if len(data) < 2:
            return
            
        # Calculate statistics
        values = [point.value for point in data]
        mean = sum(values) / len(values)
        
        # Simple anomaly detection (can be enhanced)
        if len(values) > 10:
            # Calculate standard deviation
            variance = sum((x - mean) ** 2 for x in values) / len(values)
            std_dev = variance ** 0.5
            
            # Check if latest value is anomalous
            latest_value = values[-1]
            if abs(latest_value - mean) > (self.anomaly_threshold * std_dev):
                self._trigger_anomaly_alert(data[-1], mean, std_dev)
    
    def _check_anomalies(self, metric: MonitoringMetric):
        """Check for anomalies in a single metric"""
        window = self.metrics_windows[metric.name]
        stats = window.get_statistics()
        
        # Simple threshold-based anomaly detection
        if stats.get('count', 0) > 5:  # Need enough data points
            mean = stats['mean']
            current_value = metric.value
            
            # Calculate rough standard deviation
            data = window.get_current_window()
            if len(data) > 1:
                values = [point.value for point in data]
                variance = sum((x - mean) ** 2 for x in values) / len(values)
                std_dev = variance ** 0.5
                
                if abs(current_value - mean) > (self.anomaly_threshold * std_dev):
                    self._trigger_anomaly_alert(metric, mean, std_dev)
    
    def _trigger_anomaly_alert(self, metric: MonitoringMetric, mean: float, std_dev: float):
        """Trigger an anomaly alert"""
        alert_id = f"anomaly_{metric.name}_{int(time.time())}"
        
        # Determine severity based on deviation
        deviation = abs(metric.value - mean) / std_dev if std_dev > 0 else 0
        if deviation > 4:
            severity = AlertSeverity.CRITICAL
        elif deviation > 3:
            severity = AlertSeverity.HIGH
        elif deviation > 2:
            severity = AlertSeverity.MEDIUM
        else:
            severity = AlertSeverity.LOW
        
        alert = Alert(
            id=alert_id,
            title=f"Anomaly Detected: {metric.name}",
            description=f"Metric {metric.name} value {metric.value:.2f} deviates "
                       f"{deviation:.2f} standard deviations from mean {mean:.2f}",
            severity=severity,
            metric=metric,
            threshold=mean + (self.anomaly_threshold * std_dev),
            timestamp=datetime.now(),
            tags={'type': 'anomaly', 'deviation': deviation}
        )
        
        self._send_alert(alert)
    
    def _send_alert(self, alert: Alert):
        """Send alert to all registered callbacks"""
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                self.logger.error(f"Error in alert callback: {e}")
    
    def get_metrics_summary(self) -> Dict[str, Dict[str, Any]]:
        """Get summary of all monitored metrics"""
        with self._lock:
            summary = {}
            for metric_name, window in self.metrics_windows.items():
                summary[metric_name] = {
                    'statistics': window.get_statistics(),
                    'data_points': len(window.get_current_window())
                }
            return summary
    
    def get_metric_history(self, metric_name: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Get historical data for a specific metric"""
        with self._lock:
            if metric_name not in self.metrics_windows:
                return []
            
            window = self.metrics_windows[metric_name]
            data = window.get_current_window()
            
            return [point.to_dict() for point in data[-limit:]]