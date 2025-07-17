"""
Performance Monitoring System

Specialized monitoring for Snowflake performance metrics including query performance,
warehouse response times, and performance degradation detection.
"""

import time
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import statistics
import logging

from .real_time_monitor import RealTimeMonitor, MonitoringMetric, Alert, AlertSeverity


class PerformanceMetricType(Enum):
    """Types of performance metrics"""
    QUERY_EXECUTION_TIME = "query_execution_time"
    WAREHOUSE_RESPONSE_TIME = "warehouse_response_time"
    QUEUE_WAIT_TIME = "queue_wait_time"
    THROUGHPUT = "throughput"
    COMPILATION_TIME = "compilation_time"
    EXECUTION_TIME = "execution_time"
    SPILL_TO_DISK = "spill_to_disk"
    CACHE_HIT_RATIO = "cache_hit_ratio"
    SCAN_PROGRESS = "scan_progress"
    BYTES_SCANNED = "bytes_scanned"


@dataclass
class PerformanceSLA:
    """Service Level Agreement configuration for performance metrics"""
    metric_type: PerformanceMetricType
    target_value: float
    tolerance_percentage: float = 10.0  # 10% tolerance
    measurement_period: int = 300  # 5 minutes
    enabled: bool = True
    
    def get_threshold_value(self) -> float:
        """Get threshold value based on target and tolerance"""
        return self.target_value * (1 + self.tolerance_percentage / 100)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'metric_type': self.metric_type.value,
            'target_value': self.target_value,
            'tolerance_percentage': self.tolerance_percentage,
            'measurement_period': self.measurement_period,
            'enabled': self.enabled,
            'threshold_value': self.get_threshold_value()
        }


@dataclass
class PerformanceTrend:
    """Represents a performance trend analysis"""
    metric_type: PerformanceMetricType
    trend_direction: str  # 'improving', 'degrading', 'stable'
    trend_strength: float  # 0.0 to 1.0
    time_period: int  # seconds
    start_time: datetime
    end_time: datetime
    description: str
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'metric_type': self.metric_type.value,
            'trend_direction': self.trend_direction,
            'trend_strength': self.trend_strength,
            'time_period': self.time_period,
            'start_time': self.start_time.isoformat(),
            'end_time': self.end_time.isoformat(),
            'description': self.description
        }


class PerformanceMonitor:
    """
    Performance monitoring system for Snowflake
    
    Monitors query performance, warehouse response times, and detects
    performance degradation with SLA tracking and trend analysis.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Initialize base monitor
        self.monitor = RealTimeMonitor(config)
        
        # Performance-specific configuration
        self.performance_slas = {}  # metric_type -> PerformanceSLA
        self.performance_trends = []  # List of detected trends
        self.performance_baselines = {}  # metric_type -> baseline_stats
        
        # Performance monitoring parameters
        self.sla_check_interval = self.config.get('sla_check_interval', 60)  # seconds
        self.trend_analysis_window = self.config.get('trend_analysis_window', 3600)  # 1 hour
        self.degradation_threshold = self.config.get('degradation_threshold', 0.2)  # 20% degradation
        
        # Load configuration
        self._load_performance_config()
        
        # Register alert callback
        self.monitor.add_alert_callback(self._handle_performance_alert)
        
        # Last SLA check timestamp
        self.last_sla_check = datetime.now()
        
    def _load_performance_config(self):
        """Load performance monitoring configuration"""
        # Load SLA configurations
        slas_config = self.config.get('performance_slas', {})
        for metric_type, sla_data in slas_config.items():
            if hasattr(PerformanceMetricType, metric_type.upper()):
                self.performance_slas[metric_type] = PerformanceSLA(
                    metric_type=PerformanceMetricType(metric_type.lower()),
                    target_value=sla_data.get('target_value', 1.0),
                    tolerance_percentage=sla_data.get('tolerance_percentage', 10.0),
                    measurement_period=sla_data.get('measurement_period', 300),
                    enabled=sla_data.get('enabled', True)
                )
        
        # Load baseline configurations
        baselines_config = self.config.get('performance_baselines', {})
        for metric_type, baseline_data in baselines_config.items():
            self.performance_baselines[metric_type] = baseline_data
    
    def start_monitoring(self):
        """Start performance monitoring"""
        self.monitor.start_monitoring()
        self.logger.info("Performance monitoring started")
    
    def stop_monitoring(self):
        """Stop performance monitoring"""
        self.monitor.stop_monitoring()
        self.logger.info("Performance monitoring stopped")
    
    def add_performance_metric(self, metric_type: PerformanceMetricType, value: float,
                             query_id: Optional[str] = None, warehouse_name: Optional[str] = None,
                             tags: Optional[Dict[str, Any]] = None):
        """Add a performance metric for monitoring"""
        metric_tags = tags or {}
        if query_id:
            metric_tags['query_id'] = query_id
        if warehouse_name:
            metric_tags['warehouse_name'] = warehouse_name
        
        metric = MonitoringMetric(
            name=f"performance_{metric_type.value}",
            value=value,
            timestamp=datetime.now(),
            source="performance_monitor",
            tags=metric_tags
        )
        
        self.monitor.add_metric(metric)
        
        # Check SLA compliance
        self._check_sla_compliance(metric_type, value)
        
        # Update baselines
        self._update_performance_baseline(metric_type, value)
        
        # Check for performance degradation
        self._check_performance_degradation(metric_type, value)
        
        # Periodic trend analysis
        self._check_trends()
    
    def _check_sla_compliance(self, metric_type: PerformanceMetricType, value: float):
        """Check if performance metric meets SLA requirements"""
        if metric_type.value in self.performance_slas:
            sla = self.performance_slas[metric_type.value]
            
            if sla.enabled and value > sla.get_threshold_value():
                self._create_sla_violation_alert(metric_type, value, sla)
    
    def _update_performance_baseline(self, metric_type: PerformanceMetricType, value: float):
        """Update performance baseline using exponential moving average"""
        metric_key = metric_type.value
        
        if metric_key not in self.performance_baselines:
            self.performance_baselines[metric_key] = {
                'mean': value,
                'count': 1,
                'min': value,
                'max': value,
                'p95': value,
                'p99': value
            }
        else:
            baseline = self.performance_baselines[metric_key]
            alpha = 0.1  # Smoothing factor
            
            # Update mean using exponential moving average
            baseline['mean'] = (alpha * value) + ((1 - alpha) * baseline['mean'])
            baseline['count'] += 1
            baseline['min'] = min(baseline['min'], value)
            baseline['max'] = max(baseline['max'], value)
            
            # Update percentiles (simplified approach)
            self._update_percentiles(metric_type, value)
    
    def _update_percentiles(self, metric_type: PerformanceMetricType, value: float):
        """Update percentile estimates"""
        # Get recent values for percentile calculation
        recent_values = self._get_recent_values(metric_type, hours=1)
        
        if len(recent_values) > 10:
            recent_values.sort()
            p95_idx = int(0.95 * len(recent_values))
            p99_idx = int(0.99 * len(recent_values))
            
            self.performance_baselines[metric_type.value]['p95'] = recent_values[p95_idx]
            self.performance_baselines[metric_type.value]['p99'] = recent_values[p99_idx]
    
    def _get_recent_values(self, metric_type: PerformanceMetricType, hours: int = 1) -> List[float]:
        """Get recent performance values for analysis"""
        metric_name = f"performance_{metric_type.value}"
        history = self.monitor.get_metric_history(metric_name, limit=500)
        
        # Filter to recent hours
        cutoff_time = datetime.now() - timedelta(hours=hours)
        recent_values = []
        
        for record in history:
            timestamp = datetime.fromisoformat(record['timestamp'])
            if timestamp >= cutoff_time:
                recent_values.append(record['value'])
        
        return recent_values
    
    def _check_performance_degradation(self, metric_type: PerformanceMetricType, value: float):
        """Check for performance degradation"""
        if metric_type.value in self.performance_baselines:
            baseline = self.performance_baselines[metric_type.value]
            baseline_mean = baseline['mean']
            
            # Check for significant degradation
            if baseline_mean > 0:
                degradation_ratio = (value - baseline_mean) / baseline_mean
                
                if degradation_ratio > self.degradation_threshold:
                    self._create_degradation_alert(metric_type, value, baseline_mean, degradation_ratio)
    
    def _check_trends(self):
        """Check for performance trends periodically"""
        current_time = datetime.now()
        
        # Check trends every 5 minutes
        if (current_time - self.last_sla_check).total_seconds() >= 300:
            self._analyze_all_trends()
            self.last_sla_check = current_time
    
    def _analyze_all_trends(self):
        """Analyze trends for all performance metrics"""
        for metric_type in PerformanceMetricType:
            self._analyze_trend(metric_type)
    
    def _analyze_trend(self, metric_type: PerformanceMetricType):
        """Analyze trend for a specific performance metric"""
        values = self._get_recent_values(metric_type, hours=1)
        
        if len(values) < 10:
            return
        
        # Calculate trend using linear regression
        trend = self._calculate_trend(values)
        
        if trend:
            self._record_trend(metric_type, trend)
            
            # Create alert for significant trends
            if abs(trend['slope']) > 0.1 and trend['r_squared'] > 0.7:
                self._create_trend_alert(metric_type, trend)
    
    def _calculate_trend(self, values: List[float]) -> Optional[Dict[str, float]]:
        """Calculate trend statistics using linear regression"""
        if len(values) < 2:
            return None
        
        n = len(values)
        x = list(range(n))
        y = values
        
        try:
            # Calculate linear regression
            sum_x = sum(x)
            sum_y = sum(y)
            sum_xy = sum(x[i] * y[i] for i in range(n))
            sum_x2 = sum(xi * xi for xi in x)
            sum_y2 = sum(yi * yi for yi in y)
            
            # Calculate slope and intercept
            denominator = n * sum_x2 - sum_x * sum_x
            if denominator == 0:
                return None
            
            slope = (n * sum_xy - sum_x * sum_y) / denominator
            intercept = (sum_y - slope * sum_x) / n
            
            # Calculate R-squared
            y_mean = sum_y / n
            ss_tot = sum((yi - y_mean) ** 2 for yi in y)
            ss_res = sum((y[i] - (slope * x[i] + intercept)) ** 2 for i in range(n))
            
            r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
            
            return {
                'slope': slope,
                'intercept': intercept,
                'r_squared': r_squared,
                'trend_strength': abs(slope) / (y_mean + 1e-10)
            }
        except Exception as e:
            self.logger.error(f"Error calculating trend: {e}")
            return None
    
    def _record_trend(self, metric_type: PerformanceMetricType, trend_stats: Dict[str, float]):
        """Record detected performance trend"""
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=1)
        
        trend_direction = "improving" if trend_stats['slope'] < 0 else "degrading"
        if abs(trend_stats['slope']) < 0.01:
            trend_direction = "stable"
        
        trend = PerformanceTrend(
            metric_type=metric_type,
            trend_direction=trend_direction,
            trend_strength=trend_stats['trend_strength'],
            time_period=3600,  # 1 hour
            start_time=start_time,
            end_time=end_time,
            description=f"Performance trend for {metric_type.value}: {trend_direction} "
                       f"(slope: {trend_stats['slope']:.4f}, R²: {trend_stats['r_squared']:.3f})"
        )
        
        self.performance_trends.append(trend)
        
        # Keep only recent trends (last 24 hours)
        cutoff_time = datetime.now() - timedelta(hours=24)
        self.performance_trends = [t for t in self.performance_trends if t.end_time >= cutoff_time]
    
    def _create_sla_violation_alert(self, metric_type: PerformanceMetricType, value: float, sla: PerformanceSLA):
        """Create SLA violation alert"""
        alert_id = f"sla_violation_{metric_type.value}_{int(time.time())}"
        
        # Determine severity based on violation magnitude
        violation_ratio = value / sla.target_value
        if violation_ratio >= 3.0:
            severity = AlertSeverity.CRITICAL
        elif violation_ratio >= 2.0:
            severity = AlertSeverity.HIGH
        elif violation_ratio >= 1.5:
            severity = AlertSeverity.MEDIUM
        else:
            severity = AlertSeverity.LOW
        
        metric = MonitoringMetric(
            name=f"performance_{metric_type.value}",
            value=value,
            timestamp=datetime.now(),
            source="performance_monitor",
            tags={
                'metric_type': metric_type.value,
                'sla_target': sla.target_value,
                'violation_ratio': violation_ratio
            }
        )
        
        alert = Alert(
            id=alert_id,
            title=f"SLA Violation: {metric_type.value}",
            description=f"Performance metric '{metric_type.value}' violated SLA: "
                       f"{value:.2f} > {sla.target_value:.2f} (target)",
            severity=severity,
            metric=metric,
            threshold=sla.get_threshold_value(),
            timestamp=datetime.now(),
            tags={
                'type': 'sla_violation',
                'metric_type': metric_type.value,
                'violation_ratio': violation_ratio
            }
        )
        
        self._send_alert(alert)
    
    def _create_degradation_alert(self, metric_type: PerformanceMetricType, value: float, 
                                 baseline_mean: float, degradation_ratio: float):
        """Create performance degradation alert"""
        alert_id = f"degradation_{metric_type.value}_{int(time.time())}"
        
        # Determine severity based on degradation magnitude
        if degradation_ratio >= 1.0:  # 100% degradation
            severity = AlertSeverity.CRITICAL
        elif degradation_ratio >= 0.5:  # 50% degradation
            severity = AlertSeverity.HIGH
        elif degradation_ratio >= 0.3:  # 30% degradation
            severity = AlertSeverity.MEDIUM
        else:
            severity = AlertSeverity.LOW
        
        metric = MonitoringMetric(
            name=f"performance_{metric_type.value}",
            value=value,
            timestamp=datetime.now(),
            source="performance_monitor",
            tags={
                'metric_type': metric_type.value,
                'baseline_mean': baseline_mean,
                'degradation_ratio': degradation_ratio
            }
        )
        
        alert = Alert(
            id=alert_id,
            title=f"Performance Degradation: {metric_type.value}",
            description=f"Performance metric '{metric_type.value}' degraded by {degradation_ratio:.1%}: "
                       f"{value:.2f} vs baseline {baseline_mean:.2f}",
            severity=severity,
            metric=metric,
            threshold=baseline_mean * (1 + self.degradation_threshold),
            timestamp=datetime.now(),
            tags={
                'type': 'performance_degradation',
                'metric_type': metric_type.value,
                'degradation_ratio': degradation_ratio
            }
        )
        
        self._send_alert(alert)
    
    def _create_trend_alert(self, metric_type: PerformanceMetricType, trend_stats: Dict[str, float]):
        """Create performance trend alert"""
        alert_id = f"trend_{metric_type.value}_{int(time.time())}"
        
        trend_direction = "improving" if trend_stats['slope'] < 0 else "degrading"
        
        # Only alert on degrading trends
        if trend_direction != "degrading":
            return
        
        # Determine severity based on trend strength
        trend_strength = trend_stats['trend_strength']
        if trend_strength >= 0.5:
            severity = AlertSeverity.HIGH
        elif trend_strength >= 0.3:
            severity = AlertSeverity.MEDIUM
        else:
            severity = AlertSeverity.LOW
        
        metric = MonitoringMetric(
            name=f"performance_trend_{metric_type.value}",
            value=trend_strength,
            timestamp=datetime.now(),
            source="performance_monitor",
            tags={
                'metric_type': metric_type.value,
                'trend_direction': trend_direction,
                'slope': trend_stats['slope'],
                'r_squared': trend_stats['r_squared']
            }
        )
        
        alert = Alert(
            id=alert_id,
            title=f"Performance Trend Alert: {metric_type.value}",
            description=f"Degrading performance trend detected for '{metric_type.value}': "
                       f"slope={trend_stats['slope']:.4f}, R²={trend_stats['r_squared']:.3f}",
            severity=severity,
            metric=metric,
            threshold=0.1,  # Minimum significant trend
            timestamp=datetime.now(),
            tags={
                'type': 'performance_trend',
                'metric_type': metric_type.value,
                'trend_direction': trend_direction,
                'trend_strength': trend_strength
            }
        )
        
        self._send_alert(alert)
    
    def _handle_performance_alert(self, alert: Alert):
        """Handle performance-specific alerts"""
        self.logger.info(f"Performance alert received: {alert.title}")
        
        # Add performance-specific processing
        alert_type = alert.tags.get('type')
        if alert_type == 'sla_violation':
            self._process_sla_violation(alert)
        elif alert_type == 'performance_degradation':
            self._process_degradation(alert)
        elif alert_type == 'performance_trend':
            self._process_trend_alert(alert)
    
    def _process_sla_violation(self, alert: Alert):
        """Process SLA violation alerts"""
        metric_type = alert.tags.get('metric_type')
        violation_ratio = alert.tags.get('violation_ratio', 1.0)
        
        self.logger.warning(f"SLA violation for {metric_type}: {violation_ratio:.2f}x target")
        
        # Could trigger automated remediation actions
        if violation_ratio >= 2.0:
            self._trigger_remediation_actions(metric_type, alert)
    
    def _process_degradation(self, alert: Alert):
        """Process performance degradation alerts"""
        metric_type = alert.tags.get('metric_type')
        degradation_ratio = alert.tags.get('degradation_ratio', 0.0)
        
        self.logger.warning(f"Performance degradation for {metric_type}: {degradation_ratio:.1%}")
    
    def _process_trend_alert(self, alert: Alert):
        """Process performance trend alerts"""
        metric_type = alert.tags.get('metric_type')
        trend_direction = alert.tags.get('trend_direction')
        
        self.logger.info(f"Performance trend alert for {metric_type}: {trend_direction}")
    
    def _trigger_remediation_actions(self, metric_type: str, alert: Alert):
        """Trigger automated remediation actions"""
        self.logger.info(f"Triggering remediation actions for {metric_type}")
        
        # Example remediation actions
        if metric_type == PerformanceMetricType.QUERY_EXECUTION_TIME.value:
            self._optimize_query_performance(alert)
        elif metric_type == PerformanceMetricType.WAREHOUSE_RESPONSE_TIME.value:
            self._scale_warehouse_resources(alert)
    
    def _optimize_query_performance(self, alert: Alert):
        """Optimize query performance"""
        self.logger.info("Initiating query performance optimization")
        # Could analyze query patterns, suggest optimizations, etc.
    
    def _scale_warehouse_resources(self, alert: Alert):
        """Scale warehouse resources"""
        self.logger.info("Initiating warehouse resource scaling")
        # Could trigger auto-scaling, notifications, etc.
    
    def _send_alert(self, alert: Alert):
        """Send alert through the monitoring system"""
        # Delegate to the base monitor's alert system
        for callback in self.monitor.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                self.logger.error(f"Error in alert callback: {e}")
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance monitoring summary"""
        summary = {
            'current_metrics': {},
            'slas': {},
            'trends': [],
            'baselines': self.performance_baselines,
            'sla_compliance': {}
        }
        
        # Current metrics summary
        metrics_summary = self.monitor.get_metrics_summary()
        for metric_name, metric_data in metrics_summary.items():
            if metric_name.startswith('performance_'):
                summary['current_metrics'][metric_name] = metric_data
        
        # SLA summary
        for metric_type, sla in self.performance_slas.items():
            summary['slas'][metric_type] = sla.to_dict()
        
        # Recent trends
        summary['trends'] = [t.to_dict() for t in self.performance_trends[-10:]]
        
        # SLA compliance calculation
        for metric_type, sla in self.performance_slas.items():
            if sla.enabled:
                recent_values = self._get_recent_values(
                    PerformanceMetricType(metric_type), 
                    hours=sla.measurement_period // 3600
                )
                
                if recent_values:
                    violations = sum(1 for v in recent_values if v > sla.get_threshold_value())
                    compliance_rate = ((len(recent_values) - violations) / len(recent_values)) * 100
                    
                    summary['sla_compliance'][metric_type] = {
                        'compliance_rate': compliance_rate,
                        'violations': violations,
                        'total_measurements': len(recent_values)
                    }
        
        return summary
    
    def get_performance_trend(self, metric_type: PerformanceMetricType, hours: int = 24) -> List[Dict[str, Any]]:
        """Get performance trend for a specific metric"""
        metric_name = f"performance_{metric_type.value}"
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