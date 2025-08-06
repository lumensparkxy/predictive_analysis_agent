"""
Cache performance monitor for tracking and analyzing cache metrics.
"""

import time
import threading
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timedelta
from collections import deque, defaultdict
from enum import Enum


class CacheAlert(Enum):
    """Cache performance alerts."""
    LOW_HIT_RATE = "low_hit_rate"
    HIGH_EVICTION_RATE = "high_eviction_rate"
    MEMORY_PRESSURE = "memory_pressure"
    SLOW_OPERATIONS = "slow_operations"
    ANOMALY_DETECTED = "anomaly_detected"


@dataclass
class CacheMetrics:
    """Cache performance metrics snapshot."""
    timestamp: datetime
    cache_name: str
    hit_rate: float
    miss_rate: float
    eviction_rate: float
    memory_usage_mb: float
    memory_utilization_percent: float
    avg_get_time_ms: float
    avg_put_time_ms: float
    total_keys: int
    operations_per_second: float


@dataclass
class CacheAlertEvent:
    """Cache alert event."""
    alert_type: CacheAlert
    cache_name: str
    severity: str  # 'low', 'medium', 'high', 'critical'
    message: str
    current_value: float
    threshold_value: float
    timestamp: datetime
    recommendations: List[str]


class CacheMonitor:
    """
    Cache performance monitor that tracks metrics, detects anomalies,
    and provides intelligent alerts and recommendations.
    """
    
    def __init__(self, max_history_size: int = 10000):
        """Initialize cache monitor."""
        self.max_history_size = max_history_size
        self.metrics_history: deque = deque(maxlen=max_history_size)
        self.alerts_history: List[CacheAlertEvent] = []
        self.registered_caches: Dict[str, Any] = {}
        self.alert_callbacks: List[Callable] = []
        
        self._lock = threading.Lock()
        self._monitoring_active = False
        self._monitor_thread: Optional[threading.Thread] = None
        
        # Performance thresholds
        self.thresholds = {
            'min_hit_rate': 0.6,         # 60% minimum hit rate
            'max_eviction_rate': 0.1,     # 10% max eviction rate
            'max_memory_utilization': 0.85, # 85% max memory usage
            'max_avg_operation_ms': 5.0,   # 5ms max avg operation time
            'min_ops_per_second': 10.0     # 10 ops/sec minimum
        }
        
        # Anomaly detection
        self.baseline_metrics: Dict[str, Dict[str, float]] = {}
        self.anomaly_threshold_multiplier = 2.0
    
    def register_cache(self, cache_name: str, cache_instance: Any):
        """Register cache for monitoring."""
        self.registered_caches[cache_name] = cache_instance
    
    def start_monitoring(self, interval_seconds: float = 30.0) -> bool:
        """Start continuous cache monitoring."""
        if self._monitoring_active:
            return False
        
        self._monitoring_active = True
        self._monitor_thread = threading.Thread(
            target=self._monitoring_loop,
            args=(interval_seconds,),
            daemon=True
        )
        self._monitor_thread.start()
        return True
    
    def stop_monitoring(self):
        """Stop continuous monitoring."""
        self._monitoring_active = False
        if self._monitor_thread and self._monitor_thread.is_alive():
            self._monitor_thread.join(timeout=5)
    
    def collect_metrics(self) -> List[CacheMetrics]:
        """Collect current metrics from all registered caches."""
        metrics_list = []
        current_time = datetime.now()
        
        for cache_name, cache_instance in self.registered_caches.items():
            try:
                metrics = self._extract_cache_metrics(cache_name, cache_instance, current_time)
                if metrics:
                    metrics_list.append(metrics)
            except Exception as e:
                print(f"Error collecting metrics for cache {cache_name}: {e}")
        
        # Store in history
        with self._lock:
            self.metrics_history.extend(metrics_list)
        
        return metrics_list
    
    def _extract_cache_metrics(self, 
                              cache_name: str, 
                              cache_instance: Any, 
                              timestamp: datetime) -> Optional[CacheMetrics]:
        """Extract metrics from cache instance."""
        try:
            # Try to get stats from cache instance
            stats = None
            if hasattr(cache_instance, 'get_stats'):
                stats = cache_instance.get_stats()
            elif hasattr(cache_instance, 'stats'):
                stats = cache_instance.stats
            
            if not stats:
                return None
            
            # Calculate rates and derived metrics
            total_requests = stats.get('hits', 0) + stats.get('misses', 0)
            hit_rate = (stats.get('hits', 0) / max(total_requests, 1))
            miss_rate = 1 - hit_rate
            
            # Eviction rate (simplified)
            eviction_rate = stats.get('evictions', 0) / max(total_requests, 1)
            
            # Memory metrics
            memory_usage_mb = stats.get('size_mb', 0)
            memory_utilization = stats.get('utilization_percent', 0) / 100
            
            # Operation performance (mock values if not available)
            avg_get_time_ms = stats.get('avg_get_time_ms', 1.0)
            avg_put_time_ms = stats.get('avg_put_time_ms', 1.0)
            
            # Operations per second (estimate)
            ops_per_second = total_requests / max(self._get_cache_uptime_seconds(cache_name), 1)
            
            return CacheMetrics(
                timestamp=timestamp,
                cache_name=cache_name,
                hit_rate=hit_rate,
                miss_rate=miss_rate,
                eviction_rate=eviction_rate,
                memory_usage_mb=memory_usage_mb,
                memory_utilization_percent=memory_utilization * 100,
                avg_get_time_ms=avg_get_time_ms,
                avg_put_time_ms=avg_put_time_ms,
                total_keys=stats.get('entries', 0),
                operations_per_second=ops_per_second
            )
            
        except Exception as e:
            print(f"Error extracting metrics from {cache_name}: {e}")
            return None
    
    def _get_cache_uptime_seconds(self, cache_name: str) -> float:
        """Get cache uptime in seconds (simplified)."""
        # In a real implementation, this would track cache start time
        return 3600.0  # Assume 1 hour uptime
    
    def _monitoring_loop(self, interval_seconds: float):
        """Main monitoring loop."""
        while self._monitoring_active:
            try:
                # Collect metrics
                metrics_list = self.collect_metrics()
                
                # Analyze for alerts
                for metrics in metrics_list:
                    alerts = self._analyze_metrics_for_alerts(metrics)
                    for alert in alerts:
                        self._trigger_alert(alert)
                
                # Update baselines
                self._update_baselines(metrics_list)
                
                time.sleep(interval_seconds)
            except Exception as e:
                print(f"Error in monitoring loop: {e}")
                time.sleep(interval_seconds)
    
    def _analyze_metrics_for_alerts(self, metrics: CacheMetrics) -> List[CacheAlertEvent]:
        """Analyze metrics and generate alerts."""
        alerts = []
        
        # Low hit rate alert
        if metrics.hit_rate < self.thresholds['min_hit_rate']:
            alerts.append(CacheAlertEvent(
                alert_type=CacheAlert.LOW_HIT_RATE,
                cache_name=metrics.cache_name,
                severity=self._calculate_severity(metrics.hit_rate, self.thresholds['min_hit_rate'], 'below'),
                message=f"Cache hit rate {metrics.hit_rate:.2%} below threshold {self.thresholds['min_hit_rate']:.2%}",
                current_value=metrics.hit_rate,
                threshold_value=self.thresholds['min_hit_rate'],
                timestamp=metrics.timestamp,
                recommendations=[
                    "Review cache size configuration",
                    "Analyze access patterns for optimization",
                    "Consider adjusting TTL settings"
                ]
            ))
        
        # High eviction rate alert
        if metrics.eviction_rate > self.thresholds['max_eviction_rate']:
            alerts.append(CacheAlertEvent(
                alert_type=CacheAlert.HIGH_EVICTION_RATE,
                cache_name=metrics.cache_name,
                severity=self._calculate_severity(metrics.eviction_rate, self.thresholds['max_eviction_rate'], 'above'),
                message=f"Eviction rate {metrics.eviction_rate:.2%} above threshold",
                current_value=metrics.eviction_rate,
                threshold_value=self.thresholds['max_eviction_rate'],
                timestamp=metrics.timestamp,
                recommendations=[
                    "Increase cache memory allocation",
                    "Review eviction policy effectiveness",
                    "Optimize data structures for smaller footprint"
                ]
            ))
        
        # Memory pressure alert
        if metrics.memory_utilization_percent > self.thresholds['max_memory_utilization'] * 100:
            alerts.append(CacheAlertEvent(
                alert_type=CacheAlert.MEMORY_PRESSURE,
                cache_name=metrics.cache_name,
                severity=self._calculate_severity(
                    metrics.memory_utilization_percent/100, 
                    self.thresholds['max_memory_utilization'], 
                    'above'
                ),
                message=f"Memory utilization {metrics.memory_utilization_percent:.1f}% approaching limit",
                current_value=metrics.memory_utilization_percent,
                threshold_value=self.thresholds['max_memory_utilization'] * 100,
                timestamp=metrics.timestamp,
                recommendations=[
                    "Increase memory allocation",
                    "Implement more aggressive eviction",
                    "Enable compression if supported"
                ]
            ))
        
        # Slow operations alert
        avg_op_time = (metrics.avg_get_time_ms + metrics.avg_put_time_ms) / 2
        if avg_op_time > self.thresholds['max_avg_operation_ms']:
            alerts.append(CacheAlertEvent(
                alert_type=CacheAlert.SLOW_OPERATIONS,
                cache_name=metrics.cache_name,
                severity=self._calculate_severity(avg_op_time, self.thresholds['max_avg_operation_ms'], 'above'),
                message=f"Average operation time {avg_op_time:.2f}ms exceeds threshold",
                current_value=avg_op_time,
                threshold_value=self.thresholds['max_avg_operation_ms'],
                timestamp=metrics.timestamp,
                recommendations=[
                    "Investigate cache backend performance",
                    "Review serialization overhead",
                    "Check for resource contention"
                ]
            ))
        
        # Anomaly detection
        anomaly_alerts = self._detect_anomalies(metrics)
        alerts.extend(anomaly_alerts)
        
        return alerts
    
    def _calculate_severity(self, current_value: float, threshold: float, direction: str) -> str:
        """Calculate alert severity based on how far from threshold."""
        if direction == 'above':
            ratio = current_value / threshold
        else:  # below
            ratio = threshold / current_value
        
        if ratio >= 2.0:
            return 'critical'
        elif ratio >= 1.5:
            return 'high'
        elif ratio >= 1.2:
            return 'medium'
        else:
            return 'low'
    
    def _detect_anomalies(self, metrics: CacheMetrics) -> List[CacheAlertEvent]:
        """Detect performance anomalies using baseline comparison."""
        alerts = []
        cache_name = metrics.cache_name
        
        if cache_name not in self.baseline_metrics:
            return alerts
        
        baseline = self.baseline_metrics[cache_name]
        
        # Check for significant deviations
        metrics_to_check = {
            'hit_rate': metrics.hit_rate,
            'eviction_rate': metrics.eviction_rate,
            'avg_get_time_ms': metrics.avg_get_time_ms,
            'operations_per_second': metrics.operations_per_second
        }
        
        for metric_name, current_value in metrics_to_check.items():
            baseline_value = baseline.get(metric_name, current_value)
            
            if baseline_value > 0:
                deviation_ratio = abs(current_value - baseline_value) / baseline_value
                
                if deviation_ratio > self.anomaly_threshold_multiplier:
                    alerts.append(CacheAlertEvent(
                        alert_type=CacheAlert.ANOMALY_DETECTED,
                        cache_name=cache_name,
                        severity='medium',
                        message=f"Anomaly in {metric_name}: {current_value:.3f} vs baseline {baseline_value:.3f}",
                        current_value=current_value,
                        threshold_value=baseline_value,
                        timestamp=metrics.timestamp,
                        recommendations=[
                            f"Investigate cause of {metric_name} deviation",
                            "Check for changes in access patterns",
                            "Review recent configuration changes"
                        ]
                    ))
        
        return alerts
    
    def _update_baselines(self, metrics_list: List[CacheMetrics]):
        """Update baseline metrics for anomaly detection."""
        # Update baselines with rolling average (simplified)
        for metrics in metrics_list:
            cache_name = metrics.cache_name
            
            if cache_name not in self.baseline_metrics:
                self.baseline_metrics[cache_name] = {}
            
            baseline = self.baseline_metrics[cache_name]
            alpha = 0.1  # Exponential smoothing factor
            
            metrics_to_baseline = {
                'hit_rate': metrics.hit_rate,
                'eviction_rate': metrics.eviction_rate,
                'avg_get_time_ms': metrics.avg_get_time_ms,
                'operations_per_second': metrics.operations_per_second
            }
            
            for metric_name, current_value in metrics_to_baseline.items():
                if metric_name in baseline:
                    # Exponential smoothing
                    baseline[metric_name] = (1 - alpha) * baseline[metric_name] + alpha * current_value
                else:
                    baseline[metric_name] = current_value
    
    def _trigger_alert(self, alert: CacheAlertEvent):
        """Trigger alert and notify callbacks."""
        with self._lock:
            self.alerts_history.append(alert)
            
            # Keep only recent alerts (last 1000)
            if len(self.alerts_history) > 1000:
                self.alerts_history = self.alerts_history[-1000:]
        
        # Notify callbacks
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                print(f"Error in alert callback: {e}")
    
    def add_alert_callback(self, callback: Callable[[CacheAlertEvent], None]):
        """Add callback function for alert notifications."""
        self.alert_callbacks.append(callback)
    
    def get_performance_dashboard(self) -> Dict[str, Any]:
        """Get performance dashboard data."""
        with self._lock:
            recent_metrics = list(self.metrics_history)[-100:]  # Last 100 data points
            recent_alerts = [a for a in self.alerts_history 
                           if a.timestamp >= datetime.now() - timedelta(hours=24)]
        
        if not recent_metrics:
            return {'status': 'no_data'}
        
        # Aggregate metrics by cache
        cache_summaries = {}
        for metrics in recent_metrics:
            cache_name = metrics.cache_name
            if cache_name not in cache_summaries:
                cache_summaries[cache_name] = {
                    'hit_rates': [],
                    'memory_usage': [],
                    'operations_per_second': [],
                    'latest_metrics': None
                }
            
            summary = cache_summaries[cache_name]
            summary['hit_rates'].append(metrics.hit_rate)
            summary['memory_usage'].append(metrics.memory_usage_mb)
            summary['operations_per_second'].append(metrics.operations_per_second)
            summary['latest_metrics'] = metrics
        
        # Calculate averages
        for cache_name, summary in cache_summaries.items():
            summary['avg_hit_rate'] = sum(summary['hit_rates']) / len(summary['hit_rates'])
            summary['avg_memory_usage'] = sum(summary['memory_usage']) / len(summary['memory_usage'])
            summary['avg_ops_per_second'] = sum(summary['operations_per_second']) / len(summary['operations_per_second'])
        
        # Alert summary
        alert_summary = defaultdict(int)
        for alert in recent_alerts:
            alert_summary[f"{alert.alert_type.value}_{alert.severity}"] += 1
        
        return {
            'timestamp': datetime.now().isoformat(),
            'cache_summaries': cache_summaries,
            'alert_summary': dict(alert_summary),
            'total_recent_alerts': len(recent_alerts),
            'monitoring_active': self._monitoring_active,
            'registered_caches': len(self.registered_caches)
        }
    
    def analyze_performance_trends(self, hours: int = 24) -> Dict[str, Any]:
        """Analyze performance trends over specified time period."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        with self._lock:
            relevant_metrics = [
                m for m in self.metrics_history
                if m.timestamp >= cutoff_time
            ]
        
        if len(relevant_metrics) < 2:
            return {'trend': 'insufficient_data'}
        
        # Group by cache and analyze trends
        cache_trends = {}
        
        for cache_name in set(m.cache_name for m in relevant_metrics):
            cache_metrics = [m for m in relevant_metrics if m.cache_name == cache_name]
            cache_metrics.sort(key=lambda x: x.timestamp)
            
            if len(cache_metrics) < 2:
                continue
            
            # Calculate trends
            hit_rates = [m.hit_rate for m in cache_metrics]
            memory_usage = [m.memory_usage_mb for m in cache_metrics]
            
            hit_rate_trend = self._calculate_trend(hit_rates)
            memory_trend = self._calculate_trend(memory_usage)
            
            cache_trends[cache_name] = {
                'hit_rate_trend': hit_rate_trend,
                'memory_usage_trend': memory_trend,
                'sample_count': len(cache_metrics),
                'time_span_hours': hours
            }
        
        return {
            'analysis_period_hours': hours,
            'cache_trends': cache_trends,
            'overall_health': self._assess_overall_health(cache_trends)
        }
    
    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate trend direction from list of values."""
        if len(values) < 2:
            return 'stable'
        
        # Simple linear trend calculation
        start_avg = sum(values[:3]) / min(3, len(values))
        end_avg = sum(values[-3:]) / min(3, len(values))
        
        change_percent = ((end_avg - start_avg) / start_avg * 100) if start_avg > 0 else 0
        
        if change_percent > 5:
            return 'improving'
        elif change_percent < -5:
            return 'degrading'
        else:
            return 'stable'
    
    def _assess_overall_health(self, cache_trends: Dict) -> str:
        """Assess overall cache system health."""
        if not cache_trends:
            return 'unknown'
        
        degrading_count = sum(1 for trend in cache_trends.values() 
                             if trend['hit_rate_trend'] == 'degrading')
        improving_count = sum(1 for trend in cache_trends.values() 
                             if trend['hit_rate_trend'] == 'improving')
        
        total_caches = len(cache_trends)
        
        if degrading_count > total_caches * 0.5:
            return 'poor'
        elif improving_count > total_caches * 0.5:
            return 'excellent'
        else:
            return 'good'
    
    def export_monitoring_report(self, filepath: str) -> bool:
        """Export comprehensive monitoring report."""
        try:
            report = {
                'generated_at': datetime.now().isoformat(),
                'performance_dashboard': self.get_performance_dashboard(),
                'performance_trends': self.analyze_performance_trends(),
                'recent_alerts': [
                    {
                        'alert_type': alert.alert_type.value,
                        'cache_name': alert.cache_name,
                        'severity': alert.severity,
                        'message': alert.message,
                        'timestamp': alert.timestamp.isoformat(),
                        'recommendations': alert.recommendations
                    }
                    for alert in self.alerts_history[-50:]  # Last 50 alerts
                ],
                'configuration': {
                    'thresholds': self.thresholds,
                    'monitoring_active': self._monitoring_active,
                    'registered_caches': list(self.registered_caches.keys())
                }
            }
            
            import json
            with open(filepath, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            return True
        except Exception as e:
            print(f"Error exporting monitoring report: {e}")
            return False