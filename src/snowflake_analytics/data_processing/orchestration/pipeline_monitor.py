"""
Pipeline Monitor

Comprehensive monitoring system for ML pipeline execution with real-time metrics,
performance tracking, resource monitoring, and alerting capabilities.

Key capabilities:
- Real-time pipeline execution monitoring
- Performance metrics collection and analysis
- Resource utilization tracking
- Error detection and alerting
- Historical trend analysis
- Custom metric dashboards
- Automated health checks
- Integration with external monitoring systems
"""

import os
import json
import time
import psutil
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable, Union
from pathlib import Path
from dataclasses import dataclass, asdict
from collections import deque, defaultdict
import statistics

import pandas as pd
import numpy as np
from snowflake.connector import connect

from ...config.settings import SnowflakeSettings
from ...utils.logger import SnowflakeLogger
from .ml_pipeline_orchestrator import PipelineExecution, PipelineStage, PipelineStatus


@dataclass
class PerformanceMetric:
    """Performance metric data point"""
    timestamp: datetime
    metric_name: str
    value: float
    unit: str
    stage: Optional[PipelineStage]
    execution_id: Optional[str]
    metadata: Dict[str, Any]


@dataclass
class ResourceMetrics:
    """System resource metrics"""
    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    memory_used_gb: float
    memory_available_gb: float
    disk_used_percent: float
    disk_free_gb: float
    network_bytes_sent: int
    network_bytes_recv: int
    active_connections: int


@dataclass
class AlertRule:
    """Alert rule configuration"""
    rule_id: str
    metric_name: str
    condition: str  # 'greater_than', 'less_than', 'equals', 'contains'
    threshold: Union[float, str]
    severity: str  # 'low', 'medium', 'high', 'critical'
    cooldown_minutes: int
    enabled: bool
    description: str


@dataclass
class Alert:
    """Generated alert"""
    alert_id: str
    rule_id: str
    timestamp: datetime
    severity: str
    metric_name: str
    current_value: Union[float, str]
    threshold: Union[float, str]
    message: str
    execution_id: Optional[str]
    stage: Optional[PipelineStage]
    acknowledged: bool
    resolved: bool


class PipelineMonitor:
    """
    Comprehensive monitoring system for ML pipeline execution.
    
    Provides real-time monitoring, performance tracking, resource monitoring,
    and alerting capabilities for the ML data processing pipeline.
    """
    
    def __init__(
        self,
        config_path: Optional[str] = None,
        metrics_retention_hours: int = 24,
        sampling_interval_seconds: int = 10,
        enable_resource_monitoring: bool = True,
        enable_performance_monitoring: bool = True,
        enable_alerting: bool = True
    ):
        """Initialize pipeline monitor"""
        
        # Core configuration
        self.settings = SnowflakeSettings(config_path)
        self.logger = SnowflakeLogger("PipelineMonitor").get_logger()
        
        # Monitoring configuration
        self.metrics_retention_hours = metrics_retention_hours
        self.sampling_interval_seconds = sampling_interval_seconds
        self.enable_resource_monitoring = enable_resource_monitoring
        self.enable_performance_monitoring = enable_performance_monitoring
        self.enable_alerting = enable_alerting
        
        # Data storage
        self.performance_metrics: deque = deque(maxlen=int(
            metrics_retention_hours * 3600 / sampling_interval_seconds
        ))
        self.resource_metrics: deque = deque(maxlen=int(
            metrics_retention_hours * 3600 / sampling_interval_seconds
        ))
        self.alerts: List[Alert] = []
        self.alert_rules: Dict[str, AlertRule] = {}
        
        # Monitoring state
        self.is_monitoring = False
        self.current_execution: Optional[PipelineExecution] = None
        self.monitoring_thread: Optional[threading.Thread] = None
        
        # Metrics aggregation
        self.metric_aggregators = defaultdict(list)
        self.stage_metrics = defaultdict(dict)
        
        # Initialize default alert rules
        self._initialize_default_alert_rules()
        
        self.logger.info("PipelineMonitor initialized successfully")
    
    def start_monitoring(self, execution: Optional[PipelineExecution] = None):
        """Start monitoring pipeline execution"""
        
        if self.is_monitoring:
            self.logger.warning("Monitoring already active")
            return
        
        self.current_execution = execution
        self.is_monitoring = True
        
        # Start monitoring thread
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            daemon=True
        )
        self.monitoring_thread.start()
        
        self.logger.info("Pipeline monitoring started")
    
    def stop_monitoring(self):
        """Stop monitoring pipeline execution"""
        
        self.is_monitoring = False
        
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        
        self.logger.info("Pipeline monitoring stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop"""
        
        while self.is_monitoring:
            try:
                # Collect metrics
                if self.enable_resource_monitoring:
                    self._collect_resource_metrics()
                
                if self.enable_performance_monitoring and self.current_execution:
                    self._collect_performance_metrics()
                
                # Check alert rules
                if self.enable_alerting:
                    self._check_alert_rules()
                
                # Clean up old data
                self._cleanup_old_data()
                
                # Wait for next sampling interval
                time.sleep(self.sampling_interval_seconds)
                
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {str(e)}")
                time.sleep(self.sampling_interval_seconds)
    
    def _collect_resource_metrics(self):
        """Collect system resource metrics"""
        
        try:
            # CPU metrics
            cpu_percent = psutil.cpu_percent()
            
            # Memory metrics
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            memory_used_gb = memory.used / (1024**3)
            memory_available_gb = memory.available / (1024**3)
            
            # Disk metrics
            disk = psutil.disk_usage('/')
            disk_used_percent = (disk.used / disk.total) * 100
            disk_free_gb = disk.free / (1024**3)
            
            # Network metrics
            network = psutil.net_io_counters()
            network_bytes_sent = network.bytes_sent
            network_bytes_recv = network.bytes_recv
            
            # Connection count
            try:
                active_connections = len(psutil.net_connections())
            except:
                active_connections = 0
            
            # Create resource metrics
            resource_metric = ResourceMetrics(
                timestamp=datetime.now(),
                cpu_percent=cpu_percent,
                memory_percent=memory_percent,
                memory_used_gb=memory_used_gb,
                memory_available_gb=memory_available_gb,
                disk_used_percent=disk_used_percent,
                disk_free_gb=disk_free_gb,
                network_bytes_sent=network_bytes_sent,
                network_bytes_recv=network_bytes_recv,
                active_connections=active_connections
            )
            
            self.resource_metrics.append(resource_metric)
            
        except Exception as e:
            self.logger.error(f"Failed to collect resource metrics: {str(e)}")
    
    def _collect_performance_metrics(self):
        """Collect pipeline performance metrics"""
        
        if not self.current_execution:
            return
        
        try:
            # Execution duration
            if self.current_execution.start_time:
                current_duration = (datetime.now() - self.current_execution.start_time).total_seconds()
                self._add_performance_metric(
                    "execution_duration_seconds",
                    current_duration,
                    "seconds",
                    metadata={"execution_id": self.current_execution.execution_id}
                )
            
            # Records processed
            self._add_performance_metric(
                "total_records_processed",
                self.current_execution.total_records_processed,
                "records",
                metadata={"execution_id": self.current_execution.execution_id}
            )
            
            # Stage-specific metrics
            for stage, result in self.current_execution.stage_results.items():
                if result.duration_seconds:
                    self._add_performance_metric(
                        "stage_duration_seconds",
                        result.duration_seconds,
                        "seconds",
                        stage=stage,
                        metadata={
                            "execution_id": self.current_execution.execution_id,
                            "stage": stage.value
                        }
                    )
                
                self._add_performance_metric(
                    "stage_records_processed",
                    result.records_processed,
                    "records",
                    stage=stage,
                    metadata={
                        "execution_id": self.current_execution.execution_id,
                        "stage": stage.value
                    }
                )
            
            # Processing rate
            if self.current_execution.total_duration_seconds and self.current_execution.total_duration_seconds > 0:
                processing_rate = (
                    self.current_execution.total_records_processed / 
                    self.current_execution.total_duration_seconds
                )
                self._add_performance_metric(
                    "processing_rate_records_per_second",
                    processing_rate,
                    "records/second",
                    metadata={"execution_id": self.current_execution.execution_id}
                )
            
        except Exception as e:
            self.logger.error(f"Failed to collect performance metrics: {str(e)}")
    
    def _add_performance_metric(
        self,
        metric_name: str,
        value: float,
        unit: str,
        stage: Optional[PipelineStage] = None,
        execution_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Add performance metric"""
        
        metric = PerformanceMetric(
            timestamp=datetime.now(),
            metric_name=metric_name,
            value=value,
            unit=unit,
            stage=stage,
            execution_id=execution_id or (self.current_execution.execution_id if self.current_execution else None),
            metadata=metadata or {}
        )
        
        self.performance_metrics.append(metric)
        
        # Update aggregators
        self.metric_aggregators[metric_name].append(value)
        
        # Keep only recent values for aggregation
        if len(self.metric_aggregators[metric_name]) > 100:
            self.metric_aggregators[metric_name] = self.metric_aggregators[metric_name][-100:]
    
    def _initialize_default_alert_rules(self):
        """Initialize default alert rules"""
        
        default_rules = [
            AlertRule(
                rule_id="high_cpu_usage",
                metric_name="cpu_percent",
                condition="greater_than",
                threshold=80.0,
                severity="high",
                cooldown_minutes=5,
                enabled=True,
                description="CPU usage is above 80%"
            ),
            AlertRule(
                rule_id="high_memory_usage",
                metric_name="memory_percent",
                condition="greater_than",
                threshold=85.0,
                severity="high",
                cooldown_minutes=5,
                enabled=True,
                description="Memory usage is above 85%"
            ),
            AlertRule(
                rule_id="low_disk_space",
                metric_name="disk_free_gb",
                condition="less_than",
                threshold=5.0,
                severity="critical",
                cooldown_minutes=10,
                enabled=True,
                description="Disk space is below 5GB"
            ),
            AlertRule(
                rule_id="slow_processing_rate",
                metric_name="processing_rate_records_per_second",
                condition="less_than",
                threshold=100.0,
                severity="medium",
                cooldown_minutes=15,
                enabled=True,
                description="Processing rate is below 100 records/second"
            ),
            AlertRule(
                rule_id="long_execution_time",
                metric_name="execution_duration_seconds",
                condition="greater_than",
                threshold=3600.0,  # 1 hour
                severity="medium",
                cooldown_minutes=30,
                enabled=True,
                description="Execution time exceeds 1 hour"
            )
        ]
        
        for rule in default_rules:
            self.alert_rules[rule.rule_id] = rule
    
    def add_alert_rule(self, rule: AlertRule):
        """Add custom alert rule"""
        
        self.alert_rules[rule.rule_id] = rule
        self.logger.info(f"Added alert rule: {rule.rule_id}")
    
    def remove_alert_rule(self, rule_id: str):
        """Remove alert rule"""
        
        if rule_id in self.alert_rules:
            del self.alert_rules[rule_id]
            self.logger.info(f"Removed alert rule: {rule_id}")
    
    def _check_alert_rules(self):
        """Check all alert rules against current metrics"""
        
        current_time = datetime.now()
        
        for rule in self.alert_rules.values():
            if not rule.enabled:
                continue
            
            try:
                # Check cooldown
                if self._is_rule_in_cooldown(rule, current_time):
                    continue
                
                # Get current metric value
                current_value = self._get_current_metric_value(rule.metric_name)
                if current_value is None:
                    continue
                
                # Check condition
                if self._evaluate_alert_condition(rule, current_value):
                    self._trigger_alert(rule, current_value)
                
            except Exception as e:
                self.logger.error(f"Error checking alert rule {rule.rule_id}: {str(e)}")
    
    def _get_current_metric_value(self, metric_name: str) -> Optional[Union[float, str]]:
        """Get current value for specified metric"""
        
        # Check resource metrics
        if self.resource_metrics and hasattr(self.resource_metrics[-1], metric_name):
            return getattr(self.resource_metrics[-1], metric_name)
        
        # Check performance metrics
        for metric in reversed(self.performance_metrics):
            if metric.metric_name == metric_name:
                return metric.value
        
        return None
    
    def _evaluate_alert_condition(self, rule: AlertRule, current_value: Union[float, str]) -> bool:
        """Evaluate alert condition"""
        
        try:
            if rule.condition == "greater_than":
                return float(current_value) > float(rule.threshold)
            elif rule.condition == "less_than":
                return float(current_value) < float(rule.threshold)
            elif rule.condition == "equals":
                return current_value == rule.threshold
            elif rule.condition == "contains":
                return str(rule.threshold) in str(current_value)
            else:
                self.logger.warning(f"Unknown alert condition: {rule.condition}")
                return False
        except (ValueError, TypeError):
            return False
    
    def _is_rule_in_cooldown(self, rule: AlertRule, current_time: datetime) -> bool:
        """Check if rule is in cooldown period"""
        
        # Find most recent alert for this rule
        for alert in reversed(self.alerts):
            if alert.rule_id == rule.rule_id:
                time_since_alert = current_time - alert.timestamp
                if time_since_alert < timedelta(minutes=rule.cooldown_minutes):
                    return True
                break
        
        return False
    
    def _trigger_alert(self, rule: AlertRule, current_value: Union[float, str]):
        """Trigger alert for rule"""
        
        alert_id = f"alert_{rule.rule_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        alert = Alert(
            alert_id=alert_id,
            rule_id=rule.rule_id,
            timestamp=datetime.now(),
            severity=rule.severity,
            metric_name=rule.metric_name,
            current_value=current_value,
            threshold=rule.threshold,
            message=f"{rule.description} (Current: {current_value}, Threshold: {rule.threshold})",
            execution_id=self.current_execution.execution_id if self.current_execution else None,
            stage=self._get_current_stage(),
            acknowledged=False,
            resolved=False
        )
        
        self.alerts.append(alert)
        
        # Log alert
        self.logger.warning(f"ALERT [{alert.severity.upper()}]: {alert.message}")
        
        # Send notifications
        self._send_alert_notification(alert)
    
    def _get_current_stage(self) -> Optional[PipelineStage]:
        """Get current pipeline stage"""
        
        if self.current_execution:
            return self.current_execution.current_stage
        
        return None
    
    def _send_alert_notification(self, alert: Alert):
        """Send alert notification"""
        
        # For now, just log the alert
        # In production, this could send emails, Slack messages, etc.
        self.logger.info(f"Alert notification: {alert.alert_id}")
    
    def _cleanup_old_data(self):
        """Clean up old metrics and alerts"""
        
        cutoff_time = datetime.now() - timedelta(hours=self.metrics_retention_hours)
        
        # Clean up performance metrics (deque handles this automatically)
        
        # Clean up old alerts (keep last 1000)
        if len(self.alerts) > 1000:
            self.alerts = self.alerts[-1000:]
    
    def get_current_metrics(self) -> Dict[str, Any]:
        """Get current metrics summary"""
        
        current_metrics = {}
        
        # Latest resource metrics
        if self.resource_metrics:
            latest_resource = self.resource_metrics[-1]
            current_metrics['resources'] = asdict(latest_resource)
        
        # Performance metrics summary
        if self.performance_metrics:
            current_metrics['performance'] = {}
            
            # Group by metric name
            metric_groups = defaultdict(list)
            for metric in self.performance_metrics:
                metric_groups[metric.metric_name].append(metric.value)
            
            for metric_name, values in metric_groups.items():
                if values:
                    current_metrics['performance'][metric_name] = {
                        'current': values[-1],
                        'average': statistics.mean(values),
                        'min': min(values),
                        'max': max(values),
                        'count': len(values)
                    }
        
        # Alert summary
        active_alerts = [alert for alert in self.alerts if not alert.resolved]
        current_metrics['alerts'] = {
            'total_alerts': len(self.alerts),
            'active_alerts': len(active_alerts),
            'critical_alerts': len([a for a in active_alerts if a.severity == 'critical']),
            'high_alerts': len([a for a in active_alerts if a.severity == 'high']),
            'medium_alerts': len([a for a in active_alerts if a.severity == 'medium']),
            'low_alerts': len([a for a in active_alerts if a.severity == 'low'])
        }
        
        return current_metrics
    
    def get_metrics_history(
        self,
        metric_names: Optional[List[str]] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Get historical metrics data"""
        
        if start_time is None:
            start_time = datetime.now() - timedelta(hours=1)
        if end_time is None:
            end_time = datetime.now()
        
        history = defaultdict(list)
        
        # Resource metrics
        for resource_metric in self.resource_metrics:
            if start_time <= resource_metric.timestamp <= end_time:
                resource_data = asdict(resource_metric)
                resource_data['timestamp'] = resource_metric.timestamp.isoformat()
                
                if metric_names is None or any(name in resource_data for name in metric_names):
                    history['resource_metrics'].append(resource_data)
        
        # Performance metrics
        for perf_metric in self.performance_metrics:
            if start_time <= perf_metric.timestamp <= end_time:
                if metric_names is None or perf_metric.metric_name in metric_names:
                    metric_data = asdict(perf_metric)
                    metric_data['timestamp'] = perf_metric.timestamp.isoformat()
                    metric_data['stage'] = perf_metric.stage.value if perf_metric.stage else None
                    
                    history[perf_metric.metric_name].append(metric_data)
        
        return dict(history)
    
    def get_alerts(
        self,
        severity: Optional[str] = None,
        resolved: Optional[bool] = None,
        limit: Optional[int] = None
    ) -> List[Alert]:
        """Get alerts with optional filtering"""
        
        filtered_alerts = self.alerts
        
        if severity:
            filtered_alerts = [a for a in filtered_alerts if a.severity == severity]
        
        if resolved is not None:
            filtered_alerts = [a for a in filtered_alerts if a.resolved == resolved]
        
        if limit:
            filtered_alerts = filtered_alerts[-limit:]
        
        return filtered_alerts
    
    def acknowledge_alert(self, alert_id: str) -> bool:
        """Acknowledge an alert"""
        
        for alert in self.alerts:
            if alert.alert_id == alert_id:
                alert.acknowledged = True
                self.logger.info(f"Alert acknowledged: {alert_id}")
                return True
        
        return False
    
    def resolve_alert(self, alert_id: str) -> bool:
        """Resolve an alert"""
        
        for alert in self.alerts:
            if alert.alert_id == alert_id:
                alert.resolved = True
                alert.acknowledged = True
                self.logger.info(f"Alert resolved: {alert_id}")
                return True
        
        return False
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary across all executions"""
        
        if not self.performance_metrics:
            return {}
        
        # Group metrics by execution
        execution_metrics = defaultdict(lambda: defaultdict(list))
        
        for metric in self.performance_metrics:
            if metric.execution_id:
                execution_metrics[metric.execution_id][metric.metric_name].append(metric.value)
        
        # Calculate summary statistics
        summary = {
            'total_executions': len(execution_metrics),
            'metrics_collected': len(self.performance_metrics),
            'average_execution_time': 0,
            'average_processing_rate': 0,
            'total_records_processed': 0
        }
        
        execution_times = []
        processing_rates = []
        total_records = []
        
        for exec_id, metrics in execution_metrics.items():
            if 'execution_duration_seconds' in metrics:
                execution_times.extend(metrics['execution_duration_seconds'])
            
            if 'processing_rate_records_per_second' in metrics:
                processing_rates.extend(metrics['processing_rate_records_per_second'])
            
            if 'total_records_processed' in metrics:
                total_records.extend(metrics['total_records_processed'])
        
        if execution_times:
            summary['average_execution_time'] = statistics.mean(execution_times)
        
        if processing_rates:
            summary['average_processing_rate'] = statistics.mean(processing_rates)
        
        if total_records:
            summary['total_records_processed'] = sum(total_records)
        
        return summary
    
    def export_metrics(self, output_path: str, format: str = 'json'):
        """Export metrics to file"""
        
        metrics_data = {
            'export_timestamp': datetime.now().isoformat(),
            'resource_metrics': [asdict(m) for m in self.resource_metrics],
            'performance_metrics': [asdict(m) for m in self.performance_metrics],
            'alerts': [asdict(a) for a in self.alerts],
            'summary': self.get_performance_summary()
        }
        
        # Convert datetime objects to strings
        def convert_datetime(obj):
            if isinstance(obj, datetime):
                return obj.isoformat()
            return obj
        
        output_path = Path(output_path)
        
        if format.lower() == 'json':
            with open(output_path, 'w') as f:
                json.dump(metrics_data, f, indent=2, default=convert_datetime)
        elif format.lower() == 'csv':
            # Export as separate CSV files
            base_path = output_path.parent / output_path.stem
            
            # Resource metrics
            if self.resource_metrics:
                resource_df = pd.DataFrame([asdict(m) for m in self.resource_metrics])
                resource_df.to_csv(f"{base_path}_resource_metrics.csv", index=False)
            
            # Performance metrics
            if self.performance_metrics:
                perf_df = pd.DataFrame([asdict(m) for m in self.performance_metrics])
                perf_df.to_csv(f"{base_path}_performance_metrics.csv", index=False)
            
            # Alerts
            if self.alerts:
                alerts_df = pd.DataFrame([asdict(a) for a in self.alerts])
                alerts_df.to_csv(f"{base_path}_alerts.csv", index=False)
        
        self.logger.info(f"Metrics exported to: {output_path}")
