"""
Status Monitor - Monitors collection operations and provides health status.

This module provides comprehensive monitoring and alerting for data collection
operations, including health checks, performance metrics, and status reporting.
"""

import threading
import time
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum

from ..utils.logger import get_logger
from ..storage.sqlite_store import SQLiteStore

logger = get_logger(__name__)


class HealthStatus(Enum):
    """Overall health status levels."""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


class ComponentStatus(Enum):
    """Individual component status."""
    OPERATIONAL = "operational"
    DEGRADED = "degraded"
    FAILED = "failed"
    UNKNOWN = "unknown"


@dataclass
class HealthCheck:
    """Health check definition."""
    name: str
    component: str
    check_function: Callable[[], Dict[str, Any]]
    interval_seconds: int = 300  # 5 minutes
    timeout_seconds: int = 30
    failure_threshold: int = 3  # consecutive failures before marking as failed
    warning_threshold: int = 2  # consecutive warnings before escalating
    enabled: bool = True
    
    # State tracking
    last_check_time: Optional[datetime] = None
    last_status: ComponentStatus = ComponentStatus.UNKNOWN
    consecutive_failures: int = 0
    consecutive_warnings: int = 0
    last_result: Optional[Dict[str, Any]] = None


@dataclass
class Alert:
    """Alert definition."""
    id: str
    severity: str  # info, warning, error, critical
    component: str
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    acknowledged_at: Optional[datetime] = None
    resolved_at: Optional[datetime] = None
    is_active: bool = True


@dataclass
class PerformanceMetric:
    """Performance metric tracking."""
    name: str
    component: str
    value: float
    unit: str
    timestamp: datetime = field(default_factory=datetime.now)
    tags: Dict[str, str] = field(default_factory=dict)


class StatusMonitor:
    """
    Comprehensive status monitoring for data collection operations.
    
    Features:
    - Health check scheduling and execution
    - Performance metric collection
    - Alert generation and management
    - Status dashboard data
    - Integration with storage for historical data
    """
    
    def __init__(
        self,
        storage: SQLiteStore,
        monitor_config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize status monitor.
        
        Args:
            storage: Storage backend for metrics and alerts
            monitor_config: Configuration for monitoring behavior
        """
        self.storage = storage
        self.config = monitor_config or {}
        
        # Monitor settings
        self.check_interval = self.config.get('check_interval_seconds', 60)
        self.metric_retention_days = self.config.get('metric_retention_days', 30)
        self.alert_retention_days = self.config.get('alert_retention_days', 90)
        self.enable_performance_monitoring = self.config.get('enable_performance_monitoring', True)
        
        # Health checks
        self.health_checks: Dict[str, HealthCheck] = {}
        
        # Alerts
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: List[Alert] = []
        
        # Performance metrics
        self.recent_metrics: List[PerformanceMetric] = []
        self.metric_aggregates: Dict[str, Dict[str, float]] = {}
        
        # Monitor state
        self.is_running = False
        self.shutdown_event = threading.Event()
        self.monitor_thread: Optional[threading.Thread] = None
        
        # External components (set by scheduler)
        self.connection_pool = None
        self.job_queue = None
        self.collection_scheduler = None
        self.retry_handler = None
        
        # Setup default health checks
        self._setup_default_health_checks()
        
        # Initialize storage tables
        self._initialize_storage()
    
    def set_components(self, **components):
        """Set references to other system components."""
        self.connection_pool = components.get('connection_pool')
        self.job_queue = components.get('job_queue')
        self.collection_scheduler = components.get('collection_scheduler')
        self.retry_handler = components.get('retry_handler')
    
    def _setup_default_health_checks(self):
        """Setup default health checks for system components."""
        default_checks = [
            HealthCheck(
                name='database_connectivity',
                component='storage',
                check_function=self._check_database_connectivity,
                interval_seconds=300  # 5 minutes
            ),
            HealthCheck(
                name='snowflake_connectivity',
                component='snowflake',
                check_function=self._check_snowflake_connectivity,
                interval_seconds=600  # 10 minutes
            ),
            HealthCheck(
                name='job_queue_health',
                component='scheduler',
                check_function=self._check_job_queue_health,
                interval_seconds=180  # 3 minutes
            ),
            HealthCheck(
                name='collection_performance',
                component='collection',
                check_function=self._check_collection_performance,
                interval_seconds=900  # 15 minutes
            ),
            HealthCheck(
                name='retry_handler_health',
                component='retry',
                check_function=self._check_retry_handler_health,
                interval_seconds=300  # 5 minutes
            )
        ]
        
        for check in default_checks:
            self.health_checks[check.name] = check
    
    def _initialize_storage(self):
        """Initialize storage tables for monitoring data."""
        try:
            # Performance metrics table
            self.storage.execute("""
                CREATE TABLE IF NOT EXISTS performance_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    component TEXT NOT NULL,
                    value REAL NOT NULL,
                    unit TEXT NOT NULL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    tags TEXT  -- JSON string
                )
            """)
            
            # Alerts table
            self.storage.execute("""
                CREATE TABLE IF NOT EXISTS alerts (
                    id TEXT PRIMARY KEY,
                    severity TEXT NOT NULL,
                    component TEXT NOT NULL,
                    message TEXT NOT NULL,
                    details TEXT,  -- JSON string
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    acknowledged_at DATETIME,
                    resolved_at DATETIME,
                    is_active BOOLEAN DEFAULT TRUE
                )
            """)
            
            # Health check results table
            self.storage.execute("""
                CREATE TABLE IF NOT EXISTS health_check_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    check_name TEXT NOT NULL,
                    component TEXT NOT NULL,
                    status TEXT NOT NULL,
                    result TEXT,  -- JSON string
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            self.storage.commit()
            logger.info("Monitoring storage tables initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize monitoring storage: {e}")
    
    def start(self):
        """Start the status monitor."""
        if self.is_running:
            logger.warning("Status monitor is already running")
            return
        
        self.is_running = True
        self.shutdown_event.clear()
        
        # Start monitoring thread
        self.monitor_thread = threading.Thread(
            target=self._monitor_worker,
            name="StatusMonitor",
            daemon=True
        )
        self.monitor_thread.start()
        
        logger.info("Status monitor started")
    
    def stop(self, timeout: int = 30):
        """Stop the status monitor."""
        if not self.is_running:
            logger.warning("Status monitor is not running")
            return
        
        logger.info("Stopping status monitor...")
        
        # Signal shutdown
        self.shutdown_event.set()
        self.is_running = False
        
        # Wait for monitor thread to finish
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=timeout)
        
        logger.info("Status monitor stopped")
    
    def _monitor_worker(self):
        """Main monitoring worker thread."""
        logger.info("Status monitor worker started")
        
        while not self.shutdown_event.is_set():
            try:
                current_time = datetime.now()
                
                # Run health checks
                for check_name, check in self.health_checks.items():
                    if self._should_run_check(check, current_time):
                        self._run_health_check(check)
                
                # Collect performance metrics
                if self.enable_performance_monitoring:
                    self._collect_performance_metrics()
                
                # Cleanup old data
                self._cleanup_old_data()
                
                # Sleep before next cycle
                time.sleep(self.check_interval)
                
            except Exception as e:
                logger.error(f"Error in status monitor worker: {e}")
                time.sleep(30)
        
        logger.info("Status monitor worker stopped")
    
    def _should_run_check(self, check: HealthCheck, current_time: datetime) -> bool:
        """Determine if health check should run."""
        if not check.enabled:
            return False
        
        if check.last_check_time is None:
            return True
        
        elapsed_time = (current_time - check.last_check_time).total_seconds()
        return elapsed_time >= check.interval_seconds
    
    def _run_health_check(self, check: HealthCheck):
        """Run a single health check."""
        try:
            logger.debug(f"Running health check: {check.name}")
            
            start_time = time.time()
            result = check.check_function()
            execution_time = time.time() - start_time
            
            check.last_check_time = datetime.now()
            check.last_result = result
            
            # Determine status
            status = self._evaluate_check_result(result)
            previous_status = check.last_status
            check.last_status = status
            
            # Update failure/warning counters
            if status == ComponentStatus.FAILED:
                check.consecutive_failures += 1
                check.consecutive_warnings = 0
            elif status == ComponentStatus.DEGRADED:
                check.consecutive_warnings += 1
                if check.consecutive_failures > 0:
                    check.consecutive_failures = 0
            else:
                check.consecutive_failures = 0
                check.consecutive_warnings = 0
            
            # Store result
            self._store_health_check_result(check, status, result)
            
            # Generate alerts if needed
            self._process_health_check_alerts(check, status, previous_status)
            
            # Record performance metric
            self.record_metric(
                name=f"{check.name}_execution_time",
                component=check.component,
                value=execution_time,
                unit="seconds"
            )
            
        except Exception as e:
            logger.error(f"Health check {check.name} failed: {e}")
            check.last_status = ComponentStatus.FAILED
            check.consecutive_failures += 1
            
            # Generate error alert
            self.create_alert(
                severity='error',
                component=check.component,
                message=f"Health check {check.name} execution failed",
                details={'error': str(e)}
            )
    
    def _evaluate_check_result(self, result: Dict[str, Any]) -> ComponentStatus:
        """Evaluate health check result to determine status."""
        if not isinstance(result, dict):
            return ComponentStatus.UNKNOWN
        
        # Check for explicit status
        if 'status' in result:
            status_str = result['status'].lower()
            if status_str in ['ok', 'healthy', 'operational']:
                return ComponentStatus.OPERATIONAL
            elif status_str in ['warning', 'degraded']:
                return ComponentStatus.DEGRADED
            elif status_str in ['error', 'failed', 'critical']:
                return ComponentStatus.FAILED
        
        # Check for success flag
        if 'success' in result:
            if result['success']:
                # Check for warnings
                if result.get('warnings') or result.get('warning_count', 0) > 0:
                    return ComponentStatus.DEGRADED
                return ComponentStatus.OPERATIONAL
            else:
                return ComponentStatus.FAILED
        
        # Default evaluation based on presence of errors
        if result.get('errors') or result.get('error_count', 0) > 0:
            return ComponentStatus.FAILED
        elif result.get('warnings') or result.get('warning_count', 0) > 0:
            return ComponentStatus.DEGRADED
        else:
            return ComponentStatus.OPERATIONAL
    
    def _store_health_check_result(self, check: HealthCheck, status: ComponentStatus, result: Dict[str, Any]):
        """Store health check result to database."""
        try:
            self.storage.execute(
                """INSERT INTO health_check_results 
                   (check_name, component, status, result) 
                   VALUES (?, ?, ?, ?)""",
                (check.name, check.component, status.value, json.dumps(result))
            )
            self.storage.commit()
        except Exception as e:
            logger.error(f"Failed to store health check result: {e}")
    
    def _process_health_check_alerts(self, check: HealthCheck, status: ComponentStatus, previous_status: ComponentStatus):
        """Process alerts based on health check results."""
        # Alert on status change to failed
        if status == ComponentStatus.FAILED and previous_status != ComponentStatus.FAILED:
            if check.consecutive_failures >= check.failure_threshold:
                self.create_alert(
                    severity='critical',
                    component=check.component,
                    message=f"Health check {check.name} failed {check.consecutive_failures} times",
                    details={'check_result': check.last_result}
                )
        
        # Alert on status change to degraded
        elif status == ComponentStatus.DEGRADED and previous_status == ComponentStatus.OPERATIONAL:
            if check.consecutive_warnings >= check.warning_threshold:
                self.create_alert(
                    severity='warning',
                    component=check.component,
                    message=f"Health check {check.name} showing degraded performance",
                    details={'check_result': check.last_result}
                )
        
        # Alert on recovery
        elif status == ComponentStatus.OPERATIONAL and previous_status in [ComponentStatus.FAILED, ComponentStatus.DEGRADED]:
            self.create_alert(
                severity='info',
                component=check.component,
                message=f"Health check {check.name} recovered",
                details={'check_result': check.last_result}
            )
    
    # Health check implementations
    def _check_database_connectivity(self) -> Dict[str, Any]:
        """Check database connectivity and performance."""
        try:
            start_time = time.time()
            result = self.storage.fetch_one("SELECT 1 as test")
            query_time = time.time() - start_time
            
            return {
                'success': True,
                'status': 'ok',
                'query_time_ms': round(query_time * 1000, 2),
                'connection_pool_size': getattr(self.storage, '_connection_count', 1)
            }
        except Exception as e:
            return {
                'success': False,
                'status': 'failed',
                'error': str(e)
            }
    
    def _check_snowflake_connectivity(self) -> Dict[str, Any]:
        """Check Snowflake connectivity and performance."""
        if not self.connection_pool:
            return {'success': False, 'error': 'Connection pool not available'}
        
        try:
            start_time = time.time()
            with self.connection_pool.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT CURRENT_VERSION()")
                version = cursor.fetchone()[0]
            
            connection_time = time.time() - start_time
            pool_status = self.connection_pool.get_pool_status()
            
            return {
                'success': True,
                'status': 'ok',
                'connection_time_ms': round(connection_time * 1000, 2),
                'snowflake_version': version,
                'pool_status': pool_status
            }
        except Exception as e:
            return {
                'success': False,
                'status': 'failed',
                'error': str(e)
            }
    
    def _check_job_queue_health(self) -> Dict[str, Any]:
        """Check job queue health and performance."""
        if not self.job_queue:
            return {'success': False, 'error': 'Job queue not available'}
        
        try:
            queue_status = self.job_queue.get_queue_status()
            
            # Evaluate queue health
            warnings = []
            queue_utilization = queue_status['queue_size'] / queue_status['max_queue_size']
            worker_utilization = queue_status['active_jobs'] / queue_status['max_workers']
            
            if queue_utilization > 0.8:
                warnings.append('Queue utilization high')
            if worker_utilization > 0.9:
                warnings.append('Worker utilization high')
            
            status = 'degraded' if warnings else 'ok'
            
            return {
                'success': True,
                'status': status,
                'queue_utilization': round(queue_utilization, 2),
                'worker_utilization': round(worker_utilization, 2),
                'warnings': warnings,
                'queue_stats': queue_status
            }
        except Exception as e:
            return {
                'success': False,
                'status': 'failed',
                'error': str(e)
            }
    
    def _check_collection_performance(self) -> Dict[str, Any]:
        """Check data collection performance and health."""
        if not self.collection_scheduler:
            return {'success': False, 'error': 'Collection scheduler not available'}
        
        try:
            summary = self.collection_scheduler.get_collection_summary(hours_back=2)
            
            # Evaluate collection health
            warnings = []
            if summary['recent_errors'] > 0:
                error_rate = summary['recent_errors'] / max(summary['recent_runs'], 1)
                if error_rate > 0.1:
                    warnings.append(f'High error rate: {error_rate:.1%}')
            
            if summary['recent_runs'] == 0:
                warnings.append('No recent collection runs')
            
            status = summary['collection_health']
            if status == 'warning':
                status = 'degraded'
            elif status == 'critical':
                status = 'failed'
            else:
                status = 'ok'
            
            return {
                'success': True,
                'status': status,
                'collection_summary': summary,
                'warnings': warnings
            }
        except Exception as e:
            return {
                'success': False,
                'status': 'failed',
                'error': str(e)
            }
    
    def _check_retry_handler_health(self) -> Dict[str, Any]:
        """Check retry handler health and performance."""
        if not self.retry_handler:
            return {'success': False, 'error': 'Retry handler not available'}
        
        try:
            handler_status = self.retry_handler.get_handler_status()
            
            # Evaluate retry handler health
            warnings = []
            stats = handler_status['statistics']
            
            if stats['tasks_failed'] > 0:
                failure_rate = stats['tasks_failed'] / max(stats['tasks_submitted'], 1)
                if failure_rate > 0.2:
                    warnings.append(f'High task failure rate: {failure_rate:.1%}')
            
            # Check circuit breakers
            for name, cb in handler_status['circuit_breakers'].items():
                if cb['state'] == 'open':
                    warnings.append(f'Circuit breaker open: {name}')
            
            status = 'degraded' if warnings else 'ok'
            
            return {
                'success': True,
                'status': status,
                'retry_stats': handler_status,
                'warnings': warnings
            }
        except Exception as e:
            return {
                'success': False,
                'status': 'failed',
                'error': str(e)
            }
    
    def _collect_performance_metrics(self):
        """Collect system performance metrics."""
        try:
            # Collect metrics from various components
            if self.connection_pool:
                pool_status = self.connection_pool.get_pool_status()
                self.record_metric('connection_pool_size', 'snowflake', pool_status.get('total_connections', 0), 'count')
                self.record_metric('connection_pool_active', 'snowflake', pool_status.get('active_connections', 0), 'count')
            
            if self.job_queue:
                queue_status = self.job_queue.get_queue_status()
                self.record_metric('job_queue_size', 'scheduler', queue_status.get('queue_size', 0), 'count')
                self.record_metric('job_active_count', 'scheduler', queue_status.get('active_jobs', 0), 'count')
                
                stats = queue_status.get('statistics', {})
                self.record_metric('jobs_completed_total', 'scheduler', stats.get('jobs_completed', 0), 'count')
                self.record_metric('jobs_failed_total', 'scheduler', stats.get('jobs_failed', 0), 'count')
            
            # System metrics
            self.record_metric('active_alerts_count', 'monitoring', len(self.active_alerts), 'count')
            self.record_metric('health_checks_count', 'monitoring', len(self.health_checks), 'count')
            
        except Exception as e:
            logger.error(f"Error collecting performance metrics: {e}")
    
    def _cleanup_old_data(self):
        """Cleanup old monitoring data."""
        try:
            cutoff_metrics = datetime.now() - timedelta(days=self.metric_retention_days)
            cutoff_alerts = datetime.now() - timedelta(days=self.alert_retention_days)
            
            # Clean old metrics
            self.storage.execute(
                "DELETE FROM performance_metrics WHERE timestamp < ?",
                (cutoff_metrics,)
            )
            
            # Clean old resolved alerts
            self.storage.execute(
                "DELETE FROM alerts WHERE resolved_at IS NOT NULL AND resolved_at < ?",
                (cutoff_alerts,)
            )
            
            # Clean old health check results
            self.storage.execute(
                "DELETE FROM health_check_results WHERE timestamp < ?",
                (cutoff_metrics,)
            )
            
            self.storage.commit()
            
            # Clean in-memory data
            cutoff_time = datetime.now() - timedelta(hours=1)
            self.recent_metrics = [
                m for m in self.recent_metrics 
                if m.timestamp >= cutoff_time
            ]
            
        except Exception as e:
            logger.error(f"Error cleaning old monitoring data: {e}")
    
    def record_metric(self, name: str, component: str, value: float, unit: str, tags: Optional[Dict[str, str]] = None):
        """Record a performance metric."""
        try:
            metric = PerformanceMetric(
                name=name,
                component=component,
                value=value,
                unit=unit,
                tags=tags or {}
            )
            
            # Add to recent metrics
            self.recent_metrics.append(metric)
            
            # Store in database
            self.storage.execute(
                """INSERT INTO performance_metrics 
                   (name, component, value, unit, tags) 
                   VALUES (?, ?, ?, ?, ?)""",
                (name, component, value, unit, json.dumps(tags or {}))
            )
            self.storage.commit()
            
        except Exception as e:
            logger.error(f"Failed to record metric {name}: {e}")
    
    def create_alert(self, severity: str, component: str, message: str, details: Optional[Dict[str, Any]] = None) -> str:
        """Create a new alert."""
        alert_id = f"{component}_{severity}_{int(time.time())}"
        
        alert = Alert(
            id=alert_id,
            severity=severity,
            component=component,
            message=message,
            details=details or {}
        )
        
        self.active_alerts[alert_id] = alert
        self.alert_history.append(alert)
        
        # Store in database
        try:
            self.storage.execute(
                """INSERT INTO alerts 
                   (id, severity, component, message, details) 
                   VALUES (?, ?, ?, ?, ?)""",
                (alert_id, severity, component, message, json.dumps(details or {}))
            )
            self.storage.commit()
        except Exception as e:
            logger.error(f"Failed to store alert: {e}")
        
        logger.warning(f"Alert created [{severity}] {component}: {message}")
        return alert_id
    
    def resolve_alert(self, alert_id: str) -> bool:
        """Resolve an active alert."""
        if alert_id in self.active_alerts:
            alert = self.active_alerts[alert_id]
            alert.is_active = False
            alert.resolved_at = datetime.now()
            
            del self.active_alerts[alert_id]
            
            # Update in database
            try:
                self.storage.execute(
                    "UPDATE alerts SET is_active = FALSE, resolved_at = ? WHERE id = ?",
                    (alert.resolved_at, alert_id)
                )
                self.storage.commit()
            except Exception as e:
                logger.error(f"Failed to update resolved alert: {e}")
            
            logger.info(f"Alert resolved: {alert_id}")
            return True
        
        return False
    
    def get_system_health(self) -> Dict[str, Any]:
        """Get overall system health status."""
        overall_status = HealthStatus.HEALTHY
        component_statuses = {}
        
        # Evaluate each component
        for check_name, check in self.health_checks.items():
            component_statuses[check.component] = check.last_status.value
            
            if check.last_status == ComponentStatus.FAILED:
                overall_status = HealthStatus.CRITICAL
            elif check.last_status == ComponentStatus.DEGRADED and overall_status != HealthStatus.CRITICAL:
                overall_status = HealthStatus.WARNING
        
        # Count alerts by severity
        alert_counts = {'critical': 0, 'error': 0, 'warning': 0, 'info': 0}
        for alert in self.active_alerts.values():
            alert_counts[alert.severity] = alert_counts.get(alert.severity, 0) + 1
        
        return {
            'overall_status': overall_status.value,
            'component_statuses': component_statuses,
            'active_alerts': len(self.active_alerts),
            'alert_breakdown': alert_counts,
            'last_updated': datetime.now().isoformat(),
            'monitoring_enabled': self.is_running
        }
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get comprehensive dashboard data."""
        return {
            'system_health': self.get_system_health(),
            'health_checks': {
                name: {
                    'status': check.last_status.value,
                    'last_check': check.last_check_time.isoformat() if check.last_check_time else None,
                    'consecutive_failures': check.consecutive_failures,
                    'enabled': check.enabled
                }
                for name, check in self.health_checks.items()
            },
            'active_alerts': [
                {
                    'id': alert.id,
                    'severity': alert.severity,
                    'component': alert.component,
                    'message': alert.message,
                    'created_at': alert.created_at.isoformat()
                }
                for alert in self.active_alerts.values()
            ],
            'recent_metrics': [
                {
                    'name': metric.name,
                    'component': metric.component,
                    'value': metric.value,
                    'unit': metric.unit,
                    'timestamp': metric.timestamp.isoformat()
                }
                for metric in self.recent_metrics[-50:]  # Last 50 metrics
            ]
        }
