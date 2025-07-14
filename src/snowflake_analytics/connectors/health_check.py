"""
Connection Health Checker - Monitors Snowflake connection health and provides diagnostics.

This module provides comprehensive health monitoring for Snowflake connections,
including connectivity tests, performance monitoring, and diagnostic information.
"""

import time
import threading
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta

from .snowflake_client import SnowflakeClient
from .connection_pool import ConnectionPool
from ..config.settings import SnowflakeSettings
from ..utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class HealthCheckResult:
    """Result of a health check operation."""
    timestamp: datetime
    is_healthy: bool
    response_time_ms: float
    error_message: Optional[str] = None
    connection_info: Optional[Dict[str, Any]] = None
    test_query_result: Optional[Any] = None


@dataclass
class HealthMetrics:
    """Health metrics over time."""
    total_checks: int
    successful_checks: int
    failed_checks: int
    average_response_time_ms: float
    min_response_time_ms: float
    max_response_time_ms: float
    uptime_percentage: float
    last_successful_check: Optional[datetime]
    last_failed_check: Optional[datetime]


class ConnectionHealthChecker:
    """
    Monitors and reports on Snowflake connection health.
    
    Features:
    - Periodic health checks with configurable intervals
    - Response time monitoring and statistics
    - Connection diagnostics and troubleshooting
    - Health history and metrics tracking
    - Integration with connection pools
    - Alert thresholds and notifications
    """
    
    def __init__(
        self,
        settings: Optional[SnowflakeSettings] = None,
        check_interval: int = 60,  # seconds
        max_history_size: int = 1000,
        response_time_threshold_ms: float = 5000.0,
        failure_threshold_percent: float = 10.0
    ):
        """
        Initialize health checker.
        
        Args:
            settings: Snowflake configuration settings
            check_interval: Interval between health checks (seconds)
            max_history_size: Maximum number of health check results to keep
            response_time_threshold_ms: Response time threshold for warnings
            failure_threshold_percent: Failure rate threshold for alerts
        """
        self.settings = settings or SnowflakeSettings()
        self.check_interval = check_interval
        self.max_history_size = max_history_size
        self.response_time_threshold_ms = response_time_threshold_ms
        self.failure_threshold_percent = failure_threshold_percent
        
        # Health check history
        self._health_history: List[HealthCheckResult] = []
        self._history_lock = threading.RLock()
        
        # Monitoring thread
        self._monitoring_thread: Optional[threading.Thread] = None
        self._shutdown_flag = threading.Event()
        self._is_monitoring = False
        
        # Clients and pools for testing
        self._test_client: Optional[SnowflakeClient] = None
        self._monitored_pools: List[ConnectionPool] = []
    
    def start_monitoring(self):
        """Start continuous health monitoring."""
        if self._is_monitoring:
            logger.warning("Health monitoring is already running")
            return
        
        self._shutdown_flag.clear()
        self._monitoring_thread = threading.Thread(
            target=self._monitoring_worker,
            name="ConnectionHealthMonitor",
            daemon=True
        )
        self._monitoring_thread.start()
        self._is_monitoring = True
        logger.info(f"Started connection health monitoring (interval: {self.check_interval}s)")
    
    def stop_monitoring(self):
        """Stop continuous health monitoring."""
        if not self._is_monitoring:
            return
        
        self._shutdown_flag.set()
        if self._monitoring_thread and self._monitoring_thread.is_alive():
            self._monitoring_thread.join(timeout=10)
        
        self._is_monitoring = False
        logger.info("Stopped connection health monitoring")
    
    def _monitoring_worker(self):
        """Background worker for continuous health monitoring."""
        while not self._shutdown_flag.is_set():
            try:
                self.perform_health_check()
                self._check_alert_conditions()
                time.sleep(self.check_interval)
            except Exception as e:
                logger.error(f"Error in health monitoring worker: {e}")
                time.sleep(min(self.check_interval, 30))  # Don't flood on errors
    
    def perform_health_check(self, client: Optional[SnowflakeClient] = None) -> HealthCheckResult:
        """
        Perform a single health check.
        
        Args:
            client: Specific client to test. If None, creates a new test client.
            
        Returns:
            HealthCheckResult: Result of the health check
        """
        start_time = time.time()
        timestamp = datetime.now()
        
        try:
            # Use provided client or create test client
            test_client = client or self._get_test_client()
            
            # Perform health check with test query
            test_query = "SELECT CURRENT_TIMESTAMP() as check_time, CURRENT_VERSION() as version"
            result = test_client.execute_query(test_query, retry_attempts=1)
            
            response_time_ms = (time.time() - start_time) * 1000
            
            # Get connection info
            connection_info = test_client.get_connection_info()
            
            health_result = HealthCheckResult(
                timestamp=timestamp,
                is_healthy=True,
                response_time_ms=response_time_ms,
                connection_info=connection_info,
                test_query_result=result[0] if result else None
            )
            
            logger.debug(f"Health check passed in {response_time_ms:.1f}ms")
            
        except Exception as e:
            response_time_ms = (time.time() - start_time) * 1000
            
            health_result = HealthCheckResult(
                timestamp=timestamp,
                is_healthy=False,
                response_time_ms=response_time_ms,
                error_message=str(e)
            )
            
            logger.warning(f"Health check failed after {response_time_ms:.1f}ms: {e}")
        
        # Store result in history
        self._add_to_history(health_result)
        
        return health_result
    
    def _get_test_client(self) -> SnowflakeClient:
        """Get or create a test client for health checks."""
        if not self._test_client:
            self._test_client = SnowflakeClient(self.settings)
        
        return self._test_client
    
    def _add_to_history(self, result: HealthCheckResult):
        """Add health check result to history."""
        with self._history_lock:
            self._health_history.append(result)
            
            # Trim history if too large
            if len(self._health_history) > self.max_history_size:
                self._health_history = self._health_history[-self.max_history_size:]
    
    def get_current_status(self) -> Dict[str, Any]:
        """
        Get current connection health status.
        
        Returns:
            Dictionary with current health status
        """
        with self._history_lock:
            if not self._health_history:
                return {
                    'status': 'unknown',
                    'message': 'No health checks performed yet'
                }
            
            latest_result = self._health_history[-1]
            
            return {
                'status': 'healthy' if latest_result.is_healthy else 'unhealthy',
                'last_check': latest_result.timestamp.isoformat(),
                'response_time_ms': latest_result.response_time_ms,
                'error_message': latest_result.error_message,
                'connection_info': latest_result.connection_info,
                'is_monitoring': self._is_monitoring
            }
    
    def get_health_metrics(self, hours_back: int = 24) -> HealthMetrics:
        """
        Calculate health metrics over specified time period.
        
        Args:
            hours_back: Number of hours to include in metrics calculation
            
        Returns:
            HealthMetrics: Calculated health metrics
        """
        cutoff_time = datetime.now() - timedelta(hours=hours_back)
        
        with self._history_lock:
            # Filter results within time window
            recent_results = [
                r for r in self._health_history 
                if r.timestamp >= cutoff_time
            ]
            
            if not recent_results:
                return HealthMetrics(
                    total_checks=0,
                    successful_checks=0,
                    failed_checks=0,
                    average_response_time_ms=0.0,
                    min_response_time_ms=0.0,
                    max_response_time_ms=0.0,
                    uptime_percentage=0.0,
                    last_successful_check=None,
                    last_failed_check=None
                )
            
            # Calculate metrics
            total_checks = len(recent_results)
            successful_checks = sum(1 for r in recent_results if r.is_healthy)
            failed_checks = total_checks - successful_checks
            
            response_times = [r.response_time_ms for r in recent_results]
            average_response_time = sum(response_times) / len(response_times)
            min_response_time = min(response_times)
            max_response_time = max(response_times)
            
            uptime_percentage = (successful_checks / total_checks) * 100
            
            # Find last successful and failed checks
            last_successful = None
            last_failed = None
            
            for result in reversed(recent_results):
                if result.is_healthy and last_successful is None:
                    last_successful = result.timestamp
                elif not result.is_healthy and last_failed is None:
                    last_failed = result.timestamp
                
                if last_successful and last_failed:
                    break
            
            return HealthMetrics(
                total_checks=total_checks,
                successful_checks=successful_checks,
                failed_checks=failed_checks,
                average_response_time_ms=average_response_time,
                min_response_time_ms=min_response_time,
                max_response_time_ms=max_response_time,
                uptime_percentage=uptime_percentage,
                last_successful_check=last_successful,
                last_failed_check=last_failed
            )
    
    def get_health_history(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get health check history.
        
        Args:
            limit: Maximum number of results to return
            
        Returns:
            List of health check results as dictionaries
        """
        with self._history_lock:
            history = self._health_history.copy()
            
            if limit:
                history = history[-limit:]
            
            return [asdict(result) for result in history]
    
    def add_connection_pool(self, pool: ConnectionPool):
        """
        Add a connection pool to monitor.
        
        Args:
            pool: ConnectionPool to monitor
        """
        self._monitored_pools.append(pool)
        logger.info(f"Added connection pool to health monitoring")
    
    def get_pool_health(self) -> List[Dict[str, Any]]:
        """
        Get health status of all monitored connection pools.
        
        Returns:
            List of pool health information
        """
        pool_health = []
        
        for i, pool in enumerate(self._monitored_pools):
            try:
                stats = pool.get_stats()
                
                # Test a connection from the pool
                pool_check_start = time.time()
                try:
                    with pool.get_connection() as conn:
                        conn.execute_query("SELECT 1", retry_attempts=1)
                    pool_response_time = (time.time() - pool_check_start) * 1000
                    pool_healthy = True
                    pool_error = None
                except Exception as e:
                    pool_response_time = (time.time() - pool_check_start) * 1000
                    pool_healthy = False
                    pool_error = str(e)
                
                pool_health.append({
                    'pool_id': i,
                    'is_healthy': pool_healthy,
                    'response_time_ms': pool_response_time,
                    'error_message': pool_error,
                    'stats': stats
                })
                
            except Exception as e:
                pool_health.append({
                    'pool_id': i,
                    'is_healthy': False,
                    'error_message': f"Failed to check pool health: {e}",
                    'stats': {}
                })
        
        return pool_health
    
    def _check_alert_conditions(self):
        """Check if any alert conditions are met."""
        if len(self._health_history) < 5:  # Need some history
            return
        
        # Check recent failure rate
        recent_results = self._health_history[-10:]  # Last 10 checks
        failure_rate = (1 - sum(1 for r in recent_results if r.is_healthy) / len(recent_results)) * 100
        
        if failure_rate > self.failure_threshold_percent:
            logger.warning(
                f"High failure rate detected: {failure_rate:.1f}% "
                f"(threshold: {self.failure_threshold_percent}%)"
            )
        
        # Check response time
        latest_result = self._health_history[-1]
        if (latest_result.is_healthy and 
            latest_result.response_time_ms > self.response_time_threshold_ms):
            logger.warning(
                f"Slow response time detected: {latest_result.response_time_ms:.1f}ms "
                f"(threshold: {self.response_time_threshold_ms}ms)"
            )
    
    def run_diagnostics(self) -> Dict[str, Any]:
        """
        Run comprehensive connection diagnostics.
        
        Returns:
            Dictionary with diagnostic information
        """
        diagnostics = {
            'timestamp': datetime.now().isoformat(),
            'connection_test': {},
            'performance_test': {},
            'configuration_check': {},
            'network_test': {},
            'pool_diagnostics': []
        }
        
        try:
            # Basic connection test
            client = self._get_test_client()
            health_result = self.perform_health_check(client)
            
            diagnostics['connection_test'] = {
                'success': health_result.is_healthy,
                'response_time_ms': health_result.response_time_ms,
                'error': health_result.error_message,
                'connection_info': health_result.connection_info
            }
            
            if health_result.is_healthy:
                # Performance test with larger query
                perf_start = time.time()
                try:
                    perf_query = """
                    SELECT 
                        COUNT(*) as row_count,
                        MAX(START_TIME) as latest_time,
                        MIN(START_TIME) as earliest_time
                    FROM SNOWFLAKE.ACCOUNT_USAGE.QUERY_HISTORY 
                    WHERE START_TIME >= DATEADD(day, -1, CURRENT_TIMESTAMP())
                    """
                    perf_result = client.execute_query(perf_query)
                    perf_time = (time.time() - perf_start) * 1000
                    
                    diagnostics['performance_test'] = {
                        'success': True,
                        'query_time_ms': perf_time,
                        'result': perf_result[0] if perf_result else None
                    }
                except Exception as e:
                    diagnostics['performance_test'] = {
                        'success': False,
                        'error': str(e)
                    }
                
                # Configuration check
                try:
                    config_query = """
                    SELECT 
                        CURRENT_ACCOUNT() as account,
                        CURRENT_USER() as user,
                        CURRENT_ROLE() as role,
                        CURRENT_WAREHOUSE() as warehouse,
                        CURRENT_DATABASE() as database,
                        CURRENT_SCHEMA() as schema
                    """
                    config_result = client.execute_query(config_query)
                    
                    diagnostics['configuration_check'] = {
                        'success': True,
                        'current_config': config_result[0] if config_result else None,
                        'expected_config': {
                            'account': self.settings.account,
                            'user': self.settings.user,
                            'warehouse': self.settings.warehouse,
                            'database': self.settings.database,
                            'schema': self.settings.schema,
                            'role': self.settings.role
                        }
                    }
                except Exception as e:
                    diagnostics['configuration_check'] = {
                        'success': False,
                        'error': str(e)
                    }
            
            # Pool diagnostics
            diagnostics['pool_diagnostics'] = self.get_pool_health()
            
        except Exception as e:
            logger.error(f"Error running diagnostics: {e}")
            diagnostics['error'] = str(e)
        
        return diagnostics
    
    def cleanup(self):
        """Clean up resources."""
        self.stop_monitoring()
        
        if self._test_client:
            try:
                self._test_client.close()
            except Exception as e:
                logger.warning(f"Error closing test client: {e}")
            self._test_client = None
        
        logger.info("Health checker cleanup completed")
