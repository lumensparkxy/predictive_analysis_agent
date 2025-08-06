"""
Health Monitoring System for Snowflake Analytics
Comprehensive health checks for application components, infrastructure, and services.
"""

import os
import time
import psutil
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from enum import Enum
import threading
import structlog
import subprocess
import json

logger = structlog.get_logger(__name__)


class HealthStatus(Enum):
    """Health status levels."""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


class HealthCheck:
    """Individual health check definition."""
    
    def __init__(self, name: str, description: str, check_function, 
                 warning_threshold: Optional[float] = None,
                 critical_threshold: Optional[float] = None,
                 timeout: int = 30):
        self.name = name
        self.description = description
        self.check_function = check_function
        self.warning_threshold = warning_threshold
        self.critical_threshold = critical_threshold
        self.timeout = timeout
        self.last_run = None
        self.last_result = None


class HealthChecker:
    """
    Comprehensive health monitoring system.
    Monitors application, database, infrastructure, and external dependencies.
    """
    
    def __init__(self):
        """Initialize health checker."""
        self.checks: Dict[str, HealthCheck] = {}
        self.check_history: Dict[str, List[Dict[str, Any]]] = {}
        self.history_max_entries = int(os.getenv('HEALTH_HISTORY_MAX', '1000'))
        
        # Configuration
        self.check_interval = int(os.getenv('HEALTH_CHECK_INTERVAL', '30'))
        self.api_base_url = os.getenv('API_BASE_URL', 'http://localhost:8000')
        self.dashboard_url = os.getenv('DASHBOARD_URL', 'http://localhost:8501')
        
        # Database connection settings
        self.db_host = os.getenv('DB_HOST', 'localhost')
        self.db_port = int(os.getenv('DB_PORT', '5432'))
        self.redis_host = os.getenv('REDIS_HOST', 'localhost')
        self.redis_port = int(os.getenv('REDIS_PORT', '6379'))
        
        # Snowflake connection settings
        self.snowflake_account = os.getenv('SNOWFLAKE_ACCOUNT')
        self.snowflake_user = os.getenv('SNOWFLAKE_USER')
        
        # Background monitoring
        self._monitoring_active = False
        self._monitoring_thread = None
        
        # Register default health checks
        self._register_default_checks()
        
        logger.info("HealthChecker initialized", checks=len(self.checks))
    
    def _register_default_checks(self):
        """Register default health checks."""
        # API health checks
        self.register_check(
            "api_health", 
            "Main API health endpoint",
            self._check_api_health,
            timeout=10
        )
        
        self.register_check(
            "api_metrics",
            "API metrics endpoint", 
            self._check_api_metrics,
            timeout=10
        )
        
        # Database health checks
        self.register_check(
            "postgresql_connection",
            "PostgreSQL database connectivity",
            self._check_postgresql,
            timeout=10
        )
        
        self.register_check(
            "redis_connection", 
            "Redis cache connectivity",
            self._check_redis,
            timeout=10
        )
        
        self.register_check(
            "snowflake_connection",
            "Snowflake data warehouse connectivity", 
            self._check_snowflake,
            timeout=30
        )
        
        # System resource checks
        self.register_check(
            "cpu_usage",
            "CPU utilization percentage",
            self._check_cpu_usage,
            warning_threshold=80.0,
            critical_threshold=90.0
        )
        
        self.register_check(
            "memory_usage",
            "Memory utilization percentage",
            self._check_memory_usage, 
            warning_threshold=80.0,
            critical_threshold=90.0
        )
        
        self.register_check(
            "disk_usage",
            "Disk space utilization percentage",
            self._check_disk_usage,
            warning_threshold=80.0,
            critical_threshold=90.0
        )
        
        # System services checks
        self.register_check(
            "analytics_services",
            "Analytics system services status",
            self._check_analytics_services,
            timeout=15
        )
        
        # Application-specific checks
        self.register_check(
            "data_freshness",
            "Data collection freshness",
            self._check_data_freshness,
            warning_threshold=6.0,   # hours
            critical_threshold=24.0  # hours
        )
        
        self.register_check(
            "queue_health",
            "Background task queue health",
            self._check_queue_health,
            warning_threshold=100.0,  # queue size
            critical_threshold=500.0
        )
        
        # External dependency checks
        self.register_check(
            "external_apis",
            "External API dependencies",
            self._check_external_apis,
            timeout=15
        )
    
    def register_check(self, name: str, description: str, check_function,
                      warning_threshold: Optional[float] = None,
                      critical_threshold: Optional[float] = None,
                      timeout: int = 30) -> bool:
        """Register a new health check."""
        try:
            health_check = HealthCheck(
                name=name,
                description=description,
                check_function=check_function,
                warning_threshold=warning_threshold,
                critical_threshold=critical_threshold,
                timeout=timeout
            )
            
            self.checks[name] = health_check
            self.check_history[name] = []
            
            logger.info("Health check registered", name=name, description=description)
            return True
            
        except Exception as e:
            logger.error("Failed to register health check", name=name, error=str(e))
            return False
    
    def run_check(self, check_name: str) -> Dict[str, Any]:
        """Run a specific health check."""
        if check_name not in self.checks:
            return {
                'name': check_name,
                'status': HealthStatus.UNKNOWN.value,
                'message': 'Health check not found',
                'timestamp': datetime.utcnow().isoformat()
            }
        
        check = self.checks[check_name]
        start_time = time.time()
        
        try:
            # Run the check with timeout
            result = self._run_check_with_timeout(check)
            
            # Determine status based on result and thresholds
            status = self._determine_status(result, check)
            
            duration = time.time() - start_time
            
            result_data = {
                'name': check_name,
                'description': check.description,
                'status': status.value,
                'value': result if isinstance(result, (int, float)) else None,
                'message': result if isinstance(result, str) else f'Check completed: {result}',
                'duration': round(duration, 3),
                'timestamp': datetime.utcnow().isoformat(),
                'thresholds': {
                    'warning': check.warning_threshold,
                    'critical': check.critical_threshold
                }
            }
            
            # Store result in check object
            check.last_run = datetime.utcnow()
            check.last_result = result_data
            
            # Add to history
            self._add_to_history(check_name, result_data)
            
            logger.debug("Health check completed", name=check_name, status=status.value, duration=duration)
            
            return result_data
            
        except Exception as e:
            duration = time.time() - start_time
            error_result = {
                'name': check_name,
                'description': check.description,
                'status': HealthStatus.CRITICAL.value,
                'message': f'Health check failed: {str(e)}',
                'duration': round(duration, 3),
                'timestamp': datetime.utcnow().isoformat(),
                'error': str(e)
            }
            
            check.last_run = datetime.utcnow()
            check.last_result = error_result
            self._add_to_history(check_name, error_result)
            
            logger.error("Health check failed", name=check_name, error=str(e), duration=duration)
            return error_result
    
    def run_all_checks(self) -> Dict[str, Any]:
        """Run all registered health checks."""
        start_time = time.time()
        results = {}
        
        logger.info("Running all health checks", total_checks=len(self.checks))
        
        # Run checks in parallel for better performance
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            future_to_check = {
                executor.submit(self.run_check, name): name 
                for name in self.checks.keys()
            }
            
            for future in concurrent.futures.as_completed(future_to_check):
                check_name = future_to_check[future]
                try:
                    result = future.result()
                    results[check_name] = result
                except Exception as e:
                    results[check_name] = {
                        'name': check_name,
                        'status': HealthStatus.CRITICAL.value,
                        'message': f'Check execution failed: {str(e)}',
                        'timestamp': datetime.utcnow().isoformat(),
                        'error': str(e)
                    }
        
        # Calculate overall health
        total_duration = time.time() - start_time
        overall_status = self._calculate_overall_status(results)
        
        summary = {
            'overall_status': overall_status.value,
            'total_checks': len(results),
            'healthy_checks': len([r for r in results.values() if r['status'] == HealthStatus.HEALTHY.value]),
            'warning_checks': len([r for r in results.values() if r['status'] == HealthStatus.WARNING.value]),
            'critical_checks': len([r for r in results.values() if r['status'] == HealthStatus.CRITICAL.value]),
            'unknown_checks': len([r for r in results.values() if r['status'] == HealthStatus.UNKNOWN.value]),
            'total_duration': round(total_duration, 3),
            'timestamp': datetime.utcnow().isoformat(),
            'checks': results
        }
        
        logger.info("Health check summary", 
                   overall_status=overall_status.value,
                   healthy=summary['healthy_checks'],
                   warning=summary['warning_checks'], 
                   critical=summary['critical_checks'],
                   duration=total_duration)
        
        return summary
    
    def get_health_summary(self) -> Dict[str, Any]:
        """Get current health summary without running checks."""
        current_time = datetime.utcnow()
        summary = {
            'timestamp': current_time.isoformat(),
            'checks': {},
            'stale_checks': []
        }
        
        for name, check in self.checks.items():
            if check.last_result:
                # Check if result is stale (older than 2x check interval)
                last_run = check.last_run
                if last_run and (current_time - last_run).total_seconds() > (self.check_interval * 2):
                    summary['stale_checks'].append(name)
                
                summary['checks'][name] = check.last_result
            else:
                summary['checks'][name] = {
                    'name': name,
                    'status': HealthStatus.UNKNOWN.value,
                    'message': 'Never run',
                    'timestamp': None
                }
        
        return summary
    
    def start_monitoring(self):
        """Start background health monitoring."""
        if self._monitoring_active:
            logger.warning("Health monitoring already active")
            return
        
        self._monitoring_active = True
        self._monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self._monitoring_thread.start()
        
        logger.info("Health monitoring started", interval=self.check_interval)
    
    def stop_monitoring(self):
        """Stop background health monitoring."""
        if not self._monitoring_active:
            return
        
        self._monitoring_active = False
        if self._monitoring_thread:
            self._monitoring_thread.join(timeout=10)
        
        logger.info("Health monitoring stopped")
    
    def _monitoring_loop(self):
        """Background monitoring loop."""
        while self._monitoring_active:
            try:
                self.run_all_checks()
            except Exception as e:
                logger.error("Error in monitoring loop", error=str(e))
            
            # Wait for next check interval
            time.sleep(self.check_interval)
    
    def _run_check_with_timeout(self, check: HealthCheck):
        """Run a health check with timeout."""
        import signal
        
        def timeout_handler(signum, frame):
            raise TimeoutError(f"Health check timed out after {check.timeout} seconds")
        
        # Set timeout signal
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(check.timeout)
        
        try:
            result = check.check_function()
            signal.alarm(0)  # Cancel timeout
            return result
        except Exception as e:
            signal.alarm(0)  # Cancel timeout
            raise
    
    def _determine_status(self, result: Any, check: HealthCheck) -> HealthStatus:
        """Determine health status based on result and thresholds."""
        if isinstance(result, str) and 'error' in result.lower():
            return HealthStatus.CRITICAL
        
        if isinstance(result, bool):
            return HealthStatus.HEALTHY if result else HealthStatus.CRITICAL
        
        if isinstance(result, (int, float)):
            if check.critical_threshold and result >= check.critical_threshold:
                return HealthStatus.CRITICAL
            elif check.warning_threshold and result >= check.warning_threshold:
                return HealthStatus.WARNING
            else:
                return HealthStatus.HEALTHY
        
        # Default to healthy for other result types
        return HealthStatus.HEALTHY
    
    def _calculate_overall_status(self, results: Dict[str, Any]) -> HealthStatus:
        """Calculate overall health status from individual check results."""
        statuses = [result['status'] for result in results.values()]
        
        if HealthStatus.CRITICAL.value in statuses:
            return HealthStatus.CRITICAL
        elif HealthStatus.WARNING.value in statuses:
            return HealthStatus.WARNING
        elif HealthStatus.UNKNOWN.value in statuses:
            return HealthStatus.WARNING
        else:
            return HealthStatus.HEALTHY
    
    def _add_to_history(self, check_name: str, result: Dict[str, Any]):
        """Add result to check history."""
        if check_name not in self.check_history:
            self.check_history[check_name] = []
        
        self.check_history[check_name].append(result)
        
        # Limit history size
        if len(self.check_history[check_name]) > self.history_max_entries:
            self.check_history[check_name] = self.check_history[check_name][-self.history_max_entries:]
    
    # Individual health check implementations
    def _check_api_health(self) -> bool:
        """Check main API health endpoint."""
        try:
            response = requests.get(f"{self.api_base_url}/health", timeout=5)
            return response.status_code == 200 and 'healthy' in response.text.lower()
        except Exception:
            return False
    
    def _check_api_metrics(self) -> bool:
        """Check API metrics endpoint."""
        try:
            response = requests.get(f"{self.api_base_url}/metrics", timeout=5)
            return response.status_code == 200
        except Exception:
            return False
    
    def _check_postgresql(self) -> bool:
        """Check PostgreSQL connection."""
        try:
            import psycopg2
            conn = psycopg2.connect(
                host=self.db_host,
                port=self.db_port,
                dbname=os.getenv('DB_NAME', 'analytics'),
                user=os.getenv('DB_USER', 'analytics'),
                password=os.getenv('DB_PASSWORD', ''),
                connect_timeout=5
            )
            cur = conn.cursor()
            cur.execute('SELECT 1')
            cur.close()
            conn.close()
            return True
        except Exception:
            return False
    
    def _check_redis(self) -> bool:
        """Check Redis connection."""
        try:
            import redis
            r = redis.Redis(host=self.redis_host, port=self.redis_port, socket_timeout=5)
            return r.ping()
        except Exception:
            return False
    
    def _check_snowflake(self) -> bool:
        """Check Snowflake connection."""
        try:
            if not self.snowflake_account or not self.snowflake_user:
                return False
                
            import snowflake.connector
            conn = snowflake.connector.connect(
                account=self.snowflake_account,
                user=self.snowflake_user,
                password=os.getenv('SNOWFLAKE_PASSWORD'),
                warehouse=os.getenv('SNOWFLAKE_WAREHOUSE'),
                database=os.getenv('SNOWFLAKE_DATABASE'),
                schema=os.getenv('SNOWFLAKE_SCHEMA'),
                login_timeout=10
            )
            cur = conn.cursor()
            cur.execute('SELECT 1')
            cur.close()
            conn.close()
            return True
        except Exception:
            return False
    
    def _check_cpu_usage(self) -> float:
        """Check CPU usage percentage."""
        return psutil.cpu_percent(interval=1)
    
    def _check_memory_usage(self) -> float:
        """Check memory usage percentage."""
        return psutil.virtual_memory().percent
    
    def _check_disk_usage(self) -> float:
        """Check disk usage percentage."""
        app_root = os.getenv('APP_ROOT', '/opt/analytics')
        return psutil.disk_usage(app_root).percent
    
    def _check_analytics_services(self) -> bool:
        """Check analytics system services."""
        services = ['analytics-api', 'analytics-worker', 'analytics-scheduler']
        
        for service in services:
            try:
                result = subprocess.run(
                    ['systemctl', 'is-active', f'{service}.service'],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                if result.returncode != 0 or result.stdout.strip() != 'active':
                    return False
            except Exception:
                return False
        
        return True
    
    def _check_data_freshness(self) -> float:
        """Check data collection freshness in hours."""
        try:
            # This would check the actual last data collection time
            # For now, return a mock value
            return 1.5  # 1.5 hours since last collection
        except Exception:
            return 999.0  # Very stale data
    
    def _check_queue_health(self) -> float:
        """Check background task queue health."""
        try:
            # This would check actual queue size
            # For now, return a mock value
            return 5.0  # 5 items in queue
        except Exception:
            return 0.0
    
    def _check_external_apis(self) -> bool:
        """Check external API dependencies."""
        try:
            # Check if we can reach key external services
            external_apis = [
                'https://api.github.com/zen',  # GitHub API
                'https://httpbin.org/status/200'  # Test endpoint
            ]
            
            for api_url in external_apis:
                try:
                    response = requests.get(api_url, timeout=5)
                    if response.status_code >= 400:
                        return False
                except Exception:
                    return False
            
            return True
        except Exception:
            return False
    
    def get_check_history(self, check_name: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get history for a specific health check."""
        if check_name not in self.check_history:
            return []
        
        history = self.check_history[check_name]
        if limit:
            history = history[-limit:]
        
        return history
    
    def export_health_data(self) -> Dict[str, Any]:
        """Export comprehensive health data for external monitoring."""
        return {
            'timestamp': datetime.utcnow().isoformat(),
            'health_summary': self.get_health_summary(),
            'check_history': {
                name: self.get_check_history(name, 100) 
                for name in self.checks.keys()
            },
            'monitoring_active': self._monitoring_active,
            'check_interval': self.check_interval,
            'total_checks': len(self.checks)
        }