"""
Main Data Collection Service - Orchestrates all data collection operations.

This module provides the main entry point for the Snowflake data collection
system, coordinating all components including scheduling, monitoring, and execution.
"""

import signal
import time
import threading
from datetime import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path

from .connectors.connection_pool import ConnectionPool
from .storage.sqlite_store import SQLiteStore
from .scheduler.collection_scheduler import CollectionScheduler
from .scheduler.job_queue import JobQueue
from .scheduler.retry_handler import RetryHandler
from .scheduler.status_monitor import StatusMonitor
from .config.settings import Settings, SnowflakeSettings
from .utils.logger import get_logger
from .utils.health_check import HealthChecker

logger = get_logger(__name__)


class DataCollectionService:
    """
    Main orchestrator for automated Snowflake data collection.
    
    This service coordinates all components of the data collection system:
    - Connection management
    - Scheduled data collection
    - Job queue and retry handling
    - Status monitoring and health checks
    - Configuration management
    
    Features:
    - Automated 15-minute data collection cycles
    - Comprehensive error handling and recovery
    - Real-time monitoring and alerting
    - Graceful shutdown and startup
    - RESTful API for status and control
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the data collection service.
        
        Args:
            config_path: Path to configuration directory
        """
        self.config_path = config_path or "config"
        self.is_running = False
        self.start_time: Optional[datetime] = None
        
        # Core components (initialized in start())
        self.settings: Optional[Settings] = None
        self.storage: Optional[SQLiteStore] = None
        self.connection_pool: Optional[ConnectionPool] = None
        self.job_queue: Optional[JobQueue] = None
        self.retry_handler: Optional[RetryHandler] = None
        self.collection_scheduler: Optional[CollectionScheduler] = None
        self.status_monitor: Optional[StatusMonitor] = None
        self.health_checker: Optional[HealthChecker] = None
        
        # Shutdown handling
        self.shutdown_event = threading.Event()
        self._setup_signal_handlers()
    
    def _setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown."""
        def signal_handler(signum, frame):
            logger.info(f"Received signal {signum}, initiating graceful shutdown...")
            self.shutdown()
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    def initialize(self) -> bool:
        """
        Initialize all service components.
        
        Returns:
            True if initialization successful, False otherwise
        """
        try:
            logger.info("Initializing Snowflake Data Collection Service...")
            
            # Load configuration
            from .config.settings import get_settings
            self.settings = get_settings()
            logger.info("Configuration loaded successfully")
            
            # Initialize storage
            storage_config = self.settings.get_storage_config()
            self.storage = SQLiteStore(
                db_path=storage_config.get('database_path', 'storage.db')
            )
            logger.info("Storage initialized")
            
            # Initialize connection pool
            snowflake_config = self.settings.get_snowflake_config()
            if snowflake_config:
                snowflake_settings = SnowflakeSettings.from_connection_config(snowflake_config)
                pool_config = self.settings.get_connection_pool_config()
                self.connection_pool = ConnectionPool(
                    settings=snowflake_settings,
                    max_connections=pool_config.get('max_connections', 5),
                    connection_timeout=pool_config.get('connection_timeout', 300)
                )
            else:
                raise ValueError("Snowflake configuration not found")
            logger.info("Connection pool initialized")
            
            # Initialize job queue
            queue_config = self.settings.get_scheduler_config().get('job_queue', {})
            self.job_queue = JobQueue(**queue_config)
            self.job_queue.set_collectors({})  # Will be set by scheduler
            logger.info("Job queue initialized")
            
            # Initialize retry handler
            retry_config = self.settings.get_scheduler_config().get('retry_handler', {})
            self.retry_handler = RetryHandler()
            logger.info("Retry handler initialized")
            
            # Initialize status monitor
            monitor_config = self.settings.get_scheduler_config().get('status_monitor', {})
            self.status_monitor = StatusMonitor(
                storage=self.storage,
                monitor_config=monitor_config
            )
            logger.info("Status monitor initialized")
            
            # Initialize collection scheduler
            scheduler_config = self.settings.get_scheduler_config()
            self.collection_scheduler = CollectionScheduler(
                connection_pool=self.connection_pool,
                storage=self.storage,
                scheduler_config=scheduler_config
            )
            logger.info("Collection scheduler initialized")
            
            # Initialize health checker
            self.health_checker = HealthChecker()
            logger.info("Health checker initialized")
            
            # Set up component cross-references
            self._setup_component_references()
            
            logger.info("All components initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize service: {e}")
            return False
    
    def _setup_component_references(self):
        """Setup cross-references between components."""
        # Set collectors in job queue
        if self.collection_scheduler and self.job_queue:
            self.job_queue.set_collectors(self.collection_scheduler.collectors)
        
        # Set component references in status monitor
        if self.status_monitor:
            self.status_monitor.set_components(
                connection_pool=self.connection_pool,
                job_queue=self.job_queue,
                collection_scheduler=self.collection_scheduler,
                retry_handler=self.retry_handler
            )
    
    def start(self) -> bool:
        """
        Start the data collection service.
        
        Returns:
            True if started successfully, False otherwise
        """
        if self.is_running:
            logger.warning("Service is already running")
            return True
        
        try:
            # Initialize if not already done
            if not all([self.settings, self.storage, self.connection_pool]):
                if not self.initialize():
                    return False
            
            logger.info("Starting Snowflake Data Collection Service...")
            
            # Start core components in order
            self.storage.connect()
            logger.info("Storage connected")
            
            self.connection_pool.start()
            logger.info("Connection pool started")
            
            self.job_queue.start()
            logger.info("Job queue started")
            
            self.retry_handler.start()
            logger.info("Retry handler started")
            
            self.status_monitor.start()
            logger.info("Status monitor started")
            
            self.collection_scheduler.start_scheduler()
            logger.info("Collection scheduler started")
            
            # Mark as running
            self.is_running = True
            self.start_time = datetime.now()
            
            # Run initial health check
            if self.health_checker:
                self.health_checker.run_all_checks()
            
            logger.info("✅ Snowflake Data Collection Service started successfully")
            logger.info(f"Service is now collecting data every 15 minutes as configured")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to start service: {e}")
            self.shutdown()
            return False
    
    def shutdown(self, timeout: int = 120):
        """
        Gracefully shutdown the service.
        
        Args:
            timeout: Maximum time to wait for shutdown
        """
        if not self.is_running:
            logger.warning("Service is not running")
            return
        
        logger.info("Shutting down Snowflake Data Collection Service...")
        self.is_running = False
        self.shutdown_event.set()
        
        try:
            # Stop components in reverse order
            if self.collection_scheduler:
                self.collection_scheduler.stop_scheduler()
                logger.info("Collection scheduler stopped")
            
            if self.status_monitor:
                self.status_monitor.stop(timeout=30)
                logger.info("Status monitor stopped")
            
            if self.retry_handler:
                self.retry_handler.stop(timeout=30)
                logger.info("Retry handler stopped")
            
            if self.job_queue:
                self.job_queue.stop(timeout=60)
                logger.info("Job queue stopped")
            
            if self.connection_pool:
                self.connection_pool.shutdown()
                logger.info("Connection pool shutdown")
            
            if self.storage:
                self.storage.close()
                logger.info("Storage disconnected")
            
            logger.info("✅ Snowflake Data Collection Service shutdown complete")
            
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
    
    def restart(self) -> bool:
        """
        Restart the service.
        
        Returns:
            True if restarted successfully, False otherwise
        """
        logger.info("Restarting Snowflake Data Collection Service...")
        
        if self.is_running:
            self.shutdown()
            time.sleep(5)  # Brief pause between shutdown and startup
        
        return self.start()
    
    def get_service_status(self) -> Dict[str, Any]:
        """Get comprehensive service status."""
        if not self.is_running:
            return {
                'service_status': 'stopped',
                'is_running': False,
                'start_time': None,
                'uptime': None
            }
        
        uptime = None
        if self.start_time:
            uptime_delta = datetime.now() - self.start_time
            uptime = {
                'days': uptime_delta.days,
                'hours': uptime_delta.seconds // 3600,
                'minutes': (uptime_delta.seconds % 3600) // 60,
                'total_seconds': uptime_delta.total_seconds()
            }
        
        status = {
            'service_status': 'running',
            'is_running': True,
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'uptime': uptime,
            'components': {}
        }
        
        # Get component statuses
        try:
            if self.connection_pool:
                status['components']['connection_pool'] = self.connection_pool.get_pool_status()
            
            if self.job_queue:
                status['components']['job_queue'] = self.job_queue.get_queue_status()
            
            if self.collection_scheduler:
                status['components']['scheduler'] = self.collection_scheduler.get_job_status()
            
            if self.retry_handler:
                status['components']['retry_handler'] = self.retry_handler.get_handler_status()
            
            if self.status_monitor:
                status['components']['status_monitor'] = self.status_monitor.get_system_health()
        
        except Exception as e:
            logger.error(f"Error getting component statuses: {e}")
            status['error'] = str(e)
        
        return status
    
    def get_collection_summary(self, hours_back: int = 24) -> Dict[str, Any]:
        """Get summary of data collection activities."""
        if not self.collection_scheduler:
            return {'error': 'Collection scheduler not available'}
        
        try:
            return self.collection_scheduler.get_collection_summary(hours_back=hours_back)
        except Exception as e:
            logger.error(f"Error getting collection summary: {e}")
            return {'error': str(e)}
    
    def run_job_now(self, job_name: str) -> Dict[str, Any]:
        """Run a specific collection job immediately."""
        if not self.collection_scheduler:
            return {'success': False, 'error': 'Collection scheduler not available'}
        
        try:
            return self.collection_scheduler.run_job_now(job_name)
        except Exception as e:
            logger.error(f"Error running job {job_name}: {e}")
            return {'success': False, 'error': str(e)}
    
    def enable_job(self, job_name: str) -> Dict[str, Any]:
        """Enable a scheduled job."""
        if not self.collection_scheduler:
            return {'success': False, 'error': 'Collection scheduler not available'}
        
        try:
            self.collection_scheduler.enable_job(job_name)
            return {'success': True, 'message': f'Job {job_name} enabled'}
        except Exception as e:
            logger.error(f"Error enabling job {job_name}: {e}")
            return {'success': False, 'error': str(e)}
    
    def disable_job(self, job_name: str) -> Dict[str, Any]:
        """Disable a scheduled job."""
        if not self.collection_scheduler:
            return {'success': False, 'error': 'Collection scheduler not available'}
        
        try:
            self.collection_scheduler.disable_job(job_name)
            return {'success': True, 'message': f'Job {job_name} disabled'}
        except Exception as e:
            logger.error(f"Error disabling job {job_name}: {e}")
            return {'success': False, 'error': str(e)}
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get system health status."""
        if not self.status_monitor:
            return {'error': 'Status monitor not available'}
        
        try:
            return self.status_monitor.get_system_health()
        except Exception as e:
            logger.error(f"Error getting health status: {e}")
            return {'error': str(e)}
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get comprehensive dashboard data."""
        if not self.status_monitor:
            return {'error': 'Status monitor not available'}
        
        try:
            dashboard_data = self.status_monitor.get_dashboard_data()
            dashboard_data['service_status'] = self.get_service_status()
            dashboard_data['collection_summary'] = self.get_collection_summary(hours_back=6)
            return dashboard_data
        except Exception as e:
            logger.error(f"Error getting dashboard data: {e}")
            return {'error': str(e)}
    
    def run_continuous(self):
        """
        Run the service continuously until shutdown.
        
        This method blocks and runs the service until a shutdown signal is received.
        Useful for running as a daemon or service process.
        """
        if not self.start():
            logger.error("Failed to start service")
            return False
        
        try:
            logger.info("Service running continuously. Press Ctrl+C to stop.")
            
            # Main service loop
            while not self.shutdown_event.is_set():
                time.sleep(10)  # Check every 10 seconds
                
                # Perform periodic maintenance
                self._periodic_maintenance()
            
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received")
        except Exception as e:
            logger.error(f"Error in continuous run: {e}")
        finally:
            self.shutdown()
    
    def _periodic_maintenance(self):
        """Perform periodic maintenance tasks."""
        try:
            # This could include tasks like:
            # - Checking component health
            # - Cleaning up old data
            # - Monitoring resource usage
            # - Alerting on issues
            
            # For now, just check if all components are still running
            if not all([
                self.connection_pool and self.connection_pool.is_healthy(),
                self.job_queue and self.job_queue.is_running,
                self.collection_scheduler and self.collection_scheduler.is_running,
                self.status_monitor and self.status_monitor.is_running
            ]):
                logger.warning("One or more components are not healthy")
                
        except Exception as e:
            logger.error(f"Error in periodic maintenance: {e}")


def main():
    """Main entry point for the data collection service."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Snowflake Data Collection Service')
    parser.add_argument('--config', default='config', help='Configuration directory path')
    parser.add_argument('--daemon', action='store_true', help='Run as daemon process')
    parser.add_argument('--log-level', default='INFO', help='Logging level')
    
    args = parser.parse_args()
    
    # Configure logging level
    import logging
    logging.getLogger().setLevel(getattr(logging, args.log_level.upper()))
    
    # Create and run service
    service = DataCollectionService(config_path=args.config)
    
    if args.daemon:
        # Run continuously as daemon
        service.run_continuous()
    else:
        # Start service and return control
        if service.start():
            logger.info("Service started successfully. Use service.shutdown() to stop.")
            return service
        else:
            logger.error("Failed to start service")
            return None


if __name__ == '__main__':
    main()
