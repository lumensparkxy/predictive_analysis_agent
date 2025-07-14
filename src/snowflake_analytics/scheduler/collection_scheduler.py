"""
Collection Scheduler - Main scheduler for automated data collection.

This module provides configurable scheduling for data collection operations
with support for different intervals, parallel execution, and monitoring.
"""

import threading
import time
# Note: schedule library would be used for production
# For now, using simple time-based scheduling
# import schedule
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field

from ..connectors.connection_pool import ConnectionPool
from ..storage.sqlite_store import SQLiteStore
from ..data_collection.usage_collector import UsageCollector
from ..data_collection.query_metrics import QueryMetricsCollector
from ..data_collection.warehouse_metrics import WarehouseMetricsCollector
from ..data_collection.user_activity import UserActivityCollector
from ..data_collection.cost_collector import CostCollector
from ..validation.schema_validator import SchemaValidator
from ..validation.quality_checks import DataQualityChecker
from ..utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ScheduledJob:
    """Definition of a scheduled collection job."""
    name: str
    collector_class: str
    method_name: str
    schedule_pattern: str  # e.g., "every(15).minutes", "every().hour"
    enabled: bool = True
    last_run: Optional[datetime] = None
    next_run: Optional[datetime] = None
    run_count: int = 0
    error_count: int = 0
    last_error: Optional[str] = None
    config: Dict[str, Any] = field(default_factory=dict)


class CollectionScheduler:
    """
    Main scheduler for automated data collection operations.
    
    Features:
    - Configurable collection intervals (15 minutes, hourly, daily)
    - Parallel collection for different data types
    - Job monitoring and status tracking
    - Error handling and retry logic
    - Performance monitoring and reporting
    """
    
    def __init__(
        self,
        connection_pool: ConnectionPool,
        storage: SQLiteStore,
        scheduler_config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize collection scheduler.
        
        Args:
            connection_pool: Pool of Snowflake connections
            storage: Storage backend for collected data
            scheduler_config: Configuration for scheduler behavior
        """
        self.connection_pool = connection_pool
        self.storage = storage
        self.config = scheduler_config or {}
        
        # Scheduler settings
        self.max_workers = self.config.get('max_workers', 3)
        self.job_timeout = self.config.get('job_timeout_seconds', 1800)  # 30 minutes
        self.enable_monitoring = self.config.get('enable_monitoring', True)
        
        # Job management
        self.scheduled_jobs: Dict[str, ScheduledJob] = {}
        self.running_jobs: Dict[str, threading.Thread] = {}
        self.job_executor = ThreadPoolExecutor(max_workers=self.max_workers)
        
        # Scheduler state
        self.is_running = False
        self.scheduler_thread: Optional[threading.Thread] = None
        self.shutdown_event = threading.Event()
        
        # Initialize components
        self.validators = {
            'schema': SchemaValidator(),
            'quality': DataQualityChecker()
        }
        
        # Initialize collectors
        self._initialize_collectors()
        
        # Setup default collection jobs
        self._setup_default_jobs()
    
    def _initialize_collectors(self):
        """Initialize data collectors."""
        self.collectors = {
            'usage': UsageCollector(self.connection_pool, self.storage),
            'query_metrics': QueryMetricsCollector(self.connection_pool, self.storage),
            'warehouse_metrics': WarehouseMetricsCollector(self.connection_pool, self.storage),
            'user_activity': UserActivityCollector(self.connection_pool, self.storage),
            'cost': CostCollector(self.connection_pool, self.storage)
        }
    
    def _setup_default_jobs(self):
        """Setup default collection jobs based on configuration."""
        default_jobs = [
            ScheduledJob(
                name='usage_collection',
                collector_class='usage',
                method_name='collect_all_usage_data',
                schedule_pattern='every(15).minutes',
                config={'force_full_collection': False}
            ),
            ScheduledJob(
                name='query_metrics_collection',
                collector_class='query_metrics',
                method_name='collect_query_metrics',
                schedule_pattern='every().hour',
                config={'hours_back': 2, 'include_query_text': False}
            ),
            ScheduledJob(
                name='warehouse_metrics_collection',
                collector_class='warehouse_metrics',
                method_name='collect_warehouse_metrics',
                schedule_pattern='every().hour',
                config={'days_back': 1}
            ),
            ScheduledJob(
                name='user_activity_collection',
                collector_class='user_activity',
                method_name='collect_user_activity',
                schedule_pattern='every(2).hours',
                config={'days_back': 1}
            ),
            ScheduledJob(
                name='cost_collection',
                collector_class='cost',
                method_name='collect_cost_data',
                schedule_pattern='every(6).hours',
                config={'days_back': 1}
            )
        ]
        
        for job in default_jobs:
            self.add_job(job)
    
    def add_job(self, job: ScheduledJob):
        """Add a scheduled job."""
        self.scheduled_jobs[job.name] = job
        
        # Parse and schedule the job
        self._schedule_job(job)
        
        logger.info(f"Added scheduled job: {job.name} ({job.schedule_pattern})")
    
    def _schedule_job(self, job: ScheduledJob):
        """Schedule a job with custom time-based scheduling."""
        if not job.enabled:
            return
        
        # Calculate next run time based on schedule pattern
        try:
            now = datetime.now()
            
            if 'every(15).minutes' in job.schedule_pattern:
                # Schedule for next 15-minute interval
                minutes = 15 - (now.minute % 15)
                next_run = now.replace(second=0, microsecond=0) + timedelta(minutes=minutes)
            elif 'every().hour' in job.schedule_pattern:
                # Schedule for next hour
                next_run = now.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
            elif 'every(2).hours' in job.schedule_pattern:
                # Schedule for next 2-hour interval
                hours_to_add = 2 - (now.hour % 2)
                next_run = now.replace(minute=0, second=0, microsecond=0) + timedelta(hours=hours_to_add)
            elif 'every(6).hours' in job.schedule_pattern:
                # Schedule for next 6-hour interval
                hours_to_add = 6 - (now.hour % 6)
                next_run = now.replace(minute=0, second=0, microsecond=0) + timedelta(hours=hours_to_add)
            elif 'every().day' in job.schedule_pattern:
                # Schedule for next day at midnight
                next_run = (now + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
            else:
                logger.warning(f"Unknown schedule pattern for job {job.name}: {job.schedule_pattern}")
                return
            
            job.next_run = next_run
            logger.info(f"Job {job.name} scheduled for {next_run}")
                
        except Exception as e:
            logger.error(f"Failed to schedule job {job.name}: {e}")
    
    def _execute_job_wrapper(self, job_name: str):
        """Wrapper for job execution with error handling."""
        try:
            self._execute_job(job_name)
        except Exception as e:
            logger.error(f"Job {job_name} failed: {e}")
            if job_name in self.scheduled_jobs:
                self.scheduled_jobs[job_name].error_count += 1
                self.scheduled_jobs[job_name].last_error = str(e)
    
    def _execute_job(self, job_name: str):
        """Execute a scheduled job."""
        if job_name not in self.scheduled_jobs:
            logger.error(f"Job not found: {job_name}")
            return
        
        job = self.scheduled_jobs[job_name]
        
        if not job.enabled:
            logger.debug(f"Job {job_name} is disabled, skipping")
            return
        
        # Check if job is already running
        if job_name in self.running_jobs and self.running_jobs[job_name].is_alive():
            logger.warning(f"Job {job_name} is already running, skipping this execution")
            return
        
        logger.info(f"Starting job: {job_name}")
        start_time = datetime.now()
        
        try:
            # Get collector and method
            collector = self.collectors.get(job.collector_class)
            if not collector:
                raise ValueError(f"Collector not found: {job.collector_class}")
            
            method = getattr(collector, job.method_name)
            if not method:
                raise ValueError(f"Method not found: {job.method_name}")
            
            # Execute the collection
            if job.config:
                result = method(**job.config)
            else:
                result = method()
            
            # Validate collected data if successful
            if result.get('success', False) and self.enable_monitoring:
                self._validate_collection_result(job_name, result)
            
            # Update job status
            job.last_run = start_time
            job.run_count += 1
            job.last_error = None
            
            execution_time = (datetime.now() - start_time).total_seconds()
            logger.info(f"Job {job_name} completed successfully in {execution_time:.1f}s")
            
        except Exception as e:
            job.error_count += 1
            job.last_error = str(e)
            logger.error(f"Job {job_name} failed: {e}")
            raise
        
        finally:
            # Clean up running job tracking
            if job_name in self.running_jobs:
                del self.running_jobs[job_name]
    
    def _validate_collection_result(self, job_name: str, result: Dict[str, Any]):
        """Validate collection result for quality and consistency."""
        try:
            # This would integrate with validation components
            records_collected = result.get('records_collected', 0)
            
            if records_collected == 0:
                logger.warning(f"Job {job_name} collected 0 records")
            elif records_collected > 1000000:  # > 1M records
                logger.warning(f"Job {job_name} collected unusually high number of records: {records_collected}")
            
            # Additional validation could be added here
            
        except Exception as e:
            logger.warning(f"Failed to validate collection result for job {job_name}: {e}")
    
    def start_scheduler(self):
        """Start the scheduler."""
        if self.is_running:
            logger.warning("Scheduler is already running")
            return
        
        self.is_running = True
        self.shutdown_event.clear()
        
        # Start scheduler thread
        self.scheduler_thread = threading.Thread(
            target=self._scheduler_worker,
            name="CollectionScheduler",
            daemon=True
        )
        self.scheduler_thread.start()
        
        logger.info("Collection scheduler started")
    
    def _scheduler_worker(self):
        """Main scheduler worker thread."""
        logger.info("Scheduler worker thread started")
        
        while not self.shutdown_event.is_set():
            try:
                current_time = datetime.now()
                
                # Check for jobs that need to run
                for job_name, job in list(self.scheduled_jobs.items()):
                    if (job.enabled and job.next_run and 
                        current_time >= job.next_run and
                        job_name not in self.running_jobs):
                        
                        # Start job execution
                        job_thread = threading.Thread(
                            target=self._execute_job_wrapper,
                            args=(job_name,),
                            name=f"Job-{job_name}",
                            daemon=True
                        )
                        job_thread.start()
                        self.running_jobs[job_name] = job_thread
                        
                        # Schedule next run
                        self._schedule_job(job)
                
                # Clean up finished job threads
                finished_jobs = []
                for job_name, thread in self.running_jobs.items():
                    if not thread.is_alive():
                        finished_jobs.append(job_name)
                
                for job_name in finished_jobs:
                    del self.running_jobs[job_name]
                
                # Sleep for a short interval
                time.sleep(10)  # Check every 10 seconds
                
            except Exception as e:
                logger.error(f"Error in scheduler worker: {e}")
                time.sleep(30)  # Wait longer on error
        
        logger.info("Scheduler worker thread stopped")
    
    def stop_scheduler(self):
        """Stop the scheduler."""
        if not self.is_running:
            logger.warning("Scheduler is not running")
            return
        
        logger.info("Stopping collection scheduler...")
        
        # Signal shutdown
        self.shutdown_event.set()
        self.is_running = False
        
        # Wait for scheduler thread to finish
        if self.scheduler_thread and self.scheduler_thread.is_alive():
            self.scheduler_thread.join(timeout=30)
        
        # Wait for running jobs to complete
        self._wait_for_jobs_completion(timeout=60)
        
        # Shutdown job executor
        self.job_executor.shutdown(wait=True)
        
        logger.info("Collection scheduler stopped")
    
    def _wait_for_jobs_completion(self, timeout: int = 60):
        """Wait for all running jobs to complete."""
        start_time = time.time()
        
        while self.running_jobs and (time.time() - start_time) < timeout:
            logger.info(f"Waiting for {len(self.running_jobs)} jobs to complete...")
            time.sleep(5)
        
        if self.running_jobs:
            logger.warning(f"Timeout waiting for jobs to complete: {list(self.running_jobs.keys())}")
    
    def get_job_status(self, job_name: Optional[str] = None) -> Dict[str, Any]:
        """Get status of scheduled jobs."""
        if job_name:
            if job_name not in self.scheduled_jobs:
                return {'error': f'Job not found: {job_name}'}
            
            job = self.scheduled_jobs[job_name]
            return {
                'name': job.name,
                'enabled': job.enabled,
                'schedule_pattern': job.schedule_pattern,
                'last_run': job.last_run.isoformat() if job.last_run else None,
                'next_run': job.next_run.isoformat() if job.next_run else None,
                'run_count': job.run_count,
                'error_count': job.error_count,
                'last_error': job.last_error,
                'is_running': job_name in self.running_jobs
            }
        else:
            # Return status for all jobs
            return {
                'scheduler_running': self.is_running,
                'total_jobs': len(self.scheduled_jobs),
                'enabled_jobs': sum(1 for job in self.scheduled_jobs.values() if job.enabled),
                'running_jobs': len(self.running_jobs),
                'jobs': {name: self.get_job_status(name) for name in self.scheduled_jobs.keys()}
            }
    
    def enable_job(self, job_name: str):
        """Enable a scheduled job."""
        if job_name not in self.scheduled_jobs:
            raise ValueError(f"Job not found: {job_name}")
        
        self.scheduled_jobs[job_name].enabled = True
        self._schedule_job(self.scheduled_jobs[job_name])
        logger.info(f"Enabled job: {job_name}")
    
    def disable_job(self, job_name: str):
        """Disable a scheduled job."""
        if job_name not in self.scheduled_jobs:
            raise ValueError(f"Job not found: {job_name}")
        
        self.scheduled_jobs[job_name].enabled = False
        self.scheduled_jobs[job_name].next_run = None
        
        logger.info(f"Disabled job: {job_name}")
    
    def run_job_now(self, job_name: str) -> Dict[str, Any]:
        """Run a job immediately (outside of schedule)."""
        if job_name not in self.scheduled_jobs:
            return {'success': False, 'error': f'Job not found: {job_name}'}
        
        logger.info(f"Running job immediately: {job_name}")
        
        try:
            # Submit job to executor
            future = self.job_executor.submit(self._execute_job, job_name)
            
            # Wait for completion with timeout
            result = future.result(timeout=self.job_timeout)
            
            return {
                'success': True,
                'message': f'Job {job_name} completed successfully',
                'job_status': self.get_job_status(job_name)
            }
            
        except Exception as e:
            logger.error(f"Failed to run job {job_name}: {e}")
            return {
                'success': False,
                'error': str(e),
                'job_status': self.get_job_status(job_name)
            }
    
    def get_collection_summary(self, hours_back: int = 24) -> Dict[str, Any]:
        """Get summary of collection activities."""
        cutoff_time = datetime.now() - timedelta(hours=hours_back)
        
        summary = {
            'time_window_hours': hours_back,
            'total_jobs': len(self.scheduled_jobs),
            'active_jobs': sum(1 for job in self.scheduled_jobs.values() if job.enabled),
            'jobs_with_errors': sum(1 for job in self.scheduled_jobs.values() if job.error_count > 0),
            'recent_runs': 0,
            'recent_errors': 0,
            'collection_health': 'unknown'
        }
        
        # Count recent runs and errors
        for job in self.scheduled_jobs.values():
            if job.last_run and job.last_run >= cutoff_time:
                summary['recent_runs'] += 1
                if job.last_error:
                    summary['recent_errors'] += 1
        
        # Determine collection health
        if summary['recent_errors'] == 0:
            summary['collection_health'] = 'healthy'
        elif summary['recent_errors'] / max(summary['recent_runs'], 1) < 0.1:
            summary['collection_health'] = 'warning'
        else:
            summary['collection_health'] = 'critical'
        
        return summary
