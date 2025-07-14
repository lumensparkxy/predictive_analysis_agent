"""
Job Queue Manager - Manages collection job queuing and execution.

This module provides a job queue system for managing data collection operations
with priority handling, retry logic, and concurrent execution control.
"""

import threading
import time
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Tuple
from queue import PriorityQueue, Empty
from concurrent.futures import ThreadPoolExecutor, Future, as_completed
from dataclasses import dataclass, field
from enum import Enum

from ..utils.logger import get_logger

logger = get_logger(__name__)


class JobStatus(Enum):
    """Job execution status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRYING = "retrying"
    CANCELLED = "cancelled"


class JobPriority(Enum):
    """Job priority levels."""
    LOW = 3
    NORMAL = 2
    HIGH = 1
    CRITICAL = 0


@dataclass
class QueuedJob:
    """Represents a job in the execution queue."""
    id: str
    name: str
    collector_class: str
    method_name: str
    args: Tuple = field(default_factory=tuple)
    kwargs: Dict[str, Any] = field(default_factory=dict)
    priority: JobPriority = JobPriority.NORMAL
    max_retries: int = 3
    retry_delay: int = 60  # seconds
    timeout: int = 1800  # 30 minutes
    
    # Status tracking
    status: JobStatus = JobStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    retry_count: int = 0
    last_error: Optional[str] = None
    result: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        """Initialize job with unique ID if not provided."""
        if not self.id:
            self.id = str(uuid.uuid4())
    
    def __lt__(self, other):
        """Compare jobs for priority queue ordering."""
        if not isinstance(other, QueuedJob):
            return NotImplemented
        
        # Lower priority value = higher priority in queue
        if self.priority.value != other.priority.value:
            return self.priority.value < other.priority.value
        
        # Same priority, order by creation time (FIFO)
        return self.created_at < other.created_at


class JobQueue:
    """
    Job queue manager for data collection operations.
    
    Features:
    - Priority-based job queuing
    - Concurrent job execution with limits
    - Automatic retry handling
    - Job status tracking and monitoring
    - Graceful shutdown with job completion
    """
    
    def __init__(
        self,
        max_workers: int = 3,
        max_queue_size: int = 100,
        default_timeout: int = 1800,
        enable_monitoring: bool = True
    ):
        """
        Initialize job queue manager.
        
        Args:
            max_workers: Maximum concurrent job executions
            max_queue_size: Maximum number of queued jobs
            default_timeout: Default job timeout in seconds
            enable_monitoring: Enable job monitoring and metrics
        """
        self.max_workers = max_workers
        self.max_queue_size = max_queue_size
        self.default_timeout = default_timeout
        self.enable_monitoring = enable_monitoring
        
        # Queue management
        self.job_queue = PriorityQueue(maxsize=max_queue_size)
        self.active_jobs: Dict[str, QueuedJob] = {}
        self.completed_jobs: Dict[str, QueuedJob] = {}
        self.job_futures: Dict[str, Future] = {}
        
        # Execution management
        self.executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="JobQueue")
        self.is_running = False
        self.shutdown_event = threading.Event()
        self.queue_worker_thread: Optional[threading.Thread] = None
        
        # Statistics
        self.stats = {
            'jobs_submitted': 0,
            'jobs_completed': 0,
            'jobs_failed': 0,
            'jobs_retried': 0,
            'jobs_cancelled': 0,
            'total_execution_time': 0,
            'average_execution_time': 0
        }
        
        # Collectors reference (will be set by scheduler)
        self.collectors: Dict[str, Any] = {}
    
    def set_collectors(self, collectors: Dict[str, Any]):
        """Set collectors for job execution."""
        self.collectors = collectors
    
    def start(self):
        """Start the job queue worker."""
        if self.is_running:
            logger.warning("Job queue is already running")
            return
        
        self.is_running = True
        self.shutdown_event.clear()
        
        # Start queue worker thread
        self.queue_worker_thread = threading.Thread(
            target=self._queue_worker,
            name="JobQueueWorker",
            daemon=True
        )
        self.queue_worker_thread.start()
        
        logger.info(f"Job queue started with {self.max_workers} workers")
    
    def stop(self, timeout: int = 60):
        """Stop the job queue and wait for completion."""
        if not self.is_running:
            logger.warning("Job queue is not running")
            return
        
        logger.info("Stopping job queue...")
        
        # Signal shutdown
        self.shutdown_event.set()
        self.is_running = False
        
        # Wait for queue worker to finish
        if self.queue_worker_thread and self.queue_worker_thread.is_alive():
            self.queue_worker_thread.join(timeout=30)
        
        # Cancel pending jobs
        self._cancel_pending_jobs()
        
        # Wait for active jobs to complete
        self._wait_for_active_jobs(timeout=timeout)
        
        # Shutdown executor
        self.executor.shutdown(wait=True)
        
        logger.info("Job queue stopped")
    
    def submit_job(
        self,
        name: str,
        collector_class: str,
        method_name: str,
        args: Tuple = (),
        kwargs: Optional[Dict[str, Any]] = None,
        priority: JobPriority = JobPriority.NORMAL,
        max_retries: int = 3,
        timeout: Optional[int] = None,
        job_id: Optional[str] = None
    ) -> str:
        """
        Submit a job to the queue.
        
        Args:
            name: Human-readable job name
            collector_class: Name of collector class
            method_name: Method to call on collector
            args: Positional arguments for method
            kwargs: Keyword arguments for method
            priority: Job priority level
            max_retries: Maximum retry attempts
            timeout: Job timeout in seconds
            job_id: Optional custom job ID
        
        Returns:
            Job ID for tracking
        """
        if not self.is_running:
            raise RuntimeError("Job queue is not running")
        
        if self.job_queue.full():
            raise RuntimeError("Job queue is full")
        
        # Create job
        job = QueuedJob(
            id=job_id or str(uuid.uuid4()),
            name=name,
            collector_class=collector_class,
            method_name=method_name,
            args=args,
            kwargs=kwargs or {},
            priority=priority,
            max_retries=max_retries,
            timeout=timeout or self.default_timeout
        )
        
        # Add to queue
        try:
            self.job_queue.put_nowait(job)
            self.stats['jobs_submitted'] += 1
            
            logger.info(f"Job submitted: {job.name} (ID: {job.id}, Priority: {priority.name})")
            return job.id
            
        except Exception as e:
            logger.error(f"Failed to submit job {job.name}: {e}")
            raise
    
    def _queue_worker(self):
        """Main queue worker thread."""
        logger.info("Job queue worker started")
        
        while not self.shutdown_event.is_set():
            try:
                # Get next job from queue
                try:
                    job = self.job_queue.get(timeout=1.0)
                except Empty:
                    continue
                
                # Check if we can start the job
                if len(self.active_jobs) >= self.max_workers:
                    # Put job back and wait
                    self.job_queue.put(job)
                    time.sleep(0.1)
                    continue
                
                # Start job execution
                self._start_job(job)
                
            except Exception as e:
                logger.error(f"Error in job queue worker: {e}")
                time.sleep(1)
        
        logger.info("Job queue worker stopped")
    
    def _start_job(self, job: QueuedJob):
        """Start executing a job."""
        try:
            # Update job status
            job.status = JobStatus.RUNNING
            job.started_at = datetime.now()
            self.active_jobs[job.id] = job
            
            # Submit to thread pool
            future = self.executor.submit(self._execute_job, job)
            self.job_futures[job.id] = future
            
            # Add completion callback
            future.add_done_callback(lambda f: self._job_completed(job.id, f))
            
            logger.debug(f"Started job: {job.name} (ID: {job.id})")
            
        except Exception as e:
            logger.error(f"Failed to start job {job.name}: {e}")
            job.status = JobStatus.FAILED
            job.last_error = str(e)
            self._move_to_completed(job)
    
    def _execute_job(self, job: QueuedJob) -> Dict[str, Any]:
        """Execute a job and return result."""
        try:
            # Get collector
            collector = self.collectors.get(job.collector_class)
            if not collector:
                raise ValueError(f"Collector not found: {job.collector_class}")
            
            # Get method
            method = getattr(collector, job.method_name)
            if not method:
                raise ValueError(f"Method not found: {job.method_name}")
            
            # Execute with timeout
            start_time = time.time()
            
            if job.kwargs:
                result = method(*job.args, **job.kwargs)
            else:
                result = method(*job.args)
            
            execution_time = time.time() - start_time
            
            # Ensure result is a dictionary
            if not isinstance(result, dict):
                result = {'result': result, 'success': True}
            
            result['execution_time'] = execution_time
            result['job_id'] = job.id
            
            return result
            
        except Exception as e:
            logger.error(f"Job execution failed for {job.name}: {e}")
            return {
                'success': False,
                'error': str(e),
                'job_id': job.id
            }
    
    def _job_completed(self, job_id: str, future: Future):
        """Handle job completion."""
        try:
            job = self.active_jobs.get(job_id)
            if not job:
                logger.warning(f"Job not found in active jobs: {job_id}")
                return
            
            # Get result
            try:
                result = future.result()
                job.result = result
                
                if result.get('success', True):
                    job.status = JobStatus.COMPLETED
                    self.stats['jobs_completed'] += 1
                    logger.info(f"Job completed successfully: {job.name} (ID: {job_id})")
                else:
                    # Job returned failure
                    job.status = JobStatus.FAILED
                    job.last_error = result.get('error', 'Unknown error')
                    self._handle_job_failure(job)
                    
            except Exception as e:
                # Job raised exception
                job.status = JobStatus.FAILED
                job.last_error = str(e)
                self._handle_job_failure(job)
            
            # Update completion time
            job.completed_at = datetime.now()
            
            # Update statistics
            if job.started_at:
                execution_time = (job.completed_at - job.started_at).total_seconds()
                self.stats['total_execution_time'] += execution_time
                
                completed_jobs = self.stats['jobs_completed'] + self.stats['jobs_failed']
                if completed_jobs > 0:
                    self.stats['average_execution_time'] = self.stats['total_execution_time'] / completed_jobs
            
            # Move to completed
            self._move_to_completed(job)
            
        except Exception as e:
            logger.error(f"Error handling job completion for {job_id}: {e}")
    
    def _handle_job_failure(self, job: QueuedJob):
        """Handle job failure and retry logic."""
        if job.retry_count < job.max_retries:
            # Schedule retry
            job.retry_count += 1
            job.status = JobStatus.RETRYING
            self.stats['jobs_retried'] += 1
            
            logger.info(f"Scheduling retry for job {job.name} (attempt {job.retry_count}/{job.max_retries})")
            
            # Reset timing
            job.started_at = None
            job.completed_at = None
            
            # Add back to queue with delay
            retry_thread = threading.Thread(
                target=self._schedule_retry,
                args=(job,),
                daemon=True
            )
            retry_thread.start()
            
        else:
            # Max retries reached
            self.stats['jobs_failed'] += 1
            logger.error(f"Job failed after {job.max_retries} retries: {job.name}")
    
    def _schedule_retry(self, job: QueuedJob):
        """Schedule a job retry with delay."""
        try:
            # Wait for retry delay
            time.sleep(job.retry_delay)
            
            # Reset status and re-queue
            job.status = JobStatus.PENDING
            self.job_queue.put(job)
            
            logger.debug(f"Job re-queued for retry: {job.name}")
            
        except Exception as e:
            logger.error(f"Failed to schedule retry for job {job.name}: {e}")
            job.status = JobStatus.FAILED
            self._move_to_completed(job)
    
    def _move_to_completed(self, job: QueuedJob):
        """Move job from active to completed."""
        try:
            # Remove from active jobs
            if job.id in self.active_jobs:
                del self.active_jobs[job.id]
            
            # Remove future
            if job.id in self.job_futures:
                del self.job_futures[job.id]
            
            # Add to completed (with limit)
            self.completed_jobs[job.id] = job
            
            # Maintain completed jobs limit
            max_completed = 1000
            if len(self.completed_jobs) > max_completed:
                # Remove oldest completed jobs
                oldest_jobs = sorted(
                    self.completed_jobs.values(),
                    key=lambda j: j.completed_at or datetime.min
                )[:len(self.completed_jobs) - max_completed]
                
                for old_job in oldest_jobs:
                    del self.completed_jobs[old_job.id]
            
        except Exception as e:
            logger.error(f"Error moving job to completed: {e}")
    
    def _cancel_pending_jobs(self):
        """Cancel all pending jobs in queue."""
        cancelled_count = 0
        
        while not self.job_queue.empty():
            try:
                job = self.job_queue.get_nowait()
                job.status = JobStatus.CANCELLED
                job.completed_at = datetime.now()
                self.completed_jobs[job.id] = job
                cancelled_count += 1
            except Empty:
                break
        
        self.stats['jobs_cancelled'] += cancelled_count
        
        if cancelled_count > 0:
            logger.info(f"Cancelled {cancelled_count} pending jobs")
    
    def _wait_for_active_jobs(self, timeout: int = 60):
        """Wait for active jobs to complete."""
        start_time = time.time()
        
        while self.active_jobs and (time.time() - start_time) < timeout:
            logger.info(f"Waiting for {len(self.active_jobs)} active jobs to complete...")
            time.sleep(2)
        
        if self.active_jobs:
            logger.warning(f"Timeout waiting for jobs: {list(self.active_jobs.keys())}")
            
            # Cancel remaining jobs
            for job_id in list(self.active_jobs.keys()):
                if job_id in self.job_futures:
                    self.job_futures[job_id].cancel()
    
    def get_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a specific job."""
        # Check active jobs
        if job_id in self.active_jobs:
            job = self.active_jobs[job_id]
        elif job_id in self.completed_jobs:
            job = self.completed_jobs[job_id]
        else:
            return None
        
        return {
            'id': job.id,
            'name': job.name,
            'status': job.status.value,
            'priority': job.priority.name,
            'created_at': job.created_at.isoformat(),
            'started_at': job.started_at.isoformat() if job.started_at else None,
            'completed_at': job.completed_at.isoformat() if job.completed_at else None,
            'retry_count': job.retry_count,
            'max_retries': job.max_retries,
            'last_error': job.last_error,
            'has_result': job.result is not None
        }
    
    def get_queue_status(self) -> Dict[str, Any]:
        """Get overall queue status."""
        return {
            'is_running': self.is_running,
            'queue_size': self.job_queue.qsize(),
            'max_queue_size': self.max_queue_size,
            'active_jobs': len(self.active_jobs),
            'max_workers': self.max_workers,
            'completed_jobs': len(self.completed_jobs),
            'statistics': self.stats.copy()
        }
    
    def cancel_job(self, job_id: str) -> bool:
        """Cancel a specific job."""
        # Check if job is active
        if job_id in self.active_jobs:
            if job_id in self.job_futures:
                cancelled = self.job_futures[job_id].cancel()
                if cancelled:
                    job = self.active_jobs[job_id]
                    job.status = JobStatus.CANCELLED
                    job.completed_at = datetime.now()
                    self._move_to_completed(job)
                    self.stats['jobs_cancelled'] += 1
                    logger.info(f"Cancelled active job: {job.name}")
                    return True
                else:
                    logger.warning(f"Could not cancel running job: {job_id}")
                    return False
        
        # Check if job is in queue
        temp_jobs = []
        found = False
        
        while not self.job_queue.empty():
            try:
                job = self.job_queue.get_nowait()
                if job.id == job_id:
                    job.status = JobStatus.CANCELLED
                    job.completed_at = datetime.now()
                    self.completed_jobs[job.id] = job
                    self.stats['jobs_cancelled'] += 1
                    found = True
                    logger.info(f"Cancelled queued job: {job.name}")
                    break
                else:
                    temp_jobs.append(job)
            except Empty:
                break
        
        # Put remaining jobs back
        for job in temp_jobs:
            self.job_queue.put(job)
        
        return found
    
    def clear_completed_jobs(self, older_than_hours: int = 24):
        """Clear completed jobs older than specified hours."""
        cutoff_time = datetime.now() - timedelta(hours=older_than_hours)
        
        to_remove = []
        for job_id, job in self.completed_jobs.items():
            if job.completed_at and job.completed_at < cutoff_time:
                to_remove.append(job_id)
        
        for job_id in to_remove:
            del self.completed_jobs[job_id]
        
        if to_remove:
            logger.info(f"Cleared {len(to_remove)} completed jobs older than {older_than_hours} hours")
