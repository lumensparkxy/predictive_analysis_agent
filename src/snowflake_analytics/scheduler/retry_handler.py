"""
Retry Handler - Handles retry logic for failed data collection operations.

This module provides sophisticated retry mechanisms with exponential backoff,
circuit breaker patterns, and adaptive retry strategies for different types of failures.
"""

import asyncio
import threading
import time
import random
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, field
from enum import Enum

from ..utils.logger import get_logger

logger = get_logger(__name__)


class RetryReason(Enum):
    """Types of retry reasons."""
    CONNECTION_ERROR = "connection_error"
    TIMEOUT = "timeout"
    RATE_LIMIT = "rate_limit"
    TEMPORARY_ERROR = "temporary_error"
    DATA_QUALITY_ERROR = "data_quality_error"
    AUTHENTICATION_ERROR = "authentication_error"
    RESOURCE_UNAVAILABLE = "resource_unavailable"
    UNKNOWN_ERROR = "unknown_error"


class RetryStrategy(Enum):
    """Retry strategy types."""
    FIXED_DELAY = "fixed_delay"
    EXPONENTIAL_BACKOFF = "exponential_backoff"
    LINEAR_BACKOFF = "linear_backoff"
    JITTERED_BACKOFF = "jittered_backoff"


@dataclass
class RetryConfig:
    """Configuration for retry behavior."""
    max_attempts: int = 3
    base_delay: int = 60  # seconds
    max_delay: int = 3600  # 1 hour
    strategy: RetryStrategy = RetryStrategy.EXPONENTIAL_BACKOFF
    backoff_multiplier: float = 2.0
    jitter_range: float = 0.1  # 10% jitter
    
    # Circuit breaker settings
    failure_threshold: int = 5  # failures before opening circuit
    recovery_timeout: int = 300  # 5 minutes before attempting recovery
    success_threshold: int = 3  # successes needed to close circuit
    
    # Reason-specific overrides
    reason_configs: Dict[RetryReason, Dict[str, Any]] = field(default_factory=dict)


@dataclass
class RetryAttempt:
    """Information about a retry attempt."""
    attempt_number: int
    scheduled_time: datetime
    actual_time: Optional[datetime] = None
    delay_used: int = 0
    reason: Optional[RetryReason] = None
    error_message: Optional[str] = None
    success: bool = False
    result: Optional[Any] = None


@dataclass
class RetryableTask:
    """A task that can be retried."""
    id: str
    name: str
    function: Callable
    args: tuple = field(default_factory=tuple)
    kwargs: Dict[str, Any] = field(default_factory=dict)
    config: RetryConfig = field(default_factory=RetryConfig)
    
    # State tracking
    created_at: datetime = field(default_factory=datetime.now)
    attempts: List[RetryAttempt] = field(default_factory=list)
    next_attempt_time: Optional[datetime] = None
    last_error: Optional[str] = None
    is_completed: bool = False
    is_cancelled: bool = False
    final_result: Optional[Any] = None


class CircuitBreakerState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Blocking requests
    HALF_OPEN = "half_open"  # Testing recovery


@dataclass
class CircuitBreaker:
    """Circuit breaker for a specific operation type."""
    name: str
    config: RetryConfig
    state: CircuitBreakerState = CircuitBreakerState.CLOSED
    failure_count: int = 0
    success_count: int = 0
    last_failure_time: Optional[datetime] = None
    last_success_time: Optional[datetime] = None
    opened_at: Optional[datetime] = None


class RetryHandler:
    """
    Advanced retry handler for data collection operations.
    
    Features:
    - Multiple retry strategies (exponential backoff, linear, jittered)
    - Circuit breaker pattern to prevent cascading failures
    - Adaptive retry based on error types
    - Task scheduling and execution management
    - Comprehensive retry statistics and monitoring
    """
    
    def __init__(self, default_config: Optional[RetryConfig] = None):
        """
        Initialize retry handler.
        
        Args:
            default_config: Default retry configuration
        """
        self.default_config = default_config or RetryConfig()
        
        # Task management
        self.pending_tasks: Dict[str, RetryableTask] = {}
        self.completed_tasks: Dict[str, RetryableTask] = {}
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        
        # Execution management
        self.is_running = False
        self.shutdown_event = threading.Event()
        self.retry_thread: Optional[threading.Thread] = None
        
        # Statistics
        self.stats = {
            'tasks_submitted': 0,
            'tasks_completed': 0,
            'tasks_failed': 0,
            'tasks_cancelled': 0,
            'total_attempts': 0,
            'successful_retries': 0,
            'circuit_breaker_trips': 0
        }
        
        # Default reason configurations
        self._setup_default_reason_configs()
    
    def _setup_default_reason_configs(self):
        """Setup default configurations for different retry reasons."""
        self.default_config.reason_configs = {
            RetryReason.CONNECTION_ERROR: {
                'max_attempts': 5,
                'base_delay': 30,
                'strategy': RetryStrategy.EXPONENTIAL_BACKOFF
            },
            RetryReason.TIMEOUT: {
                'max_attempts': 3,
                'base_delay': 60,
                'strategy': RetryStrategy.LINEAR_BACKOFF
            },
            RetryReason.RATE_LIMIT: {
                'max_attempts': 10,
                'base_delay': 300,  # 5 minutes
                'strategy': RetryStrategy.JITTERED_BACKOFF,
                'max_delay': 7200  # 2 hours
            },
            RetryReason.AUTHENTICATION_ERROR: {
                'max_attempts': 2,
                'base_delay': 120,
                'strategy': RetryStrategy.FIXED_DELAY
            },
            RetryReason.DATA_QUALITY_ERROR: {
                'max_attempts': 1,  # Don't retry data quality issues
                'base_delay': 0
            }
        }
    
    def start(self):
        """Start the retry handler."""
        if self.is_running:
            logger.warning("Retry handler is already running")
            return
        
        self.is_running = True
        self.shutdown_event.clear()
        
        # Start retry processing thread
        self.retry_thread = threading.Thread(
            target=self._retry_worker,
            name="RetryHandler",
            daemon=True
        )
        self.retry_thread.start()
        
        logger.info("Retry handler started")
    
    def stop(self, timeout: int = 30):
        """Stop the retry handler."""
        if not self.is_running:
            logger.warning("Retry handler is not running")
            return
        
        logger.info("Stopping retry handler...")
        
        # Signal shutdown
        self.shutdown_event.set()
        self.is_running = False
        
        # Wait for retry thread to finish
        if self.retry_thread and self.retry_thread.is_alive():
            self.retry_thread.join(timeout=timeout)
        
        logger.info("Retry handler stopped")
    
    def submit_task(
        self,
        task_id: str,
        name: str,
        function: Callable,
        args: tuple = (),
        kwargs: Optional[Dict[str, Any]] = None,
        config: Optional[RetryConfig] = None
    ) -> str:
        """
        Submit a task for retry handling.
        
        Args:
            task_id: Unique task identifier
            name: Human-readable task name
            function: Function to execute
            args: Positional arguments
            kwargs: Keyword arguments
            config: Custom retry configuration
        
        Returns:
            Task ID for tracking
        """
        if not self.is_running:
            raise RuntimeError("Retry handler is not running")
        
        if task_id in self.pending_tasks or task_id in self.completed_tasks:
            raise ValueError(f"Task already exists: {task_id}")
        
        # Create retryable task
        task = RetryableTask(
            id=task_id,
            name=name,
            function=function,
            args=args,
            kwargs=kwargs or {},
            config=config or self.default_config
        )
        
        # Schedule first attempt
        task.next_attempt_time = datetime.now()
        self.pending_tasks[task_id] = task
        self.stats['tasks_submitted'] += 1
        
        logger.info(f"Task submitted for retry handling: {name} (ID: {task_id})")
        return task_id
    
    def _retry_worker(self):
        """Main retry processing worker."""
        logger.info("Retry worker started")
        
        while not self.shutdown_event.is_set():
            try:
                current_time = datetime.now()
                ready_tasks = []
                
                # Find tasks ready for retry
                for task_id, task in list(self.pending_tasks.items()):
                    if task.next_attempt_time and task.next_attempt_time <= current_time:
                        ready_tasks.append(task)
                
                # Process ready tasks
                for task in ready_tasks:
                    self._process_task(task)
                
                # Sleep before next check
                time.sleep(5)
                
            except Exception as e:
                logger.error(f"Error in retry worker: {e}")
                time.sleep(10)
        
        logger.info("Retry worker stopped")
    
    def _process_task(self, task: RetryableTask):
        """Process a single retry task."""
        if task.is_cancelled:
            self._move_to_completed(task)
            return
        
        attempt_number = len(task.attempts) + 1
        
        # Check circuit breaker
        circuit_breaker = self._get_circuit_breaker(task.name)
        if not self._can_attempt(circuit_breaker):
            logger.warning(f"Circuit breaker open for {task.name}, postponing task {task.id}")
            self._schedule_next_attempt(task, RetryReason.RESOURCE_UNAVAILABLE)
            return
        
        # Create attempt record
        attempt = RetryAttempt(
            attempt_number=attempt_number,
            scheduled_time=task.next_attempt_time,
            actual_time=datetime.now()
        )
        task.attempts.append(attempt)
        self.stats['total_attempts'] += 1
        
        logger.info(f"Attempting task {task.name} (attempt {attempt_number}/{task.config.max_attempts})")
        
        try:
            # Execute task
            result = task.function(*task.args, **task.kwargs)
            
            # Mark as successful
            attempt.success = True
            attempt.result = result
            task.is_completed = True
            task.final_result = result
            
            # Update circuit breaker
            self._record_success(circuit_breaker)
            
            # Move to completed
            self._move_to_completed(task)
            self.stats['tasks_completed'] += 1
            
            if attempt_number > 1:
                self.stats['successful_retries'] += 1
            
            logger.info(f"Task completed successfully: {task.name} (attempt {attempt_number})")
            
        except Exception as e:
            # Mark as failed
            attempt.success = False
            attempt.error_message = str(e)
            task.last_error = str(e)
            
            # Determine retry reason
            retry_reason = self._classify_error(e)
            attempt.reason = retry_reason
            
            # Update circuit breaker
            self._record_failure(circuit_breaker)
            
            # Check if we should retry
            if attempt_number >= task.config.max_attempts:
                # Max attempts reached
                task.is_completed = True
                self._move_to_completed(task)
                self.stats['tasks_failed'] += 1
                logger.error(f"Task failed after {attempt_number} attempts: {task.name}")
            else:
                # Schedule next retry
                self._schedule_next_attempt(task, retry_reason)
    
    def _get_circuit_breaker(self, operation_name: str) -> CircuitBreaker:
        """Get or create circuit breaker for operation."""
        if operation_name not in self.circuit_breakers:
            self.circuit_breakers[operation_name] = CircuitBreaker(
                name=operation_name,
                config=self.default_config
            )
        return self.circuit_breakers[operation_name]
    
    def _can_attempt(self, circuit_breaker: CircuitBreaker) -> bool:
        """Check if circuit breaker allows attempts."""
        current_time = datetime.now()
        
        if circuit_breaker.state == CircuitBreakerState.CLOSED:
            return True
        elif circuit_breaker.state == CircuitBreakerState.HALF_OPEN:
            return True  # Allow limited attempts in half-open state
        elif circuit_breaker.state == CircuitBreakerState.OPEN:
            # Check if recovery timeout has passed
            if (circuit_breaker.opened_at and 
                current_time >= circuit_breaker.opened_at + 
                timedelta(seconds=circuit_breaker.config.recovery_timeout)):
                circuit_breaker.state = CircuitBreakerState.HALF_OPEN
                circuit_breaker.success_count = 0
                logger.info(f"Circuit breaker {circuit_breaker.name} moved to half-open state")
                return True
            return False
        
        return False
    
    def _record_success(self, circuit_breaker: CircuitBreaker):
        """Record a successful operation."""
        circuit_breaker.last_success_time = datetime.now()
        
        if circuit_breaker.state == CircuitBreakerState.HALF_OPEN:
            circuit_breaker.success_count += 1
            if circuit_breaker.success_count >= circuit_breaker.config.success_threshold:
                circuit_breaker.state = CircuitBreakerState.CLOSED
                circuit_breaker.failure_count = 0
                logger.info(f"Circuit breaker {circuit_breaker.name} closed after successful recovery")
        elif circuit_breaker.state == CircuitBreakerState.CLOSED:
            # Reset failure count on success
            circuit_breaker.failure_count = 0
    
    def _record_failure(self, circuit_breaker: CircuitBreaker):
        """Record a failed operation."""
        circuit_breaker.last_failure_time = datetime.now()
        circuit_breaker.failure_count += 1
        
        if (circuit_breaker.state == CircuitBreakerState.CLOSED and 
            circuit_breaker.failure_count >= circuit_breaker.config.failure_threshold):
            circuit_breaker.state = CircuitBreakerState.OPEN
            circuit_breaker.opened_at = datetime.now()
            self.stats['circuit_breaker_trips'] += 1
            logger.warning(f"Circuit breaker {circuit_breaker.name} opened after {circuit_breaker.failure_count} failures")
        elif circuit_breaker.state == CircuitBreakerState.HALF_OPEN:
            # Failed in half-open state, back to open
            circuit_breaker.state = CircuitBreakerState.OPEN
            circuit_breaker.opened_at = datetime.now()
            logger.warning(f"Circuit breaker {circuit_breaker.name} back to open state after failure in half-open")
    
    def _classify_error(self, error: Exception) -> RetryReason:
        """Classify error to determine retry strategy."""
        error_str = str(error).lower()
        
        if 'connection' in error_str or 'network' in error_str:
            return RetryReason.CONNECTION_ERROR
        elif 'timeout' in error_str:
            return RetryReason.TIMEOUT
        elif 'rate limit' in error_str or 'throttl' in error_str:
            return RetryReason.RATE_LIMIT
        elif 'auth' in error_str or 'permission' in error_str:
            return RetryReason.AUTHENTICATION_ERROR
        elif 'data quality' in error_str or 'validation' in error_str:
            return RetryReason.DATA_QUALITY_ERROR
        elif 'unavailable' in error_str or 'busy' in error_str:
            return RetryReason.RESOURCE_UNAVAILABLE
        elif 'temporary' in error_str:
            return RetryReason.TEMPORARY_ERROR
        else:
            return RetryReason.UNKNOWN_ERROR
    
    def _schedule_next_attempt(self, task: RetryableTask, reason: RetryReason):
        """Schedule the next retry attempt."""
        attempt_number = len(task.attempts)
        
        # Get configuration for this reason
        reason_config = task.config.reason_configs.get(reason, {})
        
        # Calculate delay
        delay = self._calculate_delay(
            attempt_number=attempt_number,
            base_delay=reason_config.get('base_delay', task.config.base_delay),
            strategy=reason_config.get('strategy', task.config.strategy),
            max_delay=reason_config.get('max_delay', task.config.max_delay),
            multiplier=task.config.backoff_multiplier,
            jitter_range=task.config.jitter_range
        )
        
        # Set next attempt time
        task.next_attempt_time = datetime.now() + timedelta(seconds=delay)
        
        if task.attempts:
            task.attempts[-1].delay_used = delay
        
        logger.info(f"Scheduled retry for task {task.name} in {delay} seconds (reason: {reason.value})")
    
    def _calculate_delay(
        self,
        attempt_number: int,
        base_delay: int,
        strategy: RetryStrategy,
        max_delay: int,
        multiplier: float,
        jitter_range: float
    ) -> int:
        """Calculate retry delay based on strategy."""
        if strategy == RetryStrategy.FIXED_DELAY:
            delay = base_delay
        elif strategy == RetryStrategy.LINEAR_BACKOFF:
            delay = base_delay * attempt_number
        elif strategy == RetryStrategy.EXPONENTIAL_BACKOFF:
            delay = base_delay * (multiplier ** (attempt_number - 1))
        elif strategy == RetryStrategy.JITTERED_BACKOFF:
            exponential_delay = base_delay * (multiplier ** (attempt_number - 1))
            jitter = exponential_delay * jitter_range * (2 * random.random() - 1)
            delay = exponential_delay + jitter
        else:
            delay = base_delay
        
        # Apply max delay limit
        delay = min(delay, max_delay)
        
        return int(delay)
    
    def _move_to_completed(self, task: RetryableTask):
        """Move task from pending to completed."""
        if task.id in self.pending_tasks:
            del self.pending_tasks[task.id]
        
        self.completed_tasks[task.id] = task
        
        # Maintain completed tasks limit
        max_completed = 500
        if len(self.completed_tasks) > max_completed:
            oldest_tasks = sorted(
                self.completed_tasks.values(),
                key=lambda t: t.created_at
            )[:len(self.completed_tasks) - max_completed]
            
            for old_task in oldest_tasks:
                del self.completed_tasks[old_task.id]
    
    def cancel_task(self, task_id: str) -> bool:
        """Cancel a pending task."""
        if task_id in self.pending_tasks:
            task = self.pending_tasks[task_id]
            task.is_cancelled = True
            self._move_to_completed(task)
            self.stats['tasks_cancelled'] += 1
            logger.info(f"Cancelled task: {task.name}")
            return True
        return False
    
    def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a specific task."""
        task = self.pending_tasks.get(task_id) or self.completed_tasks.get(task_id)
        if not task:
            return None
        
        return {
            'id': task.id,
            'name': task.name,
            'is_completed': task.is_completed,
            'is_cancelled': task.is_cancelled,
            'created_at': task.created_at.isoformat(),
            'attempts': len(task.attempts),
            'max_attempts': task.config.max_attempts,
            'next_attempt_time': task.next_attempt_time.isoformat() if task.next_attempt_time else None,
            'last_error': task.last_error,
            'has_result': task.final_result is not None
        }
    
    def get_handler_status(self) -> Dict[str, Any]:
        """Get overall retry handler status."""
        return {
            'is_running': self.is_running,
            'pending_tasks': len(self.pending_tasks),
            'completed_tasks': len(self.completed_tasks),
            'circuit_breakers': {
                name: {
                    'state': cb.state.value,
                    'failure_count': cb.failure_count,
                    'success_count': cb.success_count
                }
                for name, cb in self.circuit_breakers.items()
            },
            'statistics': self.stats.copy()
        }
    
    def reset_circuit_breaker(self, operation_name: str) -> bool:
        """Manually reset a circuit breaker."""
        if operation_name in self.circuit_breakers:
            cb = self.circuit_breakers[operation_name]
            cb.state = CircuitBreakerState.CLOSED
            cb.failure_count = 0
            cb.success_count = 0
            cb.opened_at = None
            logger.info(f"Circuit breaker reset: {operation_name}")
            return True
        return False
