"""
Pipeline Scheduler

Advanced scheduling system for ML pipeline execution with support for
cron-like scheduling, dependency management, retry mechanisms, and automated execution.

Key capabilities:
- Flexible scheduling with cron expressions
- Pipeline dependency management
- Intelligent retry mechanisms with backoff
- Resource-aware scheduling
- Execution queue management
- Schedule conflict detection
- Performance-based optimization
- Integration with monitoring system
"""

import os
import json
import time
import threading
import schedule
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable, Union
from pathlib import Path
from dataclasses import dataclass, asdict
from enum import Enum
from croniter import croniter
import queue

import pandas as pd
import numpy as np

from ...config.settings import SnowflakeSettings
from ...utils.logger import SnowflakeLogger
from .ml_pipeline_orchestrator import MLPipelineOrchestrator, PipelineExecution, PipelineStage, PipelineStatus


class ScheduleType(Enum):
    """Schedule type enumeration"""
    CRON = "cron"
    INTERVAL = "interval"
    ONE_TIME = "one_time"
    MANUAL = "manual"


class ScheduleStatus(Enum):
    """Schedule status enumeration"""
    ACTIVE = "active"
    INACTIVE = "inactive"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"


class RetryStrategy(Enum):
    """Retry strategy enumeration"""
    FIXED_DELAY = "fixed_delay"
    EXPONENTIAL_BACKOFF = "exponential_backoff"
    LINEAR_BACKOFF = "linear_backoff"
    NO_RETRY = "no_retry"


@dataclass
class ScheduleConfig:
    """Schedule configuration"""
    schedule_id: str
    name: str
    description: str
    schedule_type: ScheduleType
    schedule_expression: str  # Cron expression or interval
    enabled: bool
    start_date: Optional[datetime]
    end_date: Optional[datetime]
    max_executions: Optional[int]
    execution_timeout_minutes: int
    retry_strategy: RetryStrategy
    max_retries: int
    retry_delay_minutes: int
    priority: int  # 1-10, higher is more important
    resource_requirements: Dict[str, Any]
    dependencies: List[str]  # Other schedule IDs
    execution_config: Dict[str, Any]
    notification_config: Dict[str, Any]


@dataclass
class ScheduledExecution:
    """Scheduled execution tracking"""
    execution_id: str
    schedule_id: str
    scheduled_time: datetime
    actual_start_time: Optional[datetime]
    actual_end_time: Optional[datetime]
    status: PipelineStatus
    attempt_number: int
    next_retry_time: Optional[datetime]
    pipeline_execution: Optional[PipelineExecution]
    error_message: Optional[str]
    resource_allocation: Dict[str, Any]


@dataclass
class ScheduleHistory:
    """Schedule execution history"""
    schedule_id: str
    total_executions: int
    successful_executions: int
    failed_executions: int
    average_duration_minutes: float
    last_execution_time: Optional[datetime]
    last_successful_execution: Optional[datetime]
    last_failed_execution: Optional[datetime]
    success_rate: float
    average_retry_count: float


class PipelineScheduler:
    """
    Advanced scheduling system for ML pipeline execution.
    
    Provides comprehensive scheduling capabilities with dependency management,
    retry mechanisms, resource management, and intelligent optimization.
    """
    
    def __init__(
        self,
        orchestrator: MLPipelineOrchestrator,
        config_path: Optional[str] = None,
        max_concurrent_executions: int = 3,
        schedule_check_interval_seconds: int = 30,
        enable_resource_monitoring: bool = True
    ):
        """Initialize pipeline scheduler"""
        
        # Core configuration
        self.settings = SnowflakeSettings(config_path)
        self.logger = SnowflakeLogger("PipelineScheduler").get_logger()
        self.orchestrator = orchestrator
        
        # Scheduler configuration
        self.max_concurrent_executions = max_concurrent_executions
        self.schedule_check_interval_seconds = schedule_check_interval_seconds
        self.enable_resource_monitoring = enable_resource_monitoring
        
        # Scheduling state
        self.schedules: Dict[str, ScheduleConfig] = {}
        self.scheduled_executions: List[ScheduledExecution] = []
        self.execution_queue: queue.PriorityQueue = queue.PriorityQueue()
        self.active_executions: Dict[str, ScheduledExecution] = {}
        self.schedule_history: Dict[str, ScheduleHistory] = {}
        
        # Scheduler thread management
        self.scheduler_thread: Optional[threading.Thread] = None
        self.execution_thread: Optional[threading.Thread] = None
        self.is_running = False
        
        # Resource management
        self.resource_allocations: Dict[str, Dict[str, Any]] = {}
        self.available_resources = self._get_available_resources()
        
        self.logger.info("PipelineScheduler initialized successfully")
    
    def start_scheduler(self):
        """Start the pipeline scheduler"""
        
        if self.is_running:
            self.logger.warning("Scheduler is already running")
            return
        
        self.is_running = True
        
        # Start scheduler thread
        self.scheduler_thread = threading.Thread(
            target=self._scheduler_loop,
            daemon=True
        )
        self.scheduler_thread.start()
        
        # Start execution thread
        self.execution_thread = threading.Thread(
            target=self._execution_loop,
            daemon=True
        )
        self.execution_thread.start()
        
        self.logger.info("Pipeline scheduler started")
    
    def stop_scheduler(self):
        """Stop the pipeline scheduler"""
        
        self.is_running = False
        
        # Wait for threads to finish
        if self.scheduler_thread:
            self.scheduler_thread.join(timeout=10)
        
        if self.execution_thread:
            self.execution_thread.join(timeout=10)
        
        self.logger.info("Pipeline scheduler stopped")
    
    def add_schedule(self, schedule_config: ScheduleConfig) -> bool:
        """Add new schedule configuration"""
        
        try:
            # Validate schedule configuration
            self._validate_schedule_config(schedule_config)
            
            # Check for conflicts
            if self._has_schedule_conflicts(schedule_config):
                self.logger.error(f"Schedule conflicts detected for: {schedule_config.schedule_id}")
                return False
            
            # Add schedule
            self.schedules[schedule_config.schedule_id] = schedule_config
            
            # Initialize history
            self.schedule_history[schedule_config.schedule_id] = ScheduleHistory(
                schedule_id=schedule_config.schedule_id,
                total_executions=0,
                successful_executions=0,
                failed_executions=0,
                average_duration_minutes=0.0,
                last_execution_time=None,
                last_successful_execution=None,
                last_failed_execution=None,
                success_rate=0.0,
                average_retry_count=0.0
            )
            
            self.logger.info(f"Schedule added: {schedule_config.schedule_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to add schedule {schedule_config.schedule_id}: {str(e)}")
            return False
    
    def remove_schedule(self, schedule_id: str) -> bool:
        """Remove schedule configuration"""
        
        if schedule_id not in self.schedules:
            self.logger.warning(f"Schedule not found: {schedule_id}")
            return False
        
        # Cancel any pending executions
        self._cancel_pending_executions(schedule_id)
        
        # Remove schedule
        del self.schedules[schedule_id]
        
        self.logger.info(f"Schedule removed: {schedule_id}")
        return True
    
    def pause_schedule(self, schedule_id: str) -> bool:
        """Pause schedule execution"""
        
        if schedule_id in self.schedules:
            self.schedules[schedule_id].enabled = False
            self.logger.info(f"Schedule paused: {schedule_id}")
            return True
        
        return False
    
    def resume_schedule(self, schedule_id: str) -> bool:
        """Resume schedule execution"""
        
        if schedule_id in self.schedules:
            self.schedules[schedule_id].enabled = True
            self.logger.info(f"Schedule resumed: {schedule_id}")
            return True
        
        return False
    
    def trigger_manual_execution(
        self,
        schedule_id: str,
        execution_config_override: Optional[Dict[str, Any]] = None
    ) -> Optional[str]:
        """Trigger manual pipeline execution"""
        
        if schedule_id not in self.schedules:
            self.logger.error(f"Schedule not found: {schedule_id}")
            return None
        
        schedule_config = self.schedules[schedule_id]
        
        # Create scheduled execution
        execution_id = f"manual_{schedule_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        scheduled_execution = ScheduledExecution(
            execution_id=execution_id,
            schedule_id=schedule_id,
            scheduled_time=datetime.now(),
            actual_start_time=None,
            actual_end_time=None,
            status=PipelineStatus.NOT_STARTED,
            attempt_number=1,
            next_retry_time=None,
            pipeline_execution=None,
            error_message=None,
            resource_allocation={}
        )
        
        # Queue for execution
        priority = schedule_config.priority
        self.execution_queue.put((priority, datetime.now(), scheduled_execution))
        
        self.logger.info(f"Manual execution queued: {execution_id}")
        return execution_id
    
    def _scheduler_loop(self):
        """Main scheduler loop"""
        
        while self.is_running:
            try:
                current_time = datetime.now()
                
                # Check each schedule
                for schedule_id, schedule_config in self.schedules.items():
                    if not schedule_config.enabled:
                        continue
                    
                    # Check if execution is due
                    if self._is_execution_due(schedule_config, current_time):
                        self._queue_execution(schedule_config, current_time)
                
                # Clean up completed executions
                self._cleanup_completed_executions()
                
                # Update schedule history
                self._update_schedule_history()
                
                # Sleep until next check
                time.sleep(self.schedule_check_interval_seconds)
                
            except Exception as e:
                self.logger.error(f"Error in scheduler loop: {str(e)}")
                time.sleep(self.schedule_check_interval_seconds)
    
    def _execution_loop(self):
        """Main execution loop"""
        
        while self.is_running:
            try:
                # Check if we can start new executions
                if len(self.active_executions) >= self.max_concurrent_executions:
                    time.sleep(5)
                    continue
                
                # Get next execution from queue (blocks if empty)
                try:
                    priority, queued_time, scheduled_execution = self.execution_queue.get(timeout=5)
                    
                    # Check if execution is still valid
                    if not self._is_execution_valid(scheduled_execution):
                        continue
                    
                    # Check resource availability
                    if not self._allocate_resources(scheduled_execution):
                        # Re-queue with lower priority
                        self.execution_queue.put((priority + 1, queued_time, scheduled_execution))
                        time.sleep(1)
                        continue
                    
                    # Start execution
                    self._start_execution(scheduled_execution)
                    
                except queue.Empty:
                    continue
                
            except Exception as e:
                self.logger.error(f"Error in execution loop: {str(e)}")
                time.sleep(5)
    
    def _is_execution_due(self, schedule_config: ScheduleConfig, current_time: datetime) -> bool:
        """Check if schedule execution is due"""
        
        # Check if schedule is within active period
        if schedule_config.start_date and current_time < schedule_config.start_date:
            return False
        
        if schedule_config.end_date and current_time > schedule_config.end_date:
            return False
        
        # Check execution count limit
        if schedule_config.max_executions:
            history = self.schedule_history.get(schedule_config.schedule_id)
            if history and history.total_executions >= schedule_config.max_executions:
                return False
        
        # Check schedule type
        if schedule_config.schedule_type == ScheduleType.ONE_TIME:
            history = self.schedule_history.get(schedule_config.schedule_id)
            return not history or history.total_executions == 0
        
        elif schedule_config.schedule_type == ScheduleType.CRON:
            return self._is_cron_due(schedule_config.schedule_expression, current_time)
        
        elif schedule_config.schedule_type == ScheduleType.INTERVAL:
            return self._is_interval_due(schedule_config, current_time)
        
        return False
    
    def _is_cron_due(self, cron_expression: str, current_time: datetime) -> bool:
        """Check if cron schedule is due"""
        
        try:
            cron = croniter(cron_expression, current_time)
            next_run = cron.get_prev(datetime)
            
            # Check if we've already scheduled this execution
            for execution in self.scheduled_executions:
                if abs((execution.scheduled_time - next_run).total_seconds()) < 60:
                    return False
            
            # Check if the next run time is within the last minute
            return (current_time - next_run).total_seconds() < 60
        
        except Exception as e:
            self.logger.error(f"Invalid cron expression {cron_expression}: {str(e)}")
            return False
    
    def _is_interval_due(self, schedule_config: ScheduleConfig, current_time: datetime) -> bool:
        """Check if interval schedule is due"""
        
        history = self.schedule_history.get(schedule_config.schedule_id)
        if not history or not history.last_execution_time:
            return True
        
        # Parse interval from expression (e.g., "30m", "1h", "2d")
        interval_seconds = self._parse_interval(schedule_config.schedule_expression)
        if interval_seconds is None:
            return False
        
        time_since_last = (current_time - history.last_execution_time).total_seconds()
        return time_since_last >= interval_seconds
    
    def _parse_interval(self, interval_expression: str) -> Optional[int]:
        """Parse interval expression to seconds"""
        
        try:
            if interval_expression.endswith('s'):
                return int(interval_expression[:-1])
            elif interval_expression.endswith('m'):
                return int(interval_expression[:-1]) * 60
            elif interval_expression.endswith('h'):
                return int(interval_expression[:-1]) * 3600
            elif interval_expression.endswith('d'):
                return int(interval_expression[:-1]) * 86400
            else:
                return int(interval_expression)  # Assume seconds
        except ValueError:
            return None
    
    def _queue_execution(self, schedule_config: ScheduleConfig, scheduled_time: datetime):
        """Queue execution for schedule"""
        
        # Check dependencies
        if not self._check_dependencies(schedule_config):
            self.logger.info(f"Dependencies not met for schedule: {schedule_config.schedule_id}")
            return
        
        # Create scheduled execution
        execution_id = f"{schedule_config.schedule_id}_{scheduled_time.strftime('%Y%m%d_%H%M%S')}"
        
        scheduled_execution = ScheduledExecution(
            execution_id=execution_id,
            schedule_id=schedule_config.schedule_id,
            scheduled_time=scheduled_time,
            actual_start_time=None,
            actual_end_time=None,
            status=PipelineStatus.NOT_STARTED,
            attempt_number=1,
            next_retry_time=None,
            pipeline_execution=None,
            error_message=None,
            resource_allocation={}
        )
        
        # Add to scheduled executions
        self.scheduled_executions.append(scheduled_execution)
        
        # Queue for execution
        priority = schedule_config.priority
        self.execution_queue.put((priority, scheduled_time, scheduled_execution))
        
        self.logger.info(f"Execution queued: {execution_id}")
    
    def _check_dependencies(self, schedule_config: ScheduleConfig) -> bool:
        """Check if schedule dependencies are satisfied"""
        
        if not schedule_config.dependencies:
            return True
        
        for dependency_id in schedule_config.dependencies:
            if dependency_id not in self.schedule_history:
                return False
            
            dependency_history = self.schedule_history[dependency_id]
            
            # Check if dependency has successful recent execution
            if not dependency_history.last_successful_execution:
                return False
            
            # Check if dependency execution is recent enough (within 24 hours)
            time_since_success = (
                datetime.now() - dependency_history.last_successful_execution
            ).total_seconds()
            
            if time_since_success > 86400:  # 24 hours
                return False
        
        return True
    
    def _is_execution_valid(self, scheduled_execution: ScheduledExecution) -> bool:
        """Check if scheduled execution is still valid"""
        
        schedule_config = self.schedules.get(scheduled_execution.schedule_id)
        if not schedule_config or not schedule_config.enabled:
            return False
        
        # Check execution timeout
        if schedule_config.execution_timeout_minutes > 0:
            max_start_time = (
                scheduled_execution.scheduled_time + 
                timedelta(minutes=schedule_config.execution_timeout_minutes)
            )
            if datetime.now() > max_start_time:
                return False
        
        return True
    
    def _allocate_resources(self, scheduled_execution: ScheduledExecution) -> bool:
        """Allocate resources for execution"""
        
        schedule_config = self.schedules[scheduled_execution.schedule_id]
        required_resources = schedule_config.resource_requirements
        
        if not required_resources:
            return True
        
        # Check if resources are available
        for resource_type, required_amount in required_resources.items():
            available_amount = self.available_resources.get(resource_type, 0)
            allocated_amount = sum(
                allocation.get(resource_type, 0) 
                for allocation in self.resource_allocations.values()
            )
            
            if available_amount - allocated_amount < required_amount:
                return False
        
        # Allocate resources
        self.resource_allocations[scheduled_execution.execution_id] = required_resources.copy()
        scheduled_execution.resource_allocation = required_resources.copy()
        
        return True
    
    def _start_execution(self, scheduled_execution: ScheduledExecution):
        """Start pipeline execution"""
        
        try:
            scheduled_execution.actual_start_time = datetime.now()
            scheduled_execution.status = PipelineStatus.RUNNING
            
            # Add to active executions
            self.active_executions[scheduled_execution.execution_id] = scheduled_execution
            
            # Get schedule configuration
            schedule_config = self.schedules[scheduled_execution.schedule_id]
            
            # Prepare execution configuration
            execution_config = schedule_config.execution_config.copy()
            
            # Start execution in separate thread
            execution_thread = threading.Thread(
                target=self._execute_pipeline,
                args=(scheduled_execution, execution_config),
                daemon=True
            )
            execution_thread.start()
            
            self.logger.info(f"Pipeline execution started: {scheduled_execution.execution_id}")
            
        except Exception as e:
            self._handle_execution_error(scheduled_execution, str(e))
    
    def _execute_pipeline(self, scheduled_execution: ScheduledExecution, execution_config: Dict[str, Any]):
        """Execute pipeline in separate thread"""
        
        try:
            # Execute pipeline using orchestrator
            pipeline_execution = self.orchestrator.execute_pipeline(
                input_data=execution_config.get('input_data'),
                stages=execution_config.get('stages'),
                config_overrides=execution_config.get('config_overrides'),
                resume_from_checkpoint=execution_config.get('resume_from_checkpoint', False),
                save_checkpoints=execution_config.get('save_checkpoints', True)
            )
            
            # Update scheduled execution
            scheduled_execution.pipeline_execution = pipeline_execution
            scheduled_execution.actual_end_time = datetime.now()
            
            if pipeline_execution.status == PipelineStatus.COMPLETED:
                scheduled_execution.status = PipelineStatus.COMPLETED
                self.logger.info(f"Pipeline execution completed: {scheduled_execution.execution_id}")
            else:
                scheduled_execution.status = PipelineStatus.FAILED
                scheduled_execution.error_message = "Pipeline execution failed"
                self.logger.error(f"Pipeline execution failed: {scheduled_execution.execution_id}")
            
        except Exception as e:
            self._handle_execution_error(scheduled_execution, str(e))
        
        finally:
            # Clean up resources and active execution
            self._cleanup_execution(scheduled_execution)
    
    def _handle_execution_error(self, scheduled_execution: ScheduledExecution, error_message: str):
        """Handle execution error with retry logic"""
        
        scheduled_execution.status = PipelineStatus.FAILED
        scheduled_execution.error_message = error_message
        
        if not scheduled_execution.actual_end_time:
            scheduled_execution.actual_end_time = datetime.now()
        
        schedule_config = self.schedules[scheduled_execution.schedule_id]
        
        # Check if retry is needed
        if (scheduled_execution.attempt_number < schedule_config.max_retries and 
            schedule_config.retry_strategy != RetryStrategy.NO_RETRY):
            
            # Calculate retry delay
            retry_delay = self._calculate_retry_delay(schedule_config, scheduled_execution.attempt_number)
            scheduled_execution.next_retry_time = datetime.now() + timedelta(minutes=retry_delay)
            
            # Create retry execution
            retry_execution = ScheduledExecution(
                execution_id=f"{scheduled_execution.execution_id}_retry_{scheduled_execution.attempt_number + 1}",
                schedule_id=scheduled_execution.schedule_id,
                scheduled_time=scheduled_execution.next_retry_time,
                actual_start_time=None,
                actual_end_time=None,
                status=PipelineStatus.NOT_STARTED,
                attempt_number=scheduled_execution.attempt_number + 1,
                next_retry_time=None,
                pipeline_execution=None,
                error_message=None,
                resource_allocation={}
            )
            
            # Queue retry
            priority = schedule_config.priority + 5  # Lower priority for retries
            self.execution_queue.put((priority, scheduled_execution.next_retry_time, retry_execution))
            
            self.logger.info(f"Retry scheduled: {retry_execution.execution_id}")
        
        self.logger.error(f"Execution error: {scheduled_execution.execution_id} - {error_message}")
    
    def _calculate_retry_delay(self, schedule_config: ScheduleConfig, attempt_number: int) -> int:
        """Calculate retry delay based on strategy"""
        
        base_delay = schedule_config.retry_delay_minutes
        
        if schedule_config.retry_strategy == RetryStrategy.FIXED_DELAY:
            return base_delay
        elif schedule_config.retry_strategy == RetryStrategy.EXPONENTIAL_BACKOFF:
            return base_delay * (2 ** (attempt_number - 1))
        elif schedule_config.retry_strategy == RetryStrategy.LINEAR_BACKOFF:
            return base_delay * attempt_number
        else:
            return base_delay
    
    def _cleanup_execution(self, scheduled_execution: ScheduledExecution):
        """Clean up completed execution"""
        
        # Remove from active executions
        if scheduled_execution.execution_id in self.active_executions:
            del self.active_executions[scheduled_execution.execution_id]
        
        # Release resources
        if scheduled_execution.execution_id in self.resource_allocations:
            del self.resource_allocations[scheduled_execution.execution_id]
    
    def _cleanup_completed_executions(self):
        """Clean up old completed executions"""
        
        cutoff_time = datetime.now() - timedelta(days=7)
        
        # Keep only recent executions
        self.scheduled_executions = [
            execution for execution in self.scheduled_executions
            if execution.scheduled_time > cutoff_time or execution.status == PipelineStatus.RUNNING
        ]
    
    def _update_schedule_history(self):
        """Update schedule execution history"""
        
        for schedule_id in self.schedules:
            if schedule_id not in self.schedule_history:
                continue
            
            history = self.schedule_history[schedule_id]
            
            # Get executions for this schedule
            schedule_executions = [
                execution for execution in self.scheduled_executions
                if execution.schedule_id == schedule_id
            ]
            
            if not schedule_executions:
                continue
            
            # Calculate statistics
            total_executions = len(schedule_executions)
            successful_executions = len([
                e for e in schedule_executions 
                if e.status == PipelineStatus.COMPLETED
            ])
            failed_executions = len([
                e for e in schedule_executions 
                if e.status == PipelineStatus.FAILED
            ])
            
            # Calculate average duration
            completed_executions = [
                e for e in schedule_executions 
                if e.actual_start_time and e.actual_end_time
            ]
            
            if completed_executions:
                durations = [
                    (e.actual_end_time - e.actual_start_time).total_seconds() / 60
                    for e in completed_executions
                ]
                average_duration = sum(durations) / len(durations)
            else:
                average_duration = 0.0
            
            # Find latest execution times
            last_execution = max(
                schedule_executions, 
                key=lambda e: e.actual_start_time or e.scheduled_time,
                default=None
            )
            
            last_successful = max([
                e for e in schedule_executions 
                if e.status == PipelineStatus.COMPLETED and e.actual_end_time
            ], key=lambda e: e.actual_end_time, default=None)
            
            last_failed = max([
                e for e in schedule_executions 
                if e.status == PipelineStatus.FAILED and e.actual_end_time
            ], key=lambda e: e.actual_end_time, default=None)
            
            # Calculate retry statistics
            retry_counts = [e.attempt_number for e in schedule_executions]
            average_retry_count = sum(retry_counts) / len(retry_counts) if retry_counts else 0.0
            
            # Update history
            history.total_executions = total_executions
            history.successful_executions = successful_executions
            history.failed_executions = failed_executions
            history.average_duration_minutes = average_duration
            history.last_execution_time = last_execution.actual_start_time if last_execution else None
            history.last_successful_execution = last_successful.actual_end_time if last_successful else None
            history.last_failed_execution = last_failed.actual_end_time if last_failed else None
            history.success_rate = successful_executions / max(total_executions, 1)
            history.average_retry_count = average_retry_count
    
    def _validate_schedule_config(self, schedule_config: ScheduleConfig):
        """Validate schedule configuration"""
        
        # Validate cron expression
        if schedule_config.schedule_type == ScheduleType.CRON:
            try:
                croniter(schedule_config.schedule_expression)
            except Exception as e:
                raise ValueError(f"Invalid cron expression: {schedule_config.schedule_expression}")
        
        # Validate interval expression
        elif schedule_config.schedule_type == ScheduleType.INTERVAL:
            if self._parse_interval(schedule_config.schedule_expression) is None:
                raise ValueError(f"Invalid interval expression: {schedule_config.schedule_expression}")
        
        # Validate date range
        if (schedule_config.start_date and schedule_config.end_date and 
            schedule_config.start_date >= schedule_config.end_date):
            raise ValueError("Start date must be before end date")
        
        # Validate retry configuration
        if schedule_config.max_retries < 0:
            raise ValueError("Max retries must be non-negative")
        
        if schedule_config.retry_delay_minutes <= 0:
            raise ValueError("Retry delay must be positive")
    
    def _has_schedule_conflicts(self, schedule_config: ScheduleConfig) -> bool:
        """Check for schedule conflicts"""
        
        # For now, just check for duplicate schedule IDs
        return schedule_config.schedule_id in self.schedules
    
    def _cancel_pending_executions(self, schedule_id: str):
        """Cancel pending executions for schedule"""
        
        # Remove from queue (not directly possible with PriorityQueue)
        # Instead, mark executions as cancelled
        for execution in self.scheduled_executions:
            if (execution.schedule_id == schedule_id and 
                execution.status in [PipelineStatus.NOT_STARTED, PipelineStatus.RUNNING]):
                execution.status = PipelineStatus.FAILED
                execution.error_message = "Cancelled due to schedule removal"
    
    def _get_available_resources(self) -> Dict[str, Any]:
        """Get available system resources"""
        
        return {
            'cpu_cores': os.cpu_count(),
            'memory_gb': 8,  # Default assumption
            'disk_gb': 100,  # Default assumption
            'max_concurrent_jobs': self.max_concurrent_executions
        }
    
    def get_schedule_status(self, schedule_id: Optional[str] = None) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """Get schedule status"""
        
        if schedule_id:
            if schedule_id not in self.schedules:
                return {}
            
            schedule_config = self.schedules[schedule_id]
            history = self.schedule_history.get(schedule_id, ScheduleHistory(
                schedule_id=schedule_id,
                total_executions=0,
                successful_executions=0,
                failed_executions=0,
                average_duration_minutes=0.0,
                last_execution_time=None,
                last_successful_execution=None,
                last_failed_execution=None,
                success_rate=0.0,
                average_retry_count=0.0
            ))
            
            return {
                'config': asdict(schedule_config),
                'history': asdict(history),
                'active_executions': len([
                    e for e in self.active_executions.values()
                    if e.schedule_id == schedule_id
                ])
            }
        else:
            return [
                {
                    'schedule_id': schedule_id,
                    'enabled': config.enabled,
                    'last_execution': self.schedule_history.get(schedule_id, {}).last_execution_time,
                    'success_rate': self.schedule_history.get(schedule_id, {}).success_rate or 0.0,
                    'active_executions': len([
                        e for e in self.active_executions.values()
                        if e.schedule_id == schedule_id
                    ])
                }
                for schedule_id, config in self.schedules.items()
            ]
    
    def get_execution_queue_status(self) -> Dict[str, Any]:
        """Get execution queue status"""
        
        return {
            'queue_size': self.execution_queue.qsize(),
            'active_executions': len(self.active_executions),
            'max_concurrent_executions': self.max_concurrent_executions,
            'available_slots': self.max_concurrent_executions - len(self.active_executions)
        }
    
    def get_recent_executions(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent scheduled executions"""
        
        recent_executions = sorted(
            self.scheduled_executions,
            key=lambda e: e.scheduled_time,
            reverse=True
        )[:limit]
        
        return [asdict(execution) for execution in recent_executions]
