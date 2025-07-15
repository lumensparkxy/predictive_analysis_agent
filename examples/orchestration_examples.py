"""
Pipeline Orchestration Example

Comprehensive example demonstrating the usage of the ML Pipeline Orchestration framework
for coordinating end-to-end data processing workflows.

This example shows:
1. Setting up the orchestrator with monitoring and scheduling
2. Executing a complete ML pipeline
3. Monitoring execution progress
4. Scheduling recurring pipeline runs
5. Handling errors and retries
6. Resource management and optimization
"""

import os
import json
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import numpy as np

from src.snowflake_analytics.data_processing.orchestration.ml_pipeline_orchestrator import (
    MLPipelineOrchestrator, PipelineStage, ExecutionMode, PipelineStatus
)
from src.snowflake_analytics.data_processing.orchestration.pipeline_monitor import (
    PipelineMonitor, AlertRule
)
from src.snowflake_analytics.data_processing.orchestration.pipeline_scheduler import (
    PipelineScheduler, ScheduleConfig, ScheduleType, RetryStrategy
)


def setup_sample_data():
    """Create sample Snowflake analytics data for demonstration"""
    
    # Create sample data directory
    data_dir = Path("data/sample")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate sample query performance data
    np.random.seed(42)
    n_records = 10000
    
    sample_data = pd.DataFrame({
        'query_id': [f"query_{i:06d}" for i in range(n_records)],
        'user_name': np.random.choice(['user_a', 'user_b', 'user_c', 'user_d'], n_records),
        'warehouse_name': np.random.choice(['small_wh', 'medium_wh', 'large_wh'], n_records),
        'database_name': np.random.choice(['prod_db', 'dev_db', 'test_db'], n_records),
        'schema_name': np.random.choice(['public', 'analytics', 'staging'], n_records),
        'query_type': np.random.choice(['SELECT', 'INSERT', 'UPDATE', 'DELETE'], n_records),
        'start_time': pd.date_range('2024-01-01', periods=n_records, freq='5min'),
        'end_time': None,  # Will be calculated
        'total_elapsed_time': np.random.exponential(5000, n_records),  # milliseconds
        'queued_provisioning_time': np.random.exponential(100, n_records),
        'queued_repair_time': np.random.exponential(50, n_records),
        'queued_overload_time': np.random.exponential(200, n_records),
        'compilation_time': np.random.exponential(300, n_records),
        'execution_time': None,  # Will be calculated
        'bytes_scanned': np.random.exponential(1000000, n_records),
        'bytes_written': np.random.exponential(100000, n_records),
        'bytes_deleted': np.random.exponential(10000, n_records),
        'partitions_scanned': np.random.poisson(10, n_records),
        'partitions_total': np.random.poisson(50, n_records),
        'credits_used_cloud_services': np.random.exponential(0.1, n_records),
        'error_code': np.random.choice([None, '100001', '100002', '100003'], n_records, p=[0.95, 0.02, 0.02, 0.01]),
        'error_message': None
    })
    
    # Calculate derived fields
    sample_data['end_time'] = sample_data['start_time'] + pd.to_timedelta(sample_data['total_elapsed_time'], unit='ms')
    sample_data['execution_time'] = (
        sample_data['total_elapsed_time'] - 
        sample_data['queued_provisioning_time'] - 
        sample_data['queued_repair_time'] - 
        sample_data['queued_overload_time'] - 
        sample_data['compilation_time']
    )
    
    # Add error messages for failed queries
    error_mask = sample_data['error_code'].notna()
    sample_data.loc[error_mask, 'error_message'] = 'Sample error message'
    
    # Save sample data
    output_path = data_dir / "sample_query_history.parquet"
    sample_data.to_parquet(output_path)
    
    print(f"Sample data created: {output_path}")
    print(f"Records: {len(sample_data)}")
    print(f"Date range: {sample_data['start_time'].min()} to {sample_data['start_time'].max()}")
    
    return str(output_path)


def basic_pipeline_execution_example():
    """Example 1: Basic pipeline execution"""
    
    print("\n=== Basic Pipeline Execution Example ===")
    
    # Create sample data
    sample_data_path = setup_sample_data()
    
    # Initialize orchestrator
    orchestrator = MLPipelineOrchestrator(
        execution_mode=ExecutionMode.SEQUENTIAL,
        max_workers=2,
        checkpoint_dir="data/checkpoints",
        metrics_dir="data/metrics"
    )
    
    # Execute complete pipeline
    print("Starting pipeline execution...")
    
    execution = orchestrator.execute_pipeline(
        input_data=sample_data_path,
        stages=[
            PipelineStage.DATA_CLEANING,
            PipelineStage.FEATURE_ENGINEERING,
            PipelineStage.DATA_AGGREGATION,
            PipelineStage.DATA_VALIDATION
        ],
        config_overrides={
            'data_cleaning': {
                'remove_duplicates': True,
                'handle_missing_values': True,
                'detect_outliers': True
            },
            'feature_engineering': {
                'generate_time_features': True,
                'generate_usage_features': True,
                'generate_cost_features': True
            }
        },
        save_checkpoints=True
    )
    
    # Print execution results
    print(f"\nExecution Results:")
    print(f"- Execution ID: {execution.execution_id}")
    print(f"- Status: {execution.status.value}")
    print(f"- Duration: {execution.total_duration_seconds:.2f} seconds")
    print(f"- Records Processed: {execution.total_records_processed:,}")
    print(f"- Stages Completed: {len(execution.stages_completed)}")
    
    for stage, result in execution.stage_results.items():
        print(f"  - {stage.value}: {result.status.value} ({result.duration_seconds:.2f}s, {result.records_processed:,} records)")
    
    if execution.errors:
        print(f"- Errors: {len(execution.errors)}")
        for error in execution.errors[:3]:  # Show first 3 errors
            print(f"  - {error}")
    
    return execution


def monitoring_example():
    """Example 2: Pipeline execution with monitoring"""
    
    print("\n=== Pipeline Monitoring Example ===")
    
    # Create sample data
    sample_data_path = setup_sample_data()
    
    # Initialize orchestrator and monitor
    orchestrator = MLPipelineOrchestrator(execution_mode=ExecutionMode.SEQUENTIAL)
    monitor = PipelineMonitor(
        metrics_retention_hours=24,
        sampling_interval_seconds=5,
        enable_resource_monitoring=True,
        enable_performance_monitoring=True,
        enable_alerting=True
    )
    
    # Add custom alert rules
    custom_alert = AlertRule(
        rule_id="high_memory_custom",
        metric_name="memory_percent",
        condition="greater_than",
        threshold=70.0,
        severity="medium",
        cooldown_minutes=2,
        enabled=True,
        description="Custom memory usage alert"
    )
    monitor.add_alert_rule(custom_alert)
    
    # Start monitoring
    monitor.start_monitoring()
    
    try:
        # Execute pipeline
        print("Starting monitored pipeline execution...")
        
        execution = orchestrator.execute_pipeline(
            input_data=sample_data_path,
            stages=[PipelineStage.DATA_CLEANING, PipelineStage.FEATURE_ENGINEERING],
            save_checkpoints=True
        )
        
        # Update monitor with execution context
        monitor.current_execution = execution
        
        # Let monitoring collect some data
        time.sleep(15)
        
        # Get current metrics
        current_metrics = monitor.get_current_metrics()
        print(f"\nCurrent Metrics:")
        print(f"- CPU Usage: {current_metrics.get('resources', {}).get('cpu_percent', 'N/A')}%")
        print(f"- Memory Usage: {current_metrics.get('resources', {}).get('memory_percent', 'N/A')}%")
        print(f"- Active Alerts: {current_metrics.get('alerts', {}).get('active_alerts', 0)}")
        
        # Get performance summary
        perf_summary = monitor.get_performance_summary()
        print(f"\nPerformance Summary:")
        print(f"- Total Executions: {perf_summary.get('total_executions', 0)}")
        print(f"- Average Execution Time: {perf_summary.get('average_execution_time', 0):.2f}s")
        print(f"- Average Processing Rate: {perf_summary.get('average_processing_rate', 0):.2f} records/sec")
        
        # Get recent alerts
        recent_alerts = monitor.get_alerts(limit=5)
        if recent_alerts:
            print(f"\nRecent Alerts:")
            for alert in recent_alerts[:3]:
                print(f"- [{alert.severity.upper()}] {alert.message}")
        
        # Export metrics
        monitor.export_metrics("data/monitoring_export.json", format="json")
        print(f"\nMetrics exported to: data/monitoring_export.json")
        
    finally:
        # Stop monitoring
        monitor.stop_monitoring()
    
    return execution, monitor


def scheduling_example():
    """Example 3: Scheduled pipeline execution"""
    
    print("\n=== Pipeline Scheduling Example ===")
    
    # Create sample data
    sample_data_path = setup_sample_data()
    
    # Initialize components
    orchestrator = MLPipelineOrchestrator(execution_mode=ExecutionMode.PARALLEL)
    scheduler = PipelineScheduler(
        orchestrator=orchestrator,
        max_concurrent_executions=2,
        schedule_check_interval_seconds=10
    )
    
    # Create schedule configurations
    daily_schedule = ScheduleConfig(
        schedule_id="daily_ml_pipeline",
        name="Daily ML Pipeline",
        description="Daily execution of complete ML pipeline",
        schedule_type=ScheduleType.CRON,
        schedule_expression="0 2 * * *",  # Daily at 2 AM
        enabled=True,
        start_date=datetime.now() - timedelta(days=1),
        end_date=None,
        max_executions=None,
        execution_timeout_minutes=60,
        retry_strategy=RetryStrategy.EXPONENTIAL_BACKOFF,
        max_retries=3,
        retry_delay_minutes=10,
        priority=5,
        resource_requirements={'cpu_cores': 2, 'memory_gb': 4},
        dependencies=[],
        execution_config={
            'input_data': sample_data_path,
            'stages': [
                PipelineStage.DATA_CLEANING,
                PipelineStage.FEATURE_ENGINEERING,
                PipelineStage.DATA_AGGREGATION
            ],
            'config_overrides': {
                'data_cleaning': {'remove_duplicates': True}
            },
            'save_checkpoints': True
        },
        notification_config={'channels': ['log', 'file']}
    )
    
    hourly_schedule = ScheduleConfig(
        schedule_id="hourly_validation",
        name="Hourly Data Validation",
        description="Hourly data quality validation",
        schedule_type=ScheduleType.INTERVAL,
        schedule_expression="1h",  # Every hour
        enabled=True,
        start_date=datetime.now(),
        end_date=None,
        max_executions=None,
        execution_timeout_minutes=15,
        retry_strategy=RetryStrategy.FIXED_DELAY,
        max_retries=2,
        retry_delay_minutes=5,
        priority=3,
        resource_requirements={'cpu_cores': 1, 'memory_gb': 2},
        dependencies=["daily_ml_pipeline"],
        execution_config={
            'input_data': sample_data_path,
            'stages': [PipelineStage.DATA_VALIDATION],
            'save_checkpoints': False
        },
        notification_config={'channels': ['log']}
    )
    
    # Add schedules
    scheduler.add_schedule(daily_schedule)
    scheduler.add_schedule(hourly_schedule)
    
    # Start scheduler
    scheduler.start_scheduler()
    
    try:
        print("Scheduler started. Waiting for scheduled executions...")
        
        # Trigger manual execution for demonstration
        print("\nTriggering manual execution...")
        execution_id = scheduler.trigger_manual_execution("daily_ml_pipeline")
        print(f"Manual execution triggered: {execution_id}")
        
        # Wait and monitor
        for i in range(30):  # Wait up to 5 minutes
            time.sleep(10)
            
            # Get queue status
            queue_status = scheduler.get_execution_queue_status()
            print(f"\nQueue Status (check {i+1}):")
            print(f"- Queue Size: {queue_status['queue_size']}")
            print(f"- Active Executions: {queue_status['active_executions']}")
            print(f"- Available Slots: {queue_status['available_slots']}")
            
            # Get schedule status
            schedule_status = scheduler.get_schedule_status()
            print(f"- Active Schedules: {len([s for s in schedule_status if s['active_executions'] > 0])}")
            
            # Check if any executions completed
            recent_executions = scheduler.get_recent_executions(limit=5)
            completed_executions = [
                e for e in recent_executions 
                if e['status'] in ['completed', 'failed']
            ]
            
            if completed_executions:
                print(f"\nCompleted Executions:")
                for execution in completed_executions[:3]:
                    status = execution['status']
                    duration = None
                    if execution['actual_start_time'] and execution['actual_end_time']:
                        start = datetime.fromisoformat(execution['actual_start_time'])
                        end = datetime.fromisoformat(execution['actual_end_time'])
                        duration = (end - start).total_seconds()
                    
                    print(f"- {execution['execution_id']}: {status}")
                    if duration:
                        print(f"  Duration: {duration:.2f}s")
                
                break
        
        # Get final schedule status
        print(f"\nFinal Schedule Status:")
        for schedule_id in ['daily_ml_pipeline', 'hourly_validation']:
            status = scheduler.get_schedule_status(schedule_id)
            if status:
                history = status.get('history', {})
                print(f"- {schedule_id}:")
                print(f"  Total Executions: {history.get('total_executions', 0)}")
                print(f"  Success Rate: {history.get('success_rate', 0):.2%}")
                print(f"  Average Duration: {history.get('average_duration_minutes', 0):.2f} min")
    
    finally:
        # Stop scheduler
        scheduler.stop_scheduler()
    
    return scheduler


def error_handling_example():
    """Example 4: Error handling and recovery"""
    
    print("\n=== Error Handling and Recovery Example ===")
    
    # Initialize orchestrator
    orchestrator = MLPipelineOrchestrator(execution_mode=ExecutionMode.SEQUENTIAL)
    
    try:
        # Attempt to execute pipeline with invalid data
        print("Attempting pipeline execution with invalid input...")
        
        execution = orchestrator.execute_pipeline(
            input_data="nonexistent_file.parquet",  # This will cause an error
            stages=[PipelineStage.DATA_CLEANING],
            save_checkpoints=True
        )
        
    except Exception as e:
        print(f"Pipeline execution failed as expected: {str(e)}")
        
        # Show how to resume from checkpoint
        print("\nDemonstrating recovery mechanisms...")
        
        # Create valid sample data
        sample_data_path = setup_sample_data()
        
        # Execute with valid data and checkpointing
        execution = orchestrator.execute_pipeline(
            input_data=sample_data_path,
            stages=[PipelineStage.DATA_CLEANING, PipelineStage.FEATURE_ENGINEERING],
            resume_from_checkpoint=False,  # Start fresh
            save_checkpoints=True
        )
        
        print(f"Recovery execution completed: {execution.status.value}")
        
        # Show execution history
        history = orchestrator.get_execution_history(limit=5)
        print(f"\nExecution History ({len(history)} total):")
        for exec_hist in history:
            print(f"- {exec_hist.execution_id}: {exec_hist.status.value}")
            if exec_hist.errors:
                print(f"  Errors: {len(exec_hist.errors)}")
    
    return orchestrator


def performance_optimization_example():
    """Example 5: Performance optimization and resource management"""
    
    print("\n=== Performance Optimization Example ===")
    
    # Create larger sample dataset
    print("Creating larger sample dataset...")
    np.random.seed(42)
    n_records = 50000
    
    large_sample_data = pd.DataFrame({
        'query_id': [f"query_{i:06d}" for i in range(n_records)],
        'user_name': np.random.choice(['user_a', 'user_b', 'user_c', 'user_d'], n_records),
        'warehouse_name': np.random.choice(['small_wh', 'medium_wh', 'large_wh'], n_records),
        'total_elapsed_time': np.random.exponential(5000, n_records),
        'start_time': pd.date_range('2024-01-01', periods=n_records, freq='1min'),
        'bytes_scanned': np.random.exponential(1000000, n_records),
        'credits_used_cloud_services': np.random.exponential(0.1, n_records)
    })
    
    # Save large dataset
    large_data_dir = Path("data/large_sample")
    large_data_dir.mkdir(parents=True, exist_ok=True)
    large_data_path = large_data_dir / "large_query_history.parquet"
    large_sample_data.to_parquet(large_data_path)
    
    print(f"Large dataset created: {len(large_sample_data):,} records")
    
    # Compare different execution modes
    execution_modes = [ExecutionMode.SEQUENTIAL, ExecutionMode.PARALLEL, ExecutionMode.HYBRID]
    results = {}
    
    for mode in execution_modes:
        print(f"\nTesting {mode.value} execution mode...")
        
        orchestrator = MLPipelineOrchestrator(
            execution_mode=mode,
            max_workers=4,
            checkpoint_dir=f"data/checkpoints_{mode.value}",
            metrics_dir=f"data/metrics_{mode.value}"
        )
        
        start_time = time.time()
        
        execution = orchestrator.execute_pipeline(
            input_data=str(large_data_path),
            stages=[
                PipelineStage.DATA_CLEANING,
                PipelineStage.FEATURE_ENGINEERING
            ],
            save_checkpoints=True
        )
        
        end_time = time.time()
        total_time = end_time - start_time
        
        results[mode.value] = {
            'total_time': total_time,
            'records_processed': execution.total_records_processed,
            'processing_rate': execution.total_records_processed / max(total_time, 1),
            'status': execution.status.value,
            'stages_completed': len(execution.stages_completed)
        }
        
        print(f"- Duration: {total_time:.2f}s")
        print(f"- Processing Rate: {results[mode.value]['processing_rate']:.2f} records/sec")
        print(f"- Status: {execution.status.value}")
    
    # Compare results
    print(f"\n=== Performance Comparison ===")
    for mode, result in results.items():
        print(f"{mode.upper()}:")
        print(f"  Duration: {result['total_time']:.2f}s")
        print(f"  Processing Rate: {result['processing_rate']:.2f} records/sec")
        print(f"  Status: {result['status']}")
    
    # Find best performing mode
    best_mode = max(results.keys(), key=lambda k: results[k]['processing_rate'])
    print(f"\nBest performing mode: {best_mode.upper()}")
    
    return results


def complete_workflow_example():
    """Example 6: Complete workflow with orchestration, monitoring, and scheduling"""
    
    print("\n=== Complete Workflow Example ===")
    
    # Create sample data
    sample_data_path = setup_sample_data()
    
    # Initialize all components
    orchestrator = MLPipelineOrchestrator(
        execution_mode=ExecutionMode.HYBRID,
        max_workers=3,
        checkpoint_dir="data/workflow_checkpoints",
        metrics_dir="data/workflow_metrics"
    )
    
    monitor = PipelineMonitor(
        metrics_retention_hours=24,
        sampling_interval_seconds=10,
        enable_resource_monitoring=True,
        enable_performance_monitoring=True,
        enable_alerting=True
    )
    
    scheduler = PipelineScheduler(
        orchestrator=orchestrator,
        max_concurrent_executions=2,
        schedule_check_interval_seconds=15
    )
    
    # Add comprehensive alert rules
    alert_rules = [
        AlertRule(
            rule_id="critical_memory",
            metric_name="memory_percent",
            condition="greater_than",
            threshold=90.0,
            severity="critical",
            cooldown_minutes=5,
            enabled=True,
            description="Critical memory usage"
        ),
        AlertRule(
            rule_id="slow_processing",
            metric_name="processing_rate_records_per_second",
            condition="less_than",
            threshold=50.0,
            severity="medium",
            cooldown_minutes=10,
            enabled=True,
            description="Slow processing rate"
        )
    ]
    
    for rule in alert_rules:
        monitor.add_alert_rule(rule)
    
    # Create comprehensive schedule
    production_schedule = ScheduleConfig(
        schedule_id="production_pipeline",
        name="Production ML Pipeline",
        description="Complete production ML pipeline with all stages",
        schedule_type=ScheduleType.CRON,
        schedule_expression="0 3 * * *",  # Daily at 3 AM
        enabled=True,
        start_date=datetime.now(),
        end_date=None,
        max_executions=None,
        execution_timeout_minutes=120,
        retry_strategy=RetryStrategy.EXPONENTIAL_BACKOFF,
        max_retries=3,
        retry_delay_minutes=15,
        priority=8,
        resource_requirements={'cpu_cores': 3, 'memory_gb': 8},
        dependencies=[],
        execution_config={
            'input_data': sample_data_path,
            'stages': [
                PipelineStage.DATA_CLEANING,
                PipelineStage.FEATURE_ENGINEERING,
                PipelineStage.DATA_AGGREGATION,
                PipelineStage.DATA_VALIDATION
            ],
            'config_overrides': {
                'data_cleaning': {
                    'remove_duplicates': True,
                    'handle_missing_values': True,
                    'detect_outliers': True,
                    'convert_types': True
                },
                'feature_engineering': {
                    'generate_time_features': True,
                    'generate_usage_features': True,
                    'generate_cost_features': True,
                    'generate_rolling_features': True,
                    'generate_pattern_features': True
                },
                'data_aggregation': {
                    'temporal_aggregation': True,
                    'dimensional_aggregation': True,
                    'cost_aggregation': True,
                    'usage_aggregation': True
                },
                'data_validation': {
                    'run_quality_checks': True,
                    'run_schema_validation': True,
                    'run_ml_readiness_checks': True
                }
            },
            'save_checkpoints': True
        },
        notification_config={'channels': ['log', 'file']}
    )
    
    # Start all components
    print("Starting complete workflow...")
    monitor.start_monitoring()
    scheduler.start_scheduler()
    
    try:
        # Add schedule
        scheduler.add_schedule(production_schedule)
        
        # Trigger manual execution for demonstration
        print("Triggering production pipeline execution...")
        execution_id = scheduler.trigger_manual_execution("production_pipeline")
        
        # Monitor execution
        start_time = time.time()
        execution_completed = False
        
        while time.time() - start_time < 300 and not execution_completed:  # 5 minute timeout
            time.sleep(15)
            
            # Get current metrics
            current_metrics = monitor.get_current_metrics()
            queue_status = scheduler.get_execution_queue_status()
            
            print(f"\n--- Workflow Status ---")
            print(f"Time Elapsed: {time.time() - start_time:.0f}s")
            print(f"Queue Size: {queue_status['queue_size']}")
            print(f"Active Executions: {queue_status['active_executions']}")
            
            if 'resources' in current_metrics:
                resources = current_metrics['resources']
                print(f"CPU Usage: {resources.get('cpu_percent', 'N/A')}%")
                print(f"Memory Usage: {resources.get('memory_percent', 'N/A')}%")
            
            if 'alerts' in current_metrics:
                alerts = current_metrics['alerts']
                active_alerts = alerts.get('active_alerts', 0)
                if active_alerts > 0:
                    print(f"Active Alerts: {active_alerts}")
            
            # Check for completed executions
            recent_executions = scheduler.get_recent_executions(limit=5)
            for execution in recent_executions:
                if (execution['execution_id'] == execution_id and 
                    execution['status'] in ['completed', 'failed']):
                    execution_completed = True
                    print(f"\nExecution {execution_id} {execution['status']}!")
                    
                    if execution['actual_start_time'] and execution['actual_end_time']:
                        start = datetime.fromisoformat(execution['actual_start_time'])
                        end = datetime.fromisoformat(execution['actual_end_time'])
                        duration = (end - start).total_seconds()
                        print(f"Duration: {duration:.2f} seconds")
                    
                    break
        
        # Generate final report
        print(f"\n=== Final Workflow Report ===")
        
        # Performance summary
        perf_summary = monitor.get_performance_summary()
        print(f"Performance Summary:")
        print(f"- Total Executions: {perf_summary.get('total_executions', 0)}")
        print(f"- Average Execution Time: {perf_summary.get('average_execution_time', 0):.2f}s")
        print(f"- Total Records Processed: {perf_summary.get('total_records_processed', 0):,}")
        
        # Alert summary
        all_alerts = monitor.get_alerts(limit=10)
        if all_alerts:
            print(f"\nAlert Summary:")
            print(f"- Total Alerts: {len(all_alerts)}")
            by_severity = {}
            for alert in all_alerts:
                by_severity[alert.severity] = by_severity.get(alert.severity, 0) + 1
            for severity, count in by_severity.items():
                print(f"  - {severity.title()}: {count}")
        
        # Schedule summary
        schedule_status = scheduler.get_schedule_status("production_pipeline")
        if schedule_status and 'history' in schedule_status:
            history = schedule_status['history']
            print(f"\nSchedule Summary:")
            print(f"- Total Executions: {history.get('total_executions', 0)}")
            print(f"- Success Rate: {history.get('success_rate', 0):.2%}")
            print(f"- Average Duration: {history.get('average_duration_minutes', 0):.2f} minutes")
        
        # Export comprehensive data
        monitor.export_metrics("data/complete_workflow_metrics.json", format="json")
        print(f"\nComplete metrics exported to: data/complete_workflow_metrics.json")
        
    finally:
        # Clean shutdown
        print("\nShutting down workflow components...")
        scheduler.stop_scheduler()
        monitor.stop_monitoring()
    
    return orchestrator, monitor, scheduler


def main():
    """Run all orchestration examples"""
    
    print("Snowflake Analytics ML Pipeline Orchestration Examples")
    print("=" * 60)
    
    # Ensure data directories exist
    Path("data").mkdir(exist_ok=True)
    Path("data/checkpoints").mkdir(exist_ok=True)
    Path("data/metrics").mkdir(exist_ok=True)
    
    try:
        # Run examples
        print("\n1. Basic Pipeline Execution")
        basic_execution = basic_pipeline_execution_example()
        
        print("\n2. Pipeline Monitoring")
        monitored_execution, monitor = monitoring_example()
        
        print("\n3. Pipeline Scheduling")
        scheduler = scheduling_example()
        
        print("\n4. Error Handling and Recovery")
        error_orchestrator = error_handling_example()
        
        print("\n5. Performance Optimization")
        performance_results = performance_optimization_example()
        
        print("\n6. Complete Workflow")
        complete_orchestrator, complete_monitor, complete_scheduler = complete_workflow_example()
        
        print("\n" + "=" * 60)
        print("All orchestration examples completed successfully!")
        print("\nKey takeaways:")
        print("- Pipeline orchestration provides comprehensive workflow management")
        print("- Monitoring enables real-time tracking and alerting")
        print("- Scheduling supports automated recurring executions")
        print("- Error handling ensures robust pipeline operations")
        print("- Performance optimization maximizes resource utilization")
        print("- Complete workflows integrate all components seamlessly")
        
    except Exception as e:
        print(f"Example execution failed: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
