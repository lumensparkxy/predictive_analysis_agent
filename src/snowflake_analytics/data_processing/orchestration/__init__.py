"""
Pipeline Orchestration Package

This package provides comprehensive pipeline orchestration capabilities for
coordinating the end-to-end ML data processing pipeline from raw data to ML-ready features.
"""

from .ml_pipeline_orchestrator import MLPipelineOrchestrator
from .pipeline_monitor import PipelineMonitor
from .pipeline_scheduler import PipelineScheduler

__version__ = "1.0.0"
__author__ = "Snowflake Analytics Team"

__all__ = [
    'MLPipelineOrchestrator',
    'PipelineMonitor',
    'PipelineScheduler'
]

# Default orchestration configuration
DEFAULT_ORCHESTRATION_CONFIG = {
    'pipeline_stages': [
        'data_ingestion',
        'data_cleaning',
        'feature_engineering', 
        'data_aggregation',
        'data_validation',
        'ml_preparation'
    ],
    'execution_mode': 'sequential',  # 'sequential' or 'parallel'
    'error_handling': 'stop_on_failure',  # 'stop_on_failure' or 'continue_on_error'
    'monitoring': {
        'enabled': True,
        'metrics_collection': True,
        'performance_tracking': True,
        'resource_monitoring': True
    },
    'scheduling': {
        'enabled': False,
        'default_schedule': 'daily',
        'retry_attempts': 3,
        'retry_delay_minutes': 30
    },
    'checkpointing': {
        'enabled': True,
        'save_intermediate_results': True,
        'checkpoint_frequency': 'stage'
    },
    'notifications': {
        'enabled': True,
        'success_notifications': True,
        'failure_notifications': True,
        'channels': ['log', 'file']
    }
}
