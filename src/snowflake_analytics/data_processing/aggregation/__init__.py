"""
Data Aggregation Package

This package provides comprehensive data aggregation capabilities for creating
summary statistics and metrics from Snowflake analytics data at various temporal
and dimensional levels.
"""

from .temporal_aggregator import TemporalAggregator
from .dimensional_aggregator import DimensionalAggregator
from .cost_aggregator import CostAggregator
from .usage_aggregator import UsageAggregator
from .aggregation_pipeline import AggregationPipeline

__version__ = "1.0.0"
__author__ = "Snowflake Analytics Team"

__all__ = [
    'TemporalAggregator',
    'DimensionalAggregator',
    'CostAggregator',
    'UsageAggregator',
    'AggregationPipeline'
]

# Default aggregation configuration
DEFAULT_AGGREGATION_CONFIG = {
    'temporal_levels': ['hourly', 'daily', 'weekly', 'monthly'],
    'dimensional_groupings': ['user', 'warehouse', 'database', 'role'],
    'metrics': ['credits_used', 'execution_time_ms', 'bytes_scanned', 'rows_produced'],
    'statistics': ['sum', 'mean', 'median', 'std', 'min', 'max', 'count'],
    'percentiles': [50, 75, 90, 95, 99],
    'enable_forecasting': True,
    'enable_alerts': True
}
