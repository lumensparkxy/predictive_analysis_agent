"""
Feature Store Package

Comprehensive feature store implementation for ML-ready feature management,
versioning, lineage tracking, and serving capabilities.

This package provides enterprise-grade feature store capabilities including:
- Feature versioning and metadata management
- Data lineage tracking and audit trails
- Feature serving for online and offline inference
- Feature quality monitoring and validation
- Time-travel capabilities for feature history
- Integration with ML pipelines and model serving
"""

from .feature_store import FeatureStore
from .feature_registry import FeatureRegistry
from .feature_metadata import FeatureMetadata, FeatureGroup, FeatureVersion
from .feature_lineage import FeatureLineageTracker
from .feature_serving import FeatureServingEngine

__version__ = "1.0.0"
__author__ = "Snowflake Analytics Team"

__all__ = [
    'FeatureStore',
    'FeatureRegistry',
    'FeatureMetadata',
    'FeatureGroup',
    'FeatureVersion',
    'FeatureLineageTracker',
    'FeatureServingEngine'
]

# Default feature store configuration
DEFAULT_FEATURE_STORE_CONFIG = {
    'storage': {
        'backend': 'parquet',  # 'parquet', 'delta', 'snowflake'
        'base_path': 'data/feature_store',
        'compression': 'snappy',
        'partitioning': ['year', 'month', 'day']
    },
    'registry': {
        'backend': 'sqlite',  # 'sqlite', 'postgres', 'snowflake'
        'connection_string': 'sqlite:///data/feature_registry.db',
        'enable_metadata_cache': True
    },
    'lineage': {
        'tracking_enabled': True,
        'detailed_lineage': True,
        'lineage_storage': 'data/lineage',
        'retention_days': 365
    },
    'serving': {
        'online_store': 'redis',  # 'redis', 'dynamodb', 'memory'
        'offline_store': 'parquet',  # 'parquet', 'snowflake', 'bigquery'
        'cache_ttl_seconds': 3600,
        'batch_size': 1000
    },
    'validation': {
        'enabled': True,
        'schema_validation': True,
        'data_quality_checks': True,
        'drift_detection': True
    },
    'monitoring': {
        'enabled': True,
        'metrics_collection': True,
        'performance_tracking': True,
        'usage_analytics': True
    },
    'versioning': {
        'strategy': 'semantic',  # 'semantic', 'timestamp', 'incremental'
        'auto_versioning': True,
        'retention_policy': 'keep_all'  # 'keep_all', 'keep_latest_n', 'time_based'
    }
}
