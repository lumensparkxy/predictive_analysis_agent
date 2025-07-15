"""
Data Validation Package

This package provides comprehensive data validation capabilities for ensuring
data quality and integrity throughout the ML pipeline processing stages.
"""

from .data_validator import DataValidator
from .quality_checker import QualityChecker
from .schema_validator import SchemaValidator
from .ml_readiness_validator import MLReadinessValidator
from .validation_pipeline import ValidationPipeline

__version__ = "1.0.0"
__author__ = "Snowflake Analytics Team"

__all__ = [
    'DataValidator',
    'QualityChecker',
    'SchemaValidator',
    'MLReadinessValidator',
    'ValidationPipeline'
]

# Default validation configuration
DEFAULT_VALIDATION_CONFIG = {
    'data_quality_checks': {
        'completeness_threshold': 0.95,
        'uniqueness_threshold': 0.99,
        'validity_threshold': 0.98,
        'consistency_threshold': 0.95
    },
    'schema_validation': {
        'enforce_types': True,
        'allow_nullable': True,
        'validate_constraints': True
    },
    'ml_readiness': {
        'min_samples': 1000,
        'max_missing_percentage': 0.05,
        'min_feature_variance': 0.01,
        'max_correlation_threshold': 0.95
    },
    'validation_rules': {
        'enable_statistical_tests': True,
        'enable_distribution_checks': True,
        'enable_outlier_detection': True,
        'enable_temporal_consistency': True
    }
}
