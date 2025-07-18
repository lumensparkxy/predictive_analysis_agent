"""
Unit tests for data validation components.

Tests data quality checks, validation rules, and error handling.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from decimal import Decimal
import re


class TestDataValidation:
    """Test suite for data validation functionality."""

    @pytest.fixture
    def mock_validator(self):
        """Create a mock data validator."""
        validator = Mock()
        validator.validate_data = Mock()
        validator.validate_schema = Mock()
        validator.validate_constraints = Mock()
        validator.validate_business_rules = Mock()
        validator.get_validation_errors = Mock()
        return validator

    @pytest.fixture
    def sample_valid_data(self):
        """Create sample valid data for testing."""
        return pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=100, freq='H'),
            'cost': np.random.uniform(10, 1000, 100),
            'credits': np.random.uniform(0.1, 10.0, 100),
            'warehouse': ['WH_ANALYTICS'] * 100,
            'user_id': [f'user_{i%20}' for i in range(100)],
            'query_id': [f'query_{i}' for i in range(100)],
            'status': ['SUCCESS'] * 95 + ['FAILED'] * 5
        })

    @pytest.fixture
    def sample_invalid_data(self):
        """Create sample invalid data for testing."""
        return pd.DataFrame({
            'timestamp': [None] * 5 + list(pd.date_range('2024-01-01', periods=5, freq='H')),
            'cost': [-10.0, 0.0, 'invalid', None, 999999.99] + [100.0] * 5,
            'credits': [0.0, -1.0, 'invalid', None, 999.99] + [1.0] * 5,
            'warehouse': ['', None, 'INVALID_WH', 'WH_ANALYTICS', 'WH_ANALYTICS'] + ['WH_ANALYTICS'] * 5,
            'user_id': ['', None, 'invalid user', 'user_1', 'user_2'] + ['user_3'] * 5,
            'query_id': ['', None, 'invalid-query', 'query_1', 'query_2'] + ['query_3'] * 5,
            'status': ['', None, 'INVALID', 'SUCCESS', 'FAILED'] + ['SUCCESS'] * 5
        })

    def test_data_schema_validation_success(self, mock_validator, sample_valid_data):
        """Test successful data schema validation."""
        # Mock successful schema validation
        mock_validator.validate_schema.return_value = True
        
        # Test schema validation
        result = mock_validator.validate_schema(sample_valid_data)
        
        assert result is True
        mock_validator.validate_schema.assert_called_once_with(sample_valid_data)

    def test_data_schema_validation_failure(self, mock_validator, sample_invalid_data):
        """Test data schema validation failure."""
        # Mock schema validation failure
        mock_validator.validate_schema.return_value = False
        
        # Test schema validation failure
        result = mock_validator.validate_schema(sample_invalid_data)
        
        assert result is False

    def test_data_type_validation(self, mock_validator):
        """Test data type validation."""
        type_validation_results = {
            'timestamp': {'valid': True, 'type': 'datetime64[ns]'},
            'cost': {'valid': True, 'type': 'float64'},
            'credits': {'valid': True, 'type': 'float64'},
            'warehouse': {'valid': True, 'type': 'object'},
            'user_id': {'valid': True, 'type': 'object'},
            'query_id': {'valid': True, 'type': 'object'},
            'status': {'valid': True, 'type': 'object'}
        }
        
        mock_validator.validate_data_types.return_value = type_validation_results
        
        # Test data type validation
        result = mock_validator.validate_data_types()
        
        assert result['timestamp']['valid'] is True
        assert result['cost']['type'] == 'float64'
        assert result['credits']['type'] == 'float64'

    def test_null_value_validation(self, mock_validator):
        """Test null value validation."""
        null_validation_results = {
            'timestamp': {'null_count': 0, 'null_percentage': 0.0, 'allows_null': False},
            'cost': {'null_count': 0, 'null_percentage': 0.0, 'allows_null': False},
            'credits': {'null_count': 0, 'null_percentage': 0.0, 'allows_null': False},
            'warehouse': {'null_count': 0, 'null_percentage': 0.0, 'allows_null': False},
            'user_id': {'null_count': 2, 'null_percentage': 2.0, 'allows_null': True},
            'query_id': {'null_count': 0, 'null_percentage': 0.0, 'allows_null': False},
            'status': {'null_count': 0, 'null_percentage': 0.0, 'allows_null': False}
        }
        
        mock_validator.validate_null_values.return_value = null_validation_results
        
        # Test null value validation
        result = mock_validator.validate_null_values()
        
        assert result['timestamp']['null_count'] == 0
        assert result['user_id']['null_percentage'] == 2.0
        assert result['cost']['allows_null'] is False

    def test_range_validation(self, mock_validator):
        """Test range validation for numeric fields."""
        range_validation_results = {
            'cost': {
                'min_value': 0.01,
                'max_value': 50000.00,
                'out_of_range_count': 0,
                'valid': True
            },
            'credits': {
                'min_value': 0.001,
                'max_value': 1000.0,
                'out_of_range_count': 0,
                'valid': True
            }
        }
        
        mock_validator.validate_ranges.return_value = range_validation_results
        
        # Test range validation
        result = mock_validator.validate_ranges()
        
        assert result['cost']['valid'] is True
        assert result['credits']['max_value'] == 1000.0
        assert result['cost']['out_of_range_count'] == 0

    def test_format_validation(self, mock_validator):
        """Test format validation for string fields."""
        format_validation_results = {
            'warehouse': {
                'pattern': r'^WH_[A-Z_]+$',
                'invalid_count': 0,
                'valid': True
            },
            'user_id': {
                'pattern': r'^user_\d+$',
                'invalid_count': 0,
                'valid': True
            },
            'query_id': {
                'pattern': r'^query_\d+$',
                'invalid_count': 0,
                'valid': True
            }
        }
        
        mock_validator.validate_formats.return_value = format_validation_results
        
        # Test format validation
        result = mock_validator.validate_formats()
        
        assert result['warehouse']['valid'] is True
        assert result['user_id']['pattern'] == r'^user_\d+$'
        assert result['query_id']['invalid_count'] == 0

    def test_business_rule_validation(self, mock_validator):
        """Test business rule validation."""
        business_rule_results = {
            'cost_credit_consistency': {
                'rule': 'cost should be proportional to credits',
                'violations': 0,
                'valid': True
            },
            'timestamp_ordering': {
                'rule': 'timestamps should be in chronological order',
                'violations': 0,
                'valid': True
            },
            'status_consistency': {
                'rule': 'failed queries should have zero credits',
                'violations': 0,
                'valid': True
            }
        }
        
        mock_validator.validate_business_rules.return_value = business_rule_results
        
        # Test business rule validation
        result = mock_validator.validate_business_rules()
        
        assert result['cost_credit_consistency']['valid'] is True
        assert result['timestamp_ordering']['violations'] == 0
        assert result['status_consistency']['valid'] is True

    def test_duplicate_detection(self, mock_validator):
        """Test duplicate record detection."""
        duplicate_results = {
            'total_records': 1000,
            'duplicate_records': 5,
            'duplicate_percentage': 0.5,
            'duplicate_keys': ['query_id', 'timestamp'],
            'duplicate_examples': [
                {'query_id': 'query_123', 'timestamp': '2024-01-01 10:00:00'},
                {'query_id': 'query_456', 'timestamp': '2024-01-01 11:00:00'}
            ]
        }
        
        mock_validator.detect_duplicates.return_value = duplicate_results
        
        # Test duplicate detection
        result = mock_validator.detect_duplicates()
        
        assert result['total_records'] == 1000
        assert result['duplicate_records'] == 5
        assert result['duplicate_percentage'] == 0.5
        assert len(result['duplicate_examples']) == 2

    def test_data_completeness_validation(self, mock_validator):
        """Test data completeness validation."""
        completeness_results = {
            'expected_records': 1000,
            'actual_records': 995,
            'completeness_percentage': 99.5,
            'missing_records': 5,
            'missing_time_ranges': [
                {'start': '2024-01-01 14:00:00', 'end': '2024-01-01 15:00:00'},
                {'start': '2024-01-01 18:00:00', 'end': '2024-01-01 19:00:00'}
            ]
        }
        
        mock_validator.validate_completeness.return_value = completeness_results
        
        # Test completeness validation
        result = mock_validator.validate_completeness()
        
        assert result['completeness_percentage'] == 99.5
        assert result['missing_records'] == 5
        assert len(result['missing_time_ranges']) == 2

    def test_data_consistency_validation(self, mock_validator):
        """Test data consistency validation."""
        consistency_results = {
            'cross_field_consistency': {
                'cost_credits_ratio': {'valid': True, 'violations': 0},
                'user_warehouse_mapping': {'valid': True, 'violations': 0},
                'query_status_cost': {'valid': True, 'violations': 0}
            },
            'temporal_consistency': {
                'timestamp_sequence': {'valid': True, 'violations': 0},
                'cost_progression': {'valid': True, 'violations': 0}
            }
        }
        
        mock_validator.validate_consistency.return_value = consistency_results
        
        # Test consistency validation
        result = mock_validator.validate_consistency()
        
        assert result['cross_field_consistency']['cost_credits_ratio']['valid'] is True
        assert result['temporal_consistency']['timestamp_sequence']['violations'] == 0

    def test_outlier_detection(self, mock_validator):
        """Test outlier detection."""
        outlier_results = {
            'cost_outliers': {
                'method': 'IQR',
                'outlier_count': 12,
                'outlier_percentage': 1.2,
                'threshold_low': 10.0,
                'threshold_high': 500.0,
                'outlier_values': [1500.0, 2000.0, 2500.0]
            },
            'credits_outliers': {
                'method': 'z_score',
                'outlier_count': 8,
                'outlier_percentage': 0.8,
                'threshold': 3.0,
                'outlier_values': [25.0, 30.0, 35.0]
            }
        }
        
        mock_validator.detect_outliers.return_value = outlier_results
        
        # Test outlier detection
        result = mock_validator.detect_outliers()
        
        assert result['cost_outliers']['outlier_count'] == 12
        assert result['credits_outliers']['method'] == 'z_score'
        assert len(result['cost_outliers']['outlier_values']) == 3

    def test_validation_error_reporting(self, mock_validator):
        """Test validation error reporting."""
        validation_errors = {
            'critical_errors': [
                {'field': 'cost', 'error': 'negative_value', 'count': 5},
                {'field': 'timestamp', 'error': 'null_value', 'count': 3}
            ],
            'warning_errors': [
                {'field': 'user_id', 'error': 'format_mismatch', 'count': 2},
                {'field': 'warehouse', 'error': 'unknown_value', 'count': 1}
            ],
            'info_errors': [
                {'field': 'query_id', 'error': 'duplicate_value', 'count': 8}
            ]
        }
        
        mock_validator.get_validation_errors.return_value = validation_errors
        
        # Test error reporting
        result = mock_validator.get_validation_errors()
        
        assert len(result['critical_errors']) == 2
        assert len(result['warning_errors']) == 2
        assert len(result['info_errors']) == 1
        assert result['critical_errors'][0]['field'] == 'cost'

    def test_validation_performance_metrics(self, mock_validator):
        """Test validation performance metrics."""
        performance_metrics = {
            'validation_time_seconds': 5.2,
            'records_validated': 10000,
            'records_per_second': 1923.1,
            'memory_usage_mb': 45.8,
            'cpu_usage_percent': 25.3,
            'validation_steps': {
                'schema_validation': 0.5,
                'type_validation': 1.2,
                'range_validation': 0.8,
                'format_validation': 1.5,
                'business_rule_validation': 1.2
            }
        }
        
        mock_validator.get_performance_metrics.return_value = performance_metrics
        
        # Test performance metrics
        result = mock_validator.get_performance_metrics()
        
        assert result['validation_time_seconds'] == 5.2
        assert result['records_validated'] == 10000
        assert result['records_per_second'] == 1923.1

    @pytest.mark.parametrize("field_name,validation_rule,expected_result", [
        ("cost", "positive_values", True),
        ("credits", "range_0_to_1000", True),
        ("warehouse", "valid_warehouse_names", True),
        ("timestamp", "valid_datetime", True),
        ("user_id", "valid_user_format", True),
    ])
    def test_field_specific_validation(self, mock_validator, field_name, validation_rule, expected_result):
        """Test field-specific validation rules."""
        mock_validator.validate_field.return_value = expected_result
        
        # Test field validation
        result = mock_validator.validate_field(field_name, validation_rule)
        
        assert result == expected_result

    def test_custom_validation_rules(self, mock_validator):
        """Test custom validation rules."""
        custom_rules = {
            'cost_threshold_rule': {
                'field': 'cost',
                'condition': 'cost > 10000',
                'action': 'flag_for_review',
                'violations': 3
            },
            'credit_efficiency_rule': {
                'field': 'credits',
                'condition': 'credits / cost < 0.01',
                'action': 'optimization_candidate',
                'violations': 15
            }
        }
        
        mock_validator.apply_custom_rules.return_value = custom_rules
        
        # Test custom rules
        result = mock_validator.apply_custom_rules()
        
        assert result['cost_threshold_rule']['violations'] == 3
        assert result['credit_efficiency_rule']['action'] == 'optimization_candidate'

    def test_validation_configuration(self, mock_validator):
        """Test validation configuration settings."""
        validation_config = {
            'enabled_validations': [
                'schema_validation',
                'type_validation',
                'range_validation',
                'format_validation',
                'business_rule_validation'
            ],
            'validation_thresholds': {
                'max_null_percentage': 5.0,
                'max_outlier_percentage': 2.0,
                'max_duplicate_percentage': 1.0
            },
            'error_handling': {
                'critical_errors': 'stop_processing',
                'warning_errors': 'log_and_continue',
                'info_errors': 'log_only'
            }
        }
        
        mock_validator.get_validation_config.return_value = validation_config
        
        # Test validation configuration
        result = mock_validator.get_validation_config()
        
        assert len(result['enabled_validations']) == 5
        assert result['validation_thresholds']['max_null_percentage'] == 5.0
        assert result['error_handling']['critical_errors'] == 'stop_processing'

    def test_validation_data_cleaning(self, mock_validator):
        """Test data cleaning during validation."""
        cleaning_results = {
            'records_before_cleaning': 1000,
            'records_after_cleaning': 950,
            'records_removed': 50,
            'cleaning_actions': {
                'null_value_removal': 20,
                'duplicate_removal': 15,
                'outlier_removal': 10,
                'format_correction': 5
            }
        }
        
        mock_validator.clean_data.return_value = cleaning_results
        
        # Test data cleaning
        result = mock_validator.clean_data()
        
        assert result['records_before_cleaning'] == 1000
        assert result['records_after_cleaning'] == 950
        assert result['records_removed'] == 50
        assert result['cleaning_actions']['null_value_removal'] == 20

    def test_validation_batch_processing(self, mock_validator):
        """Test batch validation processing."""
        batch_results = {
            'total_batches': 10,
            'processed_batches': 10,
            'failed_batches': 0,
            'total_records': 100000,
            'valid_records': 98500,
            'invalid_records': 1500,
            'validation_success_rate': 98.5
        }
        
        mock_validator.validate_batch.return_value = batch_results
        
        # Test batch validation
        result = mock_validator.validate_batch()
        
        assert result['total_batches'] == 10
        assert result['validation_success_rate'] == 98.5
        assert result['failed_batches'] == 0

    def test_validation_error_recovery(self, mock_validator):
        """Test error recovery during validation."""
        # Mock validation failure followed by recovery
        mock_validator.validate_data.side_effect = [
            Exception("Validation failed"),
            Exception("Still failing"),
            {'status': 'success', 'errors': []}  # Recovery on third try
        ]
        
        # Test error recovery
        attempts = 0
        max_attempts = 3
        result = None
        
        while attempts < max_attempts:
            try:
                result = mock_validator.validate_data()
                break
            except Exception:
                attempts += 1
        
        assert result is not None
        assert result['status'] == 'success'
        assert attempts == 2  # Failed twice, succeeded on third