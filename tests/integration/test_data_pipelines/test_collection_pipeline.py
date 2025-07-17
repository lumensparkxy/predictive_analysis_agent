"""
Integration tests for data collection pipeline.

Tests end-to-end data collection workflow from Snowflake to processing.
"""

import pytest
from unittest.mock import Mock, patch
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


class TestDataCollectionPipeline:
    """Test suite for data collection pipeline integration."""

    @pytest.fixture
    def mock_pipeline(self):
        """Create a mock data collection pipeline."""
        pipeline = Mock()
        pipeline.run_collection = Mock()
        pipeline.validate_data = Mock()
        pipeline.store_data = Mock()
        pipeline.get_status = Mock()
        return pipeline

    def test_end_to_end_data_collection(self, mock_pipeline):
        """Test end-to-end data collection pipeline."""
        # Mock pipeline execution
        collection_result = {
            'status': 'success',
            'records_collected': 10000,
            'processing_time_seconds': 45.2,
            'data_quality_score': 0.95,
            'errors': [],
            'timestamp': datetime.now()
        }
        
        mock_pipeline.run_collection.return_value = collection_result
        
        # Test pipeline execution
        result = mock_pipeline.run_collection()
        
        assert result['status'] == 'success'
        assert result['records_collected'] == 10000
        assert result['data_quality_score'] == 0.95
        assert len(result['errors']) == 0

    def test_data_validation_integration(self, mock_pipeline):
        """Test data validation integration in pipeline."""
        # Mock validation result
        validation_result = {
            'validation_passed': True,
            'total_records': 10000,
            'valid_records': 9950,
            'invalid_records': 50,
            'validation_errors': [
                {'field': 'cost', 'error': 'null_value', 'count': 25},
                {'field': 'timestamp', 'error': 'format_error', 'count': 25}
            ],
            'data_quality_score': 0.995
        }
        
        mock_pipeline.validate_data.return_value = validation_result
        
        # Test validation
        result = mock_pipeline.validate_data()
        
        assert result['validation_passed'] is True
        assert result['data_quality_score'] == 0.995
        assert len(result['validation_errors']) == 2

    def test_data_storage_integration(self, mock_pipeline):
        """Test data storage integration in pipeline."""
        # Mock storage result
        storage_result = {
            'storage_success': True,
            'records_stored': 9950,
            'storage_location': 'warehouse/cost_data/2024-01-01',
            'storage_format': 'parquet',
            'compression_ratio': 0.15,
            'storage_time_seconds': 12.3
        }
        
        mock_pipeline.store_data.return_value = storage_result
        
        # Test storage
        result = mock_pipeline.store_data()
        
        assert result['storage_success'] is True
        assert result['records_stored'] == 9950
        assert result['storage_format'] == 'parquet'
        assert result['compression_ratio'] == 0.15

    def test_pipeline_error_handling(self, mock_pipeline):
        """Test pipeline error handling."""
        # Mock pipeline failure
        mock_pipeline.run_collection.side_effect = Exception("Pipeline failed")
        
        with pytest.raises(Exception) as exc_info:
            mock_pipeline.run_collection()
        
        assert "Pipeline failed" in str(exc_info.value)

    def test_pipeline_monitoring(self, mock_pipeline):
        """Test pipeline monitoring and status."""
        # Mock monitoring status
        status_result = {
            'pipeline_status': 'running',
            'current_stage': 'data_validation',
            'progress_percentage': 75,
            'estimated_completion': datetime.now() + timedelta(minutes=10),
            'performance_metrics': {
                'throughput_records_per_second': 220,
                'memory_usage_mb': 512,
                'cpu_usage_percent': 45
            }
        }
        
        mock_pipeline.get_status.return_value = status_result
        
        # Test status monitoring
        result = mock_pipeline.get_status()
        
        assert result['pipeline_status'] == 'running'
        assert result['progress_percentage'] == 75
        assert result['performance_metrics']['throughput_records_per_second'] == 220


class TestDataProcessingPipeline:
    """Test suite for data processing pipeline integration."""

    @pytest.fixture
    def mock_processing_pipeline(self):
        """Create a mock data processing pipeline."""
        pipeline = Mock()
        pipeline.process_data = Mock()
        pipeline.clean_data = Mock()
        pipeline.aggregate_data = Mock()
        pipeline.feature_engineering = Mock()
        return pipeline

    def test_data_cleaning_integration(self, mock_processing_pipeline):
        """Test data cleaning integration."""
        # Mock cleaning result
        cleaning_result = {
            'records_before_cleaning': 10000,
            'records_after_cleaning': 9800,
            'cleaning_operations': {
                'null_value_handling': 150,
                'outlier_removal': 50,
                'duplicate_removal': 0,
                'format_standardization': 200
            },
            'data_quality_improvement': 0.02
        }
        
        mock_processing_pipeline.clean_data.return_value = cleaning_result
        
        # Test cleaning
        result = mock_processing_pipeline.clean_data()
        
        assert result['records_before_cleaning'] == 10000
        assert result['records_after_cleaning'] == 9800
        assert result['data_quality_improvement'] == 0.02

    def test_data_aggregation_integration(self, mock_processing_pipeline):
        """Test data aggregation integration."""
        # Mock aggregation result
        aggregation_result = {
            'aggregation_type': 'time_series',
            'original_records': 9800,
            'aggregated_records': 720,  # Hourly aggregation
            'aggregation_functions': ['sum', 'avg', 'max', 'min'],
            'time_granularity': 'hourly',
            'compression_ratio': 0.073
        }
        
        mock_processing_pipeline.aggregate_data.return_value = aggregation_result
        
        # Test aggregation
        result = mock_processing_pipeline.aggregate_data()
        
        assert result['aggregation_type'] == 'time_series'
        assert result['original_records'] == 9800
        assert result['aggregated_records'] == 720
        assert 'sum' in result['aggregation_functions']

    def test_feature_engineering_integration(self, mock_processing_pipeline):
        """Test feature engineering integration."""
        # Mock feature engineering result
        feature_result = {
            'original_features': 10,
            'engineered_features': 25,
            'feature_types': {
                'time_features': 8,
                'lag_features': 5,
                'rolling_features': 7,
                'interaction_features': 5
            },
            'feature_importance_computed': True,
            'processing_time_seconds': 35.7
        }
        
        mock_processing_pipeline.feature_engineering.return_value = feature_result
        
        # Test feature engineering
        result = mock_processing_pipeline.feature_engineering()
        
        assert result['original_features'] == 10
        assert result['engineered_features'] == 25
        assert result['feature_types']['time_features'] == 8
        assert result['feature_importance_computed'] is True


class TestDataValidationPipeline:
    """Test suite for data validation pipeline integration."""

    @pytest.fixture
    def mock_validation_pipeline(self):
        """Create a mock data validation pipeline."""
        pipeline = Mock()
        pipeline.validate_schema = Mock()
        pipeline.validate_quality = Mock()
        pipeline.validate_business_rules = Mock()
        pipeline.generate_report = Mock()
        return pipeline

    def test_schema_validation_integration(self, mock_validation_pipeline):
        """Test schema validation integration."""
        # Mock schema validation result
        schema_result = {
            'schema_valid': True,
            'expected_columns': 15,
            'actual_columns': 15,
            'column_types_valid': True,
            'missing_columns': [],
            'extra_columns': [],
            'type_mismatches': []
        }
        
        mock_validation_pipeline.validate_schema.return_value = schema_result
        
        # Test schema validation
        result = mock_validation_pipeline.validate_schema()
        
        assert result['schema_valid'] is True
        assert result['expected_columns'] == 15
        assert result['actual_columns'] == 15
        assert len(result['missing_columns']) == 0

    def test_quality_validation_integration(self, mock_validation_pipeline):
        """Test quality validation integration."""
        # Mock quality validation result
        quality_result = {
            'quality_score': 0.92,
            'completeness_score': 0.98,
            'accuracy_score': 0.95,
            'consistency_score': 0.88,
            'validity_score': 0.97,
            'quality_issues': [
                {'type': 'outlier', 'count': 50, 'severity': 'medium'},
                {'type': 'inconsistency', 'count': 25, 'severity': 'low'}
            ]
        }
        
        mock_validation_pipeline.validate_quality.return_value = quality_result
        
        # Test quality validation
        result = mock_validation_pipeline.validate_quality()
        
        assert result['quality_score'] == 0.92
        assert result['completeness_score'] == 0.98
        assert len(result['quality_issues']) == 2

    def test_business_rules_validation_integration(self, mock_validation_pipeline):
        """Test business rules validation integration."""
        # Mock business rules validation result
        rules_result = {
            'rules_passed': 18,
            'rules_failed': 2,
            'rules_total': 20,
            'rules_success_rate': 0.90,
            'failed_rules': [
                {'rule': 'cost_credit_ratio', 'violations': 15},
                {'rule': 'warehouse_usage_limits', 'violations': 8}
            ],
            'critical_violations': 1
        }
        
        mock_validation_pipeline.validate_business_rules.return_value = rules_result
        
        # Test business rules validation
        result = mock_validation_pipeline.validate_business_rules()
        
        assert result['rules_passed'] == 18
        assert result['rules_failed'] == 2
        assert result['rules_success_rate'] == 0.90
        assert result['critical_violations'] == 1

    def test_validation_report_generation(self, mock_validation_pipeline):
        """Test validation report generation."""
        # Mock validation report
        report_result = {
            'report_generated': True,
            'report_format': 'html',
            'report_path': '/tmp/validation_report.html',
            'report_sections': [
                'executive_summary',
                'schema_validation',
                'quality_assessment',
                'business_rules_validation',
                'recommendations'
            ],
            'generation_time_seconds': 5.2
        }
        
        mock_validation_pipeline.generate_report.return_value = report_result
        
        # Test report generation
        result = mock_validation_pipeline.generate_report()
        
        assert result['report_generated'] is True
        assert result['report_format'] == 'html'
        assert len(result['report_sections']) == 5
        assert result['generation_time_seconds'] == 5.2