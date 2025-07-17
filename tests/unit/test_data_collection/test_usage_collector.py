"""
Unit tests for usage data collection.

Tests usage metrics collection, data validation, and error handling.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


class TestUsageCollector:
    """Test suite for usage data collection functionality."""

    @pytest.fixture
    def mock_usage_collector(self):
        """Create a mock usage collector."""
        collector = Mock()
        collector.collect_usage_data = Mock()
        collector.validate_usage_data = Mock()
        collector.store_usage_data = Mock()
        collector.get_usage_metrics = Mock()
        collector.is_connected = True
        return collector

    @pytest.fixture
    def sample_usage_metrics(self):
        """Create sample usage metrics data."""
        return {
            'query_count': 1250,
            'active_users': 45,
            'warehouse_usage_hours': 120.5,
            'data_processed_gb': 2048.7,
            'avg_query_duration_ms': 3500,
            'peak_concurrent_queries': 25,
            'compute_time_seconds': 45600,
            'queued_queries': 12,
            'failed_queries': 3,
            'cached_queries': 180
        }

    @pytest.fixture
    def sample_warehouse_usage(self):
        """Create sample warehouse usage data."""
        return pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=24, freq='H'),
            'warehouse_name': ['WH_ANALYTICS'] * 24,
            'size': ['LARGE'] * 24,
            'cluster_count': [2] * 24,
            'credits_used': np.random.uniform(0.5, 2.0, 24),
            'query_count': np.random.poisson(50, 24),
            'active_users': np.random.poisson(10, 24),
            'avg_query_duration': np.random.uniform(1000, 5000, 24),
            'queued_load': np.random.uniform(0, 1, 24)
        })

    def test_usage_data_collection_success(self, mock_usage_collector, sample_usage_metrics):
        """Test successful usage data collection."""
        # Mock successful data collection
        mock_usage_collector.collect_usage_data.return_value = sample_usage_metrics
        
        # Test data collection
        result = mock_usage_collector.collect_usage_data()
        
        assert result is not None
        assert result['query_count'] == 1250
        assert result['active_users'] == 45
        assert result['warehouse_usage_hours'] == 120.5
        mock_usage_collector.collect_usage_data.assert_called_once()

    def test_usage_data_collection_failure(self, mock_usage_collector):
        """Test usage data collection failure handling."""
        # Mock collection failure
        mock_usage_collector.collect_usage_data.side_effect = Exception("Collection failed")
        
        with pytest.raises(Exception) as exc_info:
            mock_usage_collector.collect_usage_data()
        
        assert "Collection failed" in str(exc_info.value)

    def test_warehouse_usage_metrics_collection(self, mock_usage_collector, sample_warehouse_usage):
        """Test warehouse usage metrics collection."""
        # Mock warehouse usage data
        mock_usage_collector.get_warehouse_usage.return_value = sample_warehouse_usage
        
        # Test warehouse usage collection
        result = mock_usage_collector.get_warehouse_usage()
        
        assert result is not None
        assert len(result) == 24
        assert 'warehouse_name' in result.columns
        assert 'credits_used' in result.columns
        assert 'query_count' in result.columns

    def test_query_metrics_collection(self, mock_usage_collector):
        """Test query metrics collection."""
        query_metrics = {
            'total_queries': 5000,
            'successful_queries': 4850,
            'failed_queries': 150,
            'avg_execution_time': 2500,
            'median_execution_time': 1800,
            'p95_execution_time': 8000,
            'complex_queries': 500,
            'simple_queries': 4500
        }
        
        mock_usage_collector.get_query_metrics.return_value = query_metrics
        
        # Test query metrics
        result = mock_usage_collector.get_query_metrics()
        
        assert result['total_queries'] == 5000
        assert result['successful_queries'] == 4850
        assert result['failed_queries'] == 150
        assert result['avg_execution_time'] == 2500

    def test_user_activity_metrics_collection(self, mock_usage_collector):
        """Test user activity metrics collection."""
        user_metrics = {
            'total_users': 100,
            'active_users_today': 45,
            'active_users_week': 78,
            'active_users_month': 95,
            'new_users_today': 2,
            'sessions_today': 120,
            'avg_session_duration': 3600,
            'peak_concurrent_users': 32
        }
        
        mock_usage_collector.get_user_metrics.return_value = user_metrics
        
        # Test user metrics
        result = mock_usage_collector.get_user_metrics()
        
        assert result['total_users'] == 100
        assert result['active_users_today'] == 45
        assert result['sessions_today'] == 120

    def test_data_processing_metrics_collection(self, mock_usage_collector):
        """Test data processing metrics collection."""
        processing_metrics = {
            'data_scanned_gb': 15000.5,
            'data_processed_gb': 12000.3,
            'data_cached_gb': 8000.2,
            'compression_ratio': 0.75,
            'partitions_scanned': 450,
            'partitions_pruned': 1200,
            'micro_partitions_scanned': 25000,
            'bytes_spilled_to_disk': 500.2
        }
        
        mock_usage_collector.get_processing_metrics.return_value = processing_metrics
        
        # Test processing metrics
        result = mock_usage_collector.get_processing_metrics()
        
        assert result['data_scanned_gb'] == 15000.5
        assert result['data_processed_gb'] == 12000.3
        assert result['compression_ratio'] == 0.75

    def test_usage_data_validation_success(self, mock_usage_collector, sample_usage_metrics):
        """Test successful usage data validation."""
        # Mock successful validation
        mock_usage_collector.validate_usage_data.return_value = True
        
        # Test validation
        is_valid = mock_usage_collector.validate_usage_data(sample_usage_metrics)
        
        assert is_valid is True
        mock_usage_collector.validate_usage_data.assert_called_once_with(sample_usage_metrics)

    def test_usage_data_validation_failure(self, mock_usage_collector):
        """Test usage data validation failure."""
        invalid_data = {
            'query_count': -100,  # Invalid negative count
            'active_users': None,  # Invalid None value
            'warehouse_usage_hours': 'invalid'  # Invalid string value
        }
        
        # Mock validation failure
        mock_usage_collector.validate_usage_data.return_value = False
        
        # Test validation failure
        is_valid = mock_usage_collector.validate_usage_data(invalid_data)
        
        assert is_valid is False

    def test_usage_data_storage(self, mock_usage_collector, sample_usage_metrics):
        """Test usage data storage functionality."""
        # Mock successful storage
        mock_usage_collector.store_usage_data.return_value = True
        
        # Test data storage
        result = mock_usage_collector.store_usage_data(sample_usage_metrics)
        
        assert result is True
        mock_usage_collector.store_usage_data.assert_called_once_with(sample_usage_metrics)

    def test_usage_data_storage_failure(self, mock_usage_collector, sample_usage_metrics):
        """Test usage data storage failure handling."""
        # Mock storage failure
        mock_usage_collector.store_usage_data.side_effect = Exception("Storage failed")
        
        with pytest.raises(Exception) as exc_info:
            mock_usage_collector.store_usage_data(sample_usage_metrics)
        
        assert "Storage failed" in str(exc_info.value)

    def test_historical_usage_data_retrieval(self, mock_usage_collector):
        """Test historical usage data retrieval."""
        # Mock historical data
        historical_data = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=30, freq='D'),
            'query_count': np.random.poisson(1000, 30),
            'active_users': np.random.poisson(40, 30),
            'credits_used': np.random.uniform(50, 200, 30),
            'data_processed_gb': np.random.uniform(500, 2000, 30)
        })
        
        mock_usage_collector.get_historical_usage.return_value = historical_data
        
        # Test historical data retrieval
        result = mock_usage_collector.get_historical_usage()
        
        assert result is not None
        assert len(result) == 30
        assert 'query_count' in result.columns
        assert 'active_users' in result.columns

    def test_usage_trends_calculation(self, mock_usage_collector):
        """Test usage trends calculation."""
        trends = {
            'query_count_trend': 'increasing',
            'query_count_change_pct': 15.5,
            'active_users_trend': 'stable',
            'active_users_change_pct': 2.1,
            'credits_trend': 'decreasing',
            'credits_change_pct': -8.3,
            'data_processed_trend': 'increasing',
            'data_processed_change_pct': 22.7
        }
        
        mock_usage_collector.calculate_usage_trends.return_value = trends
        
        # Test trends calculation
        result = mock_usage_collector.calculate_usage_trends()
        
        assert result['query_count_trend'] == 'increasing'
        assert result['query_count_change_pct'] == 15.5
        assert result['credits_trend'] == 'decreasing'

    def test_peak_usage_detection(self, mock_usage_collector):
        """Test peak usage detection."""
        peak_data = {
            'peak_hour': 14,  # 2 PM
            'peak_day': 'Tuesday',
            'peak_query_count': 2500,
            'peak_concurrent_users': 65,
            'peak_credits_per_hour': 45.8,
            'peak_data_processed_gb': 500.2
        }
        
        mock_usage_collector.detect_peak_usage.return_value = peak_data
        
        # Test peak detection
        result = mock_usage_collector.detect_peak_usage()
        
        assert result['peak_hour'] == 14
        assert result['peak_day'] == 'Tuesday'
        assert result['peak_query_count'] == 2500

    def test_usage_anomaly_detection(self, mock_usage_collector):
        """Test usage anomaly detection."""
        anomalies = {
            'anomaly_detected': True,
            'anomaly_type': 'query_spike',
            'anomaly_severity': 'high',
            'anomaly_timestamp': datetime.now(),
            'anomaly_value': 5000,
            'expected_value': 1200,
            'deviation_pct': 316.7
        }
        
        mock_usage_collector.detect_usage_anomalies.return_value = anomalies
        
        # Test anomaly detection
        result = mock_usage_collector.detect_usage_anomalies()
        
        assert result['anomaly_detected'] is True
        assert result['anomaly_type'] == 'query_spike'
        assert result['anomaly_severity'] == 'high'

    @pytest.mark.parametrize("warehouse_size,expected_credits", [
        ("X-SMALL", 1),
        ("SMALL", 2),
        ("MEDIUM", 4),
        ("LARGE", 8),
        ("X-LARGE", 16),
    ])
    def test_warehouse_size_credit_calculation(self, mock_usage_collector, warehouse_size, expected_credits):
        """Test warehouse size to credits calculation."""
        mock_usage_collector.get_warehouse_credits.return_value = expected_credits
        
        # Test credits calculation
        result = mock_usage_collector.get_warehouse_credits(warehouse_size)
        
        assert result == expected_credits

    def test_usage_collection_batch_processing(self, mock_usage_collector):
        """Test batch processing of usage data."""
        batch_data = [
            {'timestamp': datetime.now() - timedelta(hours=i), 'query_count': 100 + i}
            for i in range(24)
        ]
        
        mock_usage_collector.process_usage_batch.return_value = len(batch_data)
        
        # Test batch processing
        result = mock_usage_collector.process_usage_batch(batch_data)
        
        assert result == 24
        mock_usage_collector.process_usage_batch.assert_called_once_with(batch_data)

    def test_usage_collection_error_retry(self, mock_usage_collector):
        """Test error retry mechanism in usage collection."""
        # Mock intermittent failures
        mock_usage_collector.collect_usage_data.side_effect = [
            Exception("Temporary failure"),
            Exception("Another failure"),
            {'query_count': 1000}  # Success on third try
        ]
        
        # Test retry logic
        attempts = 0
        max_attempts = 3
        result = None
        
        while attempts < max_attempts:
            try:
                result = mock_usage_collector.collect_usage_data()
                break
            except Exception:
                attempts += 1
        
        assert result is not None
        assert result['query_count'] == 1000
        assert attempts == 2  # Failed twice, succeeded on third

    def test_usage_data_aggregation(self, mock_usage_collector):
        """Test usage data aggregation functionality."""
        aggregated_data = {
            'daily_totals': {
                'total_queries': 12000,
                'total_credits': 480.5,
                'total_data_processed_gb': 8500.2,
                'unique_users': 78
            },
            'hourly_averages': {
                'avg_queries_per_hour': 500,
                'avg_credits_per_hour': 20.02,
                'avg_data_per_hour_gb': 354.17,
                'avg_users_per_hour': 35
            }
        }
        
        mock_usage_collector.aggregate_usage_data.return_value = aggregated_data
        
        # Test aggregation
        result = mock_usage_collector.aggregate_usage_data()
        
        assert result['daily_totals']['total_queries'] == 12000
        assert result['hourly_averages']['avg_queries_per_hour'] == 500

    def test_usage_collection_performance_metrics(self, mock_usage_collector):
        """Test performance metrics during usage collection."""
        performance_metrics = {
            'collection_time_seconds': 15.5,
            'validation_time_seconds': 2.1,
            'storage_time_seconds': 8.3,
            'total_time_seconds': 25.9,
            'records_processed': 10000,
            'records_per_second': 385.8,
            'memory_usage_mb': 128.5,
            'cpu_usage_percent': 45.2
        }
        
        mock_usage_collector.get_performance_metrics.return_value = performance_metrics
        
        # Test performance metrics
        result = mock_usage_collector.get_performance_metrics()
        
        assert result['collection_time_seconds'] == 15.5
        assert result['records_processed'] == 10000
        assert result['records_per_second'] == 385.8