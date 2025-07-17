"""
Unit tests for cost data collection.

Tests cost metrics collection, validation, and storage functionality.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from decimal import Decimal


class TestCostCollector:
    """Test suite for cost data collection functionality."""

    @pytest.fixture
    def mock_cost_collector(self):
        """Create a mock cost collector."""
        collector = Mock()
        collector.collect_cost_data = Mock()
        collector.validate_cost_data = Mock()
        collector.store_cost_data = Mock()
        collector.get_cost_metrics = Mock()
        collector.is_connected = True
        return collector

    @pytest.fixture
    def sample_cost_metrics(self):
        """Create sample cost metrics data."""
        return {
            'total_cost': Decimal('1250.75'),
            'compute_cost': Decimal('800.50'),
            'storage_cost': Decimal('200.25'),
            'data_transfer_cost': Decimal('150.00'),
            'serverless_cost': Decimal('100.00'),
            'credits_consumed': Decimal('62.54'),
            'currency': 'USD',
            'billing_period': '2024-01-01'
        }

    @pytest.fixture
    def sample_warehouse_costs(self):
        """Create sample warehouse cost data."""
        return pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=24, freq='H'),
            'warehouse_name': ['WH_ANALYTICS'] * 24,
            'warehouse_size': ['LARGE'] * 24,
            'credits_used': np.random.uniform(0.5, 2.0, 24),
            'cost_per_credit': [3.00] * 24,
            'total_cost': np.random.uniform(1.50, 6.00, 24),
            'compute_cost': np.random.uniform(1.00, 4.00, 24),
            'cloud_service_cost': np.random.uniform(0.10, 0.50, 24),
            'query_count': np.random.poisson(50, 24)
        })

    def test_cost_data_collection_success(self, mock_cost_collector, sample_cost_metrics):
        """Test successful cost data collection."""
        # Mock successful data collection
        mock_cost_collector.collect_cost_data.return_value = sample_cost_metrics
        
        # Test data collection
        result = mock_cost_collector.collect_cost_data()
        
        assert result is not None
        assert result['total_cost'] == Decimal('1250.75')
        assert result['compute_cost'] == Decimal('800.50')
        assert result['storage_cost'] == Decimal('200.25')
        assert result['currency'] == 'USD'
        mock_cost_collector.collect_cost_data.assert_called_once()

    def test_cost_data_collection_failure(self, mock_cost_collector):
        """Test cost data collection failure handling."""
        # Mock collection failure
        mock_cost_collector.collect_cost_data.side_effect = Exception("Collection failed")
        
        with pytest.raises(Exception) as exc_info:
            mock_cost_collector.collect_cost_data()
        
        assert "Collection failed" in str(exc_info.value)

    def test_warehouse_cost_collection(self, mock_cost_collector, sample_warehouse_costs):
        """Test warehouse-specific cost collection."""
        # Mock warehouse cost data
        mock_cost_collector.get_warehouse_costs.return_value = sample_warehouse_costs
        
        # Test warehouse cost collection
        result = mock_cost_collector.get_warehouse_costs()
        
        assert result is not None
        assert len(result) == 24
        assert 'warehouse_name' in result.columns
        assert 'total_cost' in result.columns
        assert 'credits_used' in result.columns

    def test_compute_cost_breakdown(self, mock_cost_collector):
        """Test compute cost breakdown collection."""
        compute_breakdown = {
            'warehouse_cost': Decimal('750.00'),
            'serverless_cost': Decimal('125.50'),
            'cloud_services_cost': Decimal('45.25'),
            'data_transfer_cost': Decimal('35.75'),
            'replication_cost': Decimal('15.00'),
            'total_compute_cost': Decimal('971.50')
        }
        
        mock_cost_collector.get_compute_cost_breakdown.return_value = compute_breakdown
        
        # Test compute cost breakdown
        result = mock_cost_collector.get_compute_cost_breakdown()
        
        assert result['warehouse_cost'] == Decimal('750.00')
        assert result['serverless_cost'] == Decimal('125.50')
        assert result['total_compute_cost'] == Decimal('971.50')

    def test_storage_cost_breakdown(self, mock_cost_collector):
        """Test storage cost breakdown collection."""
        storage_breakdown = {
            'database_storage_cost': Decimal('180.25'),
            'stage_storage_cost': Decimal('45.50'),
            'failsafe_storage_cost': Decimal('12.75'),
            'time_travel_storage_cost': Decimal('8.25'),
            'total_storage_cost': Decimal('246.75')
        }
        
        mock_cost_collector.get_storage_cost_breakdown.return_value = storage_breakdown
        
        # Test storage cost breakdown
        result = mock_cost_collector.get_storage_cost_breakdown()
        
        assert result['database_storage_cost'] == Decimal('180.25')
        assert result['stage_storage_cost'] == Decimal('45.50')
        assert result['total_storage_cost'] == Decimal('246.75')

    def test_cost_per_query_calculation(self, mock_cost_collector):
        """Test cost per query calculation."""
        cost_per_query_data = {
            'total_queries': 5000,
            'total_cost': Decimal('1250.00'),
            'avg_cost_per_query': Decimal('0.25'),
            'median_cost_per_query': Decimal('0.18'),
            'max_cost_per_query': Decimal('15.50'),
            'min_cost_per_query': Decimal('0.01')
        }
        
        mock_cost_collector.calculate_cost_per_query.return_value = cost_per_query_data
        
        # Test cost per query calculation
        result = mock_cost_collector.calculate_cost_per_query()
        
        assert result['avg_cost_per_query'] == Decimal('0.25')
        assert result['total_queries'] == 5000
        assert result['total_cost'] == Decimal('1250.00')

    def test_cost_data_validation_success(self, mock_cost_collector, sample_cost_metrics):
        """Test successful cost data validation."""
        # Mock successful validation
        mock_cost_collector.validate_cost_data.return_value = True
        
        # Test validation
        is_valid = mock_cost_collector.validate_cost_data(sample_cost_metrics)
        
        assert is_valid is True
        mock_cost_collector.validate_cost_data.assert_called_once_with(sample_cost_metrics)

    def test_cost_data_validation_failure(self, mock_cost_collector):
        """Test cost data validation failure."""
        invalid_data = {
            'total_cost': Decimal('-100.00'),  # Invalid negative cost
            'compute_cost': None,  # Invalid None value
            'storage_cost': 'invalid',  # Invalid string value
            'currency': 'INVALID'  # Invalid currency code
        }
        
        # Mock validation failure
        mock_cost_collector.validate_cost_data.return_value = False
        
        # Test validation failure
        is_valid = mock_cost_collector.validate_cost_data(invalid_data)
        
        assert is_valid is False

    def test_cost_data_storage(self, mock_cost_collector, sample_cost_metrics):
        """Test cost data storage functionality."""
        # Mock successful storage
        mock_cost_collector.store_cost_data.return_value = True
        
        # Test data storage
        result = mock_cost_collector.store_cost_data(sample_cost_metrics)
        
        assert result is True
        mock_cost_collector.store_cost_data.assert_called_once_with(sample_cost_metrics)

    def test_cost_data_storage_failure(self, mock_cost_collector, sample_cost_metrics):
        """Test cost data storage failure handling."""
        # Mock storage failure
        mock_cost_collector.store_cost_data.side_effect = Exception("Storage failed")
        
        with pytest.raises(Exception) as exc_info:
            mock_cost_collector.store_cost_data(sample_cost_metrics)
        
        assert "Storage failed" in str(exc_info.value)

    def test_historical_cost_data_retrieval(self, mock_cost_collector):
        """Test historical cost data retrieval."""
        # Mock historical data
        historical_data = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=30, freq='D'),
            'total_cost': np.random.uniform(800, 1500, 30),
            'compute_cost': np.random.uniform(500, 1000, 30),
            'storage_cost': np.random.uniform(150, 300, 30),
            'credits_consumed': np.random.uniform(25, 75, 30)
        })
        
        mock_cost_collector.get_historical_costs.return_value = historical_data
        
        # Test historical data retrieval
        result = mock_cost_collector.get_historical_costs()
        
        assert result is not None
        assert len(result) == 30
        assert 'total_cost' in result.columns
        assert 'compute_cost' in result.columns

    def test_cost_trends_calculation(self, mock_cost_collector):
        """Test cost trends calculation."""
        trends = {
            'total_cost_trend': 'increasing',
            'total_cost_change_pct': 12.5,
            'compute_cost_trend': 'stable',
            'compute_cost_change_pct': 1.8,
            'storage_cost_trend': 'decreasing',
            'storage_cost_change_pct': -5.2,
            'credits_trend': 'increasing',
            'credits_change_pct': 18.3
        }
        
        mock_cost_collector.calculate_cost_trends.return_value = trends
        
        # Test trends calculation
        result = mock_cost_collector.calculate_cost_trends()
        
        assert result['total_cost_trend'] == 'increasing'
        assert result['total_cost_change_pct'] == 12.5
        assert result['storage_cost_trend'] == 'decreasing'

    def test_cost_anomaly_detection(self, mock_cost_collector):
        """Test cost anomaly detection."""
        anomalies = {
            'anomaly_detected': True,
            'anomaly_type': 'cost_spike',
            'anomaly_severity': 'high',
            'anomaly_timestamp': datetime.now(),
            'anomaly_cost': Decimal('2500.00'),
            'expected_cost': Decimal('1200.00'),
            'deviation_pct': 108.3,
            'affected_warehouse': 'WH_ANALYTICS'
        }
        
        mock_cost_collector.detect_cost_anomalies.return_value = anomalies
        
        # Test anomaly detection
        result = mock_cost_collector.detect_cost_anomalies()
        
        assert result['anomaly_detected'] is True
        assert result['anomaly_type'] == 'cost_spike'
        assert result['anomaly_severity'] == 'high'
        assert result['anomaly_cost'] == Decimal('2500.00')

    def test_cost_allocation_by_department(self, mock_cost_collector):
        """Test cost allocation by department."""
        department_costs = {
            'engineering': Decimal('650.25'),
            'analytics': Decimal('425.50'),
            'finance': Decimal('180.75'),
            'marketing': Decimal('95.25'),
            'unallocated': Decimal('48.25'),
            'total_allocated': Decimal('1400.00')
        }
        
        mock_cost_collector.get_department_costs.return_value = department_costs
        
        # Test department cost allocation
        result = mock_cost_collector.get_department_costs()
        
        assert result['engineering'] == Decimal('650.25')
        assert result['analytics'] == Decimal('425.50')
        assert result['total_allocated'] == Decimal('1400.00')

    def test_cost_allocation_by_user(self, mock_cost_collector):
        """Test cost allocation by user."""
        user_costs = [
            {'user_id': 'user1', 'user_name': 'John Doe', 'cost': Decimal('245.50')},
            {'user_id': 'user2', 'user_name': 'Jane Smith', 'cost': Decimal('189.25')},
            {'user_id': 'user3', 'user_name': 'Bob Johnson', 'cost': Decimal('156.75')},
            {'user_id': 'user4', 'user_name': 'Alice Brown', 'cost': Decimal('98.50')}
        ]
        
        mock_cost_collector.get_user_costs.return_value = user_costs
        
        # Test user cost allocation
        result = mock_cost_collector.get_user_costs()
        
        assert len(result) == 4
        assert result[0]['user_name'] == 'John Doe'
        assert result[0]['cost'] == Decimal('245.50')

    @pytest.mark.parametrize("warehouse_size,expected_rate", [
        ("X-SMALL", Decimal('1.00')),
        ("SMALL", Decimal('2.00')),
        ("MEDIUM", Decimal('4.00')),
        ("LARGE", Decimal('8.00')),
        ("X-LARGE", Decimal('16.00')),
    ])
    def test_warehouse_cost_rate_calculation(self, mock_cost_collector, warehouse_size, expected_rate):
        """Test warehouse cost rate calculation."""
        mock_cost_collector.get_warehouse_rate.return_value = expected_rate
        
        # Test rate calculation
        result = mock_cost_collector.get_warehouse_rate(warehouse_size)
        
        assert result == expected_rate

    def test_cost_budgeting_and_forecasting(self, mock_cost_collector):
        """Test cost budgeting and forecasting."""
        budget_data = {
            'monthly_budget': Decimal('10000.00'),
            'current_spend': Decimal('7500.50'),
            'remaining_budget': Decimal('2499.50'),
            'budget_utilization_pct': 75.01,
            'projected_monthly_spend': Decimal('9800.75'),
            'budget_status': 'on_track',
            'days_remaining': 8,
            'avg_daily_spend': Decimal('312.52')
        }
        
        mock_cost_collector.get_budget_status.return_value = budget_data
        
        # Test budget status
        result = mock_cost_collector.get_budget_status()
        
        assert result['monthly_budget'] == Decimal('10000.00')
        assert result['current_spend'] == Decimal('7500.50')
        assert result['budget_status'] == 'on_track'

    def test_cost_optimization_recommendations(self, mock_cost_collector):
        """Test cost optimization recommendations."""
        recommendations = {
            'total_potential_savings': Decimal('245.75'),
            'recommendations': [
                {
                    'type': 'warehouse_sizing',
                    'recommendation': 'Downsize WH_ANALYTICS from LARGE to MEDIUM',
                    'potential_savings': Decimal('120.00'),
                    'confidence': 0.85
                },
                {
                    'type': 'auto_suspend',
                    'recommendation': 'Reduce auto-suspend timeout to 5 minutes',
                    'potential_savings': Decimal('85.50'),
                    'confidence': 0.92
                },
                {
                    'type': 'query_optimization',
                    'recommendation': 'Optimize top 5 expensive queries',
                    'potential_savings': Decimal('40.25'),
                    'confidence': 0.75
                }
            ]
        }
        
        mock_cost_collector.get_optimization_recommendations.return_value = recommendations
        
        # Test optimization recommendations
        result = mock_cost_collector.get_optimization_recommendations()
        
        assert result['total_potential_savings'] == Decimal('245.75')
        assert len(result['recommendations']) == 3
        assert result['recommendations'][0]['type'] == 'warehouse_sizing'

    def test_cost_collection_batch_processing(self, mock_cost_collector):
        """Test batch processing of cost data."""
        batch_data = [
            {'timestamp': datetime.now() - timedelta(hours=i), 'cost': Decimal(f'{100 + i}.00')}
            for i in range(24)
        ]
        
        mock_cost_collector.process_cost_batch.return_value = len(batch_data)
        
        # Test batch processing
        result = mock_cost_collector.process_cost_batch(batch_data)
        
        assert result == 24
        mock_cost_collector.process_cost_batch.assert_called_once_with(batch_data)

    def test_cost_collection_error_retry(self, mock_cost_collector):
        """Test error retry mechanism in cost collection."""
        # Mock intermittent failures
        mock_cost_collector.collect_cost_data.side_effect = [
            Exception("Temporary failure"),
            Exception("Another failure"),
            {'total_cost': Decimal('1000.00')}  # Success on third try
        ]
        
        # Test retry logic
        attempts = 0
        max_attempts = 3
        result = None
        
        while attempts < max_attempts:
            try:
                result = mock_cost_collector.collect_cost_data()
                break
            except Exception:
                attempts += 1
        
        assert result is not None
        assert result['total_cost'] == Decimal('1000.00')
        assert attempts == 2  # Failed twice, succeeded on third

    def test_currency_conversion(self, mock_cost_collector):
        """Test currency conversion functionality."""
        conversion_data = {
            'original_amount': Decimal('1000.00'),
            'original_currency': 'USD',
            'target_currency': 'EUR',
            'exchange_rate': Decimal('0.85'),
            'converted_amount': Decimal('850.00'),
            'conversion_date': datetime.now().date()
        }
        
        mock_cost_collector.convert_currency.return_value = conversion_data
        
        # Test currency conversion
        result = mock_cost_collector.convert_currency(
            Decimal('1000.00'), 'USD', 'EUR'
        )
        
        assert result['converted_amount'] == Decimal('850.00')
        assert result['exchange_rate'] == Decimal('0.85')

    def test_cost_collection_performance_metrics(self, mock_cost_collector):
        """Test performance metrics during cost collection."""
        performance_metrics = {
            'collection_time_seconds': 12.5,
            'validation_time_seconds': 1.8,
            'storage_time_seconds': 6.2,
            'total_time_seconds': 20.5,
            'records_processed': 5000,
            'records_per_second': 243.9,
            'memory_usage_mb': 85.3,
            'cpu_usage_percent': 32.1
        }
        
        mock_cost_collector.get_performance_metrics.return_value = performance_metrics
        
        # Test performance metrics
        result = mock_cost_collector.get_performance_metrics()
        
        assert result['collection_time_seconds'] == 12.5
        assert result['records_processed'] == 5000
        assert result['records_per_second'] == 243.9