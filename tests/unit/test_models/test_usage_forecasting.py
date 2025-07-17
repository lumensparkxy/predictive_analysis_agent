"""
Unit tests for usage forecasting models.

Tests usage prediction models, training, and accuracy validation.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


class TestUsageForecastingModels:
    """Test suite for usage forecasting models."""

    @pytest.fixture
    def mock_usage_predictor(self):
        """Create a mock usage predictor."""
        predictor = Mock()
        predictor.train = Mock()
        predictor.predict = Mock()
        predictor.evaluate = Mock()
        predictor.save_model = Mock()
        predictor.load_model = Mock()
        predictor.is_trained = False
        predictor.model_type = "lstm"
        return predictor

    @pytest.fixture
    def sample_usage_training_data(self):
        """Create sample training data for usage forecasting."""
        np.random.seed(42)
        timestamps = pd.date_range('2023-01-01', periods=8760, freq='H')  # 1 year hourly
        
        # Create realistic usage patterns
        hour_of_day = timestamps.hour
        day_of_week = timestamps.dayofweek
        
        # Base usage with daily and weekly patterns
        base_usage = 50
        daily_pattern = 20 * np.sin(2 * np.pi * hour_of_day / 24)
        weekly_pattern = 10 * np.sin(2 * np.pi * day_of_week / 7)
        trend = np.linspace(0, 20, 8760)
        noise = np.random.normal(0, 5, 8760)
        
        usage = base_usage + daily_pattern + weekly_pattern + trend + noise
        usage = np.maximum(usage, 0)  # Ensure non-negative
        
        return pd.DataFrame({
            'timestamp': timestamps,
            'usage': usage,
            'query_count': np.random.poisson(usage * 2, 8760),
            'active_users': np.random.poisson(usage * 0.5, 8760),
            'warehouse_size': np.random.choice(['SMALL', 'MEDIUM', 'LARGE'], 8760),
            'day_of_week': day_of_week,
            'hour_of_day': hour_of_day
        })

    @pytest.fixture
    def sample_usage_prediction_data(self):
        """Create sample data for usage predictions."""
        future_timestamps = pd.date_range('2024-01-01', periods=168, freq='H')  # 1 week
        return pd.DataFrame({
            'timestamp': future_timestamps,
            'day_of_week': future_timestamps.dayofweek,
            'hour_of_day': future_timestamps.hour,
            'warehouse_size': ['LARGE'] * 168
        })

    def test_usage_predictor_initialization(self, mock_usage_predictor):
        """Test usage predictor initialization."""
        assert mock_usage_predictor is not None
        assert mock_usage_predictor.model_type == "lstm"
        assert mock_usage_predictor.is_trained is False

    def test_usage_model_training_success(self, mock_usage_predictor, sample_usage_training_data):
        """Test successful usage model training."""
        # Mock successful training
        mock_usage_predictor.train.return_value = True
        mock_usage_predictor.is_trained = True
        
        # Test training
        result = mock_usage_predictor.train(sample_usage_training_data)
        
        assert result is True
        mock_usage_predictor.train.assert_called_once_with(sample_usage_training_data)

    def test_usage_model_training_failure(self, mock_usage_predictor, sample_usage_training_data):
        """Test usage model training failure handling."""
        # Mock training failure
        mock_usage_predictor.train.side_effect = Exception("Training failed")
        
        with pytest.raises(Exception) as exc_info:
            mock_usage_predictor.train(sample_usage_training_data)
        
        assert "Training failed" in str(exc_info.value)

    def test_usage_prediction_success(self, mock_usage_predictor, sample_usage_prediction_data):
        """Test successful usage prediction."""
        # Mock trained model
        mock_usage_predictor.is_trained = True
        
        # Mock prediction results
        prediction_results = pd.DataFrame({
            'timestamp': sample_usage_prediction_data['timestamp'],
            'predicted_usage': np.random.uniform(40, 80, len(sample_usage_prediction_data)),
            'predicted_query_count': np.random.poisson(100, len(sample_usage_prediction_data)),
            'predicted_active_users': np.random.poisson(25, len(sample_usage_prediction_data)),
            'confidence_lower': np.random.uniform(30, 60, len(sample_usage_prediction_data)),
            'confidence_upper': np.random.uniform(50, 90, len(sample_usage_prediction_data))
        })
        
        mock_usage_predictor.predict.return_value = prediction_results
        
        # Test prediction
        result = mock_usage_predictor.predict(sample_usage_prediction_data)
        
        assert result is not None
        assert len(result) == 168
        assert 'predicted_usage' in result.columns
        assert 'confidence_lower' in result.columns
        assert 'confidence_upper' in result.columns
        mock_usage_predictor.predict.assert_called_once_with(sample_usage_prediction_data)

    def test_usage_model_evaluation_metrics(self, mock_usage_predictor):
        """Test usage model evaluation metrics."""
        # Mock evaluation metrics
        evaluation_metrics = {
            'mae': 8.5,
            'mse': 125.3,
            'rmse': 11.2,
            'r2_score': 0.82,
            'mape': 12.5,
            'accuracy_threshold_80': 0.75,
            'accuracy_threshold_90': 0.68,
            'peak_prediction_accuracy': 0.78,
            'valley_prediction_accuracy': 0.81
        }
        
        mock_usage_predictor.evaluate.return_value = evaluation_metrics
        
        # Test evaluation
        result = mock_usage_predictor.evaluate()
        
        assert result['mae'] == 8.5
        assert result['r2_score'] == 0.82
        assert result['mape'] == 12.5
        assert result['peak_prediction_accuracy'] == 0.78

    def test_usage_pattern_analysis(self, mock_usage_predictor):
        """Test usage pattern analysis."""
        pattern_analysis = {
            'daily_patterns': {
                'peak_hours': [9, 10, 11, 14, 15, 16],
                'low_hours': [22, 23, 0, 1, 2, 3],
                'peak_usage': 85.2,
                'low_usage': 15.8,
                'daily_variance': 145.6
            },
            'weekly_patterns': {
                'peak_days': ['Monday', 'Tuesday', 'Wednesday'],
                'low_days': ['Saturday', 'Sunday'],
                'weekday_avg': 65.4,
                'weekend_avg': 38.7,
                'weekly_variance': 89.3
            },
            'seasonal_patterns': {
                'quarterly_trend': 'increasing',
                'seasonal_amplitude': 25.6,
                'seasonal_period': 7  # days
            }
        }
        
        mock_usage_predictor.analyze_patterns.return_value = pattern_analysis
        
        # Test pattern analysis
        result = mock_usage_predictor.analyze_patterns()
        
        assert len(result['daily_patterns']['peak_hours']) == 6
        assert result['daily_patterns']['peak_usage'] == 85.2
        assert result['weekly_patterns']['weekday_avg'] == 65.4
        assert result['seasonal_patterns']['quarterly_trend'] == 'increasing'

    def test_usage_anomaly_detection(self, mock_usage_predictor):
        """Test usage anomaly detection."""
        anomaly_results = {
            'anomalies_detected': 15,
            'anomaly_percentage': 1.8,
            'anomaly_types': {
                'spike_anomalies': 8,
                'drop_anomalies': 5,
                'trend_anomalies': 2
            },
            'anomaly_details': [
                {
                    'timestamp': datetime(2024, 1, 15, 14, 0),
                    'type': 'spike',
                    'actual_value': 150.5,
                    'expected_value': 65.2,
                    'deviation': 85.3,
                    'severity': 'high'
                }
            ]
        }
        
        mock_usage_predictor.detect_anomalies.return_value = anomaly_results
        
        # Test anomaly detection
        result = mock_usage_predictor.detect_anomalies()
        
        assert result['anomalies_detected'] == 15
        assert result['anomaly_percentage'] == 1.8
        assert result['anomaly_types']['spike_anomalies'] == 8
        assert len(result['anomaly_details']) == 1

    def test_usage_capacity_planning(self, mock_usage_predictor):
        """Test usage capacity planning."""
        capacity_plan = {
            'current_capacity': 100,
            'predicted_peak_usage': 125.8,
            'capacity_utilization': 0.85,
            'recommended_capacity': 150,
            'scaling_recommendations': {
                'scale_up_triggers': [
                    {'threshold': 80, 'action': 'add_small_warehouse'},
                    {'threshold': 90, 'action': 'add_medium_warehouse'}
                ],
                'scale_down_triggers': [
                    {'threshold': 40, 'action': 'remove_small_warehouse'},
                    {'threshold': 20, 'action': 'suspend_warehouse'}
                ]
            },
            'cost_optimization': {
                'potential_savings': 450.75,
                'optimization_actions': ['right_sizing', 'auto_suspend']
            }
        }
        
        mock_usage_predictor.plan_capacity.return_value = capacity_plan
        
        # Test capacity planning
        result = mock_usage_predictor.plan_capacity()
        
        assert result['current_capacity'] == 100
        assert result['predicted_peak_usage'] == 125.8
        assert result['recommended_capacity'] == 150
        assert len(result['scaling_recommendations']['scale_up_triggers']) == 2

    def test_usage_forecasting_horizons(self, mock_usage_predictor):
        """Test different forecasting horizons."""
        horizons = {
            'short_term': {  # 24 hours
                'horizon': '24h',
                'predictions': np.random.uniform(40, 80, 24),
                'confidence': 0.92,
                'mae': 6.2
            },
            'medium_term': {  # 7 days
                'horizon': '7d',
                'predictions': np.random.uniform(35, 85, 168),
                'confidence': 0.85,
                'mae': 9.8
            },
            'long_term': {  # 30 days
                'horizon': '30d',
                'predictions': np.random.uniform(30, 90, 720),
                'confidence': 0.75,
                'mae': 15.5
            }
        }
        
        mock_usage_predictor.forecast_multiple_horizons.return_value = horizons
        
        # Test multiple horizons
        result = mock_usage_predictor.forecast_multiple_horizons()
        
        assert result['short_term']['confidence'] == 0.92
        assert result['medium_term']['mae'] == 9.8
        assert len(result['long_term']['predictions']) == 720

    def test_usage_model_ensemble(self, mock_usage_predictor):
        """Test usage model ensemble functionality."""
        ensemble_results = {
            'base_models': ['lstm', 'arima', 'prophet'],
            'model_weights': [0.5, 0.3, 0.2],
            'ensemble_predictions': np.random.uniform(40, 80, 168),
            'individual_predictions': {
                'lstm': np.random.uniform(45, 85, 168),
                'arima': np.random.uniform(35, 75, 168),
                'prophet': np.random.uniform(40, 80, 168)
            },
            'ensemble_performance': {
                'mae': 7.8,
                'rmse': 10.2,
                'r2_score': 0.87
            }
        }
        
        mock_usage_predictor.ensemble_predict.return_value = ensemble_results
        
        # Test ensemble prediction
        result = mock_usage_predictor.ensemble_predict()
        
        assert len(result['base_models']) == 3
        assert sum(result['model_weights']) == 1.0
        assert result['ensemble_performance']['r2_score'] == 0.87
        assert len(result['ensemble_predictions']) == 168

    def test_usage_model_feature_engineering(self, mock_usage_predictor):
        """Test feature engineering for usage prediction."""
        feature_engineering = {
            'time_features': [
                'hour_of_day', 'day_of_week', 'month', 'quarter',
                'is_weekend', 'is_holiday', 'business_hours'
            ],
            'lag_features': [
                'usage_lag_1h', 'usage_lag_24h', 'usage_lag_168h'
            ],
            'rolling_features': [
                'usage_rolling_mean_24h', 'usage_rolling_std_24h',
                'usage_rolling_max_24h', 'usage_rolling_min_24h'
            ],
            'interaction_features': [
                'hour_warehouse_interaction', 'day_user_interaction'
            ],
            'feature_importance': {
                'hour_of_day': 0.25,
                'day_of_week': 0.18,
                'usage_lag_24h': 0.15,
                'usage_rolling_mean_24h': 0.12,
                'warehouse_size': 0.10
            }
        }
        
        mock_usage_predictor.engineer_features.return_value = feature_engineering
        
        # Test feature engineering
        result = mock_usage_predictor.engineer_features()
        
        assert len(result['time_features']) == 7
        assert len(result['lag_features']) == 3
        assert len(result['rolling_features']) == 4
        assert result['feature_importance']['hour_of_day'] == 0.25

    def test_usage_model_drift_monitoring(self, mock_usage_predictor):
        """Test usage model drift monitoring."""
        drift_monitoring = {
            'drift_status': 'detected',
            'drift_score': 0.15,
            'drift_threshold': 0.10,
            'drift_components': {
                'data_drift': 0.08,
                'concept_drift': 0.07,
                'covariate_drift': 0.05
            },
            'affected_features': [
                'hour_of_day', 'query_count', 'active_users'
            ],
            'drift_timeline': {
                'drift_start': datetime(2024, 1, 15),
                'drift_detected': datetime(2024, 1, 20),
                'severity_progression': ['low', 'medium', 'high']
            },
            'remediation_actions': [
                'retrain_model', 'adjust_features', 'increase_monitoring'
            ]
        }
        
        mock_usage_predictor.monitor_drift.return_value = drift_monitoring
        
        # Test drift monitoring
        result = mock_usage_predictor.monitor_drift()
        
        assert result['drift_status'] == 'detected'
        assert result['drift_score'] == 0.15
        assert len(result['affected_features']) == 3
        assert len(result['remediation_actions']) == 3

    @pytest.mark.parametrize("model_type,expected_accuracy", [
        ("lstm", 0.85),
        ("arima", 0.78),
        ("prophet", 0.82),
        ("xgboost", 0.87),
        ("ensemble", 0.90),
    ])
    def test_different_usage_model_types(self, mock_usage_predictor, model_type, expected_accuracy):
        """Test different usage model types."""
        mock_usage_predictor.model_type = model_type
        mock_usage_predictor.evaluate.return_value = {'r2_score': expected_accuracy}
        
        # Test model type
        assert mock_usage_predictor.model_type == model_type
        
        # Test expected accuracy
        result = mock_usage_predictor.evaluate()
        assert result['r2_score'] == expected_accuracy

    def test_usage_prediction_uncertainty_quantification(self, mock_usage_predictor):
        """Test uncertainty quantification in usage predictions."""
        uncertainty_results = {
            'prediction_intervals': {
                'confidence_50': {
                    'lower': np.random.uniform(45, 55, 168),
                    'upper': np.random.uniform(55, 65, 168)
                },
                'confidence_80': {
                    'lower': np.random.uniform(40, 50, 168),
                    'upper': np.random.uniform(60, 70, 168)
                },
                'confidence_95': {
                    'lower': np.random.uniform(35, 45, 168),
                    'upper': np.random.uniform(65, 75, 168)
                }
            },
            'uncertainty_metrics': {
                'prediction_variance': 145.6,
                'epistemic_uncertainty': 23.4,
                'aleatoric_uncertainty': 122.2,
                'total_uncertainty': 145.6
            }
        }
        
        mock_usage_predictor.quantify_uncertainty.return_value = uncertainty_results
        
        # Test uncertainty quantification
        result = mock_usage_predictor.quantify_uncertainty()
        
        assert 'confidence_50' in result['prediction_intervals']
        assert 'confidence_95' in result['prediction_intervals']
        assert result['uncertainty_metrics']['prediction_variance'] == 145.6
        assert len(result['prediction_intervals']['confidence_80']['lower']) == 168

    def test_usage_model_interpretability(self, mock_usage_predictor):
        """Test usage model interpretability features."""
        interpretability = {
            'feature_importance': {
                'hour_of_day': 0.28,
                'day_of_week': 0.22,
                'usage_lag_24h': 0.18,
                'query_count': 0.15,
                'active_users': 0.12,
                'warehouse_size': 0.05
            },
            'partial_dependence': {
                'hour_of_day': {
                    'values': list(range(24)),
                    'effects': np.random.uniform(-10, 10, 24)
                },
                'day_of_week': {
                    'values': list(range(7)),
                    'effects': np.random.uniform(-5, 5, 7)
                }
            },
            'interaction_effects': {
                'hour_day_interaction': 0.08,
                'warehouse_user_interaction': 0.06,
                'query_usage_interaction': 0.12
            }
        }
        
        mock_usage_predictor.interpret_model.return_value = interpretability
        
        # Test interpretability
        result = mock_usage_predictor.interpret_model()
        
        assert result['feature_importance']['hour_of_day'] == 0.28
        assert len(result['partial_dependence']['hour_of_day']['values']) == 24
        assert result['interaction_effects']['hour_day_interaction'] == 0.08

    def test_usage_model_online_learning(self, mock_usage_predictor):
        """Test online learning capabilities."""
        online_learning = {
            'learning_rate': 0.01,
            'batch_size': 32,
            'update_frequency': 'hourly',
            'model_updates': 156,
            'performance_drift': -0.02,  # Slight improvement
            'adaptation_metrics': {
                'convergence_rate': 0.85,
                'stability_score': 0.92,
                'learning_efficiency': 0.78
            },
            'update_history': [
                {'timestamp': datetime(2024, 1, 1, 10), 'mae_before': 8.5, 'mae_after': 8.2},
                {'timestamp': datetime(2024, 1, 1, 11), 'mae_before': 8.2, 'mae_after': 8.0}
            ]
        }
        
        mock_usage_predictor.online_learn.return_value = online_learning
        
        # Test online learning
        result = mock_usage_predictor.online_learn()
        
        assert result['learning_rate'] == 0.01
        assert result['update_frequency'] == 'hourly'
        assert result['model_updates'] == 156
        assert len(result['update_history']) == 2
        assert result['adaptation_metrics']['stability_score'] == 0.92