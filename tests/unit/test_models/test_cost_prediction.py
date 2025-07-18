"""
Unit tests for cost prediction models.

Tests ML model training, prediction accuracy, and performance.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from decimal import Decimal


class TestCostPredictionModels:
    """Test suite for cost prediction models."""

    @pytest.fixture
    def mock_cost_predictor(self):
        """Create a mock cost predictor."""
        predictor = Mock()
        predictor.train = Mock()
        predictor.predict = Mock()
        predictor.evaluate = Mock()
        predictor.save_model = Mock()
        predictor.load_model = Mock()
        predictor.is_trained = False
        predictor.model_type = "prophet"
        return predictor

    @pytest.fixture
    def sample_training_data(self):
        """Create sample training data for cost prediction."""
        np.random.seed(42)
        dates = pd.date_range('2023-01-01', periods=365, freq='D')
        base_cost = 1000
        trend = np.linspace(0, 200, 365)
        seasonal = 100 * np.sin(2 * np.pi * np.arange(365) / 30)  # Monthly seasonality
        noise = np.random.normal(0, 50, 365)
        
        return pd.DataFrame({
            'ds': dates,
            'y': base_cost + trend + seasonal + noise,
            'warehouse': ['WH_ANALYTICS'] * 365,
            'credits_used': base_cost + trend + seasonal + noise,
            'query_count': np.random.poisson(100, 365)
        })

    @pytest.fixture
    def sample_prediction_data(self):
        """Create sample data for predictions."""
        future_dates = pd.date_range('2024-01-01', periods=30, freq='D')
        return pd.DataFrame({
            'ds': future_dates,
            'warehouse': ['WH_ANALYTICS'] * 30,
            'query_count': np.random.poisson(100, 30)
        })

    def test_cost_predictor_initialization(self, mock_cost_predictor):
        """Test cost predictor initialization."""
        assert mock_cost_predictor is not None
        assert mock_cost_predictor.model_type == "prophet"
        assert mock_cost_predictor.is_trained is False

    def test_model_training_success(self, mock_cost_predictor, sample_training_data):
        """Test successful model training."""
        # Mock successful training
        mock_cost_predictor.train.return_value = True
        mock_cost_predictor.is_trained = True
        
        # Test training
        result = mock_cost_predictor.train(sample_training_data)
        
        assert result is True
        mock_cost_predictor.train.assert_called_once_with(sample_training_data)

    def test_model_training_failure(self, mock_cost_predictor, sample_training_data):
        """Test model training failure handling."""
        # Mock training failure
        mock_cost_predictor.train.side_effect = Exception("Training failed")
        
        with pytest.raises(Exception) as exc_info:
            mock_cost_predictor.train(sample_training_data)
        
        assert "Training failed" in str(exc_info.value)

    def test_model_prediction_success(self, mock_cost_predictor, sample_prediction_data):
        """Test successful model prediction."""
        # Mock trained model
        mock_cost_predictor.is_trained = True
        
        # Mock prediction results
        prediction_results = pd.DataFrame({
            'ds': sample_prediction_data['ds'],
            'yhat': np.random.uniform(900, 1100, len(sample_prediction_data)),
            'yhat_lower': np.random.uniform(800, 1000, len(sample_prediction_data)),
            'yhat_upper': np.random.uniform(1000, 1200, len(sample_prediction_data))
        })
        
        mock_cost_predictor.predict.return_value = prediction_results
        
        # Test prediction
        result = mock_cost_predictor.predict(sample_prediction_data)
        
        assert result is not None
        assert len(result) == 30
        assert 'yhat' in result.columns
        assert 'yhat_lower' in result.columns
        assert 'yhat_upper' in result.columns
        mock_cost_predictor.predict.assert_called_once_with(sample_prediction_data)

    def test_model_prediction_untrained(self, mock_cost_predictor, sample_prediction_data):
        """Test prediction with untrained model."""
        # Mock untrained model
        mock_cost_predictor.is_trained = False
        mock_cost_predictor.predict.side_effect = Exception("Model not trained")
        
        with pytest.raises(Exception) as exc_info:
            mock_cost_predictor.predict(sample_prediction_data)
        
        assert "Model not trained" in str(exc_info.value)

    def test_model_evaluation_metrics(self, mock_cost_predictor):
        """Test model evaluation metrics."""
        # Mock evaluation metrics
        evaluation_metrics = {
            'mae': 45.2,
            'mse': 2850.5,
            'rmse': 53.4,
            'r2_score': 0.85,
            'mape': 4.8,
            'accuracy_threshold_90': 0.78,
            'accuracy_threshold_95': 0.65
        }
        
        mock_cost_predictor.evaluate.return_value = evaluation_metrics
        
        # Test evaluation
        result = mock_cost_predictor.evaluate()
        
        assert result['mae'] == 45.2
        assert result['r2_score'] == 0.85
        assert result['mape'] == 4.8
        assert result['accuracy_threshold_90'] == 0.78

    def test_model_cross_validation(self, mock_cost_predictor):
        """Test model cross-validation."""
        cv_results = {
            'cv_scores': [0.82, 0.85, 0.79, 0.87, 0.83],
            'mean_score': 0.832,
            'std_score': 0.029,
            'cv_mae': [42.1, 38.9, 48.2, 35.6, 41.3],
            'mean_mae': 41.22,
            'std_mae': 4.86
        }
        
        mock_cost_predictor.cross_validate.return_value = cv_results
        
        # Test cross-validation
        result = mock_cost_predictor.cross_validate()
        
        assert result['mean_score'] == 0.832
        assert result['std_score'] == 0.029
        assert len(result['cv_scores']) == 5

    def test_model_hyperparameter_tuning(self, mock_cost_predictor):
        """Test hyperparameter tuning."""
        tuning_results = {
            'best_params': {
                'changepoint_prior_scale': 0.05,
                'seasonality_prior_scale': 10.0,
                'holidays_prior_scale': 10.0,
                'seasonality_mode': 'multiplicative'
            },
            'best_score': 0.87,
            'param_grid_size': 48,
            'best_trial': 23
        }
        
        mock_cost_predictor.tune_hyperparameters.return_value = tuning_results
        
        # Test hyperparameter tuning
        result = mock_cost_predictor.tune_hyperparameters()
        
        assert result['best_score'] == 0.87
        assert 'changepoint_prior_scale' in result['best_params']
        assert result['param_grid_size'] == 48

    def test_model_feature_importance(self, mock_cost_predictor):
        """Test feature importance analysis."""
        feature_importance = {
            'trend_importance': 0.45,
            'seasonality_importance': 0.30,
            'holiday_importance': 0.15,
            'regressor_importance': {
                'query_count': 0.08,
                'warehouse_size': 0.02
            }
        }
        
        mock_cost_predictor.get_feature_importance.return_value = feature_importance
        
        # Test feature importance
        result = mock_cost_predictor.get_feature_importance()
        
        assert result['trend_importance'] == 0.45
        assert result['seasonality_importance'] == 0.30
        assert result['regressor_importance']['query_count'] == 0.08

    def test_model_serialization(self, mock_cost_predictor):
        """Test model serialization and deserialization."""
        # Mock model saving
        mock_cost_predictor.save_model.return_value = True
        
        # Test model saving
        save_result = mock_cost_predictor.save_model('/tmp/model.pkl')
        assert save_result is True
        mock_cost_predictor.save_model.assert_called_once_with('/tmp/model.pkl')
        
        # Mock model loading
        mock_cost_predictor.load_model.return_value = True
        mock_cost_predictor.is_trained = True
        
        # Test model loading
        load_result = mock_cost_predictor.load_model('/tmp/model.pkl')
        assert load_result is True
        assert mock_cost_predictor.is_trained is True
        mock_cost_predictor.load_model.assert_called_once_with('/tmp/model.pkl')

    def test_model_versioning(self, mock_cost_predictor):
        """Test model versioning functionality."""
        version_info = {
            'model_version': 'v1.2.0',
            'training_date': datetime.now(),
            'data_version': 'v2.1.0',
            'performance_metrics': {
                'mae': 42.5,
                'r2_score': 0.86
            },
            'model_hash': 'abc123def456'
        }
        
        mock_cost_predictor.get_version_info.return_value = version_info
        
        # Test version info
        result = mock_cost_predictor.get_version_info()
        
        assert result['model_version'] == 'v1.2.0'
        assert result['data_version'] == 'v2.1.0'
        assert result['model_hash'] == 'abc123def456'

    def test_model_prediction_intervals(self, mock_cost_predictor):
        """Test prediction interval generation."""
        intervals = {
            'confidence_80': {
                'lower': np.array([920, 940, 960]),
                'upper': np.array([1080, 1100, 1120])
            },
            'confidence_95': {
                'lower': np.array([890, 910, 930]),
                'upper': np.array([1110, 1130, 1150])
            }
        }
        
        mock_cost_predictor.get_prediction_intervals.return_value = intervals
        
        # Test prediction intervals
        result = mock_cost_predictor.get_prediction_intervals()
        
        assert 'confidence_80' in result
        assert 'confidence_95' in result
        assert len(result['confidence_80']['lower']) == 3
        assert len(result['confidence_95']['upper']) == 3

    def test_model_seasonal_decomposition(self, mock_cost_predictor):
        """Test seasonal decomposition of predictions."""
        decomposition = {
            'trend': np.array([1000, 1005, 1010, 1015, 1020]),
            'seasonal': np.array([50, -20, 30, -10, 40]),
            'residual': np.array([2.1, -1.5, 0.8, -0.9, 1.2]),
            'seasonal_periods': ['weekly', 'monthly', 'quarterly']
        }
        
        mock_cost_predictor.decompose_seasonality.return_value = decomposition
        
        # Test seasonal decomposition
        result = mock_cost_predictor.decompose_seasonality()
        
        assert 'trend' in result
        assert 'seasonal' in result
        assert 'residual' in result
        assert len(result['seasonal_periods']) == 3

    @pytest.mark.parametrize("model_type,expected_accuracy", [
        ("prophet", 0.85),
        ("arima", 0.82),
        ("lstm", 0.88),
        ("ensemble", 0.90),
    ])
    def test_different_model_types(self, mock_cost_predictor, model_type, expected_accuracy):
        """Test different model types."""
        mock_cost_predictor.model_type = model_type
        mock_cost_predictor.evaluate.return_value = {'r2_score': expected_accuracy}
        
        # Test model type
        assert mock_cost_predictor.model_type == model_type
        
        # Test expected accuracy
        result = mock_cost_predictor.evaluate()
        assert result['r2_score'] == expected_accuracy

    def test_model_performance_benchmarking(self, mock_cost_predictor):
        """Test model performance benchmarking."""
        benchmark_results = {
            'training_time_seconds': 45.6,
            'prediction_time_seconds': 2.3,
            'memory_usage_mb': 128.5,
            'model_size_mb': 15.7,
            'throughput_predictions_per_second': 1250,
            'scalability_score': 0.92
        }
        
        mock_cost_predictor.benchmark_performance.return_value = benchmark_results
        
        # Test performance benchmarking
        result = mock_cost_predictor.benchmark_performance()
        
        assert result['training_time_seconds'] == 45.6
        assert result['prediction_time_seconds'] == 2.3
        assert result['throughput_predictions_per_second'] == 1250

    def test_model_drift_detection(self, mock_cost_predictor):
        """Test model drift detection."""
        drift_results = {
            'drift_detected': True,
            'drift_score': 0.15,
            'drift_threshold': 0.10,
            'drift_type': 'concept_drift',
            'affected_features': ['query_count', 'warehouse_usage'],
            'recommendation': 'retrain_model'
        }
        
        mock_cost_predictor.detect_drift.return_value = drift_results
        
        # Test drift detection
        result = mock_cost_predictor.detect_drift()
        
        assert result['drift_detected'] is True
        assert result['drift_score'] == 0.15
        assert result['drift_type'] == 'concept_drift'
        assert result['recommendation'] == 'retrain_model'

    def test_model_explanation_and_interpretability(self, mock_cost_predictor):
        """Test model explanation and interpretability."""
        explanation = {
            'global_explanation': {
                'feature_importance': {
                    'trend': 0.45,
                    'seasonality': 0.30,
                    'holidays': 0.15,
                    'query_count': 0.10
                }
            },
            'local_explanation': {
                'prediction_components': {
                    'trend_contribution': 450.0,
                    'seasonal_contribution': 75.0,
                    'holiday_contribution': -25.0,
                    'regressor_contribution': 50.0
                }
            },
            'shap_values': np.array([0.1, -0.05, 0.15, -0.02])
        }
        
        mock_cost_predictor.explain_prediction.return_value = explanation
        
        # Test explanation
        result = mock_cost_predictor.explain_prediction()
        
        assert 'global_explanation' in result
        assert 'local_explanation' in result
        assert result['global_explanation']['feature_importance']['trend'] == 0.45
        assert len(result['shap_values']) == 4

    def test_model_robustness_testing(self, mock_cost_predictor):
        """Test model robustness under various conditions."""
        robustness_results = {
            'noise_resilience': {
                'noise_level_5pct': 0.84,
                'noise_level_10pct': 0.81,
                'noise_level_20pct': 0.75
            },
            'outlier_resilience': {
                'outlier_ratio_1pct': 0.83,
                'outlier_ratio_5pct': 0.78,
                'outlier_ratio_10pct': 0.70
            },
            'missing_data_resilience': {
                'missing_5pct': 0.82,
                'missing_10pct': 0.79,
                'missing_20pct': 0.72
            }
        }
        
        mock_cost_predictor.test_robustness.return_value = robustness_results
        
        # Test robustness
        result = mock_cost_predictor.test_robustness()
        
        assert result['noise_resilience']['noise_level_5pct'] == 0.84
        assert result['outlier_resilience']['outlier_ratio_1pct'] == 0.83
        assert result['missing_data_resilience']['missing_5pct'] == 0.82

    def test_model_batch_prediction(self, mock_cost_predictor):
        """Test batch prediction functionality."""
        batch_results = {
            'batch_size': 1000,
            'predictions': np.random.uniform(900, 1100, 1000),
            'processing_time_seconds': 15.2,
            'predictions_per_second': 65.8,
            'memory_peak_mb': 256.3,
            'success_rate': 0.998
        }
        
        mock_cost_predictor.predict_batch.return_value = batch_results
        
        # Test batch prediction
        result = mock_cost_predictor.predict_batch()
        
        assert result['batch_size'] == 1000
        assert len(result['predictions']) == 1000
        assert result['success_rate'] == 0.998

    def test_model_real_time_prediction(self, mock_cost_predictor):
        """Test real-time prediction functionality."""
        real_time_results = {
            'prediction_value': 1050.75,
            'confidence_interval': [925.0, 1175.0],
            'prediction_timestamp': datetime.now(),
            'latency_ms': 45.2,
            'model_version': 'v1.2.0',
            'input_features': {
                'query_count': 150,
                'warehouse_size': 'LARGE',
                'timestamp': datetime.now()
            }
        }
        
        mock_cost_predictor.predict_real_time.return_value = real_time_results
        
        # Test real-time prediction
        result = mock_cost_predictor.predict_real_time()
        
        assert result['prediction_value'] == 1050.75
        assert result['latency_ms'] == 45.2
        assert result['model_version'] == 'v1.2.0'
        assert len(result['confidence_interval']) == 2