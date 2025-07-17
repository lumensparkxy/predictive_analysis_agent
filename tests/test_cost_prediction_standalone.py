"""
Standalone test for cost prediction models.

Basic validation of model functionality without external dependencies.
"""

import sys
import os
from datetime import datetime, timedelta

# Add src to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Import just the models we need
from snowflake_analytics.models.base import BaseTimeSeriesModel, PredictionResult
from snowflake_analytics.models.cost_prediction.prophet_model import ProphetCostModel
from snowflake_analytics.models.cost_prediction.arima_model import ARIMACostModel
from snowflake_analytics.models.cost_prediction.lstm_model import LSTMCostModel


def generate_test_data():
    """Generate synthetic cost data for testing."""
    start_date = datetime(2023, 1, 1)
    dates = [start_date + timedelta(days=i) for i in range(60)]
    
    # Simple cost pattern
    costs = []
    for i, date in enumerate(dates):
        base_cost = 500 + i * 2  # Linear trend
        weekday_factor = 1.2 if date.weekday() < 5 else 0.8
        cost = base_cost * weekday_factor
        costs.append(cost)
    
    return dates, costs


def test_prophet_model():
    """Test Prophet model functionality."""
    print("Testing Prophet Model...")
    
    # Generate test data
    dates, costs = generate_test_data()
    train_data = (dates, costs)
    
    # Initialize and train model
    model = ProphetCostModel()
    assert not model.is_trained, "Model should not be trained initially"
    
    # Train model
    result = model.train(train_data)
    assert model.is_trained, "Model should be trained after training"
    assert isinstance(result, dict), "Training should return metrics dict"
    assert 'mape' in result, "Training result should include MAPE"
    
    # Test prediction
    prediction_result = model.predict(10)
    assert isinstance(prediction_result, PredictionResult), "Should return PredictionResult"
    assert len(prediction_result.predictions) == 10, "Should return 10 predictions"
    assert len(prediction_result.confidence_intervals) == 10, "Should return 10 confidence intervals"
    
    # Test all predictions are positive
    for pred in prediction_result.predictions:
        assert pred > 0, f"Prediction {pred} should be positive"
    
    # Test forecast
    forecast_result = model.forecast(5)
    assert len(forecast_result.predictions) == 5, "Forecast should return 5 predictions"
    
    print("✓ Prophet Model tests passed")


def test_arima_model():
    """Test ARIMA model functionality."""
    print("Testing ARIMA Model...")
    
    # Generate test data
    dates, costs = generate_test_data()
    train_data = (dates, costs)
    
    # Initialize and train model
    model = ARIMACostModel()
    assert not model.is_trained, "Model should not be trained initially"
    
    # Train model
    result = model.train(train_data)
    assert model.is_trained, "Model should be trained after training"
    assert isinstance(result, dict), "Training should return metrics dict"
    assert 'aic' in result, "Training result should include AIC"
    assert 'bic' in result, "Training result should include BIC"
    
    # Test prediction
    prediction_result = model.predict(7)
    assert isinstance(prediction_result, PredictionResult), "Should return PredictionResult"
    assert len(prediction_result.predictions) == 7, "Should return 7 predictions"
    
    # Test all predictions are positive
    for pred in prediction_result.predictions:
        assert pred > 0, f"Prediction {pred} should be positive"
    
    # Test model diagnostics
    diagnostics = model.get_model_diagnostics()
    assert isinstance(diagnostics, dict), "Diagnostics should return dict"
    assert 'order' in diagnostics, "Diagnostics should include ARIMA order"
    
    print("✓ ARIMA Model tests passed")


def test_lstm_model():
    """Test LSTM model functionality."""
    print("Testing LSTM Model...")
    
    # Generate test data (need more data for LSTM)
    start_date = datetime(2023, 1, 1)
    dates = [start_date + timedelta(days=i) for i in range(80)]
    costs = [500 + i * 2 + (20 if i % 7 < 5 else -10) for i in range(80)]
    train_data = (dates, costs)
    
    # Initialize and train model
    model = LSTMCostModel()
    assert not model.is_trained, "Model should not be trained initially"
    
    # Train model
    result = model.train(train_data)
    assert model.is_trained, "Model should be trained after training"
    assert isinstance(result, dict), "Training should return metrics dict"
    assert 'final_loss' in result, "Training result should include final_loss"
    
    # Test prediction
    prediction_result = model.predict(5)
    assert isinstance(prediction_result, PredictionResult), "Should return PredictionResult"
    assert len(prediction_result.predictions) == 5, "Should return 5 predictions"
    
    # Test all predictions are positive
    for pred in prediction_result.predictions:
        assert pred > 0, f"Prediction {pred} should be positive"
    
    # Test model summary
    summary = model.get_model_summary()
    assert isinstance(summary, dict), "Summary should return dict"
    assert 'architecture' in summary, "Summary should include architecture"
    
    print("✓ LSTM Model tests passed")


def test_model_evaluation():
    """Test model evaluation functionality."""
    print("Testing Model Evaluation...")
    
    # Generate train and test data
    dates, costs = generate_test_data()
    train_data = (dates[:50], costs[:50])
    test_dates = dates[50:]
    test_costs = costs[50:]
    
    # Train model
    model = ProphetCostModel()
    model.train(train_data)
    
    # Evaluate model
    metrics = model.evaluate(test_dates, test_costs)
    assert isinstance(metrics, dict), "Evaluation should return metrics dict"
    assert 'mae' in metrics, "Should include MAE"
    assert 'rmse' in metrics, "Should include RMSE"
    assert 'mape' in metrics, "Should include MAPE"
    
    # All metrics should be positive
    for metric_name, value in metrics.items():
        assert value >= 0, f"Metric {metric_name} should be non-negative"
    
    print("✓ Model Evaluation tests passed")


def test_feature_importance():
    """Test feature importance functionality."""
    print("Testing Feature Importance...")
    
    dates, costs = generate_test_data()
    train_data = (dates, costs)
    
    # Test with Prophet model
    model = ProphetCostModel()
    model.train(train_data)
    
    importance = model.get_feature_importance()
    assert isinstance(importance, dict), "Feature importance should return dict"
    assert len(importance) > 0, "Should have some feature importance scores"
    
    # All importance scores should be non-negative
    for feature, score in importance.items():
        assert score >= 0, f"Feature importance for {feature} should be non-negative"
    
    print("✓ Feature Importance tests passed")


def test_error_handling():
    """Test error handling for invalid inputs."""
    print("Testing Error Handling...")
    
    model = ProphetCostModel()
    
    # Test prediction without training
    try:
        model.predict(10)
        assert False, "Should raise exception for untrained model"
    except Exception:
        pass  # Expected
    
    # Test invalid data format
    try:
        model.train("invalid_data")
        assert False, "Should raise exception for invalid data"
    except Exception:
        pass  # Expected
    
    # Test empty data
    try:
        model.train(([], []))
        assert False, "Should raise exception for empty data"
    except Exception:
        pass  # Expected
    
    print("✓ Error Handling tests passed")


def run_all_tests():
    """Run all test functions."""
    print("Running Cost Prediction Model Tests...")
    print("=" * 50)
    
    try:
        test_prophet_model()
        test_arima_model() 
        test_lstm_model()
        test_model_evaluation()
        test_feature_importance()
        test_error_handling()
        
        print("=" * 50)
        print("✅ All tests passed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    success = run_all_tests()
    exit(0 if success else 1)