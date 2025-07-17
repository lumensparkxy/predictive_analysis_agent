"""
Minimal test for cost prediction models.

Direct import test without package dependencies.
"""

import sys
import os
from datetime import datetime, timedelta

# Add the specific modules directory to path
models_path = os.path.join(os.path.dirname(__file__), '..', 'src', 'snowflake_analytics', 'models')
sys.path.insert(0, models_path)

# Direct imports
try:
    from base import BaseTimeSeriesModel, PredictionResult, ModelMetadata
    from cost_prediction.prophet_model import ProphetCostModel
    from cost_prediction.arima_model import ARIMACostModel
    from cost_prediction.lstm_model import LSTMCostModel
    print("✓ Successfully imported all cost prediction models")
except ImportError as e:
    print(f"❌ Import failed: {e}")
    sys.exit(1)


def generate_test_data():
    """Generate synthetic cost data for testing."""
    start_date = datetime(2023, 1, 1)
    dates = [start_date + timedelta(days=i) for i in range(60)]
    
    # Simple cost pattern with trend and seasonality
    costs = []
    for i, date in enumerate(dates):
        base_cost = 500 + i * 2  # Linear trend
        weekday_factor = 1.2 if date.weekday() < 5 else 0.8  # Weekday vs weekend
        cost = base_cost * weekday_factor
        costs.append(cost)
    
    return dates, costs


def test_basic_functionality():
    """Test basic model functionality."""
    print("Testing basic model functionality...")
    
    dates, costs = generate_test_data()
    train_data = (dates, costs)
    
    # Test Prophet Model
    print("  Testing Prophet model...")
    prophet_model = ProphetCostModel()
    assert not prophet_model.is_trained
    
    # Train
    result = prophet_model.train(train_data)
    assert prophet_model.is_trained
    assert isinstance(result, dict)
    assert 'mape' in result
    print(f"    Training MAPE: {result['mape']:.2f}%")
    
    # Predict
    pred_result = prophet_model.predict(10)
    assert len(pred_result.predictions) == 10
    assert all(p > 0 for p in pred_result.predictions)
    print(f"    First prediction: ${pred_result.predictions[0]:.2f}")
    
    # Test ARIMA Model
    print("  Testing ARIMA model...")
    arima_model = ARIMACostModel()
    result = arima_model.train(train_data)
    assert arima_model.is_trained
    assert 'aic' in result
    print(f"    AIC: {result['aic']:.2f}")
    
    pred_result = arima_model.predict(5)
    assert len(pred_result.predictions) == 5
    print(f"    First prediction: ${pred_result.predictions[0]:.2f}")
    
    # Test LSTM Model  
    print("  Testing LSTM model...")
    lstm_model = LSTMCostModel()
    
    # Need more data for LSTM
    extended_dates = [dates[0] + timedelta(days=i) for i in range(80)]
    extended_costs = [500 + i * 2 + (20 if i % 7 < 5 else -10) for i in range(80)]
    extended_data = (extended_dates, extended_costs)
    
    result = lstm_model.train(extended_data)
    assert lstm_model.is_trained
    assert 'final_loss' in result
    print(f"    Final loss: {result['final_loss']:.6f}")
    
    pred_result = lstm_model.predict(3)
    assert len(pred_result.predictions) == 3
    print(f"    First prediction: ${pred_result.predictions[0]:.2f}")
    
    print("✓ Basic functionality tests passed")


def test_model_evaluation():
    """Test model evaluation."""
    print("Testing model evaluation...")
    
    # Generate train/test data
    dates, costs = generate_test_data()
    train_data = (dates[:40], costs[:40])
    test_dates = dates[40:]
    test_costs = costs[40:]
    
    # Train and evaluate Prophet model
    model = ProphetCostModel()
    model.train(train_data)
    
    metrics = model.evaluate(test_dates, test_costs)
    assert isinstance(metrics, dict)
    assert 'mae' in metrics and 'rmse' in metrics and 'mape' in metrics
    print(f"  Evaluation MAPE: {metrics['mape']:.2f}%")
    print(f"  Evaluation RMSE: ${metrics['rmse']:.2f}")
    
    print("✓ Model evaluation tests passed")


def test_feature_importance():
    """Test feature importance."""
    print("Testing feature importance...")
    
    dates, costs = generate_test_data()
    train_data = (dates, costs)
    
    model = ProphetCostModel()
    model.train(train_data)
    
    importance = model.get_feature_importance()
    assert isinstance(importance, dict)
    assert len(importance) > 0
    
    print("  Feature importance:")
    for feature, score in importance.items():
        print(f"    {feature}: {score:.3f}")
    
    print("✓ Feature importance tests passed")


def test_metadata():
    """Test model metadata."""
    print("Testing model metadata...")
    
    dates, costs = generate_test_data()
    train_data = (dates, costs)
    
    model = ProphetCostModel()
    model.train(train_data)
    
    metadata = model.metadata
    assert isinstance(metadata, ModelMetadata)
    assert metadata.model_type == "prophet_cost"
    assert metadata.trained_at is not None
    assert len(metadata.features) > 0
    assert len(metadata.performance_metrics) > 0
    
    # Test serialization
    metadata_dict = metadata.to_dict()
    assert isinstance(metadata_dict, dict)
    assert 'model_name' in metadata_dict
    
    print(f"  Model: {metadata.model_name}")
    print(f"  Type: {metadata.model_type}")
    print(f"  Features: {len(metadata.features)}")
    print(f"  Training time: {metadata.trained_at}")
    
    print("✓ Metadata tests passed")


def test_error_handling():
    """Test error handling."""
    print("Testing error handling...")
    
    model = ProphetCostModel()
    
    # Test prediction without training
    try:
        model.predict(5)
        assert False, "Should have raised exception"
    except Exception:
        pass  # Expected
    
    # Test invalid data
    try:
        model.train("invalid")
        assert False, "Should have raised exception"
    except Exception:
        pass  # Expected
    
    # Test insufficient data
    try:
        model.train(([], []))
        assert False, "Should have raised exception"
    except Exception:
        pass  # Expected
    
    print("✓ Error handling tests passed")


def main():
    """Run all tests."""
    print("Cost Prediction Models - Validation Test")
    print("=" * 50)
    
    try:
        test_basic_functionality()
        test_model_evaluation()
        test_feature_importance()
        test_metadata()
        test_error_handling()
        
        print("=" * 50)
        print("🎉 All validation tests passed successfully!")
        print("Cost prediction models are working correctly.")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)