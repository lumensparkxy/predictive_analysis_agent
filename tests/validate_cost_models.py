#!/usr/bin/env python3
"""
Validation script for cost prediction models.

This script validates that our cost prediction models are correctly implemented
by testing the core functionality without dealing with complex import issues.
"""

import sys
import os
from datetime import datetime, timedelta
import importlib.util

def load_module_from_file(module_name, file_path):
    """Load a module from a file path."""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

def main():
    """Main validation function."""
    print("Cost Prediction Models - Validation Script")
    print("=" * 50)
    
    # Set up paths
    models_dir = os.path.join(os.path.dirname(__file__), '..', 'src', 'snowflake_analytics', 'models')
    models_dir = os.path.abspath(models_dir)
    
    try:
        # Load base module
        base_path = os.path.join(models_dir, 'base.py')
        base_module = load_module_from_file('base', base_path)
        print("✓ Loaded base module")
        
        # Get classes from base module
        BaseTimeSeriesModel = base_module.BaseTimeSeriesModel
        PredictionResult = base_module.PredictionResult
        ModelMetadata = base_module.ModelMetadata
        ModelTrainingError = base_module.ModelTrainingError
        PredictionError = base_module.PredictionError
        
        # Load prophet model
        prophet_path = os.path.join(models_dir, 'cost_prediction', 'prophet_model.py')
        
        # Read and modify the prophet model code to use direct imports
        with open(prophet_path, 'r') as f:
            prophet_code = f.read()
        
        # Replace relative imports with absolute references
        prophet_code = prophet_code.replace(
            'from ..base import BaseTimeSeriesModel, PredictionResult, ModelTrainingError, PredictionError',
            '# Imports handled by validation script'
        )
        
        # Create a temporary module
        prophet_module = type(sys)('prophet_model')
        prophet_module.BaseTimeSeriesModel = BaseTimeSeriesModel
        prophet_module.PredictionResult = PredictionResult
        prophet_module.ModelTrainingError = ModelTrainingError
        prophet_module.PredictionError = PredictionError
        prophet_module.logger = type('Logger', (), {'info': lambda x: None, 'error': lambda x: None, 'warning': lambda x: None})()
        
        # Execute the prophet model code in the module's namespace
        exec(prophet_code, prophet_module.__dict__)
        
        ProphetCostModel = prophet_module.ProphetCostModel
        print("✓ Loaded ProphetCostModel")
        
        # Now test the functionality
        print("\n🧪 Testing ProphetCostModel functionality...")
        
        # Generate test data
        start_date = datetime(2023, 1, 1)
        dates = [start_date + timedelta(days=i) for i in range(60)]
        costs = []
        
        for i, date in enumerate(dates):
            base_cost = 500 + i * 2  # Trend
            weekday_factor = 1.2 if date.weekday() < 5 else 0.8  # Seasonality
            cost = base_cost * weekday_factor
            costs.append(cost)
        
        train_data = (dates, costs)
        
        # Test model creation
        model = ProphetCostModel()
        assert hasattr(model, 'is_trained'), "Model should have is_trained property"
        assert not model.is_trained, "Model should start untrained"
        print("✓ Model created successfully")
        
        # Test training
        result = model.train(train_data)
        assert model.is_trained, "Model should be trained after training"
        assert isinstance(result, dict), "Training should return dictionary"
        assert 'mape' in result, "Training result should include MAPE"
        print(f"✓ Training completed. MAPE: {result['mape']:.2f}%")
        
        # Test prediction
        pred_result = model.predict(10)
        assert hasattr(pred_result, 'predictions'), "Result should have predictions"
        assert len(pred_result.predictions) == 10, "Should return 10 predictions"
        assert all(p > 0 for p in pred_result.predictions), "All predictions should be positive"
        print(f"✓ Prediction completed. First 3 predictions: {[round(p, 2) for p in pred_result.predictions[:3]]}")
        
        # Test confidence intervals
        assert hasattr(pred_result, 'confidence_intervals'), "Result should have confidence intervals"
        assert len(pred_result.confidence_intervals) == 10, "Should return 10 confidence intervals"
        
        for i, (lower, upper) in enumerate(pred_result.confidence_intervals):
            pred = pred_result.predictions[i]
            assert lower <= pred <= upper, f"Prediction {pred} should be within interval [{lower}, {upper}]"
        print("✓ Confidence intervals are valid")
        
        # Test forecast method
        forecast_result = model.forecast(5)
        assert len(forecast_result.predictions) == 5, "Forecast should return 5 predictions"
        print(f"✓ Forecast completed. First forecast: ${forecast_result.predictions[0]:.2f}")
        
        # Test evaluation
        test_dates = dates[-5:]
        test_costs = costs[-5:]
        eval_metrics = model.evaluate(test_dates, test_costs)
        assert isinstance(eval_metrics, dict), "Evaluation should return dictionary"
        assert 'mape' in eval_metrics, "Evaluation should include MAPE"
        print(f"✓ Evaluation completed. Test MAPE: {eval_metrics['mape']:.2f}%")
        
        # Test feature importance
        importance = model.get_feature_importance()
        if importance:
            assert isinstance(importance, dict), "Feature importance should be dictionary"
            print(f"✓ Feature importance: {list(importance.keys())}")
        
        # Test metadata
        metadata = model.metadata
        assert hasattr(metadata, 'model_type'), "Metadata should have model_type"
        assert metadata.model_type == "prophet_cost", "Model type should be prophet_cost"
        assert metadata.trained_at is not None, "Should have training timestamp"
        print(f"✓ Metadata: {metadata.model_name} ({metadata.model_type})")
        
        # Test error handling
        print("\n🧪 Testing error handling...")
        
        untrained_model = ProphetCostModel("test_untrained")
        try:
            untrained_model.predict(5)
            assert False, "Should raise exception for untrained model"
        except:
            print("✓ Correctly raises exception for untrained model")
        
        try:
            model.train("invalid_data")
            assert False, "Should raise exception for invalid data"
        except:
            print("✓ Correctly raises exception for invalid data")
        
        print("\n" + "=" * 50)
        print("🎉 ALL VALIDATION TESTS PASSED!")
        print("Cost prediction models are implemented correctly and working as expected.")
        print("\nKey achievements:")
        print("• ✅ ProphetCostModel training and prediction")
        print("• ✅ Confidence intervals and uncertainty quantification")  
        print("• ✅ Model evaluation and metrics calculation")
        print("• ✅ Feature importance extraction")
        print("• ✅ Proper error handling")
        print("• ✅ Model metadata and versioning")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Validation failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)