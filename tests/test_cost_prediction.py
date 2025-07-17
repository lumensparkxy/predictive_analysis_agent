"""
Test suite for cost prediction models.

Tests the functionality of Prophet, ARIMA, LSTM, and Ensemble cost models.
"""

import unittest
from datetime import datetime, timedelta
import sys
import os

# Add src to Python path to import our modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from snowflake_analytics.models.cost_prediction import (
    CostPredictor, CostPredictionTarget, ModelType,
    ProphetCostModel, ARIMACostModel, LSTMCostModel, EnsembleCostModel
)


class TestCostPredictionModels(unittest.TestCase):
    """Test cases for cost prediction models."""
    
    def setUp(self):
        """Set up test data."""
        # Generate sample cost data
        start_date = datetime(2023, 1, 1)
        self.dates = [start_date + timedelta(days=i) for i in range(60)]
        
        # Simulate realistic cost data with trend and seasonality
        base_cost = 500.0
        self.costs = []
        for i, date in enumerate(self.dates):
            # Add trend
            trend = base_cost + i * 2.0
            
            # Add weekly seasonality (higher on weekdays)
            weekday_factor = 1.2 if date.weekday() < 5 else 0.8
            
            # Add some noise
            import random
            random.seed(42)  # For reproducible tests
            noise = random.uniform(-50, 50)
            
            cost = trend * weekday_factor + noise
            self.costs.append(max(0, cost))  # Ensure non-negative
        
        self.train_data = (self.dates, self.costs)
    
    def test_prophet_model_training(self):
        """Test Prophet model training."""
        model = ProphetCostModel()
        
        # Test training
        result = model.train(self.train_data)
        
        self.assertTrue(model.is_trained)
        self.assertIsInstance(result, dict)
        self.assertIn('mape', result)
        self.assertIn('mae', result)
        self.assertIn('rmse', result)
    
    def test_prophet_model_prediction(self):
        """Test Prophet model prediction."""
        model = ProphetCostModel()
        model.train(self.train_data)
        
        # Test prediction
        result = model.predict(10)  # Predict 10 periods
        
        self.assertEqual(len(result.predictions), 10)
        self.assertEqual(len(result.confidence_intervals), 10)
        self.assertIsInstance(result.predictions, list)
        
        # Test all predictions are positive
        for pred in result.predictions:
            self.assertGreater(pred, 0)
    
    def test_arima_model_training(self):
        """Test ARIMA model training."""
        model = ARIMACostModel()
        
        # Test training
        result = model.train(self.train_data)
        
        self.assertTrue(model.is_trained)
        self.assertIsInstance(result, dict)
        self.assertIn('aic', result)
        self.assertIn('bic', result)
        self.assertIn('mape', result)
    
    def test_lstm_model_training(self):
        """Test LSTM model training."""
        model = LSTMCostModel()
        
        # Test training
        result = model.train(self.train_data)
        
        self.assertTrue(model.is_trained)
        self.assertIsInstance(result, dict)
        self.assertIn('final_loss', result)
        self.assertIn('mape', result)
    
    def test_ensemble_model_training(self):
        """Test Ensemble model training."""
        model = EnsembleCostModel()
        
        # Test training
        result = model.train(self.train_data)
        
        self.assertTrue(model.is_trained)
        self.assertIsInstance(result, dict)
        self.assertIn('ensemble_metrics', result)
        self.assertIn('component_results', result)
        self.assertIn('model_weights', result)
    
    def test_cost_predictor_interface(self):
        """Test CostPredictor unified interface."""
        predictor = CostPredictor()
        
        # Test training
        result = predictor.train_model(self.train_data)
        self.assertIsInstance(result, dict)
        self.assertIn('model_type', result)
        
        # Test daily cost prediction
        daily_pred = predictor.predict_daily_costs(days=7)
        self.assertEqual(len(daily_pred.predictions), 7)
        
        # Test weekly cost prediction
        weekly_pred = predictor.predict_weekly_costs(weeks=4)
        self.assertEqual(len(weekly_pred.predictions), 4)
        
        # Test monthly cost prediction
        monthly_pred = predictor.predict_monthly_costs(months=3)
        self.assertEqual(len(monthly_pred.predictions), 3)
    
    def test_model_comparison(self):
        """Test model comparison functionality."""
        predictor = CostPredictor()
        
        # Train the default model
        predictor.train_model(self.train_data)
        
        # Create test data
        test_dates = [self.dates[-1] + timedelta(days=i+1) for i in range(5)]
        test_costs = [600 + i * 10 for i in range(5)]  # Simple test pattern
        test_data = (test_dates, test_costs)
        
        # Test evaluation
        eval_result = predictor.evaluate_model(test_data)
        self.assertIsInstance(eval_result, dict)
        self.assertIn('evaluation_metrics', eval_result)
    
    def test_model_metadata(self):
        """Test model metadata functionality."""
        model = ProphetCostModel()
        model.train(self.train_data)
        
        # Test metadata
        metadata = model.metadata
        self.assertEqual(metadata.model_type, "prophet_cost")
        self.assertIsNotNone(metadata.trained_at)
        self.assertIsInstance(metadata.features, list)
        self.assertIsInstance(metadata.performance_metrics, dict)
    
    def test_feature_importance(self):
        """Test feature importance extraction."""
        model = ProphetCostModel()
        model.train(self.train_data)
        
        importance = model.get_feature_importance()
        self.assertIsInstance(importance, dict)
        self.assertIn('trend', importance)
    
    def test_error_handling(self):
        """Test error handling for invalid inputs."""
        model = ProphetCostModel()
        
        # Test prediction without training
        with self.assertRaises(Exception):
            model.predict(10)
        
        # Test invalid data format
        with self.assertRaises(Exception):
            model.train("invalid_data")
    
    def test_ensemble_consensus(self):
        """Test ensemble consensus functionality."""
        model = EnsembleCostModel()
        model.train(self.train_data)
        
        # Test consensus metrics
        consensus = model.get_model_consensus(5)
        self.assertIsInstance(consensus, dict)
        
        if 'participating_models' in consensus:
            self.assertIsInstance(consensus['participating_models'], list)


class TestCostPredictionIntegration(unittest.TestCase):
    """Integration tests for cost prediction system."""
    
    def setUp(self):
        """Set up integration test data."""
        start_date = datetime(2023, 1, 1)
        self.dates = [start_date + timedelta(days=i) for i in range(100)]
        
        # More complex cost pattern
        self.costs = []
        for i, date in enumerate(self.dates):
            base = 500 + i * 1.5  # Trend
            seasonal = 50 * (1 + 0.5 * (date.month % 3))  # Quarterly pattern
            weekly = 30 if date.weekday() < 5 else -20  # Weekday vs weekend
            
            import random
            random.seed(i)  # Deterministic randomness
            noise = random.uniform(-25, 25)
            
            cost = base + seasonal + weekly + noise
            self.costs.append(max(0, cost))
        
        self.train_data = (self.dates, self.costs)
    
    def test_end_to_end_workflow(self):
        """Test complete end-to-end prediction workflow."""
        # Initialize predictor
        predictor = CostPredictor(default_model=ModelType.ENSEMBLE)
        
        # Train model
        train_result = predictor.train_model(self.train_data)
        self.assertIn('model_type', train_result)
        
        # Make various predictions
        daily_forecast = predictor.predict_daily_costs(30)
        weekly_forecast = predictor.predict_weekly_costs(12)
        monthly_forecast = predictor.predict_monthly_costs(6)
        
        # Validate predictions
        self.assertEqual(len(daily_forecast.predictions), 30)
        self.assertEqual(len(weekly_forecast.predictions), 12)
        self.assertEqual(len(monthly_forecast.predictions), 6)
        
        # All predictions should be positive
        for pred in daily_forecast.predictions:
            self.assertGreater(pred, 0)
    
    def test_model_switching(self):
        """Test switching between different model types."""
        predictor = CostPredictor()
        
        # Test different models
        for model_type in [ModelType.PROPHET, ModelType.ARIMA, ModelType.LSTM]:
            predictor.set_active_model(model_type)
            self.assertEqual(predictor.get_active_model(), model_type.value)
            
            # Train and predict
            predictor.train_model(self.train_data)
            result = predictor.predict_daily_costs(5)
            self.assertEqual(len(result.predictions), 5)
    
    def test_warehouse_specific_prediction(self):
        """Test warehouse-specific cost prediction."""
        predictor = CostPredictor()
        predictor.train_model(self.train_data)
        
        # Predict for specific warehouse
        warehouse_pred = predictor.predict_warehouse_costs("WH001", periods=15)
        
        self.assertEqual(len(warehouse_pred.predictions), 15)
        self.assertIn('warehouse_id', warehouse_pred.prediction_metadata)
    
    def test_prediction_confidence(self):
        """Test confidence interval generation."""
        predictor = CostPredictor()
        predictor.train_model(self.train_data)
        
        result = predictor.predict_daily_costs(10)
        
        # Validate confidence intervals
        self.assertEqual(len(result.confidence_intervals), 10)
        
        for i, (lower, upper) in enumerate(result.confidence_intervals):
            pred = result.predictions[i]
            self.assertLessEqual(lower, pred)
            self.assertLessEqual(pred, upper)
            self.assertGreater(upper - lower, 0)  # Non-zero interval


if __name__ == '__main__':
    # Run the tests
    unittest.main(verbosity=2)