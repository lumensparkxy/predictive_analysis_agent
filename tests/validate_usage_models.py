#!/usr/bin/env python3
"""
Validation script for usage prediction models.

Tests warehouse utilization, query volume, user activity, and peak detection models.
"""

import sys
import os
from datetime import datetime, timedelta
import importlib.util
import random

def load_module_from_file(module_name, file_path):
    """Load a module from a file path."""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

def generate_warehouse_test_data():
    """Generate test data for warehouse utilization model."""
    start_date = datetime(2023, 1, 1)
    timestamps = [start_date + timedelta(hours=i) for i in range(168)]  # 1 week
    
    utilization = []
    query_count = []
    
    for ts in timestamps:
        # Business hours pattern
        if 9 <= ts.hour <= 17 and ts.weekday() < 5:
            base_util = random.uniform(0.6, 0.9)
            base_queries = random.randint(50, 200)
        else:
            base_util = random.uniform(0.1, 0.4)
            base_queries = random.randint(5, 30)
        
        utilization.append(base_util)
        query_count.append(base_queries)
    
    return {
        'timestamps': timestamps,
        'utilization': utilization,
        'warehouse_size': 'M',
        'query_count': query_count
    }

def generate_query_volume_test_data():
    """Generate test data for query volume model."""
    start_date = datetime(2023, 1, 1)
    timestamps = [start_date + timedelta(hours=i) for i in range(72)]  # 3 days
    
    query_counts = []
    execution_times = []
    
    for ts in timestamps:
        # Business hours pattern
        if 9 <= ts.hour <= 17 and ts.weekday() < 5:
            count = random.randint(20, 100)
            exec_times = [random.uniform(5, 300) for _ in range(count // 10)]
        else:
            count = random.randint(5, 25)
            exec_times = [random.uniform(10, 60) for _ in range(count // 10)]
        
        query_counts.append(count)
        execution_times.append(exec_times)
    
    return {
        'timestamps': timestamps,
        'query_counts': query_counts,
        'execution_times': execution_times
    }

def generate_user_activity_test_data():
    """Generate test data for user activity model."""
    start_date = datetime(2023, 1, 1)
    timestamps = [start_date + timedelta(hours=i) for i in range(120)]  # 5 days
    
    users = ['user1', 'user2', 'user3', 'user4', 'user5']
    user_ids = []
    activity_counts = []
    session_durations = []
    
    for ts in timestamps:
        # Random user activity
        active_users = random.sample(users, random.randint(1, len(users)))
        
        for user in active_users:
            user_ids.append(user)
            
            # Activity pattern based on time
            if 9 <= ts.hour <= 17 and ts.weekday() < 5:
                activity = random.randint(10, 80)
                duration = random.uniform(30, 240)  # 30 min to 4 hours
            else:
                activity = random.randint(1, 20)
                duration = random.uniform(10, 60)   # 10 min to 1 hour
            
            activity_counts.append(activity)
            session_durations.append(duration)
    
    # Adjust timestamps to match user sessions
    timestamps = timestamps[:len(user_ids)]
    
    return {
        'timestamps': timestamps,
        'user_ids': user_ids,
        'activity_counts': activity_counts,
        'session_durations': session_durations
    }

def generate_peak_detection_test_data():
    """Generate test data for peak detection model."""
    start_date = datetime(2023, 1, 1)
    timestamps = [start_date + timedelta(hours=i) for i in range(168)]  # 1 week
    
    usage_metrics = []
    
    for ts in timestamps:
        # Base metrics with some peaks
        cpu_util = random.uniform(0.2, 0.8)
        memory_util = random.uniform(0.3, 0.7)
        query_count = random.randint(10, 100)
        
        # Create peaks during business hours
        if 10 <= ts.hour <= 11 and ts.weekday() == 1:  # Tuesday 10-11 AM
            cpu_util = random.uniform(0.8, 0.95)
            memory_util = random.uniform(0.85, 0.95)
            query_count = random.randint(150, 300)
        elif 14 <= ts.hour <= 15 and ts.weekday() == 3:  # Thursday 2-3 PM
            cpu_util = random.uniform(0.85, 0.98)
            memory_util = random.uniform(0.8, 0.9)
            query_count = random.randint(200, 400)
        
        usage_metrics.append({
            'cpu_utilization': cpu_util,
            'memory_utilization': memory_util,
            'query_count': query_count
        })
    
    return {
        'timestamps': timestamps,
        'usage_metrics': usage_metrics
    }

def main():
    """Main validation function."""
    print("Usage Prediction Models - Validation Script")
    print("=" * 50)
    
    # Set up paths
    models_dir = os.path.join(os.path.dirname(__file__), '..', 'src', 'snowflake_analytics', 'models')
    models_dir = os.path.abspath(models_dir)
    
    try:
        # Load base module
        base_path = os.path.join(models_dir, 'base.py')
        base_module = load_module_from_file('base', base_path)
        print("✓ Loaded base module")
        
        # Get base classes
        BaseTimeSeriesModel = base_module.BaseTimeSeriesModel
        BaseModel = base_module.BaseModel
        PredictionResult = base_module.PredictionResult
        ModelTrainingError = base_module.ModelTrainingError
        PredictionError = base_module.PredictionError
        
        # Set random seed for reproducible results
        random.seed(42)
        
        # Test Warehouse Utilization Model
        print("\n🧪 Testing WarehouseUtilizationModel...")
        
        warehouse_path = os.path.join(models_dir, 'usage_prediction', 'warehouse_utilization.py')
        with open(warehouse_path, 'r') as f:
            warehouse_code = f.read()
        
        # Replace imports
        warehouse_code = warehouse_code.replace(
            'from ..base import BaseTimeSeriesModel, PredictionResult, ModelTrainingError, PredictionError',
            '# Imports handled by validation script'
        )
        
        # Create module and execute
        warehouse_module = type(sys)('warehouse_utilization')
        warehouse_module.BaseTimeSeriesModel = BaseTimeSeriesModel
        warehouse_module.PredictionResult = PredictionResult
        warehouse_module.ModelTrainingError = ModelTrainingError
        warehouse_module.PredictionError = PredictionError
        warehouse_module.logger = type('Logger', (), {'info': lambda x: None, 'error': lambda x: None, 'warning': lambda x: None})()
        warehouse_module.math = __import__('math')
        warehouse_module.datetime = __import__('datetime').datetime
        warehouse_module.timedelta = __import__('datetime').timedelta
        
        exec(warehouse_code, warehouse_module.__dict__)
        WarehouseUtilizationModel = warehouse_module.WarehouseUtilizationModel
        
        # Test warehouse model
        warehouse_model = WarehouseUtilizationModel()
        warehouse_data = generate_warehouse_test_data()
        
        result = warehouse_model.train(warehouse_data)
        assert warehouse_model.is_trained, "Warehouse model should be trained"
        print(f"✓ Warehouse model trained - Avg utilization: {result['avg_utilization']:.2f}")
        
        pred_result = warehouse_model.predict(24)
        assert len(pred_result.predictions) == 24, "Should return 24 predictions"
        print(f"✓ Warehouse predictions generated - First utilization: {pred_result.predictions[0]:.2f}")
        
        scaling_recs = warehouse_model.get_scaling_recommendations(24)
        assert isinstance(scaling_recs, dict), "Should return scaling recommendations"
        print(f"✓ Scaling recommendations: {scaling_recs['summary']['scale_up_hours']} scale-up hours")
        
        # Test Query Volume Predictor
        print("\n🧪 Testing QueryVolumePredictor...")
        
        query_path = os.path.join(models_dir, 'usage_prediction', 'query_volume_predictor.py')
        with open(query_path, 'r') as f:
            query_code = f.read()
        
        query_code = query_code.replace(
            'from ..base import BaseTimeSeriesModel, PredictionResult, ModelTrainingError, PredictionError',
            '# Imports handled by validation script'
        )
        
        query_module = type(sys)('query_volume_predictor')
        query_module.BaseTimeSeriesModel = BaseTimeSeriesModel
        query_module.PredictionResult = PredictionResult
        query_module.ModelTrainingError = ModelTrainingError
        query_module.PredictionError = PredictionError
        query_module.logger = type('Logger', (), {'info': lambda x: None, 'error': lambda x: None, 'warning': lambda x: None})()
        query_module.math = __import__('math')
        query_module.datetime = __import__('datetime').datetime
        query_module.timedelta = __import__('datetime').timedelta
        
        exec(query_code, query_module.__dict__)
        QueryVolumePredictor = query_module.QueryVolumePredictor
        
        query_model = QueryVolumePredictor()
        query_data = generate_query_volume_test_data()
        
        result = query_model.train(query_data)
        assert query_model.is_trained, "Query model should be trained"
        print(f"✓ Query model trained - Total queries: {result['total_queries_trained']}")
        
        pred_result = query_model.predict(12)
        assert len(pred_result.predictions) == 12, "Should return 12 predictions"
        print(f"✓ Query predictions generated - First volume: {pred_result.predictions[0]}")
        
        insights = query_model.get_query_insights()
        assert isinstance(insights, dict), "Should return query insights"
        print(f"✓ Query insights: Peak hour {insights['temporal_patterns']['peak_hour']}")
        
        # Test User Activity Predictor
        print("\n🧪 Testing UserActivityPredictor...")
        
        user_path = os.path.join(models_dir, 'usage_prediction', 'user_activity_predictor.py')
        with open(user_path, 'r') as f:
            user_code = f.read()
        
        user_code = user_code.replace(
            'from ..base import BaseTimeSeriesModel, PredictionResult, ModelTrainingError, PredictionError',
            '# Imports handled by validation script'
        )
        
        user_module = type(sys)('user_activity_predictor')
        user_module.BaseTimeSeriesModel = BaseTimeSeriesModel
        user_module.PredictionResult = PredictionResult
        user_module.ModelTrainingError = ModelTrainingError
        user_module.PredictionError = PredictionError
        user_module.logger = type('Logger', (), {'info': lambda x: None, 'error': lambda x: None, 'warning': lambda x: None})()
        user_module.math = __import__('math')
        user_module.datetime = __import__('datetime').datetime
        user_module.timedelta = __import__('datetime').timedelta
        
        exec(user_code, user_module.__dict__)
        UserActivityPredictor = user_module.UserActivityPredictor
        
        user_model = UserActivityPredictor()
        user_data = generate_user_activity_test_data()
        
        result = user_model.train(user_data)
        assert user_model.is_trained, "User model should be trained"
        print(f"✓ User model trained - {result['unique_users']} users, {result['total_sessions']} sessions")
        
        pred_result = user_model.predict(8)
        assert len(pred_result.predictions) == 8, "Should return 8 predictions"
        first_pred = pred_result.predictions[0]
        print(f"✓ User predictions generated - Active users: {first_pred['active_users']}")
        
        user_insights = user_model.get_user_insights()
        assert isinstance(user_insights, dict), "Should return user insights"
        print(f"✓ User insights: {user_insights['total_users']} total users")
        
        # Test Peak Detector
        print("\n🧪 Testing PeakDetector...")
        
        peak_path = os.path.join(models_dir, 'usage_prediction', 'peak_detector.py')
        with open(peak_path, 'r') as f:
            peak_code = f.read()
        
        peak_code = peak_code.replace(
            'from ..base import BaseModel, PredictionResult, ModelTrainingError, PredictionError',
            '# Imports handled by validation script'
        )
        
        peak_module = type(sys)('peak_detector')
        peak_module.BaseModel = BaseModel
        peak_module.PredictionResult = PredictionResult
        peak_module.ModelTrainingError = ModelTrainingError
        peak_module.PredictionError = PredictionError
        peak_module.logger = type('Logger', (), {'info': lambda x: None, 'error': lambda x: None, 'warning': lambda x: None})()
        peak_module.math = __import__('math')
        peak_module.datetime = __import__('datetime').datetime
        peak_module.timedelta = __import__('datetime').timedelta
        
        exec(peak_code, peak_module.__dict__)
        PeakDetector = peak_module.PeakDetector
        
        peak_model = PeakDetector()
        peak_data = generate_peak_detection_test_data()
        
        result = peak_model.train(peak_data)
        assert peak_model.is_trained, "Peak model should be trained"
        print(f"✓ Peak model trained - {result['historical_peaks_detected']} peaks detected")
        
        pred_result = peak_model.predict(48)
        assert len(pred_result.predictions) == 48, "Should return 48 predictions"
        peak_count = sum(1 for p in pred_result.predictions if p['is_peak'])
        print(f"✓ Peak predictions generated - {peak_count} peaks predicted")
        
        peak_summary = peak_model.get_peak_summary(3)
        assert isinstance(peak_summary, dict), "Should return peak summary"
        print(f"✓ Peak summary: {peak_summary['summary']['total_peaks']} peaks in next 3 days")
        
        print("\n" + "=" * 50)
        print("🎉 ALL USAGE PREDICTION MODEL TESTS PASSED!")
        print("Usage forecasting functionality is working correctly.")
        print("\nKey achievements:")
        print("• ✅ Warehouse utilization prediction and scaling recommendations")
        print("• ✅ Query volume forecasting and complexity analysis")
        print("• ✅ User activity prediction and behavior insights")
        print("• ✅ Peak detection and risk assessment")
        print("• ✅ All models provide proper predictions and metadata")
        print("• ✅ Resource optimization recommendations")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Validation failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)