"""
Main usage forecasting interface.

Provides unified interface for all usage prediction models and manages
warehouse utilization, query volume, user activity, and peak detection.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import logging
from datetime import datetime, timedelta
from enum import Enum

from ..base import PredictionResult, ModelError, ModelNotTrainedError
from .warehouse_utilization import WarehouseUtilizationModel
from .query_volume_predictor import QueryVolumePredictor
from .user_activity_predictor import UserActivityPredictor
from .peak_detector import PeakDetector

logger = logging.getLogger(__name__)


class UsagePredictionTarget(Enum):
    """Usage prediction target types."""
    WAREHOUSE_UTILIZATION = "warehouse_utilization"
    QUERY_VOLUME = "query_volume"
    USER_ACTIVITY = "user_activity"
    PEAK_DETECTION = "peak_detection"
    COMPREHENSIVE = "comprehensive"  # All usage predictions


class UsageMetricType(Enum):
    """Types of usage metrics to track."""
    CPU_UTILIZATION = "cpu_utilization"
    MEMORY_UTILIZATION = "memory_utilization"
    STORAGE_USAGE = "storage_usage"
    QUERY_COUNT = "query_count"
    USER_COUNT = "user_count"
    SESSION_COUNT = "session_count"
    EXECUTION_TIME = "execution_time"
    CONCURRENCY = "concurrency"


class UsageForecaster:
    """
    Main usage forecasting interface.
    
    Manages multiple usage prediction models and provides unified
    interface for training, prediction, and analysis.
    """
    
    def __init__(self):
        self.models = {}
        self.active_models = set()
        
        # Initialize available models
        self._initialize_models()
    
    def _initialize_models(self) -> None:
        """Initialize all available usage prediction models."""
        try:
            self.models = {
                UsagePredictionTarget.WAREHOUSE_UTILIZATION: WarehouseUtilizationModel(),
                UsagePredictionTarget.QUERY_VOLUME: QueryVolumePredictor(),
                UsagePredictionTarget.USER_ACTIVITY: UserActivityPredictor(),
                UsagePredictionTarget.PEAK_DETECTION: PeakDetector()
            }
            
            logger.info("Usage forecasting models initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize usage forecasting models: {str(e)}")
            raise ModelError(f"Model initialization failed: {str(e)}")
    
    def train_models(self, training_data: Dict[str, Any], 
                    targets: Optional[List[UsagePredictionTarget]] = None) -> Dict[str, Any]:
        """
        Train usage prediction models.
        
        Args:
            training_data: Dictionary containing training data for different models
            targets: Specific models to train (if None, trains all models)
            
        Returns:
            Training results for all models
        """
        if targets is None:
            targets = list(self.models.keys())
        
        training_results = {}
        
        for target in targets:
            if target not in self.models:
                logger.warning(f"Unknown target: {target}")
                continue
            
            model = self.models[target]
            
            try:
                # Extract relevant data for this model
                model_data = self._prepare_model_data(target, training_data)
                
                if model_data:
                    logger.info(f"Training {target.value} model...")
                    result = model.train(model_data)
                    training_results[target.value] = {
                        'success': True,
                        'metrics': result,
                        'model_info': model.metadata.to_dict()
                    }
                    self.active_models.add(target)
                    logger.info(f"{target.value} model trained successfully")
                else:
                    training_results[target.value] = {
                        'success': False,
                        'error': 'No suitable training data provided'
                    }
                    
            except Exception as e:
                logger.error(f"Failed to train {target.value}: {str(e)}")
                training_results[target.value] = {
                    'success': False,
                    'error': str(e)
                }
        
        return training_results
    
    def _prepare_model_data(self, target: UsagePredictionTarget, 
                          training_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Prepare training data for specific model."""
        if target == UsagePredictionTarget.WAREHOUSE_UTILIZATION:
            required_keys = ['timestamps', 'utilization', 'warehouse_size', 'query_count']
            if all(key in training_data for key in required_keys):
                return {
                    'timestamps': training_data['timestamps'],
                    'utilization': training_data['utilization'],
                    'warehouse_size': training_data['warehouse_size'],
                    'query_count': training_data['query_count']
                }
        
        elif target == UsagePredictionTarget.QUERY_VOLUME:
            required_keys = ['timestamps', 'query_counts', 'execution_times']
            if all(key in training_data for key in required_keys):
                return {
                    'timestamps': training_data['timestamps'],
                    'query_counts': training_data['query_counts'],
                    'execution_times': training_data['execution_times'],
                    'user_ids': training_data.get('user_ids', [])
                }
        
        elif target == UsagePredictionTarget.USER_ACTIVITY:
            required_keys = ['timestamps', 'user_ids', 'activity_counts', 'session_durations']
            if all(key in training_data for key in required_keys):
                return {
                    'timestamps': training_data['timestamps'],
                    'user_ids': training_data['user_ids'],
                    'activity_counts': training_data['activity_counts'],
                    'session_durations': training_data['session_durations']
                }
        
        elif target == UsagePredictionTarget.PEAK_DETECTION:
            required_keys = ['timestamps', 'usage_metrics']
            if all(key in training_data for key in required_keys):
                return {
                    'timestamps': training_data['timestamps'],
                    'usage_metrics': training_data['usage_metrics']
                }
        
        return None
    
    def predict_usage(self, target: UsagePredictionTarget,
                     prediction_input: Any,
                     **kwargs) -> PredictionResult:
        """
        Generate usage predictions for specified target.
        
        Args:
            target: Type of usage prediction to make
            prediction_input: Input for prediction (periods or timestamps)
            **kwargs: Additional prediction parameters
            
        Returns:
            Prediction results
        """
        if target not in self.models:
            raise ValueError(f"Unknown target: {target}")
        
        if target not in self.active_models:
            raise ModelNotTrainedError(f"Model {target.value} must be trained before prediction")
        
        model = self.models[target]
        
        try:
            result = model.predict(prediction_input, **kwargs)
            
            # Add forecaster-specific metadata
            result.prediction_metadata.update({
                'forecaster_model': target.value,
                'generated_by': 'usage_forecaster'
            })
            
            logger.info(f"Usage prediction completed for {target.value}")
            return result
            
        except Exception as e:
            logger.error(f"Prediction failed for {target.value}: {str(e)}")
            raise ModelError(f"Prediction failed: {str(e)}")
    
    def predict_warehouse_utilization(self, periods: int = 24, **kwargs) -> PredictionResult:
        """
        Predict warehouse utilization for next N periods.
        
        Args:
            periods: Number of periods (hours) to predict
            **kwargs: Additional prediction parameters
            
        Returns:
            Warehouse utilization predictions
        """
        return self.predict_usage(
            UsagePredictionTarget.WAREHOUSE_UTILIZATION,
            periods,
            **kwargs
        )
    
    def predict_query_volume(self, periods: int = 24, **kwargs) -> PredictionResult:
        """
        Predict query volume for next N periods.
        
        Args:
            periods: Number of periods to predict
            **kwargs: Additional prediction parameters
            
        Returns:
            Query volume predictions
        """
        return self.predict_usage(
            UsagePredictionTarget.QUERY_VOLUME,
            periods,
            **kwargs
        )
    
    def predict_user_activity(self, periods: int = 24, **kwargs) -> PredictionResult:
        """
        Predict user activity for next N periods.
        
        Args:
            periods: Number of periods to predict
            **kwargs: Additional prediction parameters
            
        Returns:
            User activity predictions
        """
        return self.predict_usage(
            UsagePredictionTarget.USER_ACTIVITY,
            periods,
            **kwargs
        )
    
    def detect_peaks(self, periods: int = 168, **kwargs) -> PredictionResult:
        """
        Detect potential peak usage periods.
        
        Args:
            periods: Number of periods to analyze (default: 1 week)
            **kwargs: Additional detection parameters
            
        Returns:
            Peak detection results
        """
        return self.predict_usage(
            UsagePredictionTarget.PEAK_DETECTION,
            periods,
            **kwargs
        )
    
    def comprehensive_forecast(self, periods: int = 24, **kwargs) -> Dict[str, PredictionResult]:
        """
        Generate comprehensive usage forecast across all dimensions.
        
        Args:
            periods: Number of periods to forecast
            **kwargs: Additional forecasting parameters
            
        Returns:
            Dictionary of prediction results for all usage types
        """
        forecasts = {}
        
        for target in self.active_models:
            try:
                result = self.predict_usage(target, periods, **kwargs)
                forecasts[target.value] = result
            except Exception as e:
                logger.warning(f"Failed to generate forecast for {target.value}: {str(e)}")
                forecasts[target.value] = None
        
        return forecasts
    
    def evaluate_models(self, test_data: Dict[str, Any],
                       targets: Optional[List[UsagePredictionTarget]] = None) -> Dict[str, Any]:
        """
        Evaluate usage prediction models on test data.
        
        Args:
            test_data: Test data for evaluation
            targets: Specific models to evaluate (if None, evaluates all active models)
            
        Returns:
            Evaluation results for all models
        """
        if targets is None:
            targets = list(self.active_models)
        
        evaluation_results = {}
        
        for target in targets:
            if target not in self.active_models:
                continue
            
            model = self.models[target]
            
            try:
                # Prepare test data for this model
                test_model_data = self._prepare_evaluation_data(target, test_data)
                
                if test_model_data:
                    metrics = model.evaluate(
                        test_model_data['X'],
                        test_model_data['y']
                    )
                    evaluation_results[target.value] = {
                        'success': True,
                        'metrics': metrics
                    }
                else:
                    evaluation_results[target.value] = {
                        'success': False,
                        'error': 'No suitable test data provided'
                    }
                    
            except Exception as e:
                logger.error(f"Failed to evaluate {target.value}: {str(e)}")
                evaluation_results[target.value] = {
                    'success': False,
                    'error': str(e)
                }
        
        return evaluation_results
    
    def _prepare_evaluation_data(self, target: UsagePredictionTarget,
                               test_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Prepare evaluation data for specific model."""
        if target == UsagePredictionTarget.WAREHOUSE_UTILIZATION:
            if 'test_timestamps' in test_data and 'test_utilization' in test_data:
                return {
                    'X': test_data['test_timestamps'],
                    'y': test_data['test_utilization']
                }
        
        elif target == UsagePredictionTarget.QUERY_VOLUME:
            if 'test_timestamps' in test_data and 'test_query_counts' in test_data:
                return {
                    'X': test_data['test_timestamps'],
                    'y': test_data['test_query_counts']
                }
        
        elif target == UsagePredictionTarget.USER_ACTIVITY:
            if 'test_timestamps' in test_data and 'test_activity_data' in test_data:
                return {
                    'X': test_data['test_timestamps'],
                    'y': test_data['test_activity_data']
                }
        
        elif target == UsagePredictionTarget.PEAK_DETECTION:
            if 'test_timestamps' in test_data and 'test_peak_labels' in test_data:
                return {
                    'X': test_data['test_timestamps'],
                    'y': test_data['test_peak_labels']
                }
        
        return None
    
    def get_usage_insights(self, days_ahead: int = 7) -> Dict[str, Any]:
        """
        Get comprehensive usage insights and recommendations.
        
        Args:
            days_ahead: Number of days to analyze
            
        Returns:
            Usage insights and recommendations
        """
        insights = {
            'forecast_period': days_ahead,
            'insights': {},
            'recommendations': [],
            'risk_assessment': {}
        }
        
        try:
            # Get predictions from all active models
            periods = days_ahead * 24  # Convert to hours
            forecasts = self.comprehensive_forecast(periods)
            
            # Extract insights from each model
            for target in self.active_models:
                model = self.models[target]
                target_name = target.value
                
                if hasattr(model, 'get_scaling_recommendations') and target == UsagePredictionTarget.WAREHOUSE_UTILIZATION:
                    insights['insights'][target_name] = model.get_scaling_recommendations(periods)
                
                elif hasattr(model, 'get_query_insights') and target == UsagePredictionTarget.QUERY_VOLUME:
                    insights['insights'][target_name] = model.get_query_insights()
                
                elif hasattr(model, 'get_user_insights') and target == UsagePredictionTarget.USER_ACTIVITY:
                    insights['insights'][target_name] = model.get_user_insights()
                
                elif hasattr(model, 'get_peak_summary') and target == UsagePredictionTarget.PEAK_DETECTION:
                    insights['insights'][target_name] = model.get_peak_summary(days_ahead)
            
            # Generate overall recommendations
            insights['recommendations'] = self._generate_recommendations(insights['insights'])
            
            # Assess overall risk
            insights['risk_assessment'] = self._assess_overall_risk(insights['insights'])
            
        except Exception as e:
            logger.error(f"Failed to generate usage insights: {str(e)}")
            insights['error'] = str(e)
        
        return insights
    
    def _generate_recommendations(self, model_insights: Dict[str, Any]) -> List[Dict[str, str]]:
        """Generate overall recommendations based on model insights."""
        recommendations = []
        
        # Warehouse utilization recommendations
        if 'warehouse_utilization' in model_insights:
            warehouse_insights = model_insights['warehouse_utilization']
            if 'recommended_actions' in warehouse_insights:
                for action in warehouse_insights['recommended_actions']:
                    recommendations.append({
                        'category': 'warehouse_scaling',
                        'priority': action.get('priority', 'medium'),
                        'message': action.get('action', 'Consider warehouse optimization'),
                        'reason': action.get('reason', '')
                    })
        
        # Peak detection recommendations
        if 'peak_detection' in model_insights:
            peak_insights = model_insights['peak_detection']
            if 'recommendations' in peak_insights:
                for rec in peak_insights['recommendations']:
                    recommendations.append({
                        'category': 'peak_management',
                        'priority': rec.get('priority', 'medium'),
                        'message': rec.get('message', 'Monitor peak usage'),
                        'reason': 'Peak usage patterns detected'
                    })
        
        # Query volume recommendations
        if 'query_volume' in model_insights:
            query_insights = model_insights['query_volume']
            if isinstance(query_insights, dict) and 'recommendations' in query_insights:
                for rec in query_insights['recommendations']:
                    recommendations.append({
                        'category': 'query_optimization',
                        'priority': rec.get('priority', 'medium'),
                        'message': rec.get('message', 'Optimize query performance'),
                        'reason': 'Query volume patterns detected'
                    })
        
        return recommendations
    
    def _assess_overall_risk(self, model_insights: Dict[str, Any]) -> Dict[str, str]:
        """Assess overall system risk based on model insights."""
        risk_factors = []
        
        # Check warehouse utilization risks
        if 'warehouse_utilization' in model_insights:
            warehouse_insights = model_insights['warehouse_utilization']
            if 'summary' in warehouse_insights:
                summary = warehouse_insights['summary']
                if summary.get('max_utilization', 0) > 0.9:
                    risk_factors.append('high_warehouse_utilization')
        
        # Check peak detection risks
        if 'peak_detection' in model_insights:
            peak_insights = model_insights['peak_detection']
            if 'summary' in peak_insights:
                summary = peak_insights['summary']
                if summary.get('highest_risk_level') in ['high', 'critical']:
                    risk_factors.append('critical_peaks_expected')
        
        # Determine overall risk level
        if 'critical_peaks_expected' in risk_factors or 'high_warehouse_utilization' in risk_factors:
            overall_risk = 'high'
        elif len(risk_factors) > 1:
            overall_risk = 'medium'
        elif len(risk_factors) == 1:
            overall_risk = 'low'
        else:
            overall_risk = 'minimal'
        
        return {
            'overall_risk': overall_risk,
            'risk_factors': risk_factors,
            'risk_summary': f"Overall risk level: {overall_risk}. {len(risk_factors)} risk factors identified."
        }
    
    def get_model_status(self) -> Dict[str, Any]:
        """
        Get status of all usage prediction models.
        
        Returns:
            Status information for all models
        """
        status = {
            'total_models': len(self.models),
            'active_models': len(self.active_models),
            'model_details': {}
        }
        
        for target, model in self.models.items():
            status['model_details'][target.value] = {
                'is_trained': model.is_trained,
                'is_active': target in self.active_models,
                'model_type': model.metadata.model_type,
                'last_trained': model.metadata.trained_at.isoformat() if model.metadata.trained_at else None,
                'performance_metrics': model.metadata.performance_metrics
            }
        
        return status
    
    def get_supported_targets(self) -> List[str]:
        """Get list of supported prediction targets."""
        return [target.value for target in UsagePredictionTarget]
    
    def get_supported_metrics(self) -> List[str]:
        """Get list of supported usage metrics."""
        return [metric.value for metric in UsageMetricType]
    
    def predict_resource_requirements(self, days_ahead: int = 7) -> Dict[str, Any]:
        """
        Predict resource requirements based on all usage models.
        
        Args:
            days_ahead: Number of days to analyze
            
        Returns:
            Resource requirement predictions
        """
        try:
            periods = days_ahead * 24
            requirements = {
                'forecast_period': days_ahead,
                'resource_predictions': {},
                'scaling_recommendations': [],
                'capacity_planning': {}
            }
            
            # Get warehouse utilization requirements
            if UsagePredictionTarget.WAREHOUSE_UTILIZATION in self.active_models:
                warehouse_model = self.models[UsagePredictionTarget.WAREHOUSE_UTILIZATION]
                if hasattr(warehouse_model, 'get_scaling_recommendations'):
                    warehouse_reqs = warehouse_model.get_scaling_recommendations(periods)
                    requirements['resource_predictions']['warehouse'] = warehouse_reqs
            
            # Get query volume resource requirements
            if UsagePredictionTarget.QUERY_VOLUME in self.active_models:
                query_model = self.models[UsagePredictionTarget.QUERY_VOLUME]
                if hasattr(query_model, 'predict_resource_requirements'):
                    query_reqs = query_model.predict_resource_requirements(periods)
                    requirements['resource_predictions']['query_processing'] = query_reqs
            
            # Combine recommendations
            all_recommendations = []
            for pred_type, pred_data in requirements['resource_predictions'].items():
                if isinstance(pred_data, dict) and 'recommendations' in pred_data:
                    for rec in pred_data['recommendations']:
                        rec['source'] = pred_type
                        all_recommendations.append(rec)
            
            requirements['scaling_recommendations'] = all_recommendations
            
            return requirements
            
        except Exception as e:
            logger.error(f"Failed to predict resource requirements: {str(e)}")
            return {'error': str(e)}
    
    def export_model_configurations(self) -> Dict[str, Any]:
        """
        Export model configurations for backup or transfer.
        
        Returns:
            Model configuration data
        """
        configurations = {}
        
        for target, model in self.models.items():
            if target in self.active_models:
                configurations[target.value] = {
                    'model_metadata': model.metadata.to_dict(),
                    'is_trained': model.is_trained,
                    'model_type': model.metadata.model_type
                }
        
        return configurations