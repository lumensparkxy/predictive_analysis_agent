"""
Ensemble cost forecasting model.

Combines multiple cost prediction models (Prophet, ARIMA, LSTM)
using various ensemble techniques for improved accuracy.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import logging
from datetime import datetime, timedelta
import math

from ..base import BaseTimeSeriesModel, PredictionResult, ModelTrainingError, PredictionError
from .prophet_model import ProphetCostModel
from .arima_model import ARIMACostModel
from .lstm_model import LSTMCostModel

logger = logging.getLogger(__name__)


class EnsembleCostModel(BaseTimeSeriesModel):
    """
    Ensemble cost prediction model.
    
    Combines multiple individual models using weighted averaging,
    voting, or stacking techniques for improved forecasting performance.
    """
    
    def __init__(self, model_name: str = "ensemble_cost_model", ensemble_method: str = "weighted_average"):
        super().__init__(model_name, "ensemble_cost")
        
        # Ensemble configuration
        self.ensemble_method = ensemble_method  # "weighted_average", "voting", "stacking"
        self.models = {}
        self.model_weights = {}
        self.performance_scores = {}
        
        # Initialize component models
        self._initialize_models()
        
    def _initialize_models(self) -> None:
        """Initialize component models for the ensemble."""
        try:
            self.models = {
                'prophet': ProphetCostModel("ensemble_prophet"),
                'arima': ARIMACostModel("ensemble_arima"),
                'lstm': LSTMCostModel("ensemble_lstm")
            }
            
            # Default equal weights
            self.model_weights = {
                'prophet': 1.0 / 3,
                'arima': 1.0 / 3,
                'lstm': 1.0 / 3
            }
            
            logger.info("Ensemble models initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize ensemble models: {str(e)}")
            raise
    
    def _validate_data(self, data: Any) -> bool:
        """Validate input data format."""
        if not isinstance(data, (list, tuple)) or len(data) < 2:
            return False
        
        dates, costs = data
        if len(dates) != len(costs) or len(dates) < 30:  # Need minimum data for all models
            return False
            
        return True
    
    def train(self, X: Any, y: Any = None, **kwargs) -> Dict[str, Any]:
        """
        Train all component models in the ensemble.
        
        Args:
            X: Tuple of (dates, costs) or time series data
            y: Not used (cost data should be in X)
            **kwargs: Additional training parameters
            
        Returns:
            Training metrics and information for all models
        """
        try:
            if not self._validate_data(X):
                raise ModelTrainingError("Invalid data format. Expected (dates, costs) tuple")
            
            dates, costs = X
            training_results = {}
            
            # Train each component model
            for model_name, model in self.models.items():
                try:
                    logger.info(f"Training {model_name} model...")
                    result = model.train(X, **kwargs)
                    training_results[model_name] = result
                    
                    # Store performance score for weighting
                    self.performance_scores[model_name] = result.get('mape', 100.0)
                    
                    logger.info(f"{model_name} training completed: MAPE={result.get('mape', 'N/A'):.2f}%")
                    
                except Exception as e:
                    logger.error(f"Failed to train {model_name}: {str(e)}")
                    training_results[model_name] = {'error': str(e)}
                    self.performance_scores[model_name] = 100.0  # Worst possible score
            
            # Calculate optimal weights based on performance
            if self.ensemble_method == "weighted_average":
                self._calculate_performance_weights()
            
            # Update ensemble metadata
            self.metadata.trained_at = datetime.now()
            self.metadata.features = ['ensemble_of_multiple_models']
            self.metadata.hyperparameters = {
                'ensemble_method': self.ensemble_method,
                'model_weights': self.model_weights,
                'component_models': list(self.models.keys())
            }
            self.metadata.training_data_info = {
                'data_points': len(dates),
                'date_range': f"{min(dates)} to {max(dates)}" if dates else "empty",
                'cost_range': f"{min(costs):.2f} to {max(costs):.2f}" if costs else "empty"
            }
            
            # Calculate ensemble metrics
            ensemble_metrics = self._calculate_ensemble_metrics(training_results)
            self.metadata.performance_metrics = ensemble_metrics
            self._is_trained = True
            
            logger.info(f"Ensemble model trained successfully with {len(self.models)} component models")
            return {
                'ensemble_metrics': ensemble_metrics,
                'component_results': training_results,
                'model_weights': self.model_weights
            }
            
        except Exception as e:
            raise ModelTrainingError(f"Ensemble model training failed: {str(e)}")
    
    def _calculate_performance_weights(self) -> None:
        """Calculate model weights based on performance scores."""
        # Convert MAPE to weights (lower MAPE = higher weight)
        inverse_scores = {name: 1.0 / max(score, 0.1) for name, score in self.performance_scores.items()}
        total_score = sum(inverse_scores.values())
        
        self.model_weights = {name: score / total_score for name, score in inverse_scores.items()}
        
        logger.info(f"Performance-based weights calculated: {self.model_weights}")
    
    def _calculate_ensemble_metrics(self, training_results: Dict[str, Dict[str, Any]]) -> Dict[str, float]:
        """Calculate ensemble-level metrics from component results."""
        valid_results = {k: v for k, v in training_results.items() if 'error' not in v}
        
        if not valid_results:
            return {'ensemble_error': 1.0}
        
        # Average metrics across models
        metrics = {}
        for metric in ['mape', 'mae', 'rmse']:
            values = [result.get(metric, 0) for result in valid_results.values()]
            if values:
                # Weighted average
                weighted_sum = sum(
                    val * self.model_weights.get(model, 0) 
                    for model, val in zip(valid_results.keys(), values)
                )
                metrics[f'ensemble_{metric}'] = weighted_sum
        
        metrics['num_models_trained'] = len(valid_results)
        metrics['training_success_rate'] = len(valid_results) / len(self.models)
        
        return metrics
    
    def predict(self, X: Any, **kwargs) -> PredictionResult:
        """
        Generate ensemble predictions by combining individual model predictions.
        
        Args:
            X: Number of periods to predict or list of future dates
            **kwargs: Additional prediction parameters
            
        Returns:
            Ensemble prediction results with confidence intervals
        """
        if not self._is_trained:
            raise PredictionError("Ensemble model must be trained before making predictions")
        
        try:
            # Get predictions from all trained models
            model_predictions = {}
            model_confidences = {}
            
            for model_name, model in self.models.items():
                if model.is_trained:
                    try:
                        result = model.predict(X, **kwargs)
                        model_predictions[model_name] = result.predictions
                        model_confidences[model_name] = result.confidence_intervals
                    except Exception as e:
                        logger.warning(f"Failed to get predictions from {model_name}: {str(e)}")
            
            if not model_predictions:
                raise PredictionError("No trained models available for prediction")
            
            # Combine predictions based on ensemble method
            if self.ensemble_method == "weighted_average":
                predictions, confidence_intervals = self._weighted_average_ensemble(
                    model_predictions, model_confidences
                )
            elif self.ensemble_method == "voting":
                predictions, confidence_intervals = self._voting_ensemble(
                    model_predictions, model_confidences
                )
            else:
                # Default to simple average
                predictions, confidence_intervals = self._simple_average_ensemble(
                    model_predictions, model_confidences
                )
            
            result = PredictionResult(
                predictions=predictions,
                confidence_intervals=confidence_intervals,
                prediction_metadata={
                    'model_type': 'ensemble',
                    'ensemble_method': self.ensemble_method,
                    'component_models': list(model_predictions.keys()),
                    'model_weights': self.model_weights,
                    'forecast_horizon': len(predictions)
                },
                model_version=self.metadata.version
            )
            
            logger.info(f"Generated ensemble predictions using {len(model_predictions)} models")
            return result
            
        except Exception as e:
            raise PredictionError(f"Ensemble prediction failed: {str(e)}")
    
    def _weighted_average_ensemble(self, 
                                 model_predictions: Dict[str, List[float]], 
                                 model_confidences: Dict[str, List[Tuple[float, float]]]) -> Tuple[List[float], List[Tuple[float, float]]]:
        """Combine predictions using weighted averaging."""
        predictions = []
        confidence_intervals = []
        
        num_periods = len(next(iter(model_predictions.values())))
        
        for i in range(num_periods):
            # Weighted average of predictions
            weighted_pred = sum(
                pred_list[i] * self.model_weights.get(model_name, 0)
                for model_name, pred_list in model_predictions.items()
            )
            predictions.append(weighted_pred)
            
            # Combine confidence intervals
            lower_bounds = []
            upper_bounds = []
            
            for model_name, conf_list in model_confidences.items():
                if i < len(conf_list):
                    weight = self.model_weights.get(model_name, 0)
                    lower_bounds.append(conf_list[i][0] * weight)
                    upper_bounds.append(conf_list[i][1] * weight)
            
            confidence_intervals.append((sum(lower_bounds), sum(upper_bounds)))
        
        return predictions, confidence_intervals
    
    def _voting_ensemble(self, 
                        model_predictions: Dict[str, List[float]], 
                        model_confidences: Dict[str, List[Tuple[float, float]]]) -> Tuple[List[float], List[Tuple[float, float]]]:
        """Combine predictions using median voting."""
        predictions = []
        confidence_intervals = []
        
        num_periods = len(next(iter(model_predictions.values())))
        
        for i in range(num_periods):
            # Median of predictions
            period_predictions = [pred_list[i] for pred_list in model_predictions.values()]
            period_predictions.sort()
            
            n = len(period_predictions)
            if n % 2 == 0:
                median_pred = (period_predictions[n//2 - 1] + period_predictions[n//2]) / 2
            else:
                median_pred = period_predictions[n//2]
            
            predictions.append(median_pred)
            
            # Use widest confidence interval
            all_lowers = [conf_list[i][0] for conf_list in model_confidences.values() if i < len(conf_list)]
            all_uppers = [conf_list[i][1] for conf_list in model_confidences.values() if i < len(conf_list)]
            
            if all_lowers and all_uppers:
                confidence_intervals.append((min(all_lowers), max(all_uppers)))
            else:
                # Fallback to ±10% if no confidence intervals
                confidence_intervals.append((median_pred * 0.9, median_pred * 1.1))
        
        return predictions, confidence_intervals
    
    def _simple_average_ensemble(self, 
                                model_predictions: Dict[str, List[float]], 
                                model_confidences: Dict[str, List[Tuple[float, float]]]) -> Tuple[List[float], List[Tuple[float, float]]]:
        """Combine predictions using simple averaging."""
        predictions = []
        confidence_intervals = []
        
        num_periods = len(next(iter(model_predictions.values())))
        
        for i in range(num_periods):
            # Simple average of predictions
            period_predictions = [pred_list[i] for pred_list in model_predictions.values()]
            avg_pred = sum(period_predictions) / len(period_predictions)
            predictions.append(avg_pred)
            
            # Average confidence intervals
            lower_bounds = [conf_list[i][0] for conf_list in model_confidences.values() if i < len(conf_list)]
            upper_bounds = [conf_list[i][1] for conf_list in model_confidences.values() if i < len(conf_list)]
            
            if lower_bounds and upper_bounds:
                avg_lower = sum(lower_bounds) / len(lower_bounds)
                avg_upper = sum(upper_bounds) / len(upper_bounds)
                confidence_intervals.append((avg_lower, avg_upper))
            else:
                confidence_intervals.append((avg_pred * 0.9, avg_pred * 1.1))
        
        return predictions, confidence_intervals
    
    def forecast(self, periods: int, **kwargs) -> PredictionResult:
        """
        Generate ensemble forecasts for future periods.
        
        Args:
            periods: Number of periods to forecast
            **kwargs: Additional forecasting parameters
            
        Returns:
            Ensemble forecast results with confidence intervals
        """
        return self.predict(periods, **kwargs)
    
    def evaluate(self, X: Any, y: Any, **kwargs) -> Dict[str, float]:
        """
        Evaluate ensemble performance on test data.
        
        Args:
            X: Test periods or dates
            y: True cost values
            **kwargs: Additional evaluation parameters
            
        Returns:
            Performance metrics including individual model comparisons
        """
        if not self._is_trained:
            raise ValueError("Ensemble model must be trained before evaluation")
        
        try:
            # Get ensemble predictions
            ensemble_predictions = self.predict(X)
            ensemble_pred_values = ensemble_predictions.predictions
            
            # Calculate ensemble metrics
            n = len(y)
            mae = sum(abs(p - t) for p, t in zip(ensemble_pred_values, y)) / n
            mse = sum((p - t) ** 2 for p, t in zip(ensemble_pred_values, y)) / n
            rmse = math.sqrt(mse)
            mape = sum(abs((t - p) / t) for p, t in zip(ensemble_pred_values, y) if t != 0) / n * 100
            
            ensemble_metrics = {
                'ensemble_mae': mae,
                'ensemble_mse': mse,
                'ensemble_rmse': rmse,
                'ensemble_mape': mape
            }
            
            # Evaluate individual models for comparison
            for model_name, model in self.models.items():
                if model.is_trained:
                    try:
                        model_metrics = model.evaluate(X, y, **kwargs)
                        for metric, value in model_metrics.items():
                            ensemble_metrics[f'{model_name}_{metric}'] = value
                    except Exception as e:
                        logger.warning(f"Failed to evaluate {model_name}: {str(e)}")
            
            logger.info(f"Ensemble evaluation completed: MAPE={mape:.2f}%, RMSE={rmse:.2f}")
            return ensemble_metrics
            
        except Exception as e:
            raise ValueError(f"Ensemble evaluation failed: {str(e)}")
    
    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """
        Get aggregated feature importance from component models.
        
        Returns:
            Dictionary of aggregated feature importance scores
        """
        if not self._is_trained:
            return None
        
        aggregated_importance = {}
        
        # Get importance from each model and weight by model performance
        for model_name, model in self.models.items():
            if model.is_trained:
                importance = model.get_feature_importance()
                if importance:
                    weight = self.model_weights.get(model_name, 0)
                    for feature, score in importance.items():
                        key = f"{model_name}_{feature}"
                        aggregated_importance[key] = score * weight
        
        return aggregated_importance
    
    def get_model_consensus(self, X: Any) -> Dict[str, Any]:
        """
        Get consensus information across models for predictions.
        
        Args:
            X: Input data for prediction
            
        Returns:
            Consensus metrics and agreement statistics
        """
        if not self._is_trained:
            return {}
        
        try:
            model_predictions = {}
            
            for model_name, model in self.models.items():
                if model.is_trained:
                    result = model.predict(X)
                    model_predictions[model_name] = result.predictions
            
            if len(model_predictions) < 2:
                return {'consensus': 'insufficient_models'}
            
            # Calculate prediction consensus metrics
            consensus_metrics = {}
            num_periods = len(next(iter(model_predictions.values())))
            
            period_variances = []
            period_agreements = []
            
            for i in range(num_periods):
                period_preds = [pred_list[i] for pred_list in model_predictions.values()]
                
                # Calculate variance
                mean_pred = sum(period_preds) / len(period_preds)
                variance = sum((p - mean_pred) ** 2 for p in period_preds) / len(period_preds)
                period_variances.append(variance)
                
                # Calculate agreement (percentage of predictions within 10% of mean)
                agreement = sum(1 for p in period_preds if abs(p - mean_pred) <= 0.1 * mean_pred) / len(period_preds)
                period_agreements.append(agreement)
            
            consensus_metrics = {
                'average_variance': sum(period_variances) / len(period_variances),
                'average_agreement': sum(period_agreements) / len(period_agreements),
                'participating_models': list(model_predictions.keys()),
                'consensus_strength': 'high' if sum(period_agreements) / len(period_agreements) > 0.8 else 'medium' if sum(period_agreements) / len(period_agreements) > 0.6 else 'low'
            }
            
            return consensus_metrics
            
        except Exception as e:
            logger.error(f"Failed to calculate consensus: {str(e)}")
            return {'error': str(e)}
    
    def set_ensemble_method(self, method: str) -> None:
        """
        Set the ensemble combination method.
        
        Args:
            method: Ensemble method ('weighted_average', 'voting', 'simple_average')
        """
        valid_methods = ['weighted_average', 'voting', 'simple_average']
        if method not in valid_methods:
            raise ValueError(f"Invalid ensemble method. Choose from: {valid_methods}")
        
        self.ensemble_method = method
        logger.info(f"Ensemble method set to: {method}")
    
    def set_model_weights(self, weights: Dict[str, float]) -> None:
        """
        Manually set model weights for the ensemble.
        
        Args:
            weights: Dictionary of model weights (must sum to 1.0)
        """
        total_weight = sum(weights.values())
        if abs(total_weight - 1.0) > 0.001:
            raise ValueError(f"Model weights must sum to 1.0, got {total_weight}")
        
        for model_name in weights:
            if model_name not in self.models:
                raise ValueError(f"Unknown model: {model_name}")
        
        self.model_weights = weights.copy()
        logger.info(f"Model weights set manually: {weights}")
        
    def get_ensemble_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive ensemble summary.
        
        Returns:
            Ensemble configuration and performance summary
        """
        return {
            'ensemble_method': self.ensemble_method,
            'component_models': list(self.models.keys()),
            'model_weights': self.model_weights,
            'performance_scores': self.performance_scores,
            'trained_models': [name for name, model in self.models.items() if model.is_trained],
            'ensemble_metrics': self.metadata.performance_metrics,
            'training_status': self._is_trained
        }