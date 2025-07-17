"""
Main cost prediction interface.

Provides unified interface for all cost prediction models and manages
model selection, training, and prediction workflows.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import logging
from datetime import datetime, timedelta
from enum import Enum

from ..base import PredictionResult, ModelError, ModelNotTrainedError
from .prophet_model import ProphetCostModel
from .arima_model import ARIMACostModel
from .lstm_model import LSTMCostModel
from .ensemble_model import EnsembleCostModel

logger = logging.getLogger(__name__)


class CostPredictionTarget(Enum):
    """Cost prediction target types."""
    DAILY_COST = "daily_cost_forecast"
    WEEKLY_COST = "weekly_cost_forecast"
    MONTHLY_COST = "monthly_cost_forecast"
    WAREHOUSE_SPECIFIC = "warehouse_specific_costs"


class ModelType(Enum):
    """Available model types for cost prediction."""
    PROPHET = "prophet"
    ARIMA = "arima"
    LSTM = "lstm"
    ENSEMBLE = "ensemble"
    AUTO = "auto"  # Automatic model selection


class CostPredictor:
    """
    Main cost prediction interface.
    
    Manages multiple cost prediction models and provides unified
    interface for training, prediction, and evaluation.
    """
    
    def __init__(self, default_model: ModelType = ModelType.ENSEMBLE):
        self.default_model = default_model
        self.models = {}
        self.active_model = None
        self.prediction_targets = {}
        
        # Initialize available models
        self._initialize_models()
        
        # Set default active model
        self.set_active_model(default_model)
        
    def _initialize_models(self) -> None:
        """Initialize all available cost prediction models."""
        try:
            self.models = {
                ModelType.PROPHET: ProphetCostModel("cost_prophet"),
                ModelType.ARIMA: ARIMACostModel("cost_arima"),
                ModelType.LSTM: LSTMCostModel("cost_lstm"),
                ModelType.ENSEMBLE: EnsembleCostModel("cost_ensemble")
            }
            
            logger.info("Cost prediction models initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize cost prediction models: {str(e)}")
            raise ModelError(f"Model initialization failed: {str(e)}")
    
    def set_active_model(self, model_type: ModelType) -> None:
        """
        Set the active model for predictions.
        
        Args:
            model_type: Type of model to activate
        """
        if model_type == ModelType.AUTO:
            # Auto-select best model based on data characteristics
            model_type = self._auto_select_model()
        
        if model_type not in self.models:
            raise ValueError(f"Unknown model type: {model_type}")
        
        self.active_model = model_type
        logger.info(f"Active model set to: {model_type.value}")
    
    def _auto_select_model(self) -> ModelType:
        """
        Automatically select the best model based on data characteristics.
        
        Returns:
            Selected model type
        """
        # For now, default to ensemble which combines all models
        # In practice, this would analyze data characteristics
        logger.info("Auto-selecting model: defaulting to ensemble")
        return ModelType.ENSEMBLE
    
    def train_model(self, 
                   data: Any, 
                   model_type: Optional[ModelType] = None,
                   **kwargs) -> Dict[str, Any]:
        """
        Train a cost prediction model.
        
        Args:
            data: Training data (dates, costs)
            model_type: Specific model to train (uses active model if None)
            **kwargs: Additional training parameters
            
        Returns:
            Training results and metrics
        """
        target_model = model_type or self.active_model
        
        if target_model not in self.models:
            raise ValueError(f"Unknown model type: {target_model}")
        
        try:
            model = self.models[target_model]
            result = model.train(data, **kwargs)
            
            logger.info(f"Model {target_model.value} trained successfully")
            return {
                'model_type': target_model.value,
                'training_result': result,
                'model_metadata': model.metadata.to_dict()
            }
            
        except Exception as e:
            logger.error(f"Failed to train {target_model.value}: {str(e)}")
            raise ModelError(f"Training failed for {target_model.value}: {str(e)}")
    
    def predict_costs(self, 
                     target: CostPredictionTarget,
                     periods: Optional[int] = None,
                     future_dates: Optional[List[datetime]] = None,
                     model_type: Optional[ModelType] = None,
                     **kwargs) -> PredictionResult:
        """
        Predict future costs for specified target and time horizon.
        
        Args:
            target: Type of cost prediction to make
            periods: Number of periods to predict
            future_dates: Specific dates to predict (alternative to periods)
            model_type: Specific model to use (uses active model if None)
            **kwargs: Additional prediction parameters
            
        Returns:
            Prediction results with confidence intervals
        """
        target_model = model_type or self.active_model
        
        if target_model not in self.models:
            raise ValueError(f"Unknown model type: {target_model}")
        
        model = self.models[target_model]
        
        if not model.is_trained:
            raise ModelNotTrainedError(f"Model {target_model.value} must be trained before prediction")
        
        try:
            # Determine prediction input
            prediction_input = future_dates if future_dates else periods
            if prediction_input is None:
                # Default prediction horizons based on target
                prediction_input = self._get_default_horizon(target)
            
            # Make prediction
            result = model.predict(prediction_input, **kwargs)
            
            # Add target-specific metadata
            result.prediction_metadata.update({
                'prediction_target': target.value,
                'predictor_model': target_model.value
            })
            
            logger.info(f"Cost prediction completed for {target.value} using {target_model.value}")
            return result
            
        except Exception as e:
            logger.error(f"Prediction failed for {target.value}: {str(e)}")
            raise ModelError(f"Prediction failed: {str(e)}")
    
    def _get_default_horizon(self, target: CostPredictionTarget) -> int:
        """Get default prediction horizon for target type."""
        horizons = {
            CostPredictionTarget.DAILY_COST: 30,    # 30 days
            CostPredictionTarget.WEEKLY_COST: 12,   # 12 weeks
            CostPredictionTarget.MONTHLY_COST: 6,   # 6 months
            CostPredictionTarget.WAREHOUSE_SPECIFIC: 30  # 30 days
        }
        return horizons.get(target, 30)
    
    def predict_daily_costs(self, days: int = 30, **kwargs) -> PredictionResult:
        """
        Predict daily costs for the next N days.
        
        Args:
            days: Number of days to predict
            **kwargs: Additional prediction parameters
            
        Returns:
            Daily cost predictions
        """
        return self.predict_costs(
            CostPredictionTarget.DAILY_COST,
            periods=days,
            **kwargs
        )
    
    def predict_weekly_costs(self, weeks: int = 12, **kwargs) -> PredictionResult:
        """
        Predict weekly costs for the next N weeks.
        
        Args:
            weeks: Number of weeks to predict
            **kwargs: Additional prediction parameters
            
        Returns:
            Weekly cost predictions
        """
        return self.predict_costs(
            CostPredictionTarget.WEEKLY_COST,
            periods=weeks,
            **kwargs
        )
    
    def predict_monthly_costs(self, months: int = 6, **kwargs) -> PredictionResult:
        """
        Predict monthly costs for the next N months.
        
        Args:
            months: Number of months to predict
            **kwargs: Additional prediction parameters
            
        Returns:
            Monthly cost predictions
        """
        return self.predict_costs(
            CostPredictionTarget.MONTHLY_COST,
            periods=months,
            **kwargs
        )
    
    def predict_warehouse_costs(self, 
                              warehouse_id: str,
                              periods: int = 30,
                              **kwargs) -> PredictionResult:
        """
        Predict costs for a specific warehouse.
        
        Args:
            warehouse_id: Identifier for the warehouse
            periods: Number of periods to predict
            **kwargs: Additional prediction parameters
            
        Returns:
            Warehouse-specific cost predictions
        """
        kwargs['warehouse_id'] = warehouse_id
        
        return self.predict_costs(
            CostPredictionTarget.WAREHOUSE_SPECIFIC,
            periods=periods,
            **kwargs
        )
    
    def evaluate_model(self, 
                      test_data: Any,
                      model_type: Optional[ModelType] = None,
                      **kwargs) -> Dict[str, Any]:
        """
        Evaluate model performance on test data.
        
        Args:
            test_data: Test data for evaluation
            model_type: Specific model to evaluate (uses active model if None)
            **kwargs: Additional evaluation parameters
            
        Returns:
            Evaluation metrics and results
        """
        target_model = model_type or self.active_model
        
        if target_model not in self.models:
            raise ValueError(f"Unknown model type: {target_model}")
        
        model = self.models[target_model]
        
        if not model.is_trained:
            raise ModelNotTrainedError(f"Model {target_model.value} must be trained before evaluation")
        
        try:
            test_dates, test_costs = test_data
            metrics = model.evaluate(test_dates, test_costs, **kwargs)
            
            result = {
                'model_type': target_model.value,
                'evaluation_metrics': metrics,
                'test_data_size': len(test_costs),
                'evaluation_date': datetime.now()
            }
            
            logger.info(f"Model evaluation completed for {target_model.value}")
            return result
            
        except Exception as e:
            logger.error(f"Evaluation failed for {target_model.value}: {str(e)}")
            raise ModelError(f"Evaluation failed: {str(e)}")
    
    def compare_models(self, test_data: Any, **kwargs) -> Dict[str, Any]:
        """
        Compare performance of all trained models.
        
        Args:
            test_data: Test data for comparison
            **kwargs: Additional evaluation parameters
            
        Returns:
            Comparison results for all models
        """
        comparison_results = {}
        
        for model_type, model in self.models.items():
            if model.is_trained:
                try:
                    result = self.evaluate_model(test_data, model_type, **kwargs)
                    comparison_results[model_type.value] = result
                except Exception as e:
                    logger.warning(f"Failed to evaluate {model_type.value}: {str(e)}")
                    comparison_results[model_type.value] = {'error': str(e)}
        
        # Rank models by performance
        if comparison_results:
            comparison_results['ranking'] = self._rank_models(comparison_results)
        
        return comparison_results
    
    def _rank_models(self, comparison_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Rank models by MAPE performance."""
        rankings = []
        
        for model_name, result in comparison_results.items():
            if isinstance(result, dict) and 'evaluation_metrics' in result:
                metrics = result['evaluation_metrics']
                mape = metrics.get('mape', metrics.get('ensemble_mape', float('inf')))
                rankings.append({
                    'model': model_name,
                    'mape': mape,
                    'rank': 0  # Will be set below
                })
        
        # Sort by MAPE (lower is better)
        rankings.sort(key=lambda x: x['mape'])
        
        # Assign ranks
        for i, ranking in enumerate(rankings):
            ranking['rank'] = i + 1
        
        return rankings
    
    def get_model_info(self, model_type: Optional[ModelType] = None) -> Dict[str, Any]:
        """
        Get information about a specific model.
        
        Args:
            model_type: Model to get info for (uses active model if None)
            
        Returns:
            Model information and metadata
        """
        target_model = model_type or self.active_model
        
        if target_model not in self.models:
            raise ValueError(f"Unknown model type: {target_model}")
        
        model = self.models[target_model]
        
        info = {
            'model_type': target_model.value,
            'is_trained': model.is_trained,
            'metadata': model.metadata.to_dict() if hasattr(model, 'metadata') else {},
            'capabilities': self._get_model_capabilities(target_model)
        }
        
        # Add model-specific information
        if hasattr(model, 'get_feature_importance') and model.is_trained:
            info['feature_importance'] = model.get_feature_importance()
        
        if target_model == ModelType.ENSEMBLE and hasattr(model, 'get_ensemble_summary'):
            info['ensemble_summary'] = model.get_ensemble_summary()
        
        return info
    
    def _get_model_capabilities(self, model_type: ModelType) -> Dict[str, bool]:
        """Get capabilities of a specific model type."""
        capabilities = {
            ModelType.PROPHET: {
                'handles_seasonality': True,
                'handles_holidays': True,
                'handles_trend_changes': True,
                'provides_uncertainty': True,
                'handles_missing_data': True
            },
            ModelType.ARIMA: {
                'handles_seasonality': True,
                'handles_holidays': False,
                'handles_trend_changes': True,
                'provides_uncertainty': True,
                'handles_missing_data': False
            },
            ModelType.LSTM: {
                'handles_seasonality': True,
                'handles_holidays': False,
                'handles_trend_changes': True,
                'provides_uncertainty': True,
                'handles_missing_data': True
            },
            ModelType.ENSEMBLE: {
                'handles_seasonality': True,
                'handles_holidays': True,
                'handles_trend_changes': True,
                'provides_uncertainty': True,
                'handles_missing_data': True
            }
        }
        
        return capabilities.get(model_type, {})
    
    def get_supported_targets(self) -> List[str]:
        """Get list of supported prediction targets."""
        return [target.value for target in CostPredictionTarget]
    
    def get_available_models(self) -> List[str]:
        """Get list of available model types."""
        return [model_type.value for model_type in ModelType if model_type != ModelType.AUTO]
    
    def get_active_model(self) -> str:
        """Get the currently active model type."""
        return self.active_model.value if self.active_model else None
    
    def save_models(self, file_path: str) -> Dict[str, bool]:
        """
        Save all trained models to disk.
        
        Args:
            file_path: Base path for saving models
            
        Returns:
            Dictionary indicating save success for each model
        """
        save_results = {}
        
        for model_type, model in self.models.items():
            if model.is_trained:
                try:
                    model_path = f"{file_path}_{model_type.value}.pkl"
                    model.save_model(model_path)
                    save_results[model_type.value] = True
                    logger.info(f"Model {model_type.value} saved successfully")
                except Exception as e:
                    logger.error(f"Failed to save {model_type.value}: {str(e)}")
                    save_results[model_type.value] = False
            else:
                save_results[model_type.value] = False  # Not trained
        
        return save_results
    
    def load_models(self, file_path: str) -> Dict[str, bool]:
        """
        Load trained models from disk.
        
        Args:
            file_path: Base path for loading models
            
        Returns:
            Dictionary indicating load success for each model
        """
        load_results = {}
        
        for model_type in self.models:
            try:
                model_path = f"{file_path}_{model_type.value}.pkl"
                self.models[model_type].load_model(model_path)
                load_results[model_type.value] = True
                logger.info(f"Model {model_type.value} loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load {model_type.value}: {str(e)}")
                load_results[model_type.value] = False
        
        return load_results