"""
Base classes and interfaces for machine learning models.

This module provides abstract base classes that define the common interface
for all ML models in the predictive analytics system.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime
import json


@dataclass
class ModelMetadata:
    """Metadata for tracking model information."""
    model_name: str
    model_type: str
    version: str
    created_at: datetime
    trained_at: Optional[datetime] = None
    features: List[str] = field(default_factory=list)
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    training_data_info: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary."""
        return {
            'model_name': self.model_name,
            'model_type': self.model_type,
            'version': self.version,
            'created_at': self.created_at.isoformat(),
            'trained_at': self.trained_at.isoformat() if self.trained_at else None,
            'features': self.features,
            'hyperparameters': self.hyperparameters,
            'performance_metrics': self.performance_metrics,
            'training_data_info': self.training_data_info
        }


@dataclass
class PredictionResult:
    """Result of a model prediction."""
    predictions: Union[List[float], List[List[float]]]
    confidence_intervals: Optional[List[Tuple[float, float]]] = None
    prediction_metadata: Dict[str, Any] = field(default_factory=dict)
    model_version: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)


class BaseModel(ABC):
    """
    Abstract base class for all ML models.
    
    Provides common interface for training, prediction, and model management.
    """
    
    def __init__(self, model_name: str, model_type: str):
        self.metadata = ModelMetadata(
            model_name=model_name,
            model_type=model_type,
            version="1.0.0",
            created_at=datetime.now()
        )
        self._is_trained = False
        self._model = None
    
    @abstractmethod
    def train(self, X: Any, y: Any, **kwargs) -> Dict[str, Any]:
        """
        Train the model on provided data.
        
        Args:
            X: Feature data
            y: Target data
            **kwargs: Additional training parameters
            
        Returns:
            Training metrics and information
        """
        pass
    
    @abstractmethod
    def predict(self, X: Any, **kwargs) -> PredictionResult:
        """
        Make predictions on new data.
        
        Args:
            X: Feature data for prediction
            **kwargs: Additional prediction parameters
            
        Returns:
            Prediction results with confidence intervals
        """
        pass
    
    @abstractmethod
    def evaluate(self, X: Any, y: Any, **kwargs) -> Dict[str, float]:
        """
        Evaluate model performance on test data.
        
        Args:
            X: Feature data
            y: True target values
            **kwargs: Additional evaluation parameters
            
        Returns:
            Performance metrics
        """
        pass
    
    def save_model(self, file_path: str) -> None:
        """Save model to disk."""
        if not self._is_trained:
            raise ValueError("Model must be trained before saving")
        
        # Implementation would save both model and metadata
        # For now, just raise NotImplementedError
        raise NotImplementedError("Model saving not implemented yet")
    
    def load_model(self, file_path: str) -> None:
        """Load model from disk."""
        # Implementation would load both model and metadata
        # For now, just raise NotImplementedError
        raise NotImplementedError("Model loading not implemented yet")
    
    @property
    def is_trained(self) -> bool:
        """Check if model is trained."""
        return self._is_trained
    
    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """Get feature importance scores if available."""
        return None


class BaseTimeSeriesModel(BaseModel):
    """
    Abstract base class for time series models.
    
    Extends BaseModel with time series specific functionality.
    """
    
    def __init__(self, model_name: str, model_type: str):
        super().__init__(model_name, model_type)
        self.forecast_horizon = 30  # Default forecast horizon in days
    
    @abstractmethod
    def forecast(self, periods: int, **kwargs) -> PredictionResult:
        """
        Generate forecasts for future periods.
        
        Args:
            periods: Number of periods to forecast
            **kwargs: Additional forecasting parameters
            
        Returns:
            Forecast results with confidence intervals
        """
        pass
    
    def set_forecast_horizon(self, horizon: int) -> None:
        """Set the default forecast horizon."""
        self.forecast_horizon = horizon


class BaseAnomalyModel(BaseModel):
    """
    Abstract base class for anomaly detection models.
    
    Extends BaseModel with anomaly detection specific functionality.
    """
    
    def __init__(self, model_name: str, model_type: str):
        super().__init__(model_name, model_type)
        self.anomaly_threshold = 0.05  # Default threshold for anomaly detection
    
    @abstractmethod
    def detect_anomalies(self, X: Any, **kwargs) -> Dict[str, Any]:
        """
        Detect anomalies in the provided data.
        
        Args:
            X: Data to check for anomalies
            **kwargs: Additional detection parameters
            
        Returns:
            Anomaly detection results including scores and labels
        """
        pass
    
    @abstractmethod
    def score_anomalies(self, X: Any, **kwargs) -> List[float]:
        """
        Calculate anomaly scores for data points.
        
        Args:
            X: Data to score
            **kwargs: Additional scoring parameters
            
        Returns:
            Anomaly scores for each data point
        """
        pass
    
    def set_anomaly_threshold(self, threshold: float) -> None:
        """Set the anomaly detection threshold."""
        if not 0 <= threshold <= 1:
            raise ValueError("Threshold must be between 0 and 1")
        self.anomaly_threshold = threshold


class ModelError(Exception):
    """Base exception class for model-related errors."""
    pass


class ModelNotTrainedError(ModelError):
    """Raised when attempting to use an untrained model."""
    pass


class ModelTrainingError(ModelError):
    """Raised when model training fails."""
    pass


class PredictionError(ModelError):
    """Raised when prediction fails."""
    pass