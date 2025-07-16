"""
Main anomaly detection interface.

Provides unified interface for all anomaly detection models and manages
isolation forest, statistical methods, real-time scoring, and explanations.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import logging
from datetime import datetime, timedelta
from enum import Enum

from ..base import PredictionResult, ModelError, ModelNotTrainedError
from .isolation_forest import IsolationForestModel
from .statistical_anomalies import StatisticalAnomalyModel

logger = logging.getLogger(__name__)


class AnomalyType(Enum):
    """Types of anomalies to detect."""
    COST_SPIKE = "unexpected_cost_spikes"
    CREDIT_CONSUMPTION = "unusual_credit_consumption"
    STORAGE_ANOMALY = "storage_cost_anomalies"
    QUERY_PATTERN = "abnormal_query_patterns"
    WAREHOUSE_ACTIVITY = "unusual_warehouse_activity"
    USER_BEHAVIOR = "unexpected_user_behavior"
    PERFORMANCE_DEGRADATION = "performance_degradations"


class DetectionMethod(Enum):
    """Anomaly detection methods."""
    ISOLATION_FOREST = "isolation_forest"
    STATISTICAL = "statistical"
    ENSEMBLE = "ensemble"
    AUTO = "auto"


class AnomalyDetector:
    """
    Main anomaly detection interface.
    
    Manages multiple anomaly detection models and provides unified
    interface for training, detection, and explanation.
    """
    
    def __init__(self, default_method: DetectionMethod = DetectionMethod.ENSEMBLE):
        self.default_method = default_method
        self.models = {}
        self.active_models = set()
        
        # Initialize available models
        self._initialize_models()
    
    def _initialize_models(self) -> None:
        """Initialize all available anomaly detection models."""
        try:
            self.models = {
                DetectionMethod.ISOLATION_FOREST: IsolationForestModel(),
                DetectionMethod.STATISTICAL: StatisticalAnomalyModel()
            }
            
            logger.info("Anomaly detection models initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize anomaly detection models: {str(e)}")
            raise ModelError(f"Model initialization failed: {str(e)}")
    
    def train_models(self, training_data: Dict[str, Any], 
                    methods: Optional[List[DetectionMethod]] = None) -> Dict[str, Any]:
        """
        Train anomaly detection models.
        
        Args:
            training_data: Training data for anomaly detection
            methods: Specific methods to train (if None, trains all)
            
        Returns:
            Training results for all models
        """
        if methods is None:
            methods = [DetectionMethod.ISOLATION_FOREST, DetectionMethod.STATISTICAL]
        
        training_results = {}
        
        for method in methods:
            if method not in self.models:
                continue
            
            model = self.models[method]
            
            try:
                # Prepare data for specific model
                model_data = self._prepare_model_data(method, training_data)
                
                if model_data:
                    logger.info(f"Training {method.value} model...")
                    result = model.train(model_data)
                    training_results[method.value] = {
                        'success': True,
                        'metrics': result
                    }
                    self.active_models.add(method)
                else:
                    training_results[method.value] = {
                        'success': False,
                        'error': 'No suitable training data'
                    }
                    
            except Exception as e:
                logger.error(f"Failed to train {method.value}: {str(e)}")
                training_results[method.value] = {
                    'success': False,
                    'error': str(e)
                }
        
        return training_results
    
    def _prepare_model_data(self, method: DetectionMethod, 
                          training_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Prepare training data for specific model."""
        if method == DetectionMethod.ISOLATION_FOREST:
            if 'timestamps' in training_data and 'features' in training_data:
                return {
                    'timestamps': training_data['timestamps'],
                    'features': training_data['features']
                }
        
        elif method == DetectionMethod.STATISTICAL:
            if 'timestamps' in training_data and 'metrics' in training_data:
                return {
                    'timestamps': training_data['timestamps'],
                    'metrics': training_data['metrics']
                }
        
        return None
    
    def detect_anomalies(self, data: Dict[str, Any],
                        method: Optional[DetectionMethod] = None,
                        **kwargs) -> Dict[str, Any]:
        """
        Detect anomalies in provided data.
        
        Args:
            data: Data to analyze for anomalies
            method: Detection method to use
            **kwargs: Additional detection parameters
            
        Returns:
            Anomaly detection results
        """
        detection_method = method or self.default_method
        
        if detection_method == DetectionMethod.ENSEMBLE:
            return self._ensemble_detection(data, **kwargs)
        elif detection_method == DetectionMethod.AUTO:
            detection_method = self._auto_select_method(data)
        
        if detection_method not in self.active_models:
            raise ModelNotTrainedError(f"Model {detection_method.value} must be trained before detection")
        
        model = self.models[detection_method]
        
        try:
            # Prepare data for the model
            model_data = self._prepare_detection_data(detection_method, data)
            
            if not model_data:
                raise ValueError("No suitable data for detection")
            
            result = model.detect_anomalies(model_data, **kwargs)
            
            # Add detector metadata
            result['detection_method'] = detection_method.value
            result['detector_model'] = 'anomaly_detector'
            
            logger.info(f"Anomaly detection completed using {detection_method.value}")
            return result
            
        except Exception as e:
            logger.error(f"Anomaly detection failed: {str(e)}")
            raise ModelError(f"Detection failed: {str(e)}")
    
    def _ensemble_detection(self, data: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Perform ensemble anomaly detection using multiple methods."""
        all_anomalies = []
        method_results = {}
        
        for method in self.active_models:
            try:
                model = self.models[method]
                model_data = self._prepare_detection_data(method, data)
                
                if model_data:
                    result = model.detect_anomalies(model_data, **kwargs)
                    method_results[method.value] = result
                    all_anomalies.extend(result['anomalies'])
                    
            except Exception as e:
                logger.warning(f"Method {method.value} failed: {str(e)}")
        
        # Combine results
        timestamps = data.get('timestamps', [])
        combined_anomalies = self._combine_anomaly_results(all_anomalies, timestamps)
        
        return {
            'anomalies': combined_anomalies,
            'total_anomalies': len(combined_anomalies),
            'anomaly_rate': len(combined_anomalies) / max(1, len(timestamps)) * 100,
            'detection_method': 'ensemble',
            'method_results': method_results,
            'consensus_score': self._calculate_consensus_score(method_results)
        }
    
    def _combine_anomaly_results(self, all_anomalies: List[Dict[str, Any]], 
                               timestamps: List[datetime]) -> List[Dict[str, Any]]:
        """Combine anomaly results from multiple methods."""
        # Group anomalies by timestamp
        timestamp_anomalies = {}
        
        for anomaly in all_anomalies:
            ts = anomaly['timestamp']
            if ts not in timestamp_anomalies:
                timestamp_anomalies[ts] = []
            timestamp_anomalies[ts].append(anomaly)
        
        # Create combined anomalies
        combined = []
        for ts, anomaly_list in timestamp_anomalies.items():
            if len(anomaly_list) >= 1:  # At least one method detected anomaly
                # Aggregate information
                scores = []
                methods = []
                severities = []
                
                for anomaly in anomaly_list:
                    if 'anomaly_score' in anomaly:
                        scores.append(anomaly['anomaly_score'])
                    if 'z_score' in anomaly:
                        scores.append(anomaly['z_score'] / 5.0)  # Normalize
                    
                    methods.append(anomaly.get('detection_method', 'unknown'))
                    severities.append(anomaly.get('severity', 'medium'))
                
                combined_anomaly = {
                    'timestamp': ts,
                    'detection_methods': list(set(methods)),
                    'consensus_score': len(anomaly_list),
                    'avg_score': sum(scores) / len(scores) if scores else 0.5,
                    'max_score': max(scores) if scores else 0.5,
                    'severity': max(severities, key=lambda x: ['low', 'medium', 'high', 'critical'].index(x)),
                    'details': anomaly_list
                }
                
                combined.append(combined_anomaly)
        
        # Sort by timestamp
        combined.sort(key=lambda x: x['timestamp'])
        return combined
    
    def _calculate_consensus_score(self, method_results: Dict[str, Dict[str, Any]]) -> float:
        """Calculate consensus score across detection methods."""
        if not method_results:
            return 0.0
        
        total_agreement = 0
        total_comparisons = 0
        
        methods = list(method_results.keys())
        for i in range(len(methods)):
            for j in range(i + 1, len(methods)):
                method1_anomalies = set(a['timestamp'] for a in method_results[methods[i]]['anomalies'])
                method2_anomalies = set(a['timestamp'] for a in method_results[methods[j]]['anomalies'])
                
                if method1_anomalies or method2_anomalies:
                    intersection = len(method1_anomalies & method2_anomalies)
                    union = len(method1_anomalies | method2_anomalies)
                    agreement = intersection / union if union > 0 else 0
                    total_agreement += agreement
                    total_comparisons += 1
        
        return total_agreement / max(1, total_comparisons)
    
    def _prepare_detection_data(self, method: DetectionMethod, 
                              data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Prepare data for specific detection method."""
        if method == DetectionMethod.ISOLATION_FOREST:
            if 'timestamps' in data and 'features' in data:
                return {'timestamps': data['timestamps'], 'features': data['features']}
        
        elif method == DetectionMethod.STATISTICAL:
            if 'timestamps' in data and 'metrics' in data:
                return {'timestamps': data['timestamps'], 'metrics': data['metrics']}
        
        return None
    
    def _auto_select_method(self, data: Dict[str, Any]) -> DetectionMethod:
        """Automatically select best detection method based on data characteristics."""
        # Simple heuristic: use statistical for time series, isolation forest for features
        if 'metrics' in data and len(data.get('timestamps', [])) > 100:
            return DetectionMethod.STATISTICAL
        elif 'features' in data:
            return DetectionMethod.ISOLATION_FOREST
        else:
            # Default to statistical if available
            if DetectionMethod.STATISTICAL in self.active_models:
                return DetectionMethod.STATISTICAL
            elif DetectionMethod.ISOLATION_FOREST in self.active_models:
                return DetectionMethod.ISOLATION_FOREST
            else:
                raise ValueError("No suitable detection method available")
    
    def explain_anomalies(self, data: Dict[str, Any], 
                         method: Optional[DetectionMethod] = None) -> List[Dict[str, Any]]:
        """
        Get explanations for detected anomalies.
        
        Args:
            data: Data containing anomalies
            method: Detection method to use for explanations
            
        Returns:
            List of anomaly explanations
        """
        detection_method = method or self.default_method
        
        if detection_method == DetectionMethod.ENSEMBLE:
            # Use isolation forest for explanations as it provides better feature analysis
            if DetectionMethod.ISOLATION_FOREST in self.active_models:
                detection_method = DetectionMethod.ISOLATION_FOREST
            else:
                detection_method = next(iter(self.active_models))
        
        if detection_method not in self.active_models:
            return []
        
        model = self.models[detection_method]
        
        try:
            if hasattr(model, 'get_anomaly_explanations'):
                model_data = self._prepare_detection_data(detection_method, data)
                if model_data:
                    return model.get_anomaly_explanations(model_data)
            
            return []
            
        except Exception as e:
            logger.error(f"Failed to generate explanations: {str(e)}")
            return []
    
    def score_anomalies(self, data: Dict[str, Any],
                       method: Optional[DetectionMethod] = None) -> List[float]:
        """
        Calculate anomaly scores for data points.
        
        Args:
            data: Data to score
            method: Detection method to use
            
        Returns:
            Anomaly scores for each data point
        """
        detection_method = method or self.default_method
        
        if detection_method == DetectionMethod.ENSEMBLE:
            # Average scores from all active models
            all_scores = []
            
            for active_method in self.active_models:
                try:
                    model = self.models[active_method]
                    model_data = self._prepare_detection_data(active_method, data)
                    
                    if model_data:
                        scores = model.score_anomalies(model_data)
                        all_scores.append(scores)
                        
                except Exception as e:
                    logger.warning(f"Scoring failed for {active_method.value}: {str(e)}")
            
            if all_scores:
                # Average scores across methods
                n_points = len(all_scores[0])
                averaged_scores = []
                
                for i in range(n_points):
                    point_scores = [scores[i] for scores in all_scores if i < len(scores)]
                    averaged_scores.append(sum(point_scores) / len(point_scores))
                
                return averaged_scores
            else:
                return [0.0] * len(data.get('timestamps', []))
        
        if detection_method not in self.active_models:
            raise ModelNotTrainedError(f"Model {detection_method.value} must be trained")
        
        model = self.models[detection_method]
        model_data = self._prepare_detection_data(detection_method, data)
        
        if model_data:
            return model.score_anomalies(model_data)
        else:
            return [0.0] * len(data.get('timestamps', []))
    
    def evaluate_models(self, test_data: Dict[str, Any], 
                       true_labels: List[int],
                       methods: Optional[List[DetectionMethod]] = None) -> Dict[str, Any]:
        """
        Evaluate anomaly detection models.
        
        Args:
            test_data: Test data
            true_labels: True anomaly labels (1 for anomaly, 0 for normal)
            methods: Specific methods to evaluate
            
        Returns:
            Evaluation results
        """
        if methods is None:
            methods = list(self.active_models)
        
        evaluation_results = {}
        
        for method in methods:
            if method not in self.active_models:
                continue
            
            model = self.models[method]
            
            try:
                model_data = self._prepare_detection_data(method, test_data)
                
                if model_data:
                    metrics = model.evaluate(model_data, true_labels)
                    evaluation_results[method.value] = {
                        'success': True,
                        'metrics': metrics
                    }
                else:
                    evaluation_results[method.value] = {
                        'success': False,
                        'error': 'No suitable test data'
                    }
                    
            except Exception as e:
                logger.error(f"Evaluation failed for {method.value}: {str(e)}")
                evaluation_results[method.value] = {
                    'success': False,
                    'error': str(e)
                }
        
        return evaluation_results
    
    def get_model_status(self) -> Dict[str, Any]:
        """Get status of all anomaly detection models."""
        status = {
            'total_models': len(self.models),
            'active_models': len(self.active_models),
            'default_method': self.default_method.value,
            'model_details': {}
        }
        
        for method, model in self.models.items():
            status['model_details'][method.value] = {
                'is_trained': model.is_trained,
                'is_active': method in self.active_models,
                'model_type': model.metadata.model_type,
                'last_trained': model.metadata.trained_at.isoformat() if model.metadata.trained_at else None
            }
        
        return status
    
    def get_supported_anomaly_types(self) -> List[str]:
        """Get list of supported anomaly types."""
        return [anomaly_type.value for anomaly_type in AnomalyType]
    
    def get_available_methods(self) -> List[str]:
        """Get list of available detection methods."""
        return [method.value for method in DetectionMethod]