"""
Isolation Forest anomaly detection model.

Implements Isolation Forest algorithm for detecting cost anomalies
and unusual patterns in Snowflake usage data.
"""

from typing import Any, Dict, List, Optional, Tuple
import logging
from datetime import datetime, timedelta
import math

from ..base import BaseAnomalyModel, PredictionResult, ModelTrainingError, PredictionError

logger = logging.getLogger(__name__)


class IsolationForestModel(BaseAnomalyModel):
    """
    Isolation Forest anomaly detection model.
    
    Uses isolation forest algorithm to detect cost anomalies and unusual
    usage patterns by isolating anomalies in feature space.
    """
    
    def __init__(self, model_name: str = "isolation_forest_model"):
        super().__init__(model_name, "isolation_forest")
        
        # Isolation Forest parameters
        self.n_estimators = 100  # Number of isolation trees
        self.max_samples = 256   # Number of samples to train each tree
        self.contamination = 0.1  # Expected proportion of anomalies
        self.max_features = 1.0  # Number of features to draw from X
        self.random_state = 42   # Random state for reproducibility
        
        # Model components
        self._trees = []
        self._feature_stats = {}
        self._training_data = None
        
    def _validate_data(self, data: Any) -> bool:
        """Validate input data format."""
        if not isinstance(data, dict):
            return False
        
        required_keys = ['timestamps', 'features']
        return all(key in data for key in required_keys)
    
    def _normalize_features(self, features: List[Dict[str, float]]) -> List[List[float]]:
        """Normalize features for isolation forest training."""
        if not features:
            return []
        
        # Get all feature names
        feature_names = set()
        for feature_dict in features:
            feature_names.update(feature_dict.keys())
        feature_names = sorted(feature_names)
        
        # Calculate feature statistics if not available
        if not self._feature_stats:
            self._feature_stats = {}
            for feature_name in feature_names:
                values = [f.get(feature_name, 0) for f in features]
                self._feature_stats[feature_name] = {
                    'mean': sum(values) / len(values),
                    'std': math.sqrt(sum((v - sum(values) / len(values)) ** 2 for v in values) / len(values)),
                    'min': min(values),
                    'max': max(values)
                }
        
        # Normalize features
        normalized_features = []
        for feature_dict in features:
            normalized_row = []
            for feature_name in feature_names:
                value = feature_dict.get(feature_name, 0)
                stats = self._feature_stats[feature_name]
                
                # Z-score normalization
                if stats['std'] > 0:
                    normalized_value = (value - stats['mean']) / stats['std']
                else:
                    normalized_value = 0
                
                normalized_row.append(normalized_value)
            normalized_features.append(normalized_row)
        
        return normalized_features
    
    def _build_isolation_tree(self, data: List[List[float]], height_limit: int) -> Dict[str, Any]:
        """Build a single isolation tree."""
        def _build_node(data_subset: List[List[float]], current_height: int) -> Dict[str, Any]:
            # Base cases
            if current_height >= height_limit or len(data_subset) <= 1:
                return {
                    'type': 'leaf',
                    'size': len(data_subset),
                    'height': current_height
                }
            
            # All data points are identical
            if len(set(tuple(point) for point in data_subset)) == 1:
                return {
                    'type': 'leaf',
                    'size': len(data_subset),
                    'height': current_height
                }
            
            # Select random feature and split point
            import random
            random.seed(self.random_state + current_height)
            
            n_features = len(data_subset[0])
            feature_idx = random.randint(0, n_features - 1)
            
            # Get feature values for selected feature
            feature_values = [point[feature_idx] for point in data_subset]
            min_val = min(feature_values)
            max_val = max(feature_values)
            
            if min_val == max_val:
                return {
                    'type': 'leaf',
                    'size': len(data_subset),
                    'height': current_height
                }
            
            # Random split point
            split_value = random.uniform(min_val, max_val)
            
            # Split data
            left_data = [point for point in data_subset if point[feature_idx] < split_value]
            right_data = [point for point in data_subset if point[feature_idx] >= split_value]
            
            # If split doesn't separate data, create leaf
            if len(left_data) == 0 or len(right_data) == 0:
                return {
                    'type': 'leaf',
                    'size': len(data_subset),
                    'height': current_height
                }
            
            return {
                'type': 'internal',
                'feature_idx': feature_idx,
                'split_value': split_value,
                'left': _build_node(left_data, current_height + 1),
                'right': _build_node(right_data, current_height + 1)
            }
        
        return _build_node(data, 0)
    
    def _path_length(self, point: List[float], tree: Dict[str, Any]) -> float:
        """Calculate path length for a point in an isolation tree."""
        def _traverse(node: Dict[str, Any], current_height: int) -> float:
            if node['type'] == 'leaf':
                # Estimate average path length for remaining points
                size = node['size']
                if size <= 1:
                    return current_height
                else:
                    # Average path length of unsuccessful search in BST
                    return current_height + 2 * (math.log(size - 1) + 0.5772156649) - (2 * (size - 1) / size)
            
            if point[node['feature_idx']] < node['split_value']:
                return _traverse(node['left'], current_height + 1)
            else:
                return _traverse(node['right'], current_height + 1)
        
        return _traverse(tree, 0)
    
    def train(self, X: Any, y: Any = None, **kwargs) -> Dict[str, Any]:
        """
        Train the Isolation Forest model.
        
        Args:
            X: Dictionary containing timestamps and features
            y: Not used (unsupervised learning)
            **kwargs: Additional training parameters
            
        Returns:
            Training metrics and information
        """
        try:
            if not self._validate_data(X):
                raise ModelTrainingError("Invalid data format. Expected dict with timestamps and features")
            
            timestamps = X['timestamps']
            features = X['features']
            
            # Store training data
            self._training_data = {
                'timestamps': timestamps,
                'features': features
            }
            
            # Normalize features
            normalized_features = self._normalize_features(features)
            
            if not normalized_features:
                raise ModelTrainingError("No valid features found for training")
            
            # Build isolation trees
            self._trees = []
            n_samples = min(self.max_samples, len(normalized_features))
            height_limit = int(math.ceil(math.log2(n_samples)))
            
            import random
            random.seed(self.random_state)
            
            for i in range(self.n_estimators):
                # Sample data for this tree
                if len(normalized_features) > n_samples:
                    tree_data = random.sample(normalized_features, n_samples)
                else:
                    tree_data = normalized_features.copy()
                
                # Build tree
                tree = self._build_isolation_tree(tree_data, height_limit)
                self._trees.append(tree)
            
            # Calculate anomaly scores for training data
            anomaly_scores = self._calculate_anomaly_scores(normalized_features)
            
            # Determine threshold based on contamination rate
            sorted_scores = sorted(anomaly_scores, reverse=True)
            threshold_idx = int(self.contamination * len(sorted_scores))
            self.anomaly_threshold = sorted_scores[threshold_idx] if threshold_idx < len(sorted_scores) else 0.5
            
            # Update metadata
            self.metadata.trained_at = datetime.now()
            self.metadata.features = list(self._feature_stats.keys())
            self.metadata.hyperparameters = {
                'n_estimators': self.n_estimators,
                'max_samples': self.max_samples,
                'contamination': self.contamination,
                'max_features': self.max_features
            }
            self.metadata.training_data_info = {
                'data_points': len(timestamps),
                'n_features': len(self._feature_stats),
                'date_range': f"{min(timestamps)} to {max(timestamps)}" if timestamps else "empty",
                'anomaly_threshold': self.anomaly_threshold
            }
            
            # Calculate training metrics
            anomalies_detected = sum(1 for score in anomaly_scores if score > self.anomaly_threshold)
            training_metrics = {
                'n_trees_built': len(self._trees),
                'n_features_used': len(self._feature_stats),
                'anomaly_threshold': self.anomaly_threshold,
                'anomalies_in_training': anomalies_detected,
                'anomaly_rate': anomalies_detected / len(anomaly_scores) * 100,
                'avg_anomaly_score': sum(anomaly_scores) / len(anomaly_scores),
                'max_anomaly_score': max(anomaly_scores),
                'training_contamination': self.contamination * 100
            }
            
            self.metadata.performance_metrics = training_metrics
            self._is_trained = True
            
            logger.info(f"Isolation Forest model trained successfully on {len(timestamps)} data points")
            return training_metrics
            
        except Exception as e:
            raise ModelTrainingError(f"Isolation Forest training failed: {str(e)}")
    
    def _calculate_anomaly_scores(self, normalized_features: List[List[float]]) -> List[float]:
        """Calculate anomaly scores for given features."""
        scores = []
        
        for point in normalized_features:
            # Calculate average path length across all trees
            path_lengths = [self._path_length(point, tree) for tree in self._trees]
            avg_path_length = sum(path_lengths) / len(path_lengths)
            
            # Convert to anomaly score (lower path length = higher anomaly score)
            # Normalize by expected path length
            n_samples = self.max_samples
            c_n = 2 * (math.log(n_samples - 1) + 0.5772156649) - (2 * (n_samples - 1) / n_samples)
            
            if c_n > 0:
                anomaly_score = 2 ** (-avg_path_length / c_n)
            else:
                anomaly_score = 0.5
            
            scores.append(anomaly_score)
        
        return scores
    
    def predict(self, X: Any, **kwargs) -> PredictionResult:
        """
        Predict anomalies in new data.
        
        Args:
            X: Dictionary containing timestamps and features
            **kwargs: Additional prediction parameters
            
        Returns:
            Anomaly predictions with scores
        """
        if not self._is_trained:
            raise PredictionError("Model must be trained before making predictions")
        
        try:
            timestamps = X['timestamps']
            features = X['features']
            
            # Normalize features using training statistics
            normalized_features = self._normalize_features(features)
            
            # Calculate anomaly scores
            anomaly_scores = self._calculate_anomaly_scores(normalized_features)
            
            # Determine anomalies
            predictions = []
            for i, (timestamp, score) in enumerate(zip(timestamps, anomaly_scores)):
                is_anomaly = score > self.anomaly_threshold
                
                predictions.append({
                    'timestamp': timestamp,
                    'anomaly_score': score,
                    'is_anomaly': is_anomaly,
                    'severity': self._classify_severity(score),
                    'features': features[i]
                })
            
            result = PredictionResult(
                predictions=predictions,
                confidence_intervals=None,  # Not applicable for anomaly detection
                prediction_metadata={
                    'model_type': 'isolation_forest',
                    'anomaly_threshold': self.anomaly_threshold,
                    'anomalies_detected': sum(1 for p in predictions if p['is_anomaly']),
                    'max_score': max(anomaly_scores),
                    'avg_score': sum(anomaly_scores) / len(anomaly_scores)
                },
                model_version=self.metadata.version
            )
            
            logger.info(f"Isolation Forest predictions completed: {sum(1 for p in predictions if p['is_anomaly'])} anomalies detected")
            return result
            
        except Exception as e:
            raise PredictionError(f"Isolation Forest prediction failed: {str(e)}")
    
    def _classify_severity(self, anomaly_score: float) -> str:
        """Classify anomaly severity based on score."""
        if anomaly_score > 0.8:
            return 'critical'
        elif anomaly_score > 0.7:
            return 'high'
        elif anomaly_score > 0.6:
            return 'medium'
        else:
            return 'low'
    
    def detect_anomalies(self, X: Any, **kwargs) -> Dict[str, Any]:
        """
        Detect anomalies in the provided data.
        
        Args:
            X: Data to check for anomalies
            **kwargs: Additional detection parameters
            
        Returns:
            Anomaly detection results including scores and labels
        """
        predictions = self.predict(X, **kwargs)
        
        anomalies = [p for p in predictions.predictions if p['is_anomaly']]
        
        return {
            'anomalies': anomalies,
            'total_anomalies': len(anomalies),
            'anomaly_rate': len(anomalies) / len(predictions.predictions) * 100,
            'severity_distribution': {
                'critical': len([a for a in anomalies if a['severity'] == 'critical']),
                'high': len([a for a in anomalies if a['severity'] == 'high']),
                'medium': len([a for a in anomalies if a['severity'] == 'medium']),
                'low': len([a for a in anomalies if a['severity'] == 'low'])
            },
            'detection_summary': predictions.prediction_metadata
        }
    
    def score_anomalies(self, X: Any, **kwargs) -> List[float]:
        """
        Calculate anomaly scores for data points.
        
        Args:
            X: Data to score
            **kwargs: Additional scoring parameters
            
        Returns:
            Anomaly scores for each data point
        """
        predictions = self.predict(X, **kwargs)
        return [p['anomaly_score'] for p in predictions.predictions]
    
    def evaluate(self, X: Any, y: Any, **kwargs) -> Dict[str, float]:
        """
        Evaluate anomaly detection performance.
        
        Args:
            X: Test data
            y: True anomaly labels (1 for anomaly, 0 for normal)
            **kwargs: Additional evaluation parameters
            
        Returns:
            Performance metrics
        """
        if not self._is_trained:
            raise ValueError("Model must be trained before evaluation")
        
        try:
            predictions = self.predict(X)
            pred_labels = [1 if p['is_anomaly'] else 0 for p in predictions.predictions]
            
            # Calculate classification metrics
            true_positives = sum(1 for p, t in zip(pred_labels, y) if p == 1 and t == 1)
            false_positives = sum(1 for p, t in zip(pred_labels, y) if p == 1 and t == 0)
            true_negatives = sum(1 for p, t in zip(pred_labels, y) if p == 0 and t == 0)
            false_negatives = sum(1 for p, t in zip(pred_labels, y) if p == 0 and t == 1)
            
            precision = true_positives / max(1, true_positives + false_positives)
            recall = true_positives / max(1, true_positives + false_negatives)
            f1_score = 2 * precision * recall / max(0.001, precision + recall)
            accuracy = (true_positives + true_negatives) / len(y)
            
            metrics = {
                'precision': precision,
                'recall': recall,
                'f1_score': f1_score,
                'accuracy': accuracy,
                'true_positives': true_positives,
                'false_positives': false_positives,
                'false_negatives': false_negatives,
                'true_negatives': true_negatives
            }
            
            logger.info(f"Isolation Forest evaluation completed: F1={f1_score:.3f}, Precision={precision:.3f}, Recall={recall:.3f}")
            return metrics
            
        except Exception as e:
            raise ValueError(f"Evaluation failed: {str(e)}")
    
    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """
        Get feature importance based on isolation tree usage.
        
        Returns:
            Dictionary of feature importance scores
        """
        if not self._is_trained or not self._feature_stats:
            return None
        
        feature_names = list(self._feature_stats.keys())
        feature_counts = {name: 0 for name in feature_names}
        
        # Count how often each feature is used for splitting
        def _count_features(node: Dict[str, Any]) -> None:
            if node['type'] == 'internal':
                feature_idx = node['feature_idx']
                if feature_idx < len(feature_names):
                    feature_counts[feature_names[feature_idx]] += 1
                _count_features(node['left'])
                _count_features(node['right'])
        
        for tree in self._trees:
            _count_features(tree)
        
        # Normalize to get importance scores
        total_splits = sum(feature_counts.values())
        if total_splits > 0:
            importance = {name: count / total_splits for name, count in feature_counts.items()}
        else:
            importance = {name: 1.0 / len(feature_names) for name in feature_names}
        
        return importance
    
    def get_anomaly_explanations(self, X: Any, **kwargs) -> List[Dict[str, Any]]:
        """
        Get explanations for detected anomalies.
        
        Args:
            X: Data containing anomalies
            **kwargs: Additional explanation parameters
            
        Returns:
            List of anomaly explanations
        """
        if not self._is_trained:
            return []
        
        try:
            predictions = self.predict(X)
            explanations = []
            
            for pred in predictions.predictions:
                if pred['is_anomaly']:
                    # Analyze which features contributed most to anomaly score
                    features = pred['features']
                    feature_contributions = {}
                    
                    for feature_name, value in features.items():
                        if feature_name in self._feature_stats:
                            stats = self._feature_stats[feature_name]
                            
                            # Calculate how unusual this value is
                            if stats['std'] > 0:
                                z_score = abs((value - stats['mean']) / stats['std'])
                                feature_contributions[feature_name] = z_score
                            else:
                                feature_contributions[feature_name] = 0
                    
                    # Get top contributing features
                    top_features = sorted(feature_contributions.items(), key=lambda x: x[1], reverse=True)[:3]
                    
                    explanation = {
                        'timestamp': pred['timestamp'],
                        'anomaly_score': pred['anomaly_score'],
                        'severity': pred['severity'],
                        'top_contributing_features': [
                            {
                                'feature': feature_name,
                                'value': features[feature_name],
                                'z_score': contribution,
                                'normal_range': f"{self._feature_stats[feature_name]['mean']:.2f} ± {self._feature_stats[feature_name]['std']:.2f}"
                            }
                            for feature_name, contribution in top_features
                        ],
                        'explanation': self._generate_explanation_text(pred, top_features)
                    }
                    
                    explanations.append(explanation)
            
            return explanations
            
        except Exception as e:
            logger.error(f"Failed to generate anomaly explanations: {str(e)}")
            return []
    
    def _generate_explanation_text(self, prediction: Dict[str, Any], top_features: List[Tuple[str, float]]) -> str:
        """Generate human-readable explanation for anomaly."""
        severity = prediction['severity']
        score = prediction['anomaly_score']
        
        explanation = f"This {severity} severity anomaly (score: {score:.3f}) was detected due to unusual values in: "
        
        feature_descriptions = []
        for feature_name, z_score in top_features:
            if z_score > 3:
                description = f"{feature_name} (extremely unusual)"
            elif z_score > 2:
                description = f"{feature_name} (very unusual)"
            else:
                description = f"{feature_name} (moderately unusual)"
            feature_descriptions.append(description)
        
        explanation += ", ".join(feature_descriptions)
        return explanation
    
    def set_contamination_rate(self, contamination: float) -> None:
        """
        Set the expected contamination rate.
        
        Args:
            contamination: Expected proportion of anomalies (0-1)
        """
        if not 0 < contamination < 1:
            raise ValueError("Contamination rate must be between 0 and 1")
        
        self.contamination = contamination
        logger.info(f"Contamination rate set to {contamination:.3f}")
    
    def get_model_summary(self) -> Dict[str, Any]:
        """
        Get model summary and configuration.
        
        Returns:
            Model summary information
        """
        if not self._is_trained:
            return {'status': 'not_trained'}
        
        return {
            'model_type': 'isolation_forest',
            'n_estimators': self.n_estimators,
            'max_samples': self.max_samples,
            'contamination': self.contamination,
            'anomaly_threshold': self.anomaly_threshold,
            'n_features': len(self._feature_stats),
            'feature_names': list(self._feature_stats.keys()),
            'training_data_points': len(self._training_data['timestamps']),
            'performance_metrics': self.metadata.performance_metrics
        }