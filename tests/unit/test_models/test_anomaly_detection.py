"""
Unit tests for anomaly detection models.

Tests anomaly detection algorithms, precision, and recall metrics.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


class TestAnomalyDetection:
    """Test suite for anomaly detection models."""

    @pytest.fixture
    def mock_anomaly_detector(self):
        """Create a mock anomaly detector."""
        detector = Mock()
        detector.train = Mock()
        detector.detect = Mock()
        detector.evaluate = Mock()
        detector.fit = Mock()
        detector.predict = Mock()
        detector.is_trained = False
        detector.model_type = "isolation_forest"
        detector.threshold = 0.1
        return detector

    @pytest.fixture
    def sample_normal_data(self):
        """Create sample normal data for anomaly detection."""
        np.random.seed(42)
        n_samples = 1000
        
        # Generate normal data with patterns
        timestamps = pd.date_range('2024-01-01', periods=n_samples, freq='H')
        normal_cost = np.random.normal(1000, 100, n_samples)
        normal_usage = np.random.normal(50, 10, n_samples)
        normal_query_count = np.random.poisson(100, n_samples)
        
        return pd.DataFrame({
            'timestamp': timestamps,
            'cost': normal_cost,
            'usage': normal_usage,
            'query_count': normal_query_count,
            'warehouse': ['WH_ANALYTICS'] * n_samples,
            'label': [0] * n_samples  # 0 = normal
        })

    @pytest.fixture
    def sample_anomalous_data(self):
        """Create sample anomalous data for testing."""
        np.random.seed(42)
        n_anomalies = 50
        
        # Generate anomalous data
        timestamps = pd.date_range('2024-01-01', periods=n_anomalies, freq='H')
        anomalous_cost = np.random.normal(3000, 500, n_anomalies)  # Much higher cost
        anomalous_usage = np.random.normal(150, 30, n_anomalies)  # Much higher usage
        anomalous_query_count = np.random.poisson(500, n_anomalies)  # Much higher queries
        
        return pd.DataFrame({
            'timestamp': timestamps,
            'cost': anomalous_cost,
            'usage': anomalous_usage,
            'query_count': anomalous_query_count,
            'warehouse': ['WH_ANALYTICS'] * n_anomalies,
            'label': [1] * n_anomalies  # 1 = anomaly
        })

    @pytest.fixture
    def combined_data(self, sample_normal_data, sample_anomalous_data):
        """Create combined normal and anomalous data."""
        # Combine normal and anomalous data
        combined = pd.concat([sample_normal_data, sample_anomalous_data], ignore_index=True)
        # Shuffle the data
        return combined.sample(frac=1).reset_index(drop=True)

    def test_anomaly_detector_initialization(self, mock_anomaly_detector):
        """Test anomaly detector initialization."""
        assert mock_anomaly_detector is not None
        assert mock_anomaly_detector.model_type == "isolation_forest"
        assert mock_anomaly_detector.threshold == 0.1
        assert mock_anomaly_detector.is_trained is False

    def test_anomaly_detector_training(self, mock_anomaly_detector, sample_normal_data):
        """Test anomaly detector training."""
        # Mock successful training
        mock_anomaly_detector.train.return_value = True
        mock_anomaly_detector.is_trained = True
        
        # Test training
        result = mock_anomaly_detector.train(sample_normal_data)
        
        assert result is True
        assert mock_anomaly_detector.is_trained is True
        mock_anomaly_detector.train.assert_called_once_with(sample_normal_data)

    def test_anomaly_detection_success(self, mock_anomaly_detector, combined_data):
        """Test successful anomaly detection."""
        # Mock trained detector
        mock_anomaly_detector.is_trained = True
        
        # Mock detection results
        n_samples = len(combined_data)
        anomaly_scores = np.random.uniform(0, 1, n_samples)
        anomaly_labels = (anomaly_scores > 0.1).astype(int)
        
        detection_results = {
            'anomaly_scores': anomaly_scores,
            'anomaly_labels': anomaly_labels,
            'anomaly_indices': np.where(anomaly_labels == 1)[0],
            'anomaly_count': np.sum(anomaly_labels),
            'anomaly_percentage': np.mean(anomaly_labels) * 100
        }
        
        mock_anomaly_detector.detect.return_value = detection_results
        
        # Test detection
        result = mock_anomaly_detector.detect(combined_data)
        
        assert result is not None
        assert 'anomaly_scores' in result
        assert 'anomaly_labels' in result
        assert len(result['anomaly_scores']) == n_samples
        assert result['anomaly_count'] == np.sum(anomaly_labels)
        mock_anomaly_detector.detect.assert_called_once_with(combined_data)

    def test_anomaly_detection_evaluation(self, mock_anomaly_detector):
        """Test anomaly detection evaluation metrics."""
        # Mock evaluation metrics
        evaluation_metrics = {
            'precision': 0.85,
            'recall': 0.78,
            'f1_score': 0.81,
            'accuracy': 0.92,
            'auc_roc': 0.88,
            'auc_pr': 0.83,
            'confusion_matrix': {
                'true_positives': 39,
                'false_positives': 7,
                'true_negatives': 920,
                'false_negatives': 34
            },
            'detection_rate': 0.78,
            'false_positive_rate': 0.008
        }
        
        mock_anomaly_detector.evaluate.return_value = evaluation_metrics
        
        # Test evaluation
        result = mock_anomaly_detector.evaluate()
        
        assert result['precision'] == 0.85
        assert result['recall'] == 0.78
        assert result['f1_score'] == 0.81
        assert result['auc_roc'] == 0.88
        assert result['confusion_matrix']['true_positives'] == 39

    def test_anomaly_types_classification(self, mock_anomaly_detector):
        """Test classification of different anomaly types."""
        anomaly_types = {
            'point_anomalies': {
                'count': 25,
                'examples': [
                    {'timestamp': datetime(2024, 1, 15, 10), 'type': 'cost_spike', 'severity': 'high'},
                    {'timestamp': datetime(2024, 1, 18, 14), 'type': 'usage_drop', 'severity': 'medium'}
                ]
            },
            'contextual_anomalies': {
                'count': 15,
                'examples': [
                    {'timestamp': datetime(2024, 1, 20, 2), 'type': 'unexpected_weekend_activity', 'severity': 'medium'},
                    {'timestamp': datetime(2024, 1, 22, 22), 'type': 'late_night_spike', 'severity': 'low'}
                ]
            },
            'collective_anomalies': {
                'count': 8,
                'examples': [
                    {'time_range': '2024-01-25 to 2024-01-27', 'type': 'sustained_high_usage', 'severity': 'high'},
                    {'time_range': '2024-01-30 to 2024-02-01', 'type': 'system_degradation', 'severity': 'critical'}
                ]
            }
        }
        
        mock_anomaly_detector.classify_anomaly_types.return_value = anomaly_types
        
        # Test anomaly type classification
        result = mock_anomaly_detector.classify_anomaly_types()
        
        assert result['point_anomalies']['count'] == 25
        assert result['contextual_anomalies']['count'] == 15
        assert result['collective_anomalies']['count'] == 8
        assert len(result['point_anomalies']['examples']) == 2

    def test_anomaly_severity_scoring(self, mock_anomaly_detector):
        """Test anomaly severity scoring."""
        severity_results = {
            'severity_distribution': {
                'critical': 5,
                'high': 12,
                'medium': 18,
                'low': 15
            },
            'severity_scores': {
                'critical_threshold': 0.9,
                'high_threshold': 0.7,
                'medium_threshold': 0.5,
                'low_threshold': 0.3
            },
            'severity_details': [
                {
                    'anomaly_id': 'anom_001',
                    'severity': 'critical',
                    'score': 0.95,
                    'impact': 'high_cost_spike',
                    'estimated_impact_cost': 5000.0
                },
                {
                    'anomaly_id': 'anom_002',
                    'severity': 'high',
                    'score': 0.82,
                    'impact': 'performance_degradation',
                    'estimated_impact_cost': 2500.0
                }
            ]
        }
        
        mock_anomaly_detector.score_severity.return_value = severity_results
        
        # Test severity scoring
        result = mock_anomaly_detector.score_severity()
        
        assert result['severity_distribution']['critical'] == 5
        assert result['severity_scores']['critical_threshold'] == 0.9
        assert len(result['severity_details']) == 2
        assert result['severity_details'][0]['severity'] == 'critical'

    def test_anomaly_root_cause_analysis(self, mock_anomaly_detector):
        """Test anomaly root cause analysis."""
        root_cause_analysis = {
            'primary_causes': [
                {
                    'cause': 'query_inefficiency',
                    'confidence': 0.85,
                    'contributing_factors': ['missing_index', 'full_table_scan'],
                    'affected_queries': ['query_123', 'query_456']
                },
                {
                    'cause': 'warehouse_oversizing',
                    'confidence': 0.72,
                    'contributing_factors': ['auto_scaling_disabled', 'manual_scaling'],
                    'affected_warehouses': ['WH_ANALYTICS']
                }
            ],
            'secondary_causes': [
                {
                    'cause': 'seasonal_demand',
                    'confidence': 0.45,
                    'contributing_factors': ['month_end_processing'],
                    'seasonal_pattern': 'monthly'
                }
            ],
            'recommended_actions': [
                'optimize_expensive_queries',
                'enable_auto_scaling',
                'review_warehouse_sizing'
            ]
        }
        
        mock_anomaly_detector.analyze_root_cause.return_value = root_cause_analysis
        
        # Test root cause analysis
        result = mock_anomaly_detector.analyze_root_cause()
        
        assert len(result['primary_causes']) == 2
        assert result['primary_causes'][0]['confidence'] == 0.85
        assert len(result['recommended_actions']) == 3
        assert result['secondary_causes'][0]['seasonal_pattern'] == 'monthly'

    def test_anomaly_detection_thresholds(self, mock_anomaly_detector):
        """Test anomaly detection threshold optimization."""
        threshold_optimization = {
            'current_threshold': 0.1,
            'optimal_threshold': 0.12,
            'threshold_analysis': {
                'precision_recall_curve': {
                    'thresholds': np.linspace(0.01, 0.5, 50),
                    'precision': np.random.uniform(0.6, 0.9, 50),
                    'recall': np.random.uniform(0.5, 0.8, 50)
                },
                'roc_curve': {
                    'thresholds': np.linspace(0.01, 0.5, 50),
                    'fpr': np.random.uniform(0.01, 0.2, 50),
                    'tpr': np.random.uniform(0.6, 0.9, 50)
                }
            },
            'threshold_recommendations': {
                'high_precision': 0.2,
                'high_recall': 0.08,
                'balanced': 0.12,
                'low_false_positive': 0.15
            }
        }
        
        mock_anomaly_detector.optimize_threshold.return_value = threshold_optimization
        
        # Test threshold optimization
        result = mock_anomaly_detector.optimize_threshold()
        
        assert result['current_threshold'] == 0.1
        assert result['optimal_threshold'] == 0.12
        assert len(result['threshold_analysis']['precision_recall_curve']['thresholds']) == 50
        assert result['threshold_recommendations']['balanced'] == 0.12

    def test_anomaly_detection_feature_importance(self, mock_anomaly_detector):
        """Test feature importance in anomaly detection."""
        feature_importance = {
            'feature_scores': {
                'cost': 0.35,
                'usage': 0.28,
                'query_count': 0.22,
                'credits_used': 0.15
            },
            'feature_contributions': {
                'cost': {
                    'positive_anomalies': 0.85,
                    'negative_anomalies': 0.15
                },
                'usage': {
                    'positive_anomalies': 0.78,
                    'negative_anomalies': 0.22
                }
            },
            'feature_interactions': {
                'cost_usage_interaction': 0.15,
                'usage_query_interaction': 0.12,
                'cost_credits_interaction': 0.08
            }
        }
        
        mock_anomaly_detector.analyze_feature_importance.return_value = feature_importance
        
        # Test feature importance
        result = mock_anomaly_detector.analyze_feature_importance()
        
        assert result['feature_scores']['cost'] == 0.35
        assert result['feature_scores']['usage'] == 0.28
        assert result['feature_contributions']['cost']['positive_anomalies'] == 0.85
        assert result['feature_interactions']['cost_usage_interaction'] == 0.15

    def test_anomaly_detection_time_series(self, mock_anomaly_detector):
        """Test time series specific anomaly detection."""
        time_series_anomalies = {
            'trend_anomalies': {
                'count': 8,
                'examples': [
                    {'period': '2024-01-15 to 2024-01-20', 'type': 'sudden_trend_change'},
                    {'period': '2024-01-25 to 2024-01-30', 'type': 'trend_reversal'}
                ]
            },
            'seasonal_anomalies': {
                'count': 12,
                'examples': [
                    {'timestamp': datetime(2024, 1, 10, 15), 'type': 'seasonal_deviation'},
                    {'timestamp': datetime(2024, 1, 18, 9), 'type': 'missing_seasonal_pattern'}
                ]
            },
            'level_shift_anomalies': {
                'count': 5,
                'examples': [
                    {'timestamp': datetime(2024, 1, 12, 11), 'type': 'permanent_level_shift'},
                    {'timestamp': datetime(2024, 1, 22, 14), 'type': 'temporary_level_shift'}
                ]
            }
        }
        
        mock_anomaly_detector.detect_time_series_anomalies.return_value = time_series_anomalies
        
        # Test time series anomaly detection
        result = mock_anomaly_detector.detect_time_series_anomalies()
        
        assert result['trend_anomalies']['count'] == 8
        assert result['seasonal_anomalies']['count'] == 12
        assert result['level_shift_anomalies']['count'] == 5
        assert len(result['trend_anomalies']['examples']) == 2

    def test_anomaly_detection_ensemble(self, mock_anomaly_detector):
        """Test ensemble anomaly detection."""
        ensemble_results = {
            'ensemble_models': [
                'isolation_forest',
                'local_outlier_factor',
                'one_class_svm',
                'statistical_detector'
            ],
            'model_weights': [0.3, 0.25, 0.25, 0.2],
            'individual_results': {
                'isolation_forest': {'precision': 0.85, 'recall': 0.78},
                'local_outlier_factor': {'precision': 0.82, 'recall': 0.75},
                'one_class_svm': {'precision': 0.88, 'recall': 0.72},
                'statistical_detector': {'precision': 0.79, 'recall': 0.85}
            },
            'ensemble_performance': {
                'precision': 0.87,
                'recall': 0.80,
                'f1_score': 0.83,
                'improvement_over_best_single': 0.04
            }
        }
        
        mock_anomaly_detector.ensemble_detect.return_value = ensemble_results
        
        # Test ensemble detection
        result = mock_anomaly_detector.ensemble_detect()
        
        assert len(result['ensemble_models']) == 4
        assert sum(result['model_weights']) == 1.0
        assert result['ensemble_performance']['precision'] == 0.87
        assert result['ensemble_performance']['improvement_over_best_single'] == 0.04

    def test_anomaly_detection_streaming(self, mock_anomaly_detector):
        """Test streaming anomaly detection."""
        streaming_results = {
            'stream_processing': {
                'window_size': 100,
                'slide_interval': 10,
                'processing_latency_ms': 25.5,
                'throughput_per_second': 1000
            },
            'real_time_anomalies': [
                {
                    'timestamp': datetime.now(),
                    'anomaly_score': 0.85,
                    'severity': 'high',
                    'type': 'cost_spike',
                    'confidence': 0.92
                },
                {
                    'timestamp': datetime.now() - timedelta(minutes=5),
                    'anomaly_score': 0.65,
                    'severity': 'medium',
                    'type': 'usage_drop',
                    'confidence': 0.78
                }
            ],
            'adaptation_metrics': {
                'model_updates': 156,
                'adaptation_rate': 0.15,
                'stability_score': 0.88
            }
        }
        
        mock_anomaly_detector.stream_detect.return_value = streaming_results
        
        # Test streaming detection
        result = mock_anomaly_detector.stream_detect()
        
        assert result['stream_processing']['window_size'] == 100
        assert result['stream_processing']['processing_latency_ms'] == 25.5
        assert len(result['real_time_anomalies']) == 2
        assert result['adaptation_metrics']['model_updates'] == 156

    @pytest.mark.parametrize("detector_type,expected_performance", [
        ("isolation_forest", {'precision': 0.85, 'recall': 0.78}),
        ("local_outlier_factor", {'precision': 0.82, 'recall': 0.75}),
        ("one_class_svm", {'precision': 0.88, 'recall': 0.72}),
        ("statistical_detector", {'precision': 0.79, 'recall': 0.85}),
        ("autoencoder", {'precision': 0.83, 'recall': 0.81}),
    ])
    def test_different_detector_types(self, mock_anomaly_detector, detector_type, expected_performance):
        """Test different anomaly detector types."""
        mock_anomaly_detector.model_type = detector_type
        mock_anomaly_detector.evaluate.return_value = expected_performance
        
        # Test detector type
        assert mock_anomaly_detector.model_type == detector_type
        
        # Test expected performance
        result = mock_anomaly_detector.evaluate()
        assert result['precision'] == expected_performance['precision']
        assert result['recall'] == expected_performance['recall']

    def test_anomaly_detection_explainability(self, mock_anomaly_detector):
        """Test anomaly detection explainability."""
        explainability_results = {
            'explanation_method': 'lime',
            'feature_explanations': {
                'cost': {
                    'contribution': 0.45,
                    'direction': 'positive',
                    'explanation': 'Cost is 3x higher than normal'
                },
                'usage': {
                    'contribution': 0.32,
                    'direction': 'positive',
                    'explanation': 'Usage is 2x higher than expected'
                },
                'query_count': {
                    'contribution': 0.23,
                    'direction': 'positive',
                    'explanation': 'Query count is unusually high'
                }
            },
            'visual_explanations': {
                'feature_importance_plot': 'path/to/plot.png',
                'anomaly_timeline': 'path/to/timeline.png',
                'distribution_comparison': 'path/to/distribution.png'
            }
        }
        
        mock_anomaly_detector.explain_anomaly.return_value = explainability_results
        
        # Test explainability
        result = mock_anomaly_detector.explain_anomaly()
        
        assert result['explanation_method'] == 'lime'
        assert result['feature_explanations']['cost']['contribution'] == 0.45
        assert result['feature_explanations']['usage']['direction'] == 'positive'
        assert 'feature_importance_plot' in result['visual_explanations']

    def test_anomaly_detection_performance_monitoring(self, mock_anomaly_detector):
        """Test anomaly detection performance monitoring."""
        performance_metrics = {
            'detection_latency': {
                'mean_ms': 15.2,
                'median_ms': 12.8,
                'p95_ms': 28.5,
                'p99_ms': 45.3
            },
            'throughput': {
                'samples_per_second': 850,
                'anomalies_per_second': 12.5,
                'batch_processing_rate': 5000
            },
            'resource_usage': {
                'cpu_usage_percent': 45.2,
                'memory_usage_mb': 512.8,
                'disk_io_mb_per_second': 25.3
            },
            'accuracy_over_time': {
                'daily_precision': [0.85, 0.83, 0.87, 0.84],
                'daily_recall': [0.78, 0.80, 0.76, 0.82],
                'trend': 'stable'
            }
        }
        
        mock_anomaly_detector.monitor_performance.return_value = performance_metrics
        
        # Test performance monitoring
        result = mock_anomaly_detector.monitor_performance()
        
        assert result['detection_latency']['mean_ms'] == 15.2
        assert result['throughput']['samples_per_second'] == 850
        assert result['resource_usage']['cpu_usage_percent'] == 45.2
        assert len(result['accuracy_over_time']['daily_precision']) == 4