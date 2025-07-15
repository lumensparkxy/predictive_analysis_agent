"""
Pattern Detection Feature Generator

This module provides comprehensive pattern detection feature engineering capabilities
for extracting behavioral patterns and anomalies from Snowflake analytics data.
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime, timedelta
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import warnings

logger = logging.getLogger(__name__)


class PatternFeatureGenerator:
    """
    Generates comprehensive pattern detection features from analytics data.
    
    This class extracts various behavioral patterns including user activity patterns,
    query complexity patterns, resource usage patterns, and anomaly detection
    features specifically designed for Snowflake analytics workload analysis.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the PatternFeatureGenerator with optional configuration.
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        
        # Default configuration
        self.default_config = {
            'pattern_types': [
                'user_behavior', 'query_complexity', 'resource_usage',
                'temporal_patterns', 'anomaly_detection'
            ],
            'clustering_features': [
                'credits_used', 'execution_time_ms', 'bytes_scanned',
                'rows_produced', 'partitions_scanned'
            ],
            'user_columns': {
                'user_name': 'USER_NAME',
                'role_name': 'ROLE_NAME',
                'session_id': 'SESSION_ID'
            },
            'query_columns': {
                'query_type': 'QUERY_TYPE',
                'database_name': 'DATABASE_NAME',
                'warehouse_name': 'WAREHOUSE_NAME'
            },
            'time_column': 'START_TIME',
            'clustering_params': {
                'n_clusters': 5,
                'random_state': 42
            },
            'anomaly_thresholds': {
                'z_score': 3.0,
                'iqr_multiplier': 1.5,
                'percentile': 99
            },
            'pattern_windows': {
                'short': 3,   # days
                'medium': 7,  # days
                'long': 30    # days
            },
            'min_pattern_occurrences': 3,
            'similarity_threshold': 0.8,
            'enable_advanced_patterns': True
        }
        
        # Merge with provided config
        self.config = {**self.default_config, **self.config}
        
        # Initialize scalers for clustering
        self.scaler = StandardScaler()
        self.fitted = False
    
    def generate_pattern_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate comprehensive pattern detection features from the input DataFrame.
        
        Args:
            data: Input DataFrame containing analytics data
        
        Returns:
            DataFrame with generated pattern features
        """
        logger.info("Generating pattern detection features...")
        
        feature_df = pd.DataFrame(index=data.index)
        pattern_types = self.config.get('pattern_types', [])
        
        # User behavior patterns
        if 'user_behavior' in pattern_types:
            feature_df.update(self._extract_user_behavior_patterns(data))
        
        # Query complexity patterns
        if 'query_complexity' in pattern_types:
            feature_df.update(self._extract_query_complexity_patterns(data))
        
        # Resource usage patterns
        if 'resource_usage' in pattern_types:
            feature_df.update(self._extract_resource_usage_patterns(data))
        
        # Temporal patterns
        if 'temporal_patterns' in pattern_types:
            feature_df.update(self._extract_temporal_patterns(data))
        
        # Anomaly detection patterns
        if 'anomaly_detection' in pattern_types:
            feature_df.update(self._extract_anomaly_patterns(data))
        
        # Advanced pattern analysis
        if self.config.get('enable_advanced_patterns', True):
            feature_df.update(self._extract_advanced_patterns(data))
        
        # Clustering-based patterns
        feature_df.update(self._extract_clustering_patterns(data))
        
        logger.info(f"Generated {len(feature_df.columns)} pattern detection features")
        return feature_df
    
    def _extract_user_behavior_patterns(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Extract user behavior and activity patterns.
        
        Args:
            data: Input DataFrame
        
        Returns:
            DataFrame with user behavior pattern features
        """
        features = pd.DataFrame(index=data.index)
        user_cols = self.config['user_columns']
        time_col = self.config['time_column']
        
        if user_cols.get('user_name') not in data.columns:
            return features
        
        user_name = data[user_cols['user_name']]
        
        # User activity intensity patterns
        user_query_counts = user_name.value_counts()
        features['user_activity_score'] = user_name.map(user_query_counts)
        
        # Normalize activity score
        max_activity = features['user_activity_score'].max()
        features['user_activity_score_normalized'] = (
            features['user_activity_score'] / max_activity if max_activity > 0 else 0
        )
        
        # User behavior consistency (coefficient of variation of usage)
        if 'credits_used' in data.columns:
            user_credit_stats = data.groupby(user_name)['credits_used'].agg(['mean', 'std'])
            user_cv = (user_credit_stats['std'] / user_credit_stats['mean']).fillna(0)
            features['user_behavior_consistency'] = user_name.map(user_cv)
        
        # User session patterns
        if user_cols.get('session_id') in data.columns:
            session_id = data[user_cols['session_id']]
            
            # Session length (queries per session)
            session_lengths = session_id.value_counts()
            features['session_length'] = session_id.map(session_lengths)
            
            # User multi-session indicator
            user_session_counts = data.groupby(user_name)[user_cols['session_id']].nunique()
            features['user_multi_session'] = (
                user_name.map(user_session_counts) > 1
            ).astype(int)
        
        # User role behavior patterns
        if user_cols.get('role_name') in data.columns:
            role_name = data[user_cols['role_name']]
            
            # Role usage frequency
            role_counts = role_name.value_counts()
            features['role_usage_frequency'] = role_name.map(role_counts)
            
            # User-role combination frequency
            user_role = user_name + "_" + role_name.fillna('UNKNOWN')
            user_role_counts = user_role.value_counts()
            features['user_role_combination_frequency'] = user_role.map(user_role_counts)
        
        # Time-based user behavior patterns
        if time_col in data.columns:
            time_series = pd.to_datetime(data[time_col])
            
            # User preferred time of day
            user_hour_mode = data.groupby(user_name)[time_col].apply(
                lambda x: pd.to_datetime(x).dt.hour.mode().iloc[0] if len(x) > 0 else 12
            )
            features['user_preferred_hour'] = user_name.map(user_hour_mode)
            
            # User time diversity (how many different hours they use)
            user_hour_diversity = data.groupby(user_name)[time_col].apply(
                lambda x: pd.to_datetime(x).dt.hour.nunique()
            )
            features['user_time_diversity'] = user_name.map(user_hour_diversity)
            
            # User business hours preference
            business_hours = (time_series.dt.hour >= 9) & (time_series.dt.hour <= 17)
            user_business_ratio = data.groupby(user_name)[business_hours].mean()
            features['user_business_hours_preference'] = user_name.map(user_business_ratio)
        
        # User query type diversity
        if self.config['query_columns'].get('query_type') in data.columns:
            query_type = data[self.config['query_columns']['query_type']]
            user_query_diversity = data.groupby(user_name)[query_type.name].nunique()
            features['user_query_type_diversity'] = user_name.map(user_query_diversity)
        
        return features
    
    def _extract_query_complexity_patterns(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Extract query complexity and characteristics patterns.
        
        Args:
            data: Input DataFrame
        
        Returns:
            DataFrame with query complexity pattern features
        """
        features = pd.DataFrame(index=data.index)
        
        # Multi-dimensional complexity score
        complexity_components = []
        
        # Execution time component
        if 'execution_time_ms' in data.columns:
            exec_time = data['execution_time_ms']
            exec_time_norm = (exec_time - exec_time.quantile(0.1)) / (
                exec_time.quantile(0.9) - exec_time.quantile(0.1)
            )
            complexity_components.append(exec_time_norm.clip(0, 1))
        
        # Data volume component
        if 'bytes_scanned' in data.columns:
            bytes_scanned = data['bytes_scanned']
            bytes_norm = (bytes_scanned - bytes_scanned.quantile(0.1)) / (
                bytes_scanned.quantile(0.9) - bytes_scanned.quantile(0.1)
            )
            complexity_components.append(bytes_norm.clip(0, 1))
        
        # Partition complexity component
        if 'partitions_scanned' in data.columns:
            partitions = data['partitions_scanned']
            part_norm = (partitions - partitions.quantile(0.1)) / (
                partitions.quantile(0.9) - partitions.quantile(0.1)
            )
            complexity_components.append(part_norm.clip(0, 1))
        
        # Row processing component
        if 'rows_produced' in data.columns:
            rows = data['rows_produced']
            rows_norm = (rows - rows.quantile(0.1)) / (
                rows.quantile(0.9) - rows.quantile(0.1)
            )
            complexity_components.append(rows_norm.clip(0, 1))
        
        if complexity_components:
            features['query_complexity_score'] = np.mean(complexity_components, axis=0)
            
            # Complexity categories
            features['complexity_category'] = pd.cut(
                features['query_complexity_score'],
                bins=[0, 0.2, 0.5, 0.8, 1.0],
                labels=['simple', 'moderate', 'complex', 'very_complex'],
                include_lowest=True
            ).astype('category')
        
        # Resource intensity patterns
        if 'credits_used' in data.columns and 'execution_time_ms' in data.columns:
            credits = data['credits_used']
            exec_time = data['execution_time_ms']
            
            # Resource intensity (credits per unit time)
            features['resource_intensity'] = np.where(
                exec_time > 0, credits / (exec_time / 1000), 0
            )
            
            # Resource efficiency class
            features['resource_efficiency_class'] = pd.cut(
                features['resource_intensity'],
                bins=np.percentile(features['resource_intensity'], [0, 25, 50, 75, 100]),
                labels=['efficient', 'moderate', 'intensive', 'very_intensive'],
                include_lowest=True
            ).astype('category')
        
        # Query selectivity patterns
        if 'bytes_scanned' in data.columns and 'rows_produced' in data.columns:
            bytes_scanned = data['bytes_scanned']
            rows_produced = data['rows_produced']
            
            # Selectivity (proportion of data that produces results)
            features['query_selectivity'] = np.where(
                bytes_scanned > 0, rows_produced * 100 / bytes_scanned, 0  # Rough estimate
            ).clip(0, 1)
            
            # Selectivity pattern
            features['selectivity_pattern'] = pd.cut(
                features['query_selectivity'],
                bins=[0, 0.01, 0.1, 0.5, 1.0],
                labels=['highly_selective', 'selective', 'moderate', 'broad'],
                include_lowest=True
            ).astype('category')
        
        return features
    
    def _extract_resource_usage_patterns(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Extract resource usage and optimization patterns.
        
        Args:
            data: Input DataFrame
        
        Returns:
            DataFrame with resource usage pattern features
        """
        features = pd.DataFrame(index=data.index)
        
        # Warehouse utilization patterns
        if 'warehouse_name' in data.columns:
            warehouse_name = data['warehouse_name']
            
            # Warehouse load distribution
            wh_counts = warehouse_name.value_counts()
            features['warehouse_load'] = warehouse_name.map(wh_counts)
            
            # Warehouse preference score (normalized load)
            max_load = features['warehouse_load'].max()
            features['warehouse_preference_score'] = (
                features['warehouse_load'] / max_load if max_load > 0 else 0
            )
        
        # Database access patterns
        if 'database_name' in data.columns:
            database_name = data['database_name']
            
            # Database usage frequency
            db_counts = database_name.value_counts()
            features['database_usage_frequency'] = database_name.map(db_counts)
            
            # Cross-database usage indicator
            user_col = self.config['user_columns'].get('user_name')
            if user_col in data.columns:
                user_db_diversity = data.groupby(data[user_col])['database_name'].nunique()
                features['user_database_diversity'] = data[user_col].map(user_db_diversity)
                
                features['is_cross_database_user'] = (
                    features['user_database_diversity'] > 1
                ).astype(int)
        
        # Resource waste indicators
        if 'credits_used' in data.columns and 'rows_produced' in data.columns:
            credits = data['credits_used']
            rows = data['rows_produced']
            
            # High cost, low output queries
            high_cost = credits > credits.quantile(0.8)
            low_output = rows < rows.quantile(0.2)
            features['potential_waste_indicator'] = (high_cost & low_output).astype(int)
            
            # Resource efficiency score
            features['resource_efficiency_score'] = np.where(
                credits > 0, rows / credits, 0
            )
            
            # Normalize efficiency score
            eff_score = features['resource_efficiency_score']
            if eff_score.max() > 0:
                features['resource_efficiency_score_normalized'] = (
                    eff_score / eff_score.quantile(0.95)
                ).clip(0, 1)
        
        # Queue pattern analysis
        if 'queue_time_ms' in data.columns:
            queue_time = data['queue_time_ms']
            
            # Queue frequency pattern
            has_queue = (queue_time > 1000).astype(int)  # More than 1 second
            features['experiences_queuing'] = has_queue
            
            # Queue severity
            features['queue_severity'] = pd.cut(
                queue_time,
                bins=[0, 1000, 10000, 60000, float('inf')],
                labels=['no_queue', 'short', 'medium', 'long'],
                include_lowest=True
            ).astype('category')
        
        return features
    
    def _extract_temporal_patterns(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Extract temporal usage patterns and periodicities.
        
        Args:
            data: Input DataFrame
        
        Returns:
            DataFrame with temporal pattern features
        """
        features = pd.DataFrame(index=data.index)
        time_col = self.config['time_column']
        
        if time_col not in data.columns:
            return features
        
        time_series = pd.to_datetime(data[time_col])
        
        # Peak hour patterns
        hour_counts = time_series.dt.hour.value_counts()
        peak_hours = hour_counts.nlargest(3).index.tolist()
        features['is_peak_hour'] = time_series.dt.hour.isin(peak_hours).astype(int)
        
        # Off-peak hour patterns
        off_peak_hours = hour_counts.nsmallest(3).index.tolist()
        features['is_off_peak_hour'] = time_series.dt.hour.isin(off_peak_hours).astype(int)
        
        # Weekend vs weekday patterns
        features['is_weekend_activity'] = (time_series.dt.dayofweek >= 5).astype(int)
        
        # Month-end activity patterns
        features['is_month_end_activity'] = (
            time_series.dt.day >= 25
        ).astype(int)  # Last week of month
        
        # Quarterly patterns
        features['is_quarter_end_activity'] = time_series.dt.is_quarter_end.astype(int)
        
        # Burst pattern detection (high activity in short periods)
        window_size = min(10, len(data))  # 10-query window
        query_density = pd.Series(index=data.index, dtype=float)
        
        for i in range(len(data)):
            start_idx = max(0, i - window_size // 2)
            end_idx = min(len(data), i + window_size // 2)
            
            if end_idx > start_idx:
                time_window = time_series.iloc[start_idx:end_idx]
                if len(time_window) > 1:
                    time_span = (time_window.max() - time_window.min()).total_seconds()
                    density = len(time_window) / max(time_span / 60, 1)  # queries per minute
                    query_density.iloc[i] = density
        
        features['query_density'] = query_density.fillna(0)
        
        # Burst activity indicator
        density_threshold = query_density.quantile(0.9)
        features['is_burst_activity'] = (query_density > density_threshold).astype(int)
        
        # Regular vs irregular timing patterns
        if len(time_series) > 5:
            # Time between consecutive queries
            time_diffs = time_series.diff().dt.total_seconds().fillna(0)
            
            # Regularity score (inverse of coefficient of variation)
            if time_diffs.std() > 0 and time_diffs.mean() > 0:
                cv = time_diffs.std() / time_diffs.mean()
                features['timing_regularity_score'] = 1 / (1 + cv)
            else:
                features['timing_regularity_score'] = 0
            
            # Gap pattern analysis
            features['time_since_last_query'] = time_diffs
            
            # Long gap indicator (more than 1 hour)
            features['follows_long_gap'] = (time_diffs > 3600).astype(int)
        
        return features
    
    def _extract_anomaly_patterns(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Extract anomaly detection and outlier patterns.
        
        Args:
            data: Input DataFrame
        
        Returns:
            DataFrame with anomaly pattern features
        """
        features = pd.DataFrame(index=data.index)
        thresholds = self.config.get('anomaly_thresholds', {})
        
        # Multi-variate anomaly detection
        anomaly_features = ['credits_used', 'execution_time_ms', 'bytes_scanned']
        available_features = [f for f in anomaly_features if f in data.columns]
        
        if len(available_features) >= 2:
            # Z-score based anomaly detection
            anomaly_scores = []
            
            for feature in available_features:
                values = data[feature]
                if values.std() > 0:
                    z_scores = np.abs((values - values.mean()) / values.std())
                    anomaly_scores.append(z_scores)
                    
                    # Individual feature anomaly
                    z_threshold = thresholds.get('z_score', 3.0)
                    features[f'{feature}_is_anomaly'] = (z_scores > z_threshold).astype(int)
            
            if anomaly_scores:
                # Combined anomaly score
                features['multivariate_anomaly_score'] = np.mean(anomaly_scores, axis=0)
                
                # Overall anomaly indicator
                features['is_anomaly'] = (
                    features['multivariate_anomaly_score'] > thresholds.get('z_score', 3.0)
                ).astype(int)
        
        # IQR-based outlier detection
        for feature in available_features:
            values = data[feature]
            Q1, Q3 = values.quantile([0.25, 0.75])
            IQR = Q3 - Q1
            
            multiplier = thresholds.get('iqr_multiplier', 1.5)
            lower_bound = Q1 - multiplier * IQR
            upper_bound = Q3 + multiplier * IQR
            
            features[f'{feature}_is_outlier_iqr'] = (
                (values < lower_bound) | (values > upper_bound)
            ).astype(int)
        
        # Percentile-based extreme value detection
        percentile_threshold = thresholds.get('percentile', 99)
        for feature in available_features:
            values = data[feature]
            threshold_value = values.quantile(percentile_threshold / 100)
            features[f'{feature}_is_extreme'] = (values > threshold_value).astype(int)
        
        # Contextual anomalies (relative to user/warehouse norms)
        user_col = self.config['user_columns'].get('user_name')
        if user_col in data.columns:
            for feature in ['credits_used', 'execution_time_ms']:
                if feature in data.columns:
                    # User-specific anomalies
                    user_stats = data.groupby(user_col)[feature].agg(['mean', 'std'])
                    user_means = data[user_col].map(user_stats['mean'])
                    user_stds = data[user_col].map(user_stats['std'])
                    
                    user_z_scores = np.where(
                        user_stds > 0,
                        np.abs((data[feature] - user_means) / user_stds),
                        0
                    )
                    
                    features[f'{feature}_user_anomaly'] = (
                        user_z_scores > 2.0
                    ).astype(int)
        
        # Pattern-based anomalies
        if 'query_complexity_score' in features.columns:
            # Complexity anomalies
            complexity = features['query_complexity_score']
            complexity_threshold = complexity.quantile(0.95)
            features['complexity_anomaly'] = (
                complexity > complexity_threshold
            ).astype(int)
        
        return features
    
    def _extract_advanced_patterns(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Extract advanced pattern analysis features.
        
        Args:
            data: Input DataFrame
        
        Returns:
            DataFrame with advanced pattern features
        """
        features = pd.DataFrame(index=data.index)
        
        # Sequence pattern analysis
        if 'execution_time_ms' in data.columns:
            exec_times = data['execution_time_ms']
            
            # Trend patterns in recent queries
            window_size = min(5, len(data))
            
            def trend_pattern(series):
                if len(series) < 3:
                    return 'insufficient_data'
                
                # Calculate simple trend
                x = np.arange(len(series))
                slope = np.polyfit(x, series, 1)[0]
                
                if slope > series.std() * 0.1:
                    return 'increasing'
                elif slope < -series.std() * 0.1:
                    return 'decreasing'
                else:
                    return 'stable'
            
            rolling_trend = exec_times.rolling(window=window_size).apply(
                lambda x: 1 if trend_pattern(x) == 'increasing' else
                         (-1 if trend_pattern(x) == 'decreasing' else 0),
                raw=False
            )
            
            features['recent_execution_trend'] = rolling_trend.fillna(0)
        
        # Correlation pattern analysis
        if len(data.columns) >= 3:
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) >= 2:
                # Calculate rolling correlations between key metrics
                metrics = ['credits_used', 'execution_time_ms', 'bytes_scanned']
                available_metrics = [m for m in metrics if m in numeric_cols]
                
                if len(available_metrics) >= 2:
                    window_size = min(10, len(data))
                    
                    corr = data[available_metrics[0]].rolling(window=window_size).corr(
                        data[available_metrics[1]]
                    )
                    features[f'rolling_correlation_{available_metrics[0]}_{available_metrics[1]}'] = (
                        corr.fillna(0)
                    )
                    
                    # Strong correlation indicator
                    features['has_strong_correlation'] = (np.abs(corr) > 0.7).astype(int)
        
        # Workload signature patterns
        signature_features = ['credits_used', 'execution_time_ms', 'bytes_scanned', 'rows_produced']
        available_sig_features = [f for f in signature_features if f in data.columns]
        
        if len(available_sig_features) >= 2:
            # Create workload signature using normalized features
            signature_data = data[available_sig_features].copy()
            
            # Normalize features
            for col in signature_data.columns:
                values = signature_data[col]
                if values.std() > 0:
                    signature_data[col] = (values - values.mean()) / values.std()
            
            # Calculate workload signature hash (simplified)
            def signature_hash(row):
                # Discretize values and create pattern
                discretized = np.digitize(row.values, bins=[-2, -1, 0, 1, 2])
                return ''.join(map(str, discretized))
            
            features['workload_signature'] = signature_data.apply(signature_hash, axis=1)
            
            # Signature frequency (pattern repetition)
            sig_counts = features['workload_signature'].value_counts()
            features['signature_frequency'] = features['workload_signature'].map(sig_counts)
            
            # Common pattern indicator
            common_threshold = self.config.get('min_pattern_occurrences', 3)
            features['is_common_pattern'] = (
                features['signature_frequency'] >= common_threshold
            ).astype(int)
        
        return features
    
    def _extract_clustering_patterns(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Extract clustering-based pattern features.
        
        Args:
            data: Input DataFrame
        
        Returns:
            DataFrame with clustering pattern features
        """
        features = pd.DataFrame(index=data.index)
        clustering_features = self.config.get('clustering_features', [])
        available_features = [f for f in clustering_features if f in data.columns]
        
        if len(available_features) < 2:
            logger.warning("Insufficient features for clustering analysis")
            return features
        
        # Prepare data for clustering
        cluster_data = data[available_features].copy()
        
        # Handle missing values
        cluster_data = cluster_data.fillna(cluster_data.median())
        
        # Scale features
        try:
            if not self.fitted:
                scaled_data = self.scaler.fit_transform(cluster_data)
                self.fitted = True
            else:
                scaled_data = self.scaler.transform(cluster_data)
            
            # K-means clustering
            n_clusters = self.config['clustering_params'].get('n_clusters', 5)
            random_state = self.config['clustering_params'].get('random_state', 42)
            
            kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
            cluster_labels = kmeans.fit_predict(scaled_data)
            
            features['cluster_id'] = cluster_labels
            
            # Cluster characteristics
            cluster_centers = kmeans.cluster_centers_
            
            # Distance to cluster center
            distances = []
            for i, label in enumerate(cluster_labels):
                center = cluster_centers[label]
                distance = np.linalg.norm(scaled_data[i] - center)
                distances.append(distance)
            
            features['distance_to_cluster_center'] = distances
            
            # Cluster density (inverse of average distance to center)
            cluster_densities = {}
            for cluster_id in range(n_clusters):
                cluster_distances = [d for i, d in enumerate(distances) if cluster_labels[i] == cluster_id]
                if cluster_distances:
                    cluster_densities[cluster_id] = 1 / (1 + np.mean(cluster_distances))
                else:
                    cluster_densities[cluster_id] = 0
            
            features['cluster_density'] = [cluster_densities[label] for label in cluster_labels]
            
            # Outlier score (high distance from center indicates outlier)
            outlier_threshold = np.percentile(distances, 95)
            features['is_cluster_outlier'] = (
                np.array(distances) > outlier_threshold
            ).astype(int)
            
            # Cluster size
            cluster_sizes = pd.Series(cluster_labels).value_counts()
            features['cluster_size'] = pd.Series(cluster_labels).map(cluster_sizes)
            
            # Rare cluster indicator
            small_cluster_threshold = len(data) * 0.05  # Less than 5% of data
            features['is_rare_cluster'] = (
                features['cluster_size'] < small_cluster_threshold
            ).astype(int)
            
        except Exception as e:
            logger.warning(f"Error in clustering analysis: {str(e)}")
        
        return features
    
    def configure_for_snowflake_data(self):
        """
        Configure the generator for typical Snowflake analytics data.
        """
        snowflake_config = {
            'clustering_features': [
                'CREDITS_USED', 'EXECUTION_TIME_MS', 'BYTES_SCANNED',
                'ROWS_PRODUCED', 'PARTITIONS_SCANNED', 'QUEUE_TIME_MS'
            ],
            'user_columns': {
                'user_name': 'USER_NAME',
                'role_name': 'ROLE_NAME',
                'session_id': 'SESSION_ID'
            },
            'query_columns': {
                'query_type': 'QUERY_TYPE',
                'database_name': 'DATABASE_NAME',
                'warehouse_name': 'WAREHOUSE_NAME'
            },
            'anomaly_thresholds': {
                'z_score': 2.5,  # More lenient for analytics data
                'iqr_multiplier': 2.0,
                'percentile': 95
            },
            'clustering_params': {
                'n_clusters': 6,  # More clusters for diverse workloads
                'random_state': 42
            }
        }
        
        self.config.update(snowflake_config)
        logger.info("Configured PatternFeatureGenerator for Snowflake analytics data")
    
    def get_feature_descriptions(self) -> Dict[str, str]:
        """
        Get descriptions of all generated pattern features.
        
        Returns:
            Dictionary mapping feature names to descriptions
        """
        descriptions = {
            # User behavior patterns
            'user_activity_score': 'User activity level based on query frequency',
            'user_behavior_consistency': 'Consistency of user resource usage patterns',
            'user_preferred_hour': 'User\'s most common hour of activity',
            'user_business_hours_preference': 'Preference for business hours activity',
            'user_query_type_diversity': 'Diversity of query types used by user',
            
            # Query complexity patterns
            'query_complexity_score': 'Multi-dimensional query complexity score',
            'complexity_category': 'Query complexity category (simple to very_complex)',
            'resource_intensity': 'Resource consumption intensity per unit time',
            'query_selectivity': 'Data selectivity of the query',
            
            # Resource usage patterns
            'warehouse_preference_score': 'Warehouse usage preference score',
            'potential_waste_indicator': 'Indicator of potentially wasteful resource usage',
            'resource_efficiency_score': 'Resource utilization efficiency score',
            'experiences_queuing': 'Whether query experienced queuing delays',
            
            # Temporal patterns
            'is_peak_hour': 'Whether query occurred during peak hours',
            'is_burst_activity': 'Whether query is part of burst activity',
            'timing_regularity_score': 'Regularity of query timing patterns',
            'follows_long_gap': 'Whether query follows a long gap in activity',
            
            # Anomaly patterns
            'multivariate_anomaly_score': 'Multi-dimensional anomaly detection score',
            'is_anomaly': 'Whether query is detected as anomalous',
            'credits_used_is_outlier_iqr': 'Whether credits usage is an IQR outlier',
            'complexity_anomaly': 'Whether query complexity is anomalous',
            
            # Clustering patterns
            'cluster_id': 'Cluster assignment based on usage patterns',
            'distance_to_cluster_center': 'Distance to assigned cluster center',
            'is_cluster_outlier': 'Whether query is an outlier within its cluster',
            'is_rare_cluster': 'Whether query belongs to a rare cluster pattern'
        }
        
        return descriptions
