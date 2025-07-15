"""
Main Feature Engineering Pipeline

This module provides the main FeaturePipeline class that orchestrates all feature engineering
operations including time-based features, usage patterns, cost metrics, and rolling statistics.
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime
import json

from .time_features import TimeFeatureGenerator
from .usage_features import UsageFeatureGenerator
from .cost_features import CostFeatureGenerator
from .rolling_features import RollingFeatureGenerator
from .pattern_features import PatternFeatureGenerator

logger = logging.getLogger(__name__)


class FeaturePipeline:
    """
    Main feature engineering pipeline that orchestrates all feature generation operations.
    
    This class provides a unified interface for creating ML-ready features from
    raw Snowflake analytics data, including time-based features, usage patterns,
    cost metrics, and statistical features.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the FeaturePipeline with optional configuration.
        
        Args:
            config: Optional configuration dictionary with feature engineering parameters
        """
        self.config = config or {}
        
        # Default configuration
        self.default_config = {
            'features_to_generate': [
                'time_features',
                'usage_features', 
                'cost_features',
                'rolling_features',
                'pattern_features'
            ],
            'time_column': 'START_TIME',  # Primary time column
            'target_columns': {
                'cost': 'CREDITS_USED',
                'usage': 'TOTAL_ELAPSED_TIME',
                'performance': 'EXECUTION_TIME_MS'
            },
            'rolling_windows': [3, 7, 14, 30],  # Rolling window sizes in days
            'aggregation_levels': ['hourly', 'daily', 'weekly'],
            'enable_advanced_features': True,
            'feature_selection': 'auto',  # 'auto', 'manual', 'all'
            'max_features': 100,  # Maximum number of features to generate
            'remove_highly_correlated': True,
            'correlation_threshold': 0.95
        }
        
        # Merge with provided config
        self.config = {**self.default_config, **self.config}
        
        # Initialize feature generators
        self.time_generator = TimeFeatureGenerator(self.config.get('time_features', {}))
        self.usage_generator = UsageFeatureGenerator(self.config.get('usage_features', {}))
        self.cost_generator = CostFeatureGenerator(self.config.get('cost_features', {}))
        self.rolling_generator = RollingFeatureGenerator(self.config.get('rolling_features', {}))
        self.pattern_generator = PatternFeatureGenerator(self.config.get('pattern_features', {}))
        
        # Feature engineering statistics
        self.feature_stats = {
            'total_features_generated': 0,
            'features_by_type': {},
            'original_columns': 0,
            'final_columns': 0,
            'features_removed': 0,
            'processing_time': 0.0,
            'memory_usage_mb': 0.0
        }
    
    def generate_features(self, 
                         data: pd.DataFrame, 
                         feature_types: Optional[List[str]] = None) -> Tuple[pd.DataFrame, Dict]:
        """
        Generate comprehensive features from the input DataFrame.
        
        Args:
            data: Input DataFrame containing raw Snowflake data
            feature_types: Optional list of feature types to generate.
                          If None, generates all configured feature types.
        
        Returns:
            Tuple of (feature_engineered_data, feature_statistics)
        """
        start_time = datetime.now()
        
        logger.info(f"Starting feature engineering on {len(data)} rows, {len(data.columns)} columns")
        
        # Initialize statistics
        self.feature_stats['original_columns'] = len(data.columns)
        initial_memory = data.memory_usage(deep=True).sum()
        
        if feature_types is None:
            feature_types = self.config.get('features_to_generate', [])
        
        try:
            feature_data = data.copy()
            features_generated = {}
            
            # Validate required columns
            if not self._validate_required_columns(feature_data):
                logger.warning("Some required columns missing, proceeding with available features")
            
            # Generate time-based features
            if 'time_features' in feature_types:
                logger.info("Generating time-based features...")
                time_features = self.time_generator.generate_time_features(feature_data)
                feature_data = pd.concat([feature_data, time_features], axis=1)
                features_generated['time_features'] = len(time_features.columns)
                logger.info(f"Generated {len(time_features.columns)} time-based features")
            
            # Generate usage pattern features
            if 'usage_features' in feature_types:
                logger.info("Generating usage pattern features...")
                usage_features = self.usage_generator.generate_usage_features(feature_data)
                feature_data = pd.concat([feature_data, usage_features], axis=1)
                features_generated['usage_features'] = len(usage_features.columns)
                logger.info(f"Generated {len(usage_features.columns)} usage pattern features")
            
            # Generate cost-related features
            if 'cost_features' in feature_types:
                logger.info("Generating cost-related features...")
                cost_features = self.cost_generator.generate_cost_features(feature_data)
                feature_data = pd.concat([feature_data, cost_features], axis=1)
                features_generated['cost_features'] = len(cost_features.columns)
                logger.info(f"Generated {len(cost_features.columns)} cost-related features")
            
            # Generate rolling statistics features
            if 'rolling_features' in feature_types:
                logger.info("Generating rolling statistics features...")
                rolling_features = self.rolling_generator.generate_rolling_features(feature_data)
                feature_data = pd.concat([feature_data, rolling_features], axis=1)
                features_generated['rolling_features'] = len(rolling_features.columns)
                logger.info(f"Generated {len(rolling_features.columns)} rolling statistics features")
            
            # Generate pattern detection features
            if 'pattern_features' in feature_types:
                logger.info("Generating pattern detection features...")
                pattern_features = self.pattern_generator.generate_pattern_features(feature_data)
                feature_data = pd.concat([feature_data, pattern_features], axis=1)
                features_generated['pattern_features'] = len(pattern_features.columns)
                logger.info(f"Generated {len(pattern_features.columns)} pattern detection features")
            
            # Post-processing: remove highly correlated features
            if self.config.get('remove_highly_correlated', True):
                feature_data = self._remove_highly_correlated_features(feature_data)
            
            # Feature selection if enabled
            if self.config.get('feature_selection') != 'all':
                feature_data = self._select_features(feature_data)
            
            # Update final statistics
            final_memory = feature_data.memory_usage(deep=True).sum()
            self.feature_stats.update({
                'total_features_generated': sum(features_generated.values()),
                'features_by_type': features_generated,
                'final_columns': len(feature_data.columns),
                'processing_time': (datetime.now() - start_time).total_seconds(),
                'memory_usage_mb': final_memory / 1024 / 1024
            })
            
            logger.info(f"Feature engineering completed in {self.feature_stats['processing_time']:.2f} seconds")
            logger.info(f"Final dataset: {len(feature_data)} rows, {len(feature_data.columns)} columns")
            logger.info(f"Memory usage: {self.feature_stats['memory_usage_mb']:.2f} MB")
            
            return feature_data, self.feature_stats.copy()
            
        except Exception as e:
            logger.error(f"Error during feature engineering: {str(e)}")
            raise
    
    def _validate_required_columns(self, data: pd.DataFrame) -> bool:
        """
        Validate that required columns are present in the data.
        
        Args:
            data: Input DataFrame
        
        Returns:
            True if all required columns are present
        """
        required_columns = [
            self.config.get('time_column', 'START_TIME')
        ]
        
        # Add target columns
        target_columns = self.config.get('target_columns', {})
        required_columns.extend(target_columns.values())
        
        missing_columns = [col for col in required_columns if col not in data.columns]
        
        if missing_columns:
            logger.warning(f"Missing required columns: {missing_columns}")
            return False
        
        return True
    
    def _remove_highly_correlated_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Remove highly correlated features to reduce redundancy.
        
        Args:
            data: Input DataFrame with features
        
        Returns:
            DataFrame with highly correlated features removed
        """
        threshold = self.config.get('correlation_threshold', 0.95)
        
        # Only consider numeric columns for correlation
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        
        if len(numeric_columns) < 2:
            return data
        
        logger.info("Removing highly correlated features...")
        
        # Calculate correlation matrix
        correlation_matrix = data[numeric_columns].corr().abs()
        
        # Find highly correlated pairs
        high_corr_pairs = []
        for i in range(len(correlation_matrix.columns)):
            for j in range(i + 1, len(correlation_matrix.columns)):
                if correlation_matrix.iloc[i, j] > threshold:
                    col1 = correlation_matrix.columns[i]
                    col2 = correlation_matrix.columns[j]
                    high_corr_pairs.append((col1, col2, correlation_matrix.iloc[i, j]))
        
        # Remove features with high correlation
        columns_to_remove = set()
        for col1, col2, corr_value in high_corr_pairs:
            # Remove the column with more missing values, or the second one if equal
            if data[col1].isnull().sum() > data[col2].isnull().sum():
                columns_to_remove.add(col1)
            else:
                columns_to_remove.add(col2)
        
        if columns_to_remove:
            logger.info(f"Removing {len(columns_to_remove)} highly correlated features")
            data = data.drop(columns=list(columns_to_remove))
            self.feature_stats['features_removed'] = len(columns_to_remove)
        
        return data
    
    def _select_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Select the most important features based on the configured method.
        
        Args:
            data: Input DataFrame with features
        
        Returns:
            DataFrame with selected features
        """
        selection_method = self.config.get('feature_selection', 'auto')
        max_features = self.config.get('max_features', 100)
        
        if len(data.columns) <= max_features:
            return data
        
        logger.info(f"Selecting top {max_features} features using {selection_method} method")
        
        if selection_method == 'auto':
            # Use variance-based selection for automatic feature selection
            return self._select_features_by_variance(data, max_features)
        elif selection_method == 'manual':
            # Manual selection based on predefined feature importance
            return self._select_features_manually(data, max_features)
        
        return data
    
    def _select_features_by_variance(self, data: pd.DataFrame, max_features: int) -> pd.DataFrame:
        """
        Select features based on variance (remove low-variance features).
        
        Args:
            data: Input DataFrame
            max_features: Maximum number of features to keep
        
        Returns:
            DataFrame with selected features
        """
        # Calculate variance for numeric columns
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        non_numeric_columns = data.select_dtypes(exclude=[np.number]).columns
        
        if len(numeric_columns) == 0:
            return data
        
        # Calculate variance and sort
        variances = data[numeric_columns].var().sort_values(ascending=False)
        
        # Select top features by variance
        top_numeric_features = variances.head(max_features - len(non_numeric_columns)).index
        
        # Combine with non-numeric columns
        selected_columns = list(non_numeric_columns) + list(top_numeric_features)
        
        return data[selected_columns]
    
    def _select_features_manually(self, data: pd.DataFrame, max_features: int) -> pd.DataFrame:
        """
        Manually select features based on domain knowledge.
        
        Args:
            data: Input DataFrame
            max_features: Maximum number of features to keep
        
        Returns:
            DataFrame with selected features
        """
        # Define feature priority for Snowflake analytics data
        priority_features = [
            # Time features
            'hour_of_day', 'day_of_week', 'month_of_year', 'is_weekend', 'is_business_hours',
            # Cost features  
            'credits_used', 'cost_per_query', 'credits_per_gb_scanned',
            # Usage features
            'execution_time_ms', 'queue_time_ms', 'bytes_scanned',
            # Rolling features
            'credits_used_rolling_7d', 'execution_time_rolling_3d',
            # Pattern features
            'query_complexity_score', 'user_activity_score'
        ]
        
        # Find existing priority features
        existing_priority = [col for col in priority_features if col in data.columns]
        
        # Add remaining columns up to max_features
        remaining_columns = [col for col in data.columns if col not in existing_priority]
        additional_needed = max_features - len(existing_priority)
        
        if additional_needed > 0:
            selected_columns = existing_priority + remaining_columns[:additional_needed]
        else:
            selected_columns = existing_priority[:max_features]
        
        return data[selected_columns]
    
    def generate_feature_summary(self, data: pd.DataFrame) -> Dict:
        """
        Generate a summary of features in the dataset.
        
        Args:
            data: DataFrame with features
        
        Returns:
            Dictionary with feature summary
        """
        summary = {
            'total_features': len(data.columns),
            'feature_types': {},
            'missing_values': {},
            'data_types': {},
            'memory_usage': {}
        }
        
        # Categorize features by type
        feature_categories = {
            'time': ['hour', 'day', 'week', 'month', 'quarter', 'season'],
            'cost': ['cost', 'credit', 'dollar', 'price'],
            'usage': ['usage', 'execution', 'queue', 'elapsed'],
            'rolling': ['rolling', 'moving', 'avg', 'mean'],
            'pattern': ['score', 'ratio', 'rate', 'complexity']
        }
        
        for category, keywords in feature_categories.items():
            matching_features = []
            for col in data.columns:
                if any(keyword in col.lower() for keyword in keywords):
                    matching_features.append(col)
            summary['feature_types'][category] = len(matching_features)
        
        # Calculate missing values and data types
        for col in data.columns:
            summary['missing_values'][col] = data[col].isnull().sum()
            summary['data_types'][col] = str(data[col].dtype)
            summary['memory_usage'][col] = data[col].memory_usage(deep=True)
        
        return summary
    
    def get_feature_importance(self, data: pd.DataFrame, target_column: str) -> Dict:
        """
        Calculate feature importance using various methods.
        
        Args:
            data: DataFrame with features
            target_column: Name of target column
        
        Returns:
            Dictionary with feature importance scores
        """
        if target_column not in data.columns:
            logger.warning(f"Target column {target_column} not found in data")
            return {}
        
        # Select numeric features only
        numeric_features = data.select_dtypes(include=[np.number]).columns
        numeric_features = [col for col in numeric_features if col != target_column]
        
        if len(numeric_features) == 0:
            return {}
        
        importance_scores = {}
        
        try:
            # Correlation-based importance
            correlations = data[numeric_features].corrwith(data[target_column]).abs()
            importance_scores['correlation'] = correlations.to_dict()
            
            # Mutual information (simplified)
            # For a full implementation, you'd use sklearn.feature_selection.mutual_info_regression
            
        except Exception as e:
            logger.warning(f"Error calculating feature importance: {str(e)}")
        
        return importance_scores
    
    def save_feature_config(self, filepath: str):
        """
        Save the current feature engineering configuration to a file.
        
        Args:
            filepath: Path to save the configuration
        """
        config_to_save = {
            'pipeline_config': self.config,
            'feature_stats': self.feature_stats
        }
        
        try:
            with open(filepath, 'w') as f:
                json.dump(config_to_save, f, indent=2, default=str)
            logger.info(f"Feature configuration saved to {filepath}")
        except Exception as e:
            logger.error(f"Error saving feature configuration: {str(e)}")
    
    def load_feature_config(self, filepath: str):
        """
        Load feature engineering configuration from a file.
        
        Args:
            filepath: Path to load the configuration from
        """
        try:
            with open(filepath, 'r') as f:
                loaded_config = json.load(f)
            
            self.config.update(loaded_config.get('pipeline_config', {}))
            logger.info(f"Feature configuration loaded from {filepath}")
        except Exception as e:
            logger.error(f"Error loading feature configuration: {str(e)}")
    
    def get_feature_engineering_summary(self) -> Dict:
        """
        Get summary of the last feature engineering operation.
        
        Returns:
            Dictionary with feature engineering statistics
        """
        return self.feature_stats.copy()
    
    def configure_for_snowflake_data(self):
        """
        Configure the pipeline for typical Snowflake analytics data.
        """
        # Snowflake-specific configuration
        snowflake_config = {
            'time_column': 'START_TIME',
            'target_columns': {
                'cost': 'CREDITS_USED',
                'usage': 'TOTAL_ELAPSED_TIME', 
                'performance': 'EXECUTION_TIME_MS',
                'bytes': 'BYTES_SCANNED'
            },
            'rolling_windows': [1, 3, 7, 14, 30],  # Days
            'aggregation_levels': ['hourly', 'daily', 'weekly'],
            'enable_advanced_features': True,
            'max_features': 150,  # More features for analytics
            'correlation_threshold': 0.90  # Less strict for analytics
        }
        
        self.config.update(snowflake_config)
        
        # Configure sub-generators for Snowflake data
        self.time_generator.configure_for_snowflake_data()
        self.usage_generator.configure_for_snowflake_data()
        self.cost_generator.configure_for_snowflake_data()
        self.rolling_generator.configure_for_snowflake_data()
        self.pattern_generator.configure_for_snowflake_data()
        
        logger.info("Configured feature pipeline for Snowflake analytics data")
