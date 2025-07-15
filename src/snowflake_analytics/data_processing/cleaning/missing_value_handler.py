"""
Missing Value Handler

This module provides comprehensive missing value detection and handling strategies
for Snowflake analytics data with multiple imputation and treatment methods.
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Any
from datetime import datetime, timedelta
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

logger = logging.getLogger(__name__)


class MissingValueHandler:
    """
    Handler for detecting and treating missing values in data.
    
    Provides multiple strategies including removal, imputation, and interpolation
    with support for different data types and business logic.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the MissingValueHandler with configuration.
        
        Args:
            config: Configuration dictionary with missing value handling parameters
        """
        self.config = config or {}
        
        # Default configuration
        self.default_config = {
            'strategy': 'auto',  # 'remove', 'impute', 'interpolate', 'auto'
            'threshold': 0.5,  # Remove rows/columns with >50% missing values
            'numeric_strategy': 'median',  # 'mean', 'median', 'mode', 'knn', 'iterative'
            'categorical_strategy': 'mode',  # 'mode', 'constant', 'unknown'
            'datetime_strategy': 'forward_fill',  # 'forward_fill', 'backward_fill', 'interpolate'
            'constant_value': 'UNKNOWN',  # Value for constant imputation
            'remove_threshold': 0.8,  # Remove columns with >80% missing values
            'interpolation_method': 'linear',  # 'linear', 'polynomial', 'spline'
            'knn_neighbors': 5  # Number of neighbors for KNN imputation
        }
        
        # Merge with provided config
        self.config = {**self.default_config, **self.config}
        
        # Statistics tracking
        self.missing_stats = {
            'missing_values_before': 0,
            'missing_values_after': 0,
            'columns_removed': 0,
            'rows_removed': 0,
            'values_imputed': 0,
            'strategy_used': {},
            'processing_time': 0.0
        }
    
    def handle_missing_values(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values in the DataFrame based on configured strategy.
        
        Args:
            data: Input DataFrame to process
        
        Returns:
            DataFrame with missing values handled
        """
        start_time = datetime.now()
        
        logger.info(f"Starting missing value handling with strategy: {self.config['strategy']}")
        
        # Calculate initial missing value statistics
        self.missing_stats['missing_values_before'] = data.isnull().sum().sum()
        initial_shape = data.shape
        
        try:
            if self.config['strategy'] == 'remove':
                cleaned_data = self._remove_missing_values(data)
            elif self.config['strategy'] == 'impute':
                cleaned_data = self._impute_missing_values(data)
            elif self.config['strategy'] == 'interpolate':
                cleaned_data = self._interpolate_missing_values(data)
            elif self.config['strategy'] == 'auto':
                cleaned_data = self._auto_handle_missing_values(data)
            else:
                logger.warning(f"Unknown strategy {self.config['strategy']}, using auto")
                cleaned_data = self._auto_handle_missing_values(data)
            
            # Update statistics
            self.missing_stats['missing_values_after'] = cleaned_data.isnull().sum().sum()
            self.missing_stats['columns_removed'] = initial_shape[1] - cleaned_data.shape[1]
            self.missing_stats['rows_removed'] = initial_shape[0] - cleaned_data.shape[0]
            self.missing_stats['values_imputed'] = (
                self.missing_stats['missing_values_before'] - 
                self.missing_stats['missing_values_after'] - 
                (self.missing_stats['rows_removed'] * initial_shape[1])
            )
            self.missing_stats['processing_time'] = (datetime.now() - start_time).total_seconds()
            
            logger.info(f"Missing value handling completed in {self.missing_stats['processing_time']:.2f} seconds")
            logger.info(f"Missing values: {self.missing_stats['missing_values_before']} → "
                       f"{self.missing_stats['missing_values_after']}")
            
            return cleaned_data
            
        except Exception as e:
            logger.error(f"Error handling missing values: {str(e)}")
            raise
    
    def _remove_missing_values(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Remove rows and columns with excessive missing values.
        
        Args:
            data: Input DataFrame
        
        Returns:
            DataFrame with missing values removed
        """
        threshold = self.config.get('threshold', 0.5)
        remove_threshold = self.config.get('remove_threshold', 0.8)
        
        cleaned_data = data.copy()
        
        # Remove columns with too many missing values
        missing_col_ratio = cleaned_data.isnull().sum() / len(cleaned_data)
        cols_to_remove = missing_col_ratio[missing_col_ratio > remove_threshold].index
        
        if len(cols_to_remove) > 0:
            logger.info(f"Removing {len(cols_to_remove)} columns with >{remove_threshold*100}% missing values")
            cleaned_data = cleaned_data.drop(columns=cols_to_remove)
        
        # Remove rows with too many missing values
        missing_row_ratio = cleaned_data.isnull().sum(axis=1) / len(cleaned_data.columns)
        rows_to_remove = missing_row_ratio > threshold
        
        if rows_to_remove.sum() > 0:
            logger.info(f"Removing {rows_to_remove.sum()} rows with >{threshold*100}% missing values")
            cleaned_data = cleaned_data[~rows_to_remove]
        
        return cleaned_data
    
    def _impute_missing_values(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Impute missing values using various strategies based on data type.
        
        Args:
            data: Input DataFrame
        
        Returns:
            DataFrame with missing values imputed
        """
        cleaned_data = data.copy()
        
        # Separate columns by data type
        numeric_cols = cleaned_data.select_dtypes(include=[np.number]).columns
        categorical_cols = cleaned_data.select_dtypes(include=['object', 'category']).columns
        datetime_cols = cleaned_data.select_dtypes(include=['datetime64']).columns
        
        # Impute numeric columns
        if len(numeric_cols) > 0:
            cleaned_data = self._impute_numeric_columns(cleaned_data, numeric_cols)
        
        # Impute categorical columns
        if len(categorical_cols) > 0:
            cleaned_data = self._impute_categorical_columns(cleaned_data, categorical_cols)
        
        # Impute datetime columns
        if len(datetime_cols) > 0:
            cleaned_data = self._impute_datetime_columns(cleaned_data, datetime_cols)
        
        return cleaned_data
    
    def _impute_numeric_columns(self, data: pd.DataFrame, numeric_cols: List[str]) -> pd.DataFrame:
        """
        Impute missing values in numeric columns.
        
        Args:
            data: Input DataFrame
            numeric_cols: List of numeric column names
        
        Returns:
            DataFrame with numeric missing values imputed
        """
        strategy = self.config.get('numeric_strategy', 'median')
        
        for col in numeric_cols:
            if data[col].isnull().sum() == 0:
                continue
            
            if strategy in ['mean', 'median', 'most_frequent']:
                imputer = SimpleImputer(strategy=strategy)
                data[col] = imputer.fit_transform(data[[col]]).flatten()
                
            elif strategy == 'knn':
                # KNN imputation for numeric columns
                knn_neighbors = self.config.get('knn_neighbors', 5)
                imputer = KNNImputer(n_neighbors=knn_neighbors)
                
                # Use only numeric columns for KNN
                numeric_data = data[numeric_cols]
                imputed_data = imputer.fit_transform(numeric_data)
                data[numeric_cols] = imputed_data
                break  # KNN handles all numeric columns at once
                
            elif strategy == 'iterative':
                # Iterative imputation
                imputer = IterativeImputer(random_state=42)
                numeric_data = data[numeric_cols]
                imputed_data = imputer.fit_transform(numeric_data)
                data[numeric_cols] = imputed_data
                break  # Iterative handles all numeric columns at once
            
            self.missing_stats['strategy_used'][col] = strategy
        
        return data
    
    def _impute_categorical_columns(self, data: pd.DataFrame, categorical_cols: List[str]) -> pd.DataFrame:
        """
        Impute missing values in categorical columns.
        
        Args:
            data: Input DataFrame
            categorical_cols: List of categorical column names
        
        Returns:
            DataFrame with categorical missing values imputed
        """
        strategy = self.config.get('categorical_strategy', 'mode')
        constant_value = self.config.get('constant_value', 'UNKNOWN')
        
        for col in categorical_cols:
            if data[col].isnull().sum() == 0:
                continue
            
            if strategy == 'mode':
                # Use most frequent value
                mode_value = data[col].mode()
                if len(mode_value) > 0:
                    data[col] = data[col].fillna(mode_value[0])
                else:
                    data[col] = data[col].fillna(constant_value)
                    
            elif strategy == 'constant':
                data[col] = data[col].fillna(constant_value)
                
            elif strategy == 'unknown':
                data[col] = data[col].fillna('UNKNOWN')
            
            self.missing_stats['strategy_used'][col] = strategy
        
        return data
    
    def _impute_datetime_columns(self, data: pd.DataFrame, datetime_cols: List[str]) -> pd.DataFrame:
        """
        Impute missing values in datetime columns.
        
        Args:
            data: Input DataFrame
            datetime_cols: List of datetime column names
        
        Returns:
            DataFrame with datetime missing values imputed
        """
        strategy = self.config.get('datetime_strategy', 'forward_fill')
        
        for col in datetime_cols:
            if data[col].isnull().sum() == 0:
                continue
            
            if strategy == 'forward_fill':
                data[col] = data[col].fillna(method='ffill')
                
            elif strategy == 'backward_fill':
                data[col] = data[col].fillna(method='bfill')
                
            elif strategy == 'interpolate':
                # Sort by datetime column for interpolation
                data = data.sort_values(col)
                data[col] = data[col].interpolate(method='time')
            
            self.missing_stats['strategy_used'][col] = strategy
        
        return data
    
    def _interpolate_missing_values(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Interpolate missing values using various interpolation methods.
        
        Args:
            data: Input DataFrame
        
        Returns:
            DataFrame with missing values interpolated
        """
        method = self.config.get('interpolation_method', 'linear')
        cleaned_data = data.copy()
        
        # Only interpolate numeric columns
        numeric_cols = cleaned_data.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if cleaned_data[col].isnull().sum() > 0:
                cleaned_data[col] = cleaned_data[col].interpolate(method=method)
        
        return cleaned_data
    
    def _auto_handle_missing_values(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Automatically handle missing values based on data characteristics.
        
        Args:
            data: Input DataFrame
        
        Returns:
            DataFrame with missing values handled automatically
        """
        cleaned_data = data.copy()
        
        # First, remove columns with excessive missing values
        remove_threshold = self.config.get('remove_threshold', 0.8)
        missing_col_ratio = cleaned_data.isnull().sum() / len(cleaned_data)
        cols_to_remove = missing_col_ratio[missing_col_ratio > remove_threshold].index
        
        if len(cols_to_remove) > 0:
            logger.info(f"Auto-removing {len(cols_to_remove)} columns with >{remove_threshold*100}% missing values")
            cleaned_data = cleaned_data.drop(columns=cols_to_remove)
        
        # Then impute remaining missing values
        cleaned_data = self._impute_missing_values(cleaned_data)
        
        return cleaned_data
    
    def analyze_missing_patterns(self, data: pd.DataFrame) -> Dict:
        """
        Analyze missing value patterns in the data.
        
        Args:
            data: Input DataFrame to analyze
        
        Returns:
            Dictionary with missing value analysis
        """
        missing_info = {}
        
        # Overall missing statistics
        total_values = data.size
        missing_values = data.isnull().sum().sum()
        
        missing_info['overall'] = {
            'total_values': total_values,
            'missing_values': missing_values,
            'missing_percentage': (missing_values / total_values * 100) if total_values > 0 else 0
        }
        
        # Per-column missing statistics
        column_missing = {}
        for col in data.columns:
            col_missing = data[col].isnull().sum()
            col_total = len(data)
            
            column_missing[col] = {
                'missing_count': col_missing,
                'missing_percentage': (col_missing / col_total * 100) if col_total > 0 else 0,
                'data_type': str(data[col].dtype)
            }
        
        missing_info['by_column'] = column_missing
        
        # Missing value patterns
        missing_patterns = data.isnull().value_counts()
        missing_info['patterns'] = missing_patterns.to_dict()
        
        return missing_info
    
    def get_missing_value_summary(self) -> Dict:
        """
        Get summary of the last missing value handling operation.
        
        Returns:
            Dictionary with missing value handling statistics
        """
        return self.missing_stats.copy()
    
    def configure_for_snowflake_data(self):
        """
        Configure the handler for typical Snowflake analytics data.
        """
        # Snowflake-specific configuration
        snowflake_config = {
            'strategy': 'auto',
            'numeric_strategy': 'median',  # More robust for skewed data
            'categorical_strategy': 'unknown',  # Clear indicator of missing data
            'datetime_strategy': 'forward_fill',  # Reasonable for time series
            'remove_threshold': 0.9,  # More lenient for analytics data
            'constant_value': 'UNKNOWN'
        }
        
        self.config.update(snowflake_config)
        logger.info("Configured for Snowflake analytics data")
