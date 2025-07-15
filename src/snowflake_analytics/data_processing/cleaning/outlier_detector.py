"""
Outlier Detection and Handling

This module provides comprehensive outlier detection and handling capabilities
using statistical methods, machine learning approaches, and domain-specific rules.
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class OutlierDetector:
    """
    Handler for detecting and treating outliers in data.
    
    Provides multiple detection methods including statistical approaches,
    machine learning methods, and domain-specific rules.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the OutlierDetector with configuration.
        
        Args:
            config: Configuration dictionary with outlier detection parameters
        """
        self.config = config or {}
        
        # Default configuration
        self.default_config = {
            'method': 'iqr',  # 'iqr', 'zscore', 'modified_zscore', 'isolation_forest', 'percentile'
            'threshold': 1.5,  # IQR multiplier or z-score threshold
            'action': 'cap',  # 'remove', 'cap', 'flag', 'transform'
            'percentile_bounds': (1, 99),  # Lower and upper percentile bounds
            'isolation_forest_contamination': 0.1,  # Expected outlier proportion
            'zscore_threshold': 3.0,  # Z-score threshold for outlier detection
            'modified_zscore_threshold': 3.5,  # Modified z-score threshold
            'columns_to_check': None,  # Specific columns to check, None for all numeric
            'preserve_extreme_values': False  # Keep extreme values that might be valid
        }
        
        # Merge with provided config
        self.config = {**self.default_config, **self.config}
        
        # Statistics tracking
        self.outlier_stats = {
            'outliers_detected': 0,
            'outliers_removed': 0,
            'outliers_capped': 0,
            'outliers_flagged': 0,
            'method_used': '',
            'columns_processed': [],
            'processing_time': 0.0
        }
    
    def detect_and_handle_outliers(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
        """
        Detect and handle outliers in the DataFrame.
        
        Args:
            data: Input DataFrame to process
        
        Returns:
            Tuple of (processed_data, outlier_count)
        """
        start_time = datetime.now()
        
        logger.info(f"Starting outlier detection with method: {self.config['method']}")
        
        # Get columns to process
        columns_to_check = self._get_columns_to_check(data)
        
        if not columns_to_check:
            logger.info("No numeric columns found for outlier detection")
            return data, 0
        
        try:
            processed_data = data.copy()
            total_outliers = 0
            
            for col in columns_to_check:
                outliers_in_col = self._detect_outliers_in_column(processed_data, col)
                total_outliers += len(outliers_in_col)
                
                if len(outliers_in_col) > 0:
                    processed_data = self._handle_outliers_in_column(
                        processed_data, col, outliers_in_col
                    )
            
            # Update statistics
            self.outlier_stats['outliers_detected'] = total_outliers
            self.outlier_stats['method_used'] = self.config['method']
            self.outlier_stats['columns_processed'] = columns_to_check
            self.outlier_stats['processing_time'] = (datetime.now() - start_time).total_seconds()
            
            logger.info(f"Detected and handled {total_outliers} outliers across "
                       f"{len(columns_to_check)} columns in "
                       f"{self.outlier_stats['processing_time']:.2f} seconds")
            
            return processed_data, total_outliers
            
        except Exception as e:
            logger.error(f"Error detecting outliers: {str(e)}")
            raise
    
    def _get_columns_to_check(self, data: pd.DataFrame) -> List[str]:
        """
        Get the list of columns to check for outliers.
        
        Args:
            data: Input DataFrame
        
        Returns:
            List of column names to check
        """
        if self.config.get('columns_to_check'):
            # Use specified columns
            specified_cols = self.config['columns_to_check']
            existing_cols = [col for col in specified_cols if col in data.columns]
            return existing_cols
        else:
            # Use all numeric columns
            return data.select_dtypes(include=[np.number]).columns.tolist()
    
    def _detect_outliers_in_column(self, data: pd.DataFrame, column: str) -> pd.Index:
        """
        Detect outliers in a specific column using the configured method.
        
        Args:
            data: Input DataFrame
            column: Column name to analyze
        
        Returns:
            Index of outlier rows
        """
        method = self.config['method']
        
        if method == 'iqr':
            return self._detect_outliers_iqr(data, column)
        elif method == 'zscore':
            return self._detect_outliers_zscore(data, column)
        elif method == 'modified_zscore':
            return self._detect_outliers_modified_zscore(data, column)
        elif method == 'isolation_forest':
            return self._detect_outliers_isolation_forest(data, column)
        elif method == 'percentile':
            return self._detect_outliers_percentile(data, column)
        else:
            logger.warning(f"Unknown method {method}, using IQR")
            return self._detect_outliers_iqr(data, column)
    
    def _detect_outliers_iqr(self, data: pd.DataFrame, column: str) -> pd.Index:
        """
        Detect outliers using the IQR (Interquartile Range) method.
        
        Args:
            data: Input DataFrame
            column: Column name to analyze
        
        Returns:
            Index of outlier rows
        """
        threshold = self.config.get('threshold', 1.5)
        
        Q1 = data[column].quantile(0.25)
        Q3 = data[column].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        
        outlier_mask = (data[column] < lower_bound) | (data[column] > upper_bound)
        return data[outlier_mask].index
    
    def _detect_outliers_zscore(self, data: pd.DataFrame, column: str) -> pd.Index:
        """
        Detect outliers using the Z-score method.
        
        Args:
            data: Input DataFrame
            column: Column name to analyze
        
        Returns:
            Index of outlier rows
        """
        threshold = self.config.get('zscore_threshold', 3.0)
        
        z_scores = np.abs(stats.zscore(data[column].dropna()))
        outlier_mask = z_scores > threshold
        
        # Map back to original index
        valid_indices = data[column].dropna().index
        outlier_indices = valid_indices[outlier_mask]
        
        return outlier_indices
    
    def _detect_outliers_modified_zscore(self, data: pd.DataFrame, column: str) -> pd.Index:
        """
        Detect outliers using the Modified Z-score method (using median).
        
        Args:
            data: Input DataFrame
            column: Column name to analyze
        
        Returns:
            Index of outlier rows
        """
        threshold = self.config.get('modified_zscore_threshold', 3.5)
        
        median = data[column].median()
        mad = np.median(np.abs(data[column] - median))
        
        # Avoid division by zero
        if mad == 0:
            mad = np.mean(np.abs(data[column] - median))
        
        if mad == 0:
            return pd.Index([])  # No outliers if no variation
        
        modified_z_scores = 0.6745 * (data[column] - median) / mad
        outlier_mask = np.abs(modified_z_scores) > threshold
        
        return data[outlier_mask].index
    
    def _detect_outliers_isolation_forest(self, data: pd.DataFrame, column: str) -> pd.Index:
        """
        Detect outliers using Isolation Forest.
        
        Args:
            data: Input DataFrame
            column: Column name to analyze
        
        Returns:
            Index of outlier rows
        """
        contamination = self.config.get('isolation_forest_contamination', 0.1)
        
        # Remove NaN values for isolation forest
        valid_data = data[column].dropna()
        
        if len(valid_data) < 10:  # Need sufficient data for isolation forest
            logger.warning(f"Insufficient data for isolation forest on column {column}")
            return pd.Index([])
        
        # Reshape for sklearn
        X = valid_data.values.reshape(-1, 1)
        
        # Fit isolation forest
        iso_forest = IsolationForest(contamination=contamination, random_state=42)
        outlier_labels = iso_forest.fit_predict(X)
        
        # Get outlier indices
        outlier_mask = outlier_labels == -1
        outlier_indices = valid_data.index[outlier_mask]
        
        return outlier_indices
    
    def _detect_outliers_percentile(self, data: pd.DataFrame, column: str) -> pd.Index:
        """
        Detect outliers using percentile bounds.
        
        Args:
            data: Input DataFrame
            column: Column name to analyze
        
        Returns:
            Index of outlier rows
        """
        bounds = self.config.get('percentile_bounds', (1, 99))
        
        lower_percentile = bounds[0]
        upper_percentile = bounds[1]
        
        lower_bound = data[column].quantile(lower_percentile / 100)
        upper_bound = data[column].quantile(upper_percentile / 100)
        
        outlier_mask = (data[column] < lower_bound) | (data[column] > upper_bound)
        return data[outlier_mask].index
    
    def _handle_outliers_in_column(self, data: pd.DataFrame, column: str, 
                                 outlier_indices: pd.Index) -> pd.DataFrame:
        """
        Handle detected outliers based on the configured action.
        
        Args:
            data: Input DataFrame
            column: Column name
            outlier_indices: Indices of outlier rows
        
        Returns:
            DataFrame with outliers handled
        """
        action = self.config.get('action', 'cap')
        
        if action == 'remove':
            data = data.drop(outlier_indices)
            self.outlier_stats['outliers_removed'] += len(outlier_indices)
            
        elif action == 'cap':
            data = self._cap_outliers(data, column, outlier_indices)
            self.outlier_stats['outliers_capped'] += len(outlier_indices)
            
        elif action == 'flag':
            data = self._flag_outliers(data, column, outlier_indices)
            self.outlier_stats['outliers_flagged'] += len(outlier_indices)
            
        elif action == 'transform':
            data = self._transform_outliers(data, column, outlier_indices)
        
        return data
    
    def _cap_outliers(self, data: pd.DataFrame, column: str, 
                     outlier_indices: pd.Index) -> pd.DataFrame:
        """
        Cap outliers to the nearest non-outlier values.
        
        Args:
            data: Input DataFrame
            column: Column name
            outlier_indices: Indices of outlier rows
        
        Returns:
            DataFrame with outliers capped
        """
        # Calculate bounds using IQR method
        Q1 = data[column].quantile(0.25)
        Q3 = data[column].quantile(0.75)
        IQR = Q3 - Q1
        threshold = self.config.get('threshold', 1.5)
        
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        
        # Cap outliers
        data.loc[outlier_indices, column] = data.loc[outlier_indices, column].clip(
            lower=lower_bound, upper=upper_bound
        )
        
        return data
    
    def _flag_outliers(self, data: pd.DataFrame, column: str, 
                      outlier_indices: pd.Index) -> pd.DataFrame:
        """
        Flag outliers by adding a boolean column.
        
        Args:
            data: Input DataFrame
            column: Column name
            outlier_indices: Indices of outlier rows
        
        Returns:
            DataFrame with outlier flag column added
        """
        flag_column = f"{column}_outlier_flag"
        data[flag_column] = False
        data.loc[outlier_indices, flag_column] = True
        
        return data
    
    def _transform_outliers(self, data: pd.DataFrame, column: str, 
                          outlier_indices: pd.Index) -> pd.DataFrame:
        """
        Transform outliers using log transformation or other methods.
        
        Args:
            data: Input DataFrame
            column: Column name
            outlier_indices: Indices of outlier rows
        
        Returns:
            DataFrame with outliers transformed
        """
        # Apply log transformation if values are positive
        if (data[column] > 0).all():
            data[column] = np.log1p(data[column])
        else:
            # Use square root transformation for non-negative values
            data[column] = np.sqrt(np.abs(data[column]))
        
        return data
    
    def analyze_outliers(self, data: pd.DataFrame) -> Dict:
        """
        Analyze outlier patterns in the data.
        
        Args:
            data: Input DataFrame to analyze
        
        Returns:
            Dictionary with outlier analysis
        """
        analysis = {}
        
        # Get numeric columns
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            col_analysis = {}
            
            # Basic statistics
            col_analysis['count'] = data[col].count()
            col_analysis['mean'] = data[col].mean()
            col_analysis['median'] = data[col].median()
            col_analysis['std'] = data[col].std()
            col_analysis['min'] = data[col].min()
            col_analysis['max'] = data[col].max()
            
            # Detect outliers using different methods
            outlier_methods = ['iqr', 'zscore', 'percentile']
            outlier_counts = {}
            
            for method in outlier_methods:
                original_method = self.config['method']
                self.config['method'] = method
                
                try:
                    outlier_indices = self._detect_outliers_in_column(data, col)
                    outlier_counts[method] = len(outlier_indices)
                except Exception as e:
                    outlier_counts[method] = 0
                    logger.warning(f"Error detecting outliers with {method}: {str(e)}")
                
                self.config['method'] = original_method
            
            col_analysis['outlier_counts'] = outlier_counts
            col_analysis['outlier_percentage'] = {
                method: (count / col_analysis['count'] * 100) if col_analysis['count'] > 0 else 0
                for method, count in outlier_counts.items()
            }
            
            analysis[col] = col_analysis
        
        return analysis
    
    def get_outlier_summary(self) -> Dict:
        """
        Get summary of the last outlier detection operation.
        
        Returns:
            Dictionary with outlier detection statistics
        """
        return self.outlier_stats.copy()
    
    def configure_for_snowflake_data(self):
        """
        Configure the detector for typical Snowflake analytics data.
        """
        # Snowflake-specific configuration
        snowflake_config = {
            'method': 'iqr',  # IQR is robust for skewed data
            'threshold': 2.0,  # More lenient threshold for analytics data
            'action': 'flag',  # Flag outliers rather than remove for analysis
            'preserve_extreme_values': True,  # Keep extreme values that might be valid
            'columns_to_check': [
                'CREDITS_USED', 'BYTES_SCANNED', 'ROWS_PRODUCED',
                'EXECUTION_TIME_MS', 'QUEUE_TIME_MS', 'TOTAL_ELAPSED_TIME'
            ]
        }
        
        self.config.update(snowflake_config)
        logger.info("Configured for Snowflake analytics data")
