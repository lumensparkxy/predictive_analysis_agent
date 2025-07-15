"""
Data Type Conversion and Standardization

This module provides comprehensive data type conversion and standardization
capabilities for Snowflake analytics data with intelligent type detection.
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any
from datetime import datetime
import re

logger = logging.getLogger(__name__)


class TypeConverter:
    """
    Handler for converting and standardizing data types.
    
    Provides intelligent type detection and conversion with special handling
    for common data patterns in Snowflake analytics data.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the TypeConverter with configuration.
        
        Args:
            config: Configuration dictionary with type conversion parameters
        """
        self.config = config or {}
        
        # Default configuration
        self.default_config = {
            'auto_detect': True,  # Automatically detect optimal types
            'datetime_formats': [
                '%Y-%m-%d %H:%M:%S',
                '%Y-%m-%d %H:%M:%S.%f',
                '%Y-%m-%dT%H:%M:%S',
                '%Y-%m-%dT%H:%M:%S.%f',
                '%Y-%m-%d',
                'ISO8601'
            ],
            'boolean_values': {
                'true_values': ['true', 'True', 'TRUE', '1', 'yes', 'Yes', 'YES', 'on', 'On', 'ON'],
                'false_values': ['false', 'False', 'FALSE', '0', 'no', 'No', 'NO', 'off', 'Off', 'OFF']
            },
            'numeric_threshold': 0.8,  # 80% of values must be numeric to convert to numeric
            'downcast_integers': True,  # Downcast integers to save memory
            'downcast_floats': True,  # Downcast floats to save memory
            'force_category': False,  # Force string columns to category
            'category_threshold': 0.5,  # Convert to category if unique values < 50% of total
            'timezone': 'UTC'  # Default timezone for datetime conversion
        }
        
        # Merge with provided config
        self.config = {**self.default_config, **self.config}
        
        # Statistics tracking
        self.conversion_stats = {
            'columns_converted': 0,
            'datetime_conversions': 0,
            'numeric_conversions': 0,
            'boolean_conversions': 0,
            'category_conversions': 0,
            'memory_saved_bytes': 0,
            'conversion_errors': {},
            'processing_time': 0.0
        }
    
    def convert_types(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
        """
        Convert and standardize data types in the DataFrame.
        
        Args:
            data: Input DataFrame to process
        
        Returns:
            Tuple of (converted_data, conversion_count)
        """
        start_time = datetime.now()
        
        logger.info("Starting data type conversion and standardization")
        
        # Calculate initial memory usage
        initial_memory = data.memory_usage(deep=True).sum()
        
        try:
            converted_data = data.copy()
            conversion_count = 0
            
            if self.config.get('auto_detect', True):
                converted_data, conversion_count = self._auto_detect_and_convert(converted_data)
            else:
                converted_data, conversion_count = self._manual_convert_types(converted_data)
            
            # Calculate final memory usage
            final_memory = converted_data.memory_usage(deep=True).sum()
            memory_saved = initial_memory - final_memory
            
            # Update statistics
            self.conversion_stats['columns_converted'] = conversion_count
            self.conversion_stats['memory_saved_bytes'] = memory_saved
            self.conversion_stats['processing_time'] = (datetime.now() - start_time).total_seconds()
            
            logger.info(f"Type conversion completed in {self.conversion_stats['processing_time']:.2f} seconds")
            logger.info(f"Converted {conversion_count} columns, saved {memory_saved / 1024 / 1024:.2f} MB")
            
            return converted_data, conversion_count
            
        except Exception as e:
            logger.error(f"Error converting types: {str(e)}")
            raise
    
    def _auto_detect_and_convert(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
        """
        Automatically detect and convert data types.
        
        Args:
            data: Input DataFrame
        
        Returns:
            Tuple of (converted_data, conversion_count)
        """
        converted_data = data.copy()
        conversion_count = 0
        
        for col in data.columns:
            original_dtype = converted_data[col].dtype
            
            # Skip if already optimal type
            if pd.api.types.is_numeric_dtype(converted_data[col]) and \
               not pd.api.types.is_object_dtype(converted_data[col]):
                # Try to downcast numeric types
                if self._should_downcast_numeric(converted_data[col]):
                    converted_data[col] = self._downcast_numeric(converted_data[col])
                    if converted_data[col].dtype != original_dtype:
                        conversion_count += 1
                continue
            
            # Try different conversion strategies
            converted_col = self._try_convert_column(converted_data[col])
            
            if converted_col.dtype != original_dtype:
                converted_data[col] = converted_col
                conversion_count += 1
                logger.debug(f"Converted {col}: {original_dtype} → {converted_col.dtype}")
        
        return converted_data, conversion_count
    
    def _try_convert_column(self, series: pd.Series) -> pd.Series:
        """
        Try to convert a column to the most appropriate type.
        
        Args:
            series: Input series to convert
        
        Returns:
            Series with converted type
        """
        # Skip if not object type (already converted)
        if not pd.api.types.is_object_dtype(series):
            return series
        
        # Remove null values for type detection
        non_null_series = series.dropna()
        
        if len(non_null_series) == 0:
            return series
        
        # Try datetime conversion first
        datetime_series = self._try_convert_datetime(series)
        if not pd.api.types.is_object_dtype(datetime_series):
            self.conversion_stats['datetime_conversions'] += 1
            return datetime_series
        
        # Try boolean conversion
        boolean_series = self._try_convert_boolean(series)
        if not pd.api.types.is_object_dtype(boolean_series):
            self.conversion_stats['boolean_conversions'] += 1
            return boolean_series
        
        # Try numeric conversion
        numeric_series = self._try_convert_numeric(series)
        if not pd.api.types.is_object_dtype(numeric_series):
            self.conversion_stats['numeric_conversions'] += 1
            return numeric_series
        
        # Try category conversion
        category_series = self._try_convert_category(series)
        if pd.api.types.is_categorical_dtype(category_series):
            self.conversion_stats['category_conversions'] += 1
            return category_series
        
        return series
    
    def _try_convert_datetime(self, series: pd.Series) -> pd.Series:
        """
        Try to convert series to datetime.
        
        Args:
            series: Input series
        
        Returns:
            Series with datetime conversion attempted
        """
        # Check if series looks like datetime
        if not self._looks_like_datetime(series):
            return series
        
        datetime_formats = self.config.get('datetime_formats', [])
        timezone = self.config.get('timezone', 'UTC')
        
        # Try pandas automatic datetime parsing first
        try:
            converted = pd.to_datetime(series, utc=True, errors='coerce')
            
            # Check if conversion was successful for most values
            success_rate = (converted.notna().sum() / len(series.dropna()))
            if success_rate > 0.8:  # 80% success rate
                return converted
        except Exception as e:
            logger.debug(f"Automatic datetime conversion failed: {str(e)}")
        
        # Try specific formats
        for fmt in datetime_formats:
            try:
                if fmt == 'ISO8601':
                    converted = pd.to_datetime(series, format='ISO8601', utc=True, errors='coerce')
                else:
                    converted = pd.to_datetime(series, format=fmt, utc=True, errors='coerce')
                
                success_rate = (converted.notna().sum() / len(series.dropna()))
                if success_rate > 0.8:
                    return converted
                    
            except Exception as e:
                logger.debug(f"Datetime conversion with format {fmt} failed: {str(e)}")
                continue
        
        return series
    
    def _try_convert_boolean(self, series: pd.Series) -> pd.Series:
        """
        Try to convert series to boolean.
        
        Args:
            series: Input series
        
        Returns:
            Series with boolean conversion attempted
        """
        if not self._looks_like_boolean(series):
            return series
        
        try:
            true_values = self.config['boolean_values']['true_values']
            false_values = self.config['boolean_values']['false_values']
            
            # Create boolean mapping
            bool_map = {}
            for val in true_values:
                bool_map[val] = True
            for val in false_values:
                bool_map[val] = False
            
            # Convert using mapping
            converted = series.map(bool_map)
            
            # Check success rate
            success_rate = (converted.notna().sum() / len(series.dropna()))
            if success_rate > 0.9:  # 90% success rate for boolean
                return converted
                
        except Exception as e:
            logger.debug(f"Boolean conversion failed: {str(e)}")
        
        return series
    
    def _try_convert_numeric(self, series: pd.Series) -> pd.Series:
        """
        Try to convert series to numeric.
        
        Args:
            series: Input series
        
        Returns:
            Series with numeric conversion attempted
        """
        if not self._looks_like_numeric(series):
            return series
        
        try:
            # Try integer conversion first
            converted_int = pd.to_numeric(series, errors='coerce', downcast='integer')
            
            # Check if all non-null values are integers
            non_null_original = series.dropna()
            non_null_converted = converted_int.dropna()
            
            if len(non_null_converted) / len(non_null_original) > self.config.get('numeric_threshold', 0.8):
                # Check if values are actually integers
                if (converted_int.dropna() == converted_int.dropna().astype(int)).all():
                    return converted_int
                else:
                    # Try float conversion
                    converted_float = pd.to_numeric(series, errors='coerce', downcast='float')
                    return converted_float
                    
        except Exception as e:
            logger.debug(f"Numeric conversion failed: {str(e)}")
        
        return series
    
    def _try_convert_category(self, series: pd.Series) -> pd.Series:
        """
        Try to convert series to category if it has low cardinality.
        
        Args:
            series: Input series
        
        Returns:
            Series with category conversion attempted
        """
        if not self.config.get('force_category', False):
            # Only convert if cardinality is low
            unique_ratio = series.nunique() / len(series)
            threshold = self.config.get('category_threshold', 0.5)
            
            if unique_ratio > threshold:
                return series
        
        try:
            return series.astype('category')
        except Exception as e:
            logger.debug(f"Category conversion failed: {str(e)}")
            return series
    
    def _looks_like_datetime(self, series: pd.Series) -> bool:
        """
        Check if series looks like datetime data.
        
        Args:
            series: Input series
        
        Returns:
            True if series looks like datetime
        """
        # Sample a few values to check
        sample_values = series.dropna().head(10).astype(str)
        
        if len(sample_values) == 0:
            return False
        
        datetime_patterns = [
            r'\d{4}-\d{2}-\d{2}',  # YYYY-MM-DD
            r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}',  # ISO format
            r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}',  # Standard datetime
        ]
        
        for pattern in datetime_patterns:
            matches = sum(1 for val in sample_values if re.search(pattern, val))
            if matches / len(sample_values) > 0.7:  # 70% match
                return True
        
        return False
    
    def _looks_like_boolean(self, series: pd.Series) -> bool:
        """
        Check if series looks like boolean data.
        
        Args:
            series: Input series
        
        Returns:
            True if series looks like boolean
        """
        unique_values = set(series.dropna().astype(str).str.lower())
        
        all_boolean_values = set()
        all_boolean_values.update([v.lower() for v in self.config['boolean_values']['true_values']])
        all_boolean_values.update([v.lower() for v in self.config['boolean_values']['false_values']])
        
        return unique_values.issubset(all_boolean_values)
    
    def _looks_like_numeric(self, series: pd.Series) -> bool:
        """
        Check if series looks like numeric data.
        
        Args:
            series: Input series
        
        Returns:
            True if series looks like numeric
        """
        # Try to convert a sample and check success rate
        sample = series.dropna().head(100)
        
        if len(sample) == 0:
            return False
        
        try:
            converted = pd.to_numeric(sample, errors='coerce')
            success_rate = converted.notna().sum() / len(sample)
            return success_rate > self.config.get('numeric_threshold', 0.8)
        except:
            return False
    
    def _should_downcast_numeric(self, series: pd.Series) -> bool:
        """
        Check if numeric series should be downcasted.
        
        Args:
            series: Input series
        
        Returns:
            True if should downcast
        """
        if pd.api.types.is_integer_dtype(series):
            return self.config.get('downcast_integers', True)
        elif pd.api.types.is_float_dtype(series):
            return self.config.get('downcast_floats', True)
        
        return False
    
    def _downcast_numeric(self, series: pd.Series) -> pd.Series:
        """
        Downcast numeric series to save memory.
        
        Args:
            series: Input numeric series
        
        Returns:
            Downcasted series
        """
        try:
            if pd.api.types.is_integer_dtype(series):
                return pd.to_numeric(series, downcast='integer')
            elif pd.api.types.is_float_dtype(series):
                return pd.to_numeric(series, downcast='float')
        except Exception as e:
            logger.debug(f"Downcasting failed: {str(e)}")
        
        return series
    
    def _manual_convert_types(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
        """
        Manually convert types based on column names or patterns.
        
        Args:
            data: Input DataFrame
        
        Returns:
            Tuple of (converted_data, conversion_count)
        """
        converted_data = data.copy()
        conversion_count = 0
        
        # Define column patterns for Snowflake data
        datetime_patterns = [
            'START_TIME', 'END_TIME', 'CREATED_TIME', 'MODIFIED_TIME',
            'LAST_SUCCESS_TIME', 'EXECUTION_TIME'
        ]
        
        boolean_patterns = [
            'IS_', 'HAS_', 'CAN_', 'SHOULD_', 'ENABLED', 'DISABLED'
        ]
        
        for col in data.columns:
            original_dtype = converted_data[col].dtype
            
            # Try datetime conversion for known datetime columns
            if any(pattern in col.upper() for pattern in datetime_patterns):
                try:
                    converted_data[col] = pd.to_datetime(converted_data[col], utc=True)
                    if converted_data[col].dtype != original_dtype:
                        conversion_count += 1
                except Exception as e:
                    self.conversion_stats['conversion_errors'][col] = str(e)
            
            # Try boolean conversion for boolean-like columns
            elif any(col.upper().startswith(pattern) for pattern in boolean_patterns):
                boolean_series = self._try_convert_boolean(converted_data[col])
                if not pd.api.types.is_object_dtype(boolean_series):
                    converted_data[col] = boolean_series
                    conversion_count += 1
        
        return converted_data, conversion_count
    
    def analyze_types(self, data: pd.DataFrame) -> Dict:
        """
        Analyze data types and conversion opportunities.
        
        Args:
            data: Input DataFrame to analyze
        
        Returns:
            Dictionary with type analysis
        """
        analysis = {
            'current_types': {},
            'recommended_types': {},
            'memory_usage': {},
            'conversion_opportunities': []
        }
        
        for col in data.columns:
            current_dtype = data[col].dtype
            analysis['current_types'][col] = str(current_dtype)
            analysis['memory_usage'][col] = data[col].memory_usage(deep=True)
            
            # Analyze potential conversions
            if pd.api.types.is_object_dtype(data[col]):
                recommended_type = self._recommend_type(data[col])
                analysis['recommended_types'][col] = recommended_type
                
                if recommended_type != 'object':
                    analysis['conversion_opportunities'].append({
                        'column': col,
                        'current_type': str(current_dtype),
                        'recommended_type': recommended_type,
                        'potential_memory_savings': 'TBD'
                    })
        
        return analysis
    
    def _recommend_type(self, series: pd.Series) -> str:
        """
        Recommend the best type for a series.
        
        Args:
            series: Input series
        
        Returns:
            Recommended type name
        """
        if self._looks_like_datetime(series):
            return 'datetime64[ns]'
        elif self._looks_like_boolean(series):
            return 'bool'
        elif self._looks_like_numeric(series):
            return 'numeric'
        elif series.nunique() / len(series) < 0.5:
            return 'category'
        else:
            return 'object'
    
    def get_conversion_summary(self) -> Dict:
        """
        Get summary of the last type conversion operation.
        
        Returns:
            Dictionary with conversion statistics
        """
        return self.conversion_stats.copy()
    
    def configure_for_snowflake_data(self):
        """
        Configure the converter for typical Snowflake analytics data.
        """
        # Snowflake-specific configuration
        snowflake_config = {
            'auto_detect': True,
            'downcast_integers': True,
            'downcast_floats': True,
            'category_threshold': 0.3,  # Lower threshold for analytics data
            'timezone': 'UTC',
            'datetime_formats': [
                '%Y-%m-%d %H:%M:%S.%f %Z',
                '%Y-%m-%d %H:%M:%S.%f',
                '%Y-%m-%d %H:%M:%S %Z',
                '%Y-%m-%d %H:%M:%S',
                '%Y-%m-%dT%H:%M:%S.%f%z',
                '%Y-%m-%dT%H:%M:%S%z',
                'ISO8601'
            ]
        }
        
        self.config.update(snowflake_config)
        logger.info("Configured for Snowflake analytics data")
