"""
Temporal Aggregator

This module provides comprehensive temporal aggregation capabilities for creating
time-based summaries and trend analysis from Snowflake analytics data.
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Any
from datetime import datetime, timedelta
import warnings

logger = logging.getLogger(__name__)


class TemporalAggregator:
    """
    Handles temporal aggregation of analytics data across various time periods.
    
    This class provides methods for aggregating data by hour, day, week, month,
    quarter, and custom time periods, with support for trend analysis,
    seasonality detection, and forecasting-ready outputs.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the TemporalAggregator with optional configuration.
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        
        # Default configuration
        self.default_config = {
            'time_column': 'START_TIME',
            'temporal_levels': ['hourly', 'daily', 'weekly', 'monthly', 'quarterly'],
            'metrics_to_aggregate': [
                'credits_used', 'execution_time_ms', 'bytes_scanned',
                'rows_produced', 'query_count'
            ],
            'aggregation_functions': {
                'sum': ['credits_used', 'execution_time_ms', 'bytes_scanned', 'rows_produced', 'query_count'],
                'mean': ['credits_used', 'execution_time_ms', 'bytes_scanned', 'rows_produced'],
                'median': ['credits_used', 'execution_time_ms', 'bytes_scanned'],
                'std': ['credits_used', 'execution_time_ms', 'bytes_scanned'],
                'min': ['credits_used', 'execution_time_ms', 'bytes_scanned'],
                'max': ['credits_used', 'execution_time_ms', 'bytes_scanned'],
                'count': ['query_count']
            },
            'percentiles': [25, 50, 75, 90, 95, 99],
            'include_trends': True,
            'include_seasonality': True,
            'include_growth_rates': True,
            'include_moving_averages': True,
            'moving_average_windows': [3, 7, 14, 30],  # days/periods
            'business_hours': (9, 17),
            'weekend_days': [5, 6],  # Saturday, Sunday
            'fill_missing_periods': True,
            'interpolate_missing': False
        }
        
        # Merge with provided config
        self.config = {**self.default_config, **self.config}
    
    def aggregate_by_time_period(self, 
                                data: pd.DataFrame, 
                                time_level: str,
                                metrics: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Aggregate data by specified time period.
        
        Args:
            data: Input DataFrame with time series data
            time_level: Time aggregation level ('hourly', 'daily', 'weekly', 'monthly', 'quarterly')
            metrics: Optional list of metrics to aggregate. If None, uses configured metrics.
        
        Returns:
            DataFrame with temporal aggregations
        """
        logger.info(f"Performing {time_level} temporal aggregation...")
        
        time_col = self.config['time_column']
        if time_col not in data.columns:
            raise ValueError(f"Time column {time_col} not found in data")
        
        # Ensure datetime format
        data = data.copy()
        data[time_col] = pd.to_datetime(data[time_col])
        
        if metrics is None:
            metrics = self.config['metrics_to_aggregate']
        
        # Create time grouping column
        time_group_col = self._create_time_grouping_column(data, time_level)
        
        # Perform aggregation
        aggregated = self._perform_temporal_aggregation(data, time_group_col, metrics)
        
        # Add temporal features
        aggregated = self._add_temporal_features(aggregated, time_level)
        
        # Add trend analysis if enabled
        if self.config.get('include_trends', True):
            aggregated = self._add_trend_analysis(aggregated, metrics)
        
        # Add seasonality features if enabled
        if self.config.get('include_seasonality', True):
            aggregated = self._add_seasonality_features(aggregated, time_level)
        
        # Add growth rates if enabled
        if self.config.get('include_growth_rates', True):
            aggregated = self._add_growth_rates(aggregated, metrics)
        
        # Add moving averages if enabled
        if self.config.get('include_moving_averages', True):
            aggregated = self._add_moving_averages(aggregated, metrics, time_level)
        
        # Fill missing periods if enabled
        if self.config.get('fill_missing_periods', True):
            aggregated = self._fill_missing_periods(aggregated, time_level)
        
        logger.info(f"Generated {time_level} aggregation with {len(aggregated)} periods and {len(aggregated.columns)} metrics")
        return aggregated
    
    def _create_time_grouping_column(self, data: pd.DataFrame, time_level: str) -> str:
        """
        Create appropriate time grouping column based on aggregation level.
        
        Args:
            data: Input DataFrame
            time_level: Time aggregation level
        
        Returns:
            Name of the created grouping column
        """
        time_col = self.config['time_column']
        time_series = data[time_col]
        
        if time_level == 'hourly':
            data['time_group'] = time_series.dt.floor('H')
            group_col = 'time_group'
        elif time_level == 'daily':
            data['time_group'] = time_series.dt.date
            group_col = 'time_group'
        elif time_level == 'weekly':
            data['time_group'] = time_series.dt.to_period('W').dt.start_time
            group_col = 'time_group'
        elif time_level == 'monthly':
            data['time_group'] = time_series.dt.to_period('M').dt.start_time
            group_col = 'time_group'
        elif time_level == 'quarterly':
            data['time_group'] = time_series.dt.to_period('Q').dt.start_time
            group_col = 'time_group'
        else:
            raise ValueError(f"Unsupported time level: {time_level}")
        
        return group_col
    
    def _perform_temporal_aggregation(self, 
                                    data: pd.DataFrame, 
                                    group_col: str, 
                                    metrics: List[str]) -> pd.DataFrame:
        """
        Perform the actual temporal aggregation.
        
        Args:
            data: Input DataFrame
            group_col: Column to group by
            metrics: List of metrics to aggregate
        
        Returns:
            Aggregated DataFrame
        """
        # Filter metrics that exist in data
        available_metrics = [m for m in metrics if m in data.columns]
        
        if not available_metrics:
            logger.warning("No specified metrics found in data")
            # Create basic aggregation with query count
            agg_data = data.groupby(group_col).size().reset_index(name='query_count')
            return agg_data
        
        # Define aggregation functions
        agg_functions = {}
        
        for metric in available_metrics:
            metric_aggs = []
            
            # Add configured aggregation functions
            for func_name, func_metrics in self.config['aggregation_functions'].items():
                if metric in func_metrics:
                    metric_aggs.append(func_name)
            
            if metric_aggs:
                agg_functions[metric] = metric_aggs
        
        # Perform aggregation
        try:
            aggregated = data.groupby(group_col).agg(agg_functions)
            
            # Flatten column names
            new_columns = []
            for col in aggregated.columns:
                if isinstance(col, tuple):
                    new_columns.append(f"{col[0]}_{col[1]}")
                else:
                    new_columns.append(col)
            
            aggregated.columns = new_columns
            aggregated = aggregated.reset_index()
            
            # Add percentile aggregations
            percentiles = self.config.get('percentiles', [])
            if percentiles:
                for metric in available_metrics:
                    if metric in data.columns:
                        for p in percentiles:
                            percentile_values = data.groupby(group_col)[metric].quantile(p/100)
                            aggregated[f"{metric}_p{p}"] = aggregated[group_col].map(percentile_values)
            
        except Exception as e:
            logger.error(f"Error in temporal aggregation: {str(e)}")
            # Fallback to simple aggregation
            aggregated = data.groupby(group_col)[available_metrics].agg(['sum', 'mean', 'count']).reset_index()
            aggregated.columns = [f"{col[0]}_{col[1]}" if isinstance(col, tuple) else col for col in aggregated.columns]
        
        return aggregated
    
    def _add_temporal_features(self, data: pd.DataFrame, time_level: str) -> pd.DataFrame:
        """
        Add temporal features like day of week, month, etc.
        
        Args:
            data: Aggregated DataFrame
            time_level: Time aggregation level
        
        Returns:
            DataFrame with added temporal features
        """
        if 'time_group' not in data.columns:
            return data
        
        data = data.copy()
        time_group = pd.to_datetime(data['time_group'])
        
        # Add basic temporal features
        data['year'] = time_group.dt.year
        data['month'] = time_group.dt.month
        data['quarter'] = time_group.dt.quarter
        
        if time_level in ['hourly', 'daily']:
            data['day_of_week'] = time_group.dt.dayofweek
            data['day_of_month'] = time_group.dt.day
            data['day_of_year'] = time_group.dt.dayofyear
            data['week_of_year'] = time_group.dt.isocalendar().week
            
            # Business day indicator
            data['is_business_day'] = (~time_group.dt.dayofweek.isin(self.config['weekend_days'])).astype(int)
            data['is_weekend'] = time_group.dt.dayofweek.isin(self.config['weekend_days']).astype(int)
        
        if time_level == 'hourly':
            data['hour'] = time_group.dt.hour
            
            # Business hours indicator
            business_hours = self.config['business_hours']
            data['is_business_hours'] = (
                (time_group.dt.hour >= business_hours[0]) & 
                (time_group.dt.hour < business_hours[1])
            ).astype(int)
        
        # Season indicators
        data['season'] = time_group.dt.month.map({
            12: 'winter', 1: 'winter', 2: 'winter',
            3: 'spring', 4: 'spring', 5: 'spring',
            6: 'summer', 7: 'summer', 8: 'summer',
            9: 'fall', 10: 'fall', 11: 'fall'
        })
        
        # Month/quarter boundary indicators
        data['is_month_start'] = time_group.dt.is_month_start.astype(int)
        data['is_month_end'] = time_group.dt.is_month_end.astype(int)
        data['is_quarter_start'] = time_group.dt.is_quarter_start.astype(int)
        data['is_quarter_end'] = time_group.dt.is_quarter_end.astype(int)
        data['is_year_start'] = time_group.dt.is_year_start.astype(int)
        data['is_year_end'] = time_group.dt.is_year_end.astype(int)
        
        return data
    
    def _add_trend_analysis(self, data: pd.DataFrame, metrics: List[str]) -> pd.DataFrame:
        """
        Add trend analysis features.
        
        Args:
            data: Aggregated DataFrame
            metrics: List of metrics to analyze trends for
        
        Returns:
            DataFrame with trend features
        """
        if len(data) < 3:
            return data
        
        data = data.copy()
        data = data.sort_values('time_group')
        
        for metric in metrics:
            metric_cols = [col for col in data.columns if col.startswith(f"{metric}_")]
            
            for col in metric_cols:
                if col in data.columns and data[col].dtype in ['int64', 'float64']:
                    # Simple trend (difference from previous period)
                    data[f"{col}_trend"] = data[col].diff()
                    
                    # Trend direction
                    data[f"{col}_trend_direction"] = np.sign(data[f"{col}_trend"])
                    
                    # Percentage change
                    data[f"{col}_pct_change"] = data[col].pct_change().fillna(0)
                    
                    # Rolling trend (slope over last few periods)
                    if len(data) >= 5:
                        window_size = min(5, len(data))
                        rolling_trend = data[col].rolling(window=window_size, min_periods=2).apply(
                            lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) > 1 else 0,
                            raw=True
                        )
                        data[f"{col}_rolling_trend"] = rolling_trend
        
        return data
    
    def _add_seasonality_features(self, data: pd.DataFrame, time_level: str) -> pd.DataFrame:
        """
        Add seasonality detection features.
        
        Args:
            data: Aggregated DataFrame
            time_level: Time aggregation level
        
        Returns:
            DataFrame with seasonality features
        """
        if len(data) < 7:  # Need minimum data for seasonality
            return data
        
        data = data.copy()
        time_group = pd.to_datetime(data['time_group'])
        
        # Cyclical encoding for seasonality
        if time_level == 'hourly':
            # Hour of day cyclical
            data['hour_sin'] = np.sin(2 * np.pi * time_group.dt.hour / 24)
            data['hour_cos'] = np.cos(2 * np.pi * time_group.dt.hour / 24)
            
            # Day of week cyclical
            data['dow_sin'] = np.sin(2 * np.pi * time_group.dt.dayofweek / 7)
            data['dow_cos'] = np.cos(2 * np.pi * time_group.dt.dayofweek / 7)
        
        elif time_level == 'daily':
            # Day of week cyclical
            data['dow_sin'] = np.sin(2 * np.pi * time_group.dt.dayofweek / 7)
            data['dow_cos'] = np.cos(2 * np.pi * time_group.dt.dayofweek / 7)
            
            # Day of month cyclical
            data['dom_sin'] = np.sin(2 * np.pi * time_group.dt.day / 30.5)
            data['dom_cos'] = np.cos(2 * np.pi * time_group.dt.day / 30.5)
        
        elif time_level == 'weekly':
            # Week of year cyclical
            data['woy_sin'] = np.sin(2 * np.pi * time_group.dt.isocalendar().week / 52)
            data['woy_cos'] = np.cos(2 * np.pi * time_group.dt.isocalendar().week / 52)
        
        # Month cyclical (for all levels except hourly)
        if time_level != 'hourly':
            data['month_sin'] = np.sin(2 * np.pi * time_group.dt.month / 12)
            data['month_cos'] = np.cos(2 * np.pi * time_group.dt.month / 12)
        
        # Quarter cyclical
        data['quarter_sin'] = np.sin(2 * np.pi * time_group.dt.quarter / 4)
        data['quarter_cos'] = np.cos(2 * np.pi * time_group.dt.quarter / 4)
        
        return data
    
    def _add_growth_rates(self, data: pd.DataFrame, metrics: List[str]) -> pd.DataFrame:
        """
        Add growth rate calculations.
        
        Args:
            data: Aggregated DataFrame
            metrics: List of metrics to calculate growth rates for
        
        Returns:
            DataFrame with growth rate features
        """
        if len(data) < 2:
            return data
        
        data = data.copy()
        data = data.sort_values('time_group')
        
        for metric in metrics:
            metric_cols = [col for col in data.columns if col.startswith(f"{metric}_sum") or col.startswith(f"{metric}_mean")]
            
            for col in metric_cols:
                if col in data.columns and data[col].dtype in ['int64', 'float64']:
                    # Period-over-period growth rate
                    data[f"{col}_growth_rate"] = data[col].pct_change().fillna(0)
                    
                    # Year-over-year growth rate (if we have enough data)
                    if len(data) >= 52:  # Roughly a year of weekly data
                        periods_per_year = self._get_periods_per_year(data)
                        if len(data) >= periods_per_year:
                            data[f"{col}_yoy_growth"] = data[col].pct_change(periods=periods_per_year).fillna(0)
                    
                    # Compound annual growth rate (CAGR) - simplified
                    if len(data) >= 4:  # At least 4 periods
                        def calculate_cagr(series, periods):
                            if periods < 2 or series.iloc[0] <= 0:
                                return 0
                            try:
                                return (series.iloc[-1] / series.iloc[0]) ** (1/periods) - 1
                            except:
                                return 0
                        
                        window_size = min(12, len(data))  # Use 12 periods or available data
                        rolling_cagr = data[col].rolling(window=window_size, min_periods=4).apply(
                            lambda x: calculate_cagr(x, len(x)), raw=False
                        )
                        data[f"{col}_cagr"] = rolling_cagr
        
        return data
    
    def _add_moving_averages(self, data: pd.DataFrame, metrics: List[str], time_level: str) -> pd.DataFrame:
        """
        Add moving average calculations.
        
        Args:
            data: Aggregated DataFrame
            metrics: List of metrics to calculate moving averages for
            time_level: Time aggregation level
        
        Returns:
            DataFrame with moving average features
        """
        if len(data) < 3:
            return data
        
        data = data.copy()
        data = data.sort_values('time_group')
        
        # Adjust window sizes based on time level
        windows = self._adjust_windows_for_time_level(time_level)
        
        for metric in metrics:
            metric_cols = [col for col in data.columns if col.startswith(f"{metric}_")]
            
            for col in metric_cols:
                if col in data.columns and data[col].dtype in ['int64', 'float64']:
                    for window in windows:
                        if window < len(data):
                            # Simple moving average
                            data[f"{col}_ma_{window}"] = data[col].rolling(window=window, min_periods=1).mean()
                            
                            # Exponential moving average
                            data[f"{col}_ema_{window}"] = data[col].ewm(span=window).mean()
                            
                            # Moving standard deviation
                            data[f"{col}_std_{window}"] = data[col].rolling(window=window, min_periods=1).std()
        
        return data
    
    def _fill_missing_periods(self, data: pd.DataFrame, time_level: str) -> pd.DataFrame:
        """
        Fill missing time periods with zero or interpolated values.
        
        Args:
            data: Aggregated DataFrame
            time_level: Time aggregation level
        
        Returns:
            DataFrame with filled missing periods
        """
        if len(data) == 0 or 'time_group' not in data.columns:
            return data
        
        data = data.copy()
        data['time_group'] = pd.to_datetime(data['time_group'])
        data = data.sort_values('time_group')
        
        # Create complete time range
        start_time = data['time_group'].min()
        end_time = data['time_group'].max()
        
        if time_level == 'hourly':
            time_range = pd.date_range(start=start_time, end=end_time, freq='H')
        elif time_level == 'daily':
            time_range = pd.date_range(start=start_time, end=end_time, freq='D')
        elif time_level == 'weekly':
            time_range = pd.date_range(start=start_time, end=end_time, freq='W')
        elif time_level == 'monthly':
            time_range = pd.date_range(start=start_time, end=end_time, freq='MS')
        elif time_level == 'quarterly':
            time_range = pd.date_range(start=start_time, end=end_time, freq='QS')
        else:
            return data
        
        # Create complete DataFrame
        complete_df = pd.DataFrame({'time_group': time_range})
        
        # Merge with existing data
        filled_data = complete_df.merge(data, on='time_group', how='left')
        
        # Fill missing values
        numeric_columns = filled_data.select_dtypes(include=[np.number]).columns
        
        if self.config.get('interpolate_missing', False):
            # Interpolate numeric columns
            filled_data[numeric_columns] = filled_data[numeric_columns].interpolate(method='linear')
        else:
            # Fill with zeros
            filled_data[numeric_columns] = filled_data[numeric_columns].fillna(0)
        
        # Fill categorical columns with appropriate values
        categorical_columns = filled_data.select_dtypes(include=['object', 'category']).columns
        for col in categorical_columns:
            if col != 'time_group':
                filled_data[col] = filled_data[col].fillna('Missing')
        
        return filled_data
    
    def _get_periods_per_year(self, data: pd.DataFrame) -> int:
        """
        Estimate number of periods per year based on data frequency.
        
        Args:
            data: Aggregated DataFrame
        
        Returns:
            Estimated periods per year
        """
        if len(data) < 2:
            return 365
        
        time_group = pd.to_datetime(data['time_group'])
        time_diff = (time_group.iloc[-1] - time_group.iloc[0]).total_seconds()
        periods = len(data) - 1
        
        if periods == 0:
            return 365
        
        avg_period_seconds = time_diff / periods
        
        # Estimate based on average period length
        if avg_period_seconds <= 3700:  # ~1 hour
            return 8760  # hours per year
        elif avg_period_seconds <= 86500:  # ~1 day
            return 365
        elif avg_period_seconds <= 605000:  # ~1 week
            return 52
        elif avg_period_seconds <= 2629800:  # ~1 month
            return 12
        else:  # quarterly or longer
            return 4
    
    def _adjust_windows_for_time_level(self, time_level: str) -> List[int]:
        """
        Adjust moving average windows based on time aggregation level.
        
        Args:
            time_level: Time aggregation level
        
        Returns:
            List of appropriate window sizes
        """
        base_windows = self.config.get('moving_average_windows', [3, 7, 14, 30])
        
        if time_level == 'hourly':
            # For hourly data, use smaller windows
            return [3, 6, 12, 24, 168]  # 3h, 6h, 12h, 1d, 1w
        elif time_level == 'daily':
            return base_windows  # 3d, 7d, 14d, 30d
        elif time_level == 'weekly':
            return [2, 4, 8, 12, 26]  # 2w, 4w, 8w, 12w, 26w
        elif time_level == 'monthly':
            return [2, 3, 6, 12]  # 2m, 3m, 6m, 12m
        elif time_level == 'quarterly':
            return [2, 4, 8]  # 2q, 4q, 8q
        else:
            return base_windows
    
    def configure_for_snowflake_data(self):
        """
        Configure the aggregator for typical Snowflake analytics data.
        """
        snowflake_config = {
            'time_column': 'START_TIME',
            'metrics_to_aggregate': [
                'CREDITS_USED', 'EXECUTION_TIME_MS', 'BYTES_SCANNED',
                'ROWS_PRODUCED', 'PARTITIONS_SCANNED', 'query_count'
            ],
            'aggregation_functions': {
                'sum': ['CREDITS_USED', 'EXECUTION_TIME_MS', 'BYTES_SCANNED', 'ROWS_PRODUCED', 'query_count'],
                'mean': ['CREDITS_USED', 'EXECUTION_TIME_MS', 'BYTES_SCANNED', 'ROWS_PRODUCED'],
                'median': ['CREDITS_USED', 'EXECUTION_TIME_MS', 'BYTES_SCANNED'],
                'std': ['CREDITS_USED', 'EXECUTION_TIME_MS', 'BYTES_SCANNED'],
                'min': ['CREDITS_USED', 'EXECUTION_TIME_MS', 'BYTES_SCANNED'],
                'max': ['CREDITS_USED', 'EXECUTION_TIME_MS', 'BYTES_SCANNED'],
                'count': ['query_count']
            },
            'business_hours': (8, 18),  # Typical business hours for analytics
            'percentiles': [50, 75, 90, 95, 99]
        }
        
        self.config.update(snowflake_config)
        logger.info("Configured TemporalAggregator for Snowflake analytics data")
