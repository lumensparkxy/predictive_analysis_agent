"""
Rolling Statistics Feature Generator

This module provides comprehensive rolling statistics feature engineering capabilities
for extracting temporal patterns and trends from Snowflake analytics data.
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime, timedelta
import warnings

logger = logging.getLogger(__name__)


class RollingFeatureGenerator:
    """
    Generates comprehensive rolling statistics features from time series data.
    
    This class extracts various rolling window statistics including moving averages,
    trends, volatility measures, and change indicators specifically designed
    for analytics workload temporal patterns.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the RollingFeatureGenerator with optional configuration.
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        
        # Default configuration
        self.default_config = {
            'time_column': 'START_TIME',
            'rolling_windows': {
                'short': [3, 6, 12],      # Hours
                'medium': [1, 3, 7],      # Days
                'long': [14, 30, 90]      # Days
            },
            'rolling_metrics': [
                'credits_used', 'execution_time_ms', 'bytes_scanned',
                'rows_produced', 'queue_time_ms'
            ],
            'statistics_types': [
                'mean', 'sum', 'std', 'min', 'max', 'median',
                'count', 'skew', 'kurt'
            ],
            'trend_windows': [7, 14, 30],  # Days for trend analysis
            'change_windows': [1, 3, 7],   # Days for change detection
            'volatility_windows': [7, 14, 30],  # Days for volatility measures
            'percentiles': [25, 75, 90, 95],
            'min_periods': 1,  # Minimum periods for rolling calculations
            'groupby_columns': ['user_name', 'warehouse_name', 'database_name'],
            'enable_cross_metric_features': True,
            'enable_seasonal_features': True,
            'enable_lag_features': True,
            'lag_periods': [1, 2, 3, 7, 14]  # Days for lag features
        }
        
        # Merge with provided config
        self.config = {**self.default_config, **self.config}
    
    def generate_rolling_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate comprehensive rolling statistics features from the input DataFrame.
        
        Args:
            data: Input DataFrame containing time series data
        
        Returns:
            DataFrame with generated rolling features
        """
        logger.info("Generating rolling statistics features...")
        
        feature_df = pd.DataFrame(index=data.index)
        time_col = self.config['time_column']
        
        if time_col not in data.columns:
            logger.warning(f"Time column {time_col} not found in data")
            return feature_df
        
        # Ensure data is sorted by time
        data_sorted = data.sort_values(time_col).copy()
        
        # Basic rolling statistics
        feature_df.update(self._extract_basic_rolling_stats(data_sorted))
        
        # Rolling trends and momentum
        feature_df.update(self._extract_rolling_trends(data_sorted))
        
        # Rolling volatility and stability measures
        feature_df.update(self._extract_rolling_volatility(data_sorted))
        
        # Rolling change and acceleration features
        feature_df.update(self._extract_rolling_changes(data_sorted))
        
        # Cross-metric rolling relationships
        if self.config.get('enable_cross_metric_features', True):
            feature_df.update(self._extract_cross_metric_rolling_features(data_sorted))
        
        # Seasonal rolling patterns
        if self.config.get('enable_seasonal_features', True):
            feature_df.update(self._extract_seasonal_rolling_features(data_sorted))
        
        # Lag features
        if self.config.get('enable_lag_features', True):
            feature_df.update(self._extract_lag_features(data_sorted))
        
        # Group-based rolling features
        feature_df.update(self._extract_grouped_rolling_features(data_sorted))
        
        # Reindex to match original data order
        feature_df = feature_df.reindex(data.index)
        
        logger.info(f"Generated {len(feature_df.columns)} rolling statistics features")
        return feature_df
    
    def _extract_basic_rolling_stats(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Extract basic rolling statistics for key metrics.
        
        Args:
            data: Input DataFrame sorted by time
        
        Returns:
            DataFrame with basic rolling statistics
        """
        features = pd.DataFrame(index=data.index)
        metrics = self.config['rolling_metrics']
        windows = self.config['rolling_windows']
        stats_types = self.config['statistics_types']
        min_periods = self.config.get('min_periods', 1)
        
        # Convert all window periods to number of rows
        time_col = self.config['time_column']
        time_series = pd.to_datetime(data[time_col])
        
        for metric in metrics:
            if metric not in data.columns:
                continue
            
            metric_values = data[metric]
            
            # Short-term windows (hours - convert to approximate rows)
            for window_hours in windows.get('short', []):
                # Estimate rows per hour (rough approximation)
                estimated_rows = max(1, int(window_hours * len(data) / 
                                           (time_series.max() - time_series.min()).total_seconds() * 3600))
                
                for stat_type in stats_types:
                    feature_name = f'{metric}_rolling_{window_hours}h_{stat_type}'
                    
                    if stat_type == 'mean':
                        features[feature_name] = metric_values.rolling(
                            window=estimated_rows, min_periods=min_periods
                        ).mean()
                    elif stat_type == 'sum':
                        features[feature_name] = metric_values.rolling(
                            window=estimated_rows, min_periods=min_periods
                        ).sum()
                    elif stat_type == 'std':
                        features[feature_name] = metric_values.rolling(
                            window=estimated_rows, min_periods=min_periods
                        ).std()
                    elif stat_type == 'min':
                        features[feature_name] = metric_values.rolling(
                            window=estimated_rows, min_periods=min_periods
                        ).min()
                    elif stat_type == 'max':
                        features[feature_name] = metric_values.rolling(
                            window=estimated_rows, min_periods=min_periods
                        ).max()
                    elif stat_type == 'median':
                        features[feature_name] = metric_values.rolling(
                            window=estimated_rows, min_periods=min_periods
                        ).median()
                    elif stat_type == 'count':
                        features[feature_name] = metric_values.rolling(
                            window=estimated_rows, min_periods=min_periods
                        ).count()
                    elif stat_type == 'skew':
                        features[feature_name] = metric_values.rolling(
                            window=estimated_rows, min_periods=min_periods
                        ).skew()
                    elif stat_type == 'kurt':
                        features[feature_name] = metric_values.rolling(
                            window=estimated_rows, min_periods=min_periods
                        ).kurt()
            
            # Medium and long-term windows (days)
            for window_type in ['medium', 'long']:
                for window_days in windows.get(window_type, []):
                    # Use time-based rolling for daily windows
                    time_window = f'{window_days}D'
                    
                    # Set time index temporarily
                    temp_data = data.set_index(time_col)
                    temp_metric = temp_data[metric]
                    
                    for stat_type in stats_types:
                        feature_name = f'{metric}_rolling_{window_days}d_{stat_type}'
                        
                        try:
                            if stat_type == 'mean':
                                rolling_stat = temp_metric.rolling(
                                    window=time_window, min_periods=min_periods
                                ).mean()
                            elif stat_type == 'sum':
                                rolling_stat = temp_metric.rolling(
                                    window=time_window, min_periods=min_periods
                                ).sum()
                            elif stat_type == 'std':
                                rolling_stat = temp_metric.rolling(
                                    window=time_window, min_periods=min_periods
                                ).std()
                            elif stat_type == 'min':
                                rolling_stat = temp_metric.rolling(
                                    window=time_window, min_periods=min_periods
                                ).min()
                            elif stat_type == 'max':
                                rolling_stat = temp_metric.rolling(
                                    window=time_window, min_periods=min_periods
                                ).max()
                            elif stat_type == 'median':
                                rolling_stat = temp_metric.rolling(
                                    window=time_window, min_periods=min_periods
                                ).median()
                            elif stat_type == 'count':
                                rolling_stat = temp_metric.rolling(
                                    window=time_window, min_periods=min_periods
                                ).count()
                            elif stat_type == 'skew':
                                rolling_stat = temp_metric.rolling(
                                    window=time_window, min_periods=min_periods
                                ).skew()
                            elif stat_type == 'kurt':
                                rolling_stat = temp_metric.rolling(
                                    window=time_window, min_periods=min_periods
                                ).kurt()
                            
                            # Reset index to match original
                            features[feature_name] = rolling_stat.reset_index(drop=True)
                            
                        except Exception as e:
                            logger.warning(f"Error calculating {feature_name}: {str(e)}")
                            continue
            
            # Percentile-based rolling features
            percentiles = self.config.get('percentiles', [25, 75, 90, 95])
            for window_days in [7, 14, 30]:
                time_window = f'{window_days}D'
                temp_data = data.set_index(time_col)
                temp_metric = temp_data[metric]
                
                for p in percentiles:
                    feature_name = f'{metric}_rolling_{window_days}d_p{p}'
                    try:
                        rolling_percentile = temp_metric.rolling(
                            window=time_window, min_periods=min_periods
                        ).quantile(p/100)
                        features[feature_name] = rolling_percentile.reset_index(drop=True)
                    except Exception as e:
                        logger.warning(f"Error calculating {feature_name}: {str(e)}")
                        continue
        
        return features
    
    def _extract_rolling_trends(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Extract rolling trend and momentum features.
        
        Args:
            data: Input DataFrame sorted by time
        
        Returns:
            DataFrame with rolling trend features
        """
        features = pd.DataFrame(index=data.index)
        metrics = self.config['rolling_metrics']
        trend_windows = self.config.get('trend_windows', [7, 14, 30])
        time_col = self.config['time_column']
        
        for metric in metrics:
            if metric not in data.columns:
                continue
            
            metric_values = data[metric]
            
            for window_days in trend_windows:
                window_size = min(window_days, len(data))
                
                # Linear trend (slope of linear regression)
                feature_name = f'{metric}_trend_{window_days}d'
                rolling_trend = metric_values.rolling(window=window_size, min_periods=2).apply(
                    lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) > 1 else 0,
                    raw=True
                )
                features[feature_name] = rolling_trend
                
                # Trend direction
                features[f'{metric}_trend_direction_{window_days}d'] = np.sign(rolling_trend)
                
                # Trend strength (R-squared of linear fit)
                def trend_strength(x):
                    if len(x) < 3:
                        return 0
                    try:
                        slope, intercept = np.polyfit(range(len(x)), x, 1)
                        predicted = slope * np.arange(len(x)) + intercept
                        ss_tot = np.sum((x - np.mean(x)) ** 2)
                        ss_res = np.sum((x - predicted) ** 2)
                        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
                    except:
                        return 0
                
                features[f'{metric}_trend_strength_{window_days}d'] = metric_values.rolling(
                    window=window_size, min_periods=3
                ).apply(trend_strength, raw=True)
                
                # Momentum (rate of change)
                features[f'{metric}_momentum_{window_days}d'] = (
                    metric_values / metric_values.shift(window_days) - 1
                ).fillna(0)
                
                # Moving average convergence/divergence (MACD-like)
                if window_days >= 14:
                    short_ma = metric_values.rolling(window=window_days//2).mean()
                    long_ma = metric_values.rolling(window=window_days).mean()
                    features[f'{metric}_macd_{window_days}d'] = short_ma - long_ma
                    
                    # Signal line (moving average of MACD)
                    signal_window = max(3, window_days//4)
                    features[f'{metric}_macd_signal_{window_days}d'] = (
                        features[f'{metric}_macd_{window_days}d'].rolling(window=signal_window).mean()
                    )
        
        return features
    
    def _extract_rolling_volatility(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Extract rolling volatility and stability measures.
        
        Args:
            data: Input DataFrame sorted by time
        
        Returns:
            DataFrame with rolling volatility features
        """
        features = pd.DataFrame(index=data.index)
        metrics = self.config['rolling_metrics']
        volatility_windows = self.config.get('volatility_windows', [7, 14, 30])
        
        for metric in metrics:
            if metric not in data.columns:
                continue
            
            metric_values = data[metric]
            
            for window_days in volatility_windows:
                window_size = min(window_days, len(data))
                
                # Standard volatility (coefficient of variation)
                rolling_mean = metric_values.rolling(window=window_size).mean()
                rolling_std = metric_values.rolling(window=window_size).std()
                features[f'{metric}_volatility_{window_days}d'] = (
                    rolling_std / rolling_mean
                ).replace([np.inf, -np.inf], 0).fillna(0)
                
                # Relative range (max - min) / mean
                rolling_min = metric_values.rolling(window=window_size).min()
                rolling_max = metric_values.rolling(window=window_size).max()
                features[f'{metric}_relative_range_{window_days}d'] = (
                    (rolling_max - rolling_min) / rolling_mean
                ).replace([np.inf, -np.inf], 0).fillna(0)
                
                # Stability score (inverse of volatility)
                features[f'{metric}_stability_{window_days}d'] = (
                    1 / (1 + features[f'{metric}_volatility_{window_days}d'])
                )
                
                # Number of direction changes (measure of consistency)
                changes = np.sign(metric_values.diff())
                direction_changes = changes.rolling(window=window_size).apply(
                    lambda x: np.sum(np.diff(x) != 0), raw=True
                )
                features[f'{metric}_direction_changes_{window_days}d'] = direction_changes
                
                # Outlier frequency in rolling window
                if window_size >= 5:
                    def outlier_count(x):
                        if len(x) < 3:
                            return 0
                        q75, q25 = np.percentile(x, [75, 25])
                        iqr = q75 - q25
                        lower_bound = q25 - 1.5 * iqr
                        upper_bound = q75 + 1.5 * iqr
                        return np.sum((x < lower_bound) | (x > upper_bound))
                    
                    features[f'{metric}_outlier_count_{window_days}d'] = metric_values.rolling(
                        window=window_size
                    ).apply(outlier_count, raw=True)
        
        return features
    
    def _extract_rolling_changes(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Extract rolling change and acceleration features.
        
        Args:
            data: Input DataFrame sorted by time
        
        Returns:
            DataFrame with rolling change features
        """
        features = pd.DataFrame(index=data.index)
        metrics = self.config['rolling_metrics']
        change_windows = self.config.get('change_windows', [1, 3, 7])
        
        for metric in metrics:
            if metric not in data.columns:
                continue
            
            metric_values = data[metric]
            
            for window_days in change_windows:
                # Absolute change
                features[f'{metric}_change_{window_days}d'] = (
                    metric_values - metric_values.shift(window_days)
                )
                
                # Percentage change
                features[f'{metric}_pct_change_{window_days}d'] = (
                    metric_values.pct_change(periods=window_days).fillna(0)
                )
                
                # Log change (for positive values)
                positive_mask = (metric_values > 0) & (metric_values.shift(window_days) > 0)
                log_change = np.log(metric_values / metric_values.shift(window_days))
                features[f'{metric}_log_change_{window_days}d'] = np.where(
                    positive_mask, log_change, 0
                )
                
                # Change acceleration (second derivative)
                if window_days >= 2:
                    change_series = features[f'{metric}_change_{window_days}d']
                    features[f'{metric}_acceleration_{window_days}d'] = (
                        change_series - change_series.shift(1)
                    )
                
                # Change magnitude
                features[f'{metric}_change_magnitude_{window_days}d'] = np.abs(
                    features[f'{metric}_change_{window_days}d']
                )
                
                # Change direction
                features[f'{metric}_change_direction_{window_days}d'] = np.sign(
                    features[f'{metric}_change_{window_days}d']
                )
                
                # Cumulative change over window
                features[f'{metric}_cumulative_change_{window_days}d'] = (
                    metric_values.diff().rolling(window=window_days).sum()
                )
        
        return features
    
    def _extract_cross_metric_rolling_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Extract rolling features across multiple metrics.
        
        Args:
            data: Input DataFrame sorted by time
        
        Returns:
            DataFrame with cross-metric rolling features
        """
        features = pd.DataFrame(index=data.index)
        metrics = self.config['rolling_metrics']
        windows = [7, 14, 30]  # Focus on medium-term windows for cross-metric
        
        # Common metric pairs for correlation analysis
        metric_pairs = [
            ('credits_used', 'execution_time_ms'),
            ('credits_used', 'bytes_scanned'),
            ('execution_time_ms', 'bytes_scanned'),
            ('execution_time_ms', 'rows_produced')
        ]
        
        for metric1, metric2 in metric_pairs:
            if metric1 not in data.columns or metric2 not in data.columns:
                continue
            
            for window_days in windows:
                window_size = min(window_days, len(data))
                
                # Rolling correlation
                rolling_corr = data[metric1].rolling(window=window_size).corr(
                    data[metric2]
                )
                features[f'{metric1}_{metric2}_corr_{window_days}d'] = rolling_corr.fillna(0)
                
                # Rolling ratio
                features[f'{metric1}_{metric2}_ratio_{window_days}d'] = (
                    data[metric1].rolling(window=window_size).mean() /
                    data[metric2].rolling(window=window_size).mean()
                ).replace([np.inf, -np.inf], 0).fillna(0)
                
                # Rolling efficiency (for relevant pairs)
                if metric1 == 'credits_used' and metric2 in ['bytes_scanned', 'rows_produced']:
                    # Efficiency: output per credit
                    efficiency = (
                        data[metric2].rolling(window=window_size).mean() /
                        data[metric1].rolling(window=window_size).mean()
                    ).replace([np.inf, -np.inf], 0).fillna(0)
                    features[f'{metric2}_per_credit_{window_days}d'] = efficiency
        
        # Rolling sum ratios (resource utilization)
        if 'credits_used' in data.columns and 'execution_time_ms' in data.columns:
            for window_days in windows:
                window_size = min(window_days, len(data))
                
                total_credits = data['credits_used'].rolling(window=window_size).sum()
                total_time_hours = (
                    data['execution_time_ms'].rolling(window=window_size).sum() / (1000 * 3600)
                )
                
                features[f'credits_per_hour_{window_days}d'] = (
                    total_credits / total_time_hours
                ).replace([np.inf, -np.inf], 0).fillna(0)
        
        return features
    
    def _extract_seasonal_rolling_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Extract seasonal rolling patterns.
        
        Args:
            data: Input DataFrame sorted by time
        
        Returns:
            DataFrame with seasonal rolling features
        """
        features = pd.DataFrame(index=data.index)
        metrics = self.config['rolling_metrics']
        time_col = self.config['time_column']
        
        if time_col not in data.columns:
            return features
        
        time_series = pd.to_datetime(data[time_col])
        
        for metric in metrics:
            if metric not in data.columns:
                continue
            
            metric_values = data[metric]
            
            # Day-of-week rolling patterns
            for dow in range(7):  # 0=Monday, 6=Sunday
                dow_mask = time_series.dt.dayofweek == dow
                if dow_mask.sum() > 0:
                    # Rolling average for this day of week
                    dow_values = metric_values.where(dow_mask)
                    features[f'{metric}_rolling_dow_{dow}_mean'] = (
                        dow_values.rolling(window=len(data), min_periods=1).mean()
                    )
            
            # Hour-of-day rolling patterns
            for hour in [9, 12, 15, 18]:  # Key business hours
                hour_mask = time_series.dt.hour == hour
                if hour_mask.sum() > 0:
                    hour_values = metric_values.where(hour_mask)
                    features[f'{metric}_rolling_hour_{hour}_mean'] = (
                        hour_values.rolling(window=len(data), min_periods=1).mean()
                    )
            
            # Monthly rolling patterns (for datasets spanning months)
            if (time_series.max() - time_series.min()).days > 60:
                for month in time_series.dt.month.unique():
                    month_mask = time_series.dt.month == month
                    if month_mask.sum() > 0:
                        month_values = metric_values.where(month_mask)
                        features[f'{metric}_rolling_month_{month}_mean'] = (
                            month_values.rolling(window=len(data), min_periods=1).mean()
                        )
            
            # Business vs weekend rolling patterns
            business_mask = time_series.dt.dayofweek < 5
            weekend_mask = time_series.dt.dayofweek >= 5
            
            if business_mask.sum() > 0:
                business_values = metric_values.where(business_mask)
                features[f'{metric}_rolling_business_mean'] = (
                    business_values.rolling(window=len(data), min_periods=1).mean()
                )
            
            if weekend_mask.sum() > 0:
                weekend_values = metric_values.where(weekend_mask)
                features[f'{metric}_rolling_weekend_mean'] = (
                    weekend_values.rolling(window=len(data), min_periods=1).mean()
                )
        
        return features
    
    def _extract_lag_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Extract lag features for time series prediction.
        
        Args:
            data: Input DataFrame sorted by time
        
        Returns:
            DataFrame with lag features
        """
        features = pd.DataFrame(index=data.index)
        metrics = self.config['rolling_metrics']
        lag_periods = self.config.get('lag_periods', [1, 2, 3, 7, 14])
        
        for metric in metrics:
            if metric not in data.columns:
                continue
            
            metric_values = data[metric]
            
            for lag in lag_periods:
                # Simple lag
                features[f'{metric}_lag_{lag}'] = metric_values.shift(lag)
                
                # Lag difference
                features[f'{metric}_lag_diff_{lag}'] = (
                    metric_values - metric_values.shift(lag)
                )
                
                # Lag ratio
                features[f'{metric}_lag_ratio_{lag}'] = (
                    metric_values / metric_values.shift(lag)
                ).replace([np.inf, -np.inf], 1).fillna(1)
        
        return features
    
    def _extract_grouped_rolling_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Extract rolling features grouped by categorical variables.
        
        Args:
            data: Input DataFrame sorted by time
        
        Returns:
            DataFrame with grouped rolling features
        """
        features = pd.DataFrame(index=data.index)
        metrics = ['credits_used', 'execution_time_ms']  # Focus on key metrics
        groupby_cols = self.config.get('groupby_columns', [])
        windows = [7, 14, 30]
        
        for group_col in groupby_cols:
            if group_col not in data.columns:
                continue
            
            for metric in metrics:
                if metric not in data.columns:
                    continue
                
                for window_days in windows:
                    try:
                        # Group-wise rolling mean
                        grouped_rolling = data.groupby(group_col)[metric].rolling(
                            window=f'{window_days}D', min_periods=1
                        ).mean().reset_index(level=0, drop=True)
                        
                        features[f'{metric}_group_{group_col}_rolling_{window_days}d'] = (
                            grouped_rolling.reindex(data.index)
                        )
                        
                        # Difference from group average
                        features[f'{metric}_vs_group_{group_col}_{window_days}d'] = (
                            data[metric] - grouped_rolling.reindex(data.index)
                        )
                        
                    except Exception as e:
                        logger.warning(f"Error calculating grouped rolling feature for {group_col}: {str(e)}")
                        continue
        
        return features
    
    def configure_for_snowflake_data(self):
        """
        Configure the generator for typical Snowflake analytics data.
        """
        snowflake_config = {
            'rolling_metrics': [
                'CREDITS_USED', 'EXECUTION_TIME_MS', 'BYTES_SCANNED',
                'ROWS_PRODUCED', 'QUEUE_TIME_MS', 'PARTITIONS_SCANNED'
            ],
            'groupby_columns': ['USER_NAME', 'WAREHOUSE_NAME', 'DATABASE_NAME', 'ROLE_NAME'],
            'rolling_windows': {
                'short': [1, 3, 6, 12],      # Hours
                'medium': [1, 3, 7, 14],     # Days
                'long': [30, 60, 90]         # Days
            },
            'trend_windows': [7, 14, 30, 60],
            'volatility_windows': [7, 14, 30],
            'lag_periods': [1, 3, 7, 14, 30]
        }
        
        self.config.update(snowflake_config)
        logger.info("Configured RollingFeatureGenerator for Snowflake analytics data")
    
    def get_feature_descriptions(self) -> Dict[str, str]:
        """
        Get descriptions of all generated rolling features.
        
        Returns:
            Dictionary mapping feature names to descriptions
        """
        descriptions = {
            # Basic rolling statistics
            'credits_used_rolling_7d_mean': 'Rolling 7-day average of credits used',
            'credits_used_rolling_7d_sum': 'Rolling 7-day sum of credits used',
            'credits_used_rolling_7d_std': 'Rolling 7-day standard deviation of credits used',
            'execution_time_ms_rolling_7d_mean': 'Rolling 7-day average execution time',
            'bytes_scanned_rolling_7d_mean': 'Rolling 7-day average bytes scanned',
            
            # Trend features
            'credits_used_trend_7d': 'Linear trend slope for credits over 7 days',
            'credits_used_trend_direction_7d': 'Trend direction for credits over 7 days',
            'credits_used_momentum_7d': 'Momentum (rate of change) for credits over 7 days',
            'credits_used_macd_14d': 'MACD indicator for credits over 14 days',
            
            # Volatility features
            'credits_used_volatility_7d': 'Coefficient of variation for credits over 7 days',
            'credits_used_stability_7d': 'Stability score (inverse volatility) for credits over 7 days',
            'credits_used_direction_changes_7d': 'Number of direction changes over 7 days',
            
            # Change features
            'credits_used_change_1d': 'Day-over-day change in credits used',
            'credits_used_pct_change_7d': 'Percentage change in credits over 7 days',
            'credits_used_acceleration_3d': 'Acceleration (second derivative) of credits over 3 days',
            
            # Cross-metric features
            'credits_used_execution_time_ms_corr_7d': 'Rolling 7-day correlation between credits and execution time',
            'bytes_scanned_per_credit_7d': 'Rolling 7-day efficiency: bytes scanned per credit',
            'credits_per_hour_7d': 'Rolling 7-day credits consumption rate per hour',
            
            # Lag features
            'credits_used_lag_1': '1-day lag of credits used',
            'credits_used_lag_diff_7': '7-day lag difference in credits used',
            'execution_time_ms_lag_ratio_3': '3-day lag ratio of execution time'
        }
        
        return descriptions
