"""
Cost-based Feature Generator

This module provides comprehensive cost-based feature engineering capabilities
for extracting financial insights from Snowflake analytics cost data.
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from scipy import stats

logger = logging.getLogger(__name__)


class CostFeatureGenerator:
    """
    Generates comprehensive cost-based features from Snowflake analytics data.
    
    This class extracts various cost patterns including credit consumption,
    cost efficiency metrics, budget analysis, and financial optimization
    indicators specifically designed for Snowflake cost analytics.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the CostFeatureGenerator with optional configuration.
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        
        # Default configuration
        self.default_config = {
            'cost_columns': {
                'credits_used': 'CREDITS_USED',
                'credits_used_compute': 'CREDITS_USED_COMPUTE',
                'credits_used_cloud_services': 'CREDITS_USED_CLOUD_SERVICES',
                'total_elapsed_time': 'TOTAL_ELAPSED_TIME_MS',
                'warehouse_size': 'WAREHOUSE_SIZE'
            },
            'usage_columns': {
                'execution_time': 'EXECUTION_TIME_MS',
                'bytes_scanned': 'BYTES_SCANNED',
                'rows_produced': 'ROWS_PRODUCED',
                'partitions_scanned': 'PARTITIONS_SCANNED'
            },
            'user_columns': {
                'user_name': 'USER_NAME',
                'warehouse_name': 'WAREHOUSE_NAME',
                'database_name': 'DATABASE_NAME'
            },
            'time_column': 'START_TIME',
            'credit_rate_usd': 2.0,  # Default credit rate in USD
            'warehouse_credits_per_hour': {
                'X-SMALL': 1, 'SMALL': 2, 'MEDIUM': 4, 'LARGE': 8,
                'X-LARGE': 16, '2X-LARGE': 32, '3X-LARGE': 64, '4X-LARGE': 128
            },
            'cost_categories': {
                'low': (0, 1),      # 0-1 credits
                'medium': (1, 10),   # 1-10 credits
                'high': (10, 100),   # 10-100 credits
                'very_high': (100, float('inf'))  # 100+ credits
            },
            'efficiency_thresholds': {
                'high_efficiency': 0.8,
                'medium_efficiency': 0.5,
                'low_efficiency': 0.2
            },
            'rolling_windows': [1, 3, 7, 30],  # days for rolling calculations
            'percentiles': [25, 50, 75, 90, 95, 99]
        }
        
        # Merge with provided config
        self.config = {**self.default_config, **self.config}
    
    def generate_cost_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate comprehensive cost-based features from the input DataFrame.
        
        Args:
            data: Input DataFrame containing Snowflake cost data
        
        Returns:
            DataFrame with generated cost features
        """
        logger.info("Generating cost-based features...")
        
        feature_df = pd.DataFrame(index=data.index)
        
        # Basic cost metrics
        feature_df.update(self._extract_basic_cost_metrics(data))
        
        # Cost efficiency metrics
        feature_df.update(self._extract_cost_efficiency_metrics(data))
        
        # Cost distribution and statistical features
        feature_df.update(self._extract_cost_distribution_features(data))
        
        # Warehouse cost analysis
        feature_df.update(self._extract_warehouse_cost_features(data))
        
        # User/workload cost patterns
        feature_df.update(self._extract_user_cost_patterns(data))
        
        # Cost optimization indicators
        feature_df.update(self._extract_cost_optimization_features(data))
        
        # Comparative cost analysis
        feature_df.update(self._extract_comparative_cost_features(data))
        
        # Cost forecasting features
        feature_df.update(self._extract_cost_forecasting_features(data))
        
        logger.info(f"Generated {len(feature_df.columns)} cost-based features")
        return feature_df
    
    def _extract_basic_cost_metrics(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Extract basic cost metrics and transformations.
        
        Args:
            data: Input DataFrame
        
        Returns:
            DataFrame with basic cost metrics
        """
        features = pd.DataFrame(index=data.index)
        cost_cols = self.config['cost_columns']
        credit_rate = self.config.get('credit_rate_usd', 2.0)
        
        # Credits used features
        if cost_cols['credits_used'] in data.columns:
            credits = data[cost_cols['credits_used']]
            features['credits_used'] = credits
            features['cost_usd'] = credits * credit_rate
            features['log_credits_used'] = np.log1p(credits)
            features['log_cost_usd'] = np.log1p(features['cost_usd'])
            
            # Cost categories
            cost_categories = self.config.get('cost_categories', {})
            features['cost_category'] = pd.cut(
                credits,
                bins=[cat[0] for cat in cost_categories.values()] + [float('inf')],
                labels=list(cost_categories.keys()),
                include_lowest=True
            ).astype('category')
            
            # Zero cost indicator
            features['is_zero_cost'] = (credits == 0).astype(int)
            
            # High cost indicator (top 10%)
            high_cost_threshold = credits.quantile(0.9)
            features['is_high_cost'] = (credits > high_cost_threshold).astype(int)
        
        # Compute vs cloud services credits
        if cost_cols.get('credits_used_compute') in data.columns:
            compute_credits = data[cost_cols['credits_used_compute']]
            features['credits_used_compute'] = compute_credits
            features['cost_usd_compute'] = compute_credits * credit_rate
            
            if cost_cols.get('credits_used_cloud_services') in data.columns:
                cloud_credits = data[cost_cols['credits_used_cloud_services']]
                features['credits_used_cloud_services'] = cloud_credits
                features['cost_usd_cloud_services'] = cloud_credits * credit_rate
                
                # Compute to cloud services ratio
                total_credits = compute_credits + cloud_credits
                features['compute_credits_ratio'] = np.where(
                    total_credits > 0, compute_credits / total_credits, 0
                )
                features['cloud_services_credits_ratio'] = np.where(
                    total_credits > 0, cloud_credits / total_credits, 0
                )
        
        # Time-based cost metrics
        if cost_cols.get('total_elapsed_time') in data.columns:
            elapsed_time = data[cost_cols['total_elapsed_time']]
            features['total_elapsed_time_ms'] = elapsed_time
            features['total_elapsed_time_hours'] = elapsed_time / (1000 * 3600)
            
            # Cost per hour
            if 'credits_used' in features.columns:
                features['credits_per_hour'] = np.where(
                    features['total_elapsed_time_hours'] > 0,
                    features['credits_used'] / features['total_elapsed_time_hours'],
                    0
                )
                features['cost_per_hour_usd'] = features['credits_per_hour'] * credit_rate
        
        return features
    
    def _extract_cost_efficiency_metrics(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Extract cost efficiency and performance per dollar metrics.
        
        Args:
            data: Input DataFrame
        
        Returns:
            DataFrame with cost efficiency metrics
        """
        features = pd.DataFrame(index=data.index)
        cost_cols = self.config['cost_columns']
        usage_cols = self.config['usage_columns']
        
        credits_col = cost_cols.get('credits_used')
        if credits_col not in data.columns:
            return features
        
        credits = data[credits_col]
        
        # Cost per unit of work metrics
        if usage_cols.get('bytes_scanned') in data.columns:
            bytes_scanned = data[usage_cols['bytes_scanned']]
            
            # Credits per GB scanned
            gb_scanned = bytes_scanned / (1024**3)
            features['credits_per_gb_scanned'] = np.where(
                gb_scanned > 0, credits / gb_scanned, 0
            )
            
            # Cost efficiency: GB per credit
            features['gb_per_credit'] = np.where(
                credits > 0, gb_scanned / credits, 0
            )
        
        if usage_cols.get('rows_produced') in data.columns:
            rows_produced = data[usage_cols['rows_produced']]
            
            # Credits per million rows
            million_rows = rows_produced / 1000000
            features['credits_per_million_rows'] = np.where(
                million_rows > 0, credits / million_rows, 0
            )
            
            # Rows per credit
            features['rows_per_credit'] = np.where(
                credits > 0, rows_produced / credits, 0
            )
        
        if usage_cols.get('execution_time') in data.columns:
            execution_time = data[usage_cols['execution_time']]
            execution_hours = execution_time / (1000 * 3600)
            
            # Credits per execution hour
            features['credits_per_execution_hour'] = np.where(
                execution_hours > 0, credits / execution_hours, 0
            )
            
            # Execution efficiency: execution time per credit
            features['execution_time_per_credit'] = np.where(
                credits > 0, execution_time / credits, 0
            )
        
        # Partitions efficiency
        if usage_cols.get('partitions_scanned') in data.columns:
            partitions = data[usage_cols['partitions_scanned']]
            
            features['credits_per_partition'] = np.where(
                partitions > 0, credits / partitions, 0
            )
            
            features['partitions_per_credit'] = np.where(
                credits > 0, partitions / credits, 0
            )
        
        # Overall cost efficiency score
        efficiency_components = []
        
        # Data efficiency component
        if 'gb_per_credit' in features.columns:
            gb_eff = features['gb_per_credit']
            gb_eff_norm = (gb_eff - gb_eff.quantile(0.1)) / (
                gb_eff.quantile(0.9) - gb_eff.quantile(0.1)
            )
            efficiency_components.append(gb_eff_norm.clip(0, 1))
        
        # Performance efficiency component
        if 'execution_time_per_credit' in features.columns:
            perf_eff = features['execution_time_per_credit']
            # Invert because lower execution time per credit is better
            perf_eff_norm = 1 - (perf_eff - perf_eff.quantile(0.1)) / (
                perf_eff.quantile(0.9) - perf_eff.quantile(0.1)
            )
            efficiency_components.append(perf_eff_norm.clip(0, 1))
        
        if efficiency_components:
            features['cost_efficiency_score'] = np.mean(efficiency_components, axis=0)
            
            # Efficiency categories
            thresholds = self.config.get('efficiency_thresholds', {})
            features['efficiency_category'] = pd.cut(
                features['cost_efficiency_score'],
                bins=[0, thresholds.get('low_efficiency', 0.2),
                     thresholds.get('medium_efficiency', 0.5),
                     thresholds.get('high_efficiency', 0.8), 1],
                labels=['low', 'medium', 'high', 'very_high'],
                include_lowest=True
            ).astype('category')
        
        return features
    
    def _extract_cost_distribution_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Extract cost distribution and statistical features.
        
        Args:
            data: Input DataFrame
        
        Returns:
            DataFrame with cost distribution features
        """
        features = pd.DataFrame(index=data.index)
        cost_cols = self.config['cost_columns']
        percentiles = self.config.get('percentiles', [25, 50, 75, 90, 95])
        
        if cost_cols['credits_used'] not in data.columns:
            return features
        
        credits = data[cost_cols['credits_used']]
        
        # Percentile-based features
        for p in percentiles:
            percentile_val = credits.quantile(p/100)
            features[f'cost_above_p{p}'] = (credits > percentile_val).astype(int)
            features[f'cost_ratio_to_p{p}'] = credits / percentile_val if percentile_val > 0 else 0
        
        # Z-score and outlier detection
        if len(credits) > 1 and credits.std() > 0:
            z_scores = (credits - credits.mean()) / credits.std()
            features['cost_z_score'] = z_scores
            features['cost_is_outlier'] = (np.abs(z_scores) > 3).astype(int)
        
        # Distribution shape
        if len(credits.dropna()) >= 10:
            features['cost_skewness'] = stats.skew(credits.dropna())
            features['cost_kurtosis'] = stats.kurtosis(credits.dropna())
        
        # Relative position in distribution
        features['cost_percentile_rank'] = credits.rank(pct=True)
        
        # Cost concentration (Gini-like coefficient)
        sorted_credits = np.sort(credits.dropna())
        n = len(sorted_credits)
        if n > 0:
            cumsum = np.cumsum(sorted_credits)
            features['cost_concentration'] = (n + 1 - 2 * np.sum(cumsum) / cumsum[-1]) / n
        
        return features
    
    def _extract_warehouse_cost_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Extract warehouse-specific cost analysis features.
        
        Args:
            data: Input DataFrame
        
        Returns:
            DataFrame with warehouse cost features
        """
        features = pd.DataFrame(index=data.index)
        cost_cols = self.config['cost_columns']
        user_cols = self.config['user_columns']
        
        # Warehouse size cost analysis
        if cost_cols.get('warehouse_size') in data.columns:
            wh_size = data[cost_cols['warehouse_size']]
            features['warehouse_size'] = wh_size.astype('category')
            
            # Expected credits per hour by warehouse size
            wh_credits_per_hour = self.config.get('warehouse_credits_per_hour', {})
            features['expected_credits_per_hour'] = wh_size.map(wh_credits_per_hour).fillna(0)
            
            # Warehouse size numeric encoding
            size_mapping = {
                'X-SMALL': 1, 'SMALL': 2, 'MEDIUM': 3, 'LARGE': 4,
                'X-LARGE': 5, '2X-LARGE': 6, '3X-LARGE': 7, '4X-LARGE': 8
            }
            features['warehouse_size_numeric'] = wh_size.map(size_mapping).fillna(0)
            
            # Actual vs expected cost comparison
            if (cost_cols['credits_used'] in data.columns and 
                cost_cols.get('total_elapsed_time') in data.columns):
                
                credits = data[cost_cols['credits_used']]
                elapsed_time = data[cost_cols['total_elapsed_time']]
                elapsed_hours = elapsed_time / (1000 * 3600)
                
                expected_credits = features['expected_credits_per_hour'] * elapsed_hours
                features['actual_vs_expected_credits_ratio'] = np.where(
                    expected_credits > 0, credits / expected_credits, 0
                )
                
                # Cost variance from expected
                features['credits_variance_from_expected'] = credits - expected_credits
                
                # Efficiency relative to warehouse size
                features['warehouse_efficiency'] = np.where(
                    features['actual_vs_expected_credits_ratio'] > 0,
                    1 / features['actual_vs_expected_credits_ratio'],
                    0
                )
        
        # Warehouse usage patterns
        if user_cols.get('warehouse_name') in data.columns:
            wh_name = data[user_cols['warehouse_name']]
            
            # Warehouse usage frequency
            wh_counts = wh_name.value_counts()
            features['warehouse_usage_frequency'] = wh_name.map(wh_counts)
            
            # Is primary warehouse
            most_used_wh = wh_counts.index[0] if len(wh_counts) > 0 else None
            if most_used_wh:
                features['is_primary_warehouse'] = (wh_name == most_used_wh).astype(int)
            
            # Warehouse cost distribution
            if cost_cols['credits_used'] in data.columns:
                credits = data[cost_cols['credits_used']]
                wh_cost_means = credits.groupby(wh_name).mean()
                features['warehouse_avg_cost'] = wh_name.map(wh_cost_means)
                
                # Above warehouse average
                features['above_warehouse_avg_cost'] = (
                    credits > features['warehouse_avg_cost']
                ).astype(int)
        
        return features
    
    def _extract_user_cost_patterns(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Extract user and workload cost pattern features.
        
        Args:
            data: Input DataFrame
        
        Returns:
            DataFrame with user cost pattern features
        """
        features = pd.DataFrame(index=data.index)
        cost_cols = self.config['cost_columns']
        user_cols = self.config['user_columns']
        
        if cost_cols['credits_used'] not in data.columns:
            return features
        
        credits = data[cost_cols['credits_used']]
        
        # User cost patterns
        if user_cols.get('user_name') in data.columns:
            user_name = data[user_cols['user_name']]
            
            # User total and average costs
            user_total_costs = credits.groupby(user_name).sum()
            user_avg_costs = credits.groupby(user_name).mean()
            user_query_counts = user_name.value_counts()
            
            features['user_total_cost'] = user_name.map(user_total_costs)
            features['user_avg_cost'] = user_name.map(user_avg_costs)
            features['user_cost_per_query'] = features['user_total_cost'] / user_name.map(user_query_counts)
            
            # User cost rank
            user_cost_ranks = user_total_costs.rank(method='dense', ascending=False)
            features['user_cost_rank'] = user_name.map(user_cost_ranks)
            
            # High cost user indicator
            high_cost_user_threshold = user_total_costs.quantile(0.8)
            features['is_high_cost_user'] = (
                features['user_total_cost'] >= high_cost_user_threshold
            ).astype(int)
            
            # User cost efficiency
            features['user_cost_vs_avg_ratio'] = credits / features['user_avg_cost']
            features['above_user_avg_cost'] = (
                features['user_cost_vs_avg_ratio'] > 1
            ).astype(int)
        
        # Database cost patterns
        if user_cols.get('database_name') in data.columns:
            db_name = data[user_cols['database_name']]
            
            # Database cost statistics
            db_total_costs = credits.groupby(db_name).sum()
            db_avg_costs = credits.groupby(db_name).mean()
            
            features['database_total_cost'] = db_name.map(db_total_costs)
            features['database_avg_cost'] = db_name.map(db_avg_costs)
            
            # Database cost share
            total_all_costs = credits.sum()
            features['database_cost_share'] = features['database_total_cost'] / total_all_costs
            
            # Above database average
            features['above_database_avg_cost'] = (
                credits > features['database_avg_cost']
            ).astype(int)
        
        return features
    
    def _extract_cost_optimization_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Extract cost optimization and waste indicators.
        
        Args:
            data: Input DataFrame
        
        Returns:
            DataFrame with cost optimization features
        """
        features = pd.DataFrame(index=data.index)
        cost_cols = self.config['cost_columns']
        usage_cols = self.config['usage_columns']
        
        if cost_cols['credits_used'] not in data.columns:
            return features
        
        credits = data[cost_cols['credits_used']]
        
        # Query efficiency indicators
        if usage_cols.get('execution_time') in data.columns:
            exec_time = data[usage_cols['execution_time']]
            
            # Potential optimization opportunities
            # High cost + long execution time
            high_cost = credits > credits.quantile(0.8)
            long_exec = exec_time > exec_time.quantile(0.8)
            features['optimization_candidate'] = (high_cost & long_exec).astype(int)
            
            # Cost vs performance efficiency
            if 'cost_efficiency_score' in features.columns:
                features['needs_optimization'] = (
                    features['cost_efficiency_score'] < 0.3
                ).astype(int)
        
        # Warehouse right-sizing opportunities
        if (cost_cols.get('warehouse_size') in data.columns and 
            'actual_vs_expected_credits_ratio' in features.columns):
            
            # Under-utilized warehouse (actual < 50% of expected)
            features['warehouse_underutilized'] = (
                features['actual_vs_expected_credits_ratio'] < 0.5
            ).astype(int)
            
            # Over-utilized warehouse (actual > 150% of expected)
            features['warehouse_overutilized'] = (
                features['actual_vs_expected_credits_ratio'] > 1.5
            ).astype(int)
        
        # Waste indicators
        # Zero result queries with high cost
        if usage_cols.get('rows_produced') in data.columns:
            rows_produced = data[usage_cols['rows_produced']]
            features['high_cost_zero_results'] = (
                (credits > credits.quantile(0.5)) & (rows_produced == 0)
            ).astype(int)
        
        # Repeated expensive queries (same user, similar cost, short time window)
        if 'user_name' in data.columns and self.config.get('time_column') in data.columns:
            # This would require more complex time-windowed analysis
            # For now, just flag queries above user average as potential repeats
            if 'above_user_avg_cost' in features.columns:
                features['potential_repeated_expensive_query'] = features['above_user_avg_cost']
        
        # Cost anomaly score
        anomaly_components = []
        
        if 'cost_z_score' in features.columns:
            # High z-score indicates cost anomaly
            anomaly_components.append(
                np.abs(features['cost_z_score']).clip(0, 3) / 3
            )
        
        if 'cost_efficiency_score' in features.columns:
            # Low efficiency indicates potential issue
            anomaly_components.append(1 - features['cost_efficiency_score'])
        
        if anomaly_components:
            features['cost_anomaly_score'] = np.mean(anomaly_components, axis=0)
            features['is_cost_anomaly'] = (features['cost_anomaly_score'] > 0.7).astype(int)
        
        return features
    
    def _extract_comparative_cost_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Extract comparative cost analysis features.
        
        Args:
            data: Input DataFrame
        
        Returns:
            DataFrame with comparative cost features
        """
        features = pd.DataFrame(index=data.index)
        cost_cols = self.config['cost_columns']
        
        if cost_cols['credits_used'] not in data.columns:
            return features
        
        credits = data[cost_cols['credits_used']]
        
        # Global cost comparisons
        global_median = credits.median()
        global_mean = credits.mean()
        global_std = credits.std()
        
        features['cost_vs_global_median'] = credits / global_median if global_median > 0 else 0
        features['cost_vs_global_mean'] = credits / global_mean if global_mean > 0 else 0
        
        # Cost relative to distribution
        features['cost_below_median'] = (credits < global_median).astype(int)
        features['cost_below_mean'] = (credits < global_mean).astype(int)
        
        # Distance from typical cost ranges
        q25, q75 = credits.quantile(0.25), credits.quantile(0.75)
        iqr = q75 - q25
        
        features['cost_within_iqr'] = (
            (credits >= q25) & (credits <= q75)
        ).astype(int)
        
        features['cost_above_q75'] = (credits > q75).astype(int)
        features['cost_below_q25'] = (credits < q25).astype(int)
        
        # Cost deviation from median
        features['cost_deviation_from_median'] = np.abs(credits - global_median)
        features['cost_deviation_normalized'] = (
            features['cost_deviation_from_median'] / global_std if global_std > 0 else 0
        )
        
        return features
    
    def _extract_cost_forecasting_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Extract features useful for cost forecasting and budgeting.
        
        Args:
            data: Input DataFrame
        
        Returns:
            DataFrame with cost forecasting features
        """
        features = pd.DataFrame(index=data.index)
        cost_cols = self.config['cost_columns']
        time_col = self.config.get('time_column')
        
        if cost_cols['credits_used'] not in data.columns or time_col not in data.columns:
            return features
        
        credits = data[cost_cols['credits_used']]
        time_series = pd.to_datetime(data[time_col])
        
        # Time-based cost patterns
        features['hour_of_day'] = time_series.dt.hour
        features['day_of_week'] = time_series.dt.dayofweek
        features['day_of_month'] = time_series.dt.day
        features['month_of_year'] = time_series.dt.month
        
        # Cumulative cost trends
        cumulative_cost = credits.cumsum()
        features['cumulative_cost'] = cumulative_cost
        
        # Moving averages for trend analysis
        if len(data) >= 7:
            features['cost_ma_7'] = credits.rolling(window=7, min_periods=1).mean()
        if len(data) >= 30:
            features['cost_ma_30'] = credits.rolling(window=30, min_periods=1).mean()
        
        # Cost velocity (rate of change)
        if len(data) >= 2:
            features['cost_velocity'] = credits.diff().fillna(0)
            features['cost_acceleration'] = features['cost_velocity'].diff().fillna(0)
        
        # Seasonal cost patterns
        # Day of week cost patterns
        dow_costs = credits.groupby(time_series.dt.dayofweek).mean()
        features['typical_dow_cost'] = time_series.dt.dayofweek.map(dow_costs)
        features['cost_vs_typical_dow'] = credits / features['typical_dow_cost']
        
        # Hour of day cost patterns
        hour_costs = credits.groupby(time_series.dt.hour).mean()
        features['typical_hour_cost'] = time_series.dt.hour.map(hour_costs)
        features['cost_vs_typical_hour'] = credits / features['typical_hour_cost']
        
        # Budget period indicators
        features['is_month_start'] = time_series.dt.is_month_start.astype(int)
        features['is_month_end'] = time_series.dt.is_month_end.astype(int)
        features['is_quarter_start'] = time_series.dt.is_quarter_start.astype(int)
        features['is_quarter_end'] = time_series.dt.is_quarter_end.astype(int)
        
        # Days since start of month/quarter (for budget tracking)
        features['days_since_month_start'] = (
            time_series - time_series.dt.to_period('M').dt.start_time
        ).dt.days
        
        features['days_since_quarter_start'] = (
            time_series - time_series.dt.to_period('Q').dt.start_time
        ).dt.days
        
        return features
    
    def configure_for_snowflake_data(self):
        """
        Configure the generator for typical Snowflake analytics data.
        """
        snowflake_config = {
            'cost_columns': {
                'credits_used': 'CREDITS_USED',
                'credits_used_compute': 'CREDITS_USED_COMPUTE',
                'credits_used_cloud_services': 'CREDITS_USED_CLOUD_SERVICES',
                'total_elapsed_time': 'TOTAL_ELAPSED_TIME_MS',
                'warehouse_size': 'WAREHOUSE_SIZE'
            },
            'credit_rate_usd': 2.0,  # Standard Snowflake credit rate
            'cost_categories': {
                'micro': (0, 0.1),
                'low': (0.1, 1),
                'medium': (1, 10),
                'high': (10, 100),
                'very_high': (100, float('inf'))
            },
            'efficiency_thresholds': {
                'high_efficiency': 0.75,
                'medium_efficiency': 0.45,
                'low_efficiency': 0.2
            }
        }
        
        self.config.update(snowflake_config)
        logger.info("Configured CostFeatureGenerator for Snowflake analytics data")
    
    def get_feature_descriptions(self) -> Dict[str, str]:
        """
        Get descriptions of all generated cost features.
        
        Returns:
            Dictionary mapping feature names to descriptions
        """
        descriptions = {
            # Basic cost metrics
            'credits_used': 'Total Snowflake credits consumed',
            'cost_usd': 'Total cost in USD',
            'log_credits_used': 'Log-transformed credits used',
            'cost_category': 'Cost category (micro/low/medium/high/very_high)',
            'is_zero_cost': 'Whether the query had zero cost',
            'is_high_cost': 'Whether the cost is in top 10%',
            
            # Efficiency metrics
            'credits_per_gb_scanned': 'Credits consumed per gigabyte scanned',
            'gb_per_credit': 'Gigabytes processed per credit (efficiency)',
            'credits_per_million_rows': 'Credits per million rows produced',
            'cost_efficiency_score': 'Overall cost efficiency score (0-1)',
            'efficiency_category': 'Efficiency level (low/medium/high/very_high)',
            
            # Warehouse metrics
            'warehouse_size_numeric': 'Warehouse size as numeric value',
            'actual_vs_expected_credits_ratio': 'Actual vs expected credits ratio',
            'warehouse_efficiency': 'Warehouse utilization efficiency',
            'warehouse_underutilized': 'Whether warehouse is underutilized',
            'warehouse_overutilized': 'Whether warehouse is overutilized',
            
            # User patterns
            'user_total_cost': 'Total cost for this user',
            'user_avg_cost': 'Average cost per query for this user',
            'user_cost_rank': 'User rank by total cost',
            'is_high_cost_user': 'Whether user is in top 20% by cost',
            
            # Optimization indicators
            'optimization_candidate': 'Query candidate for optimization',
            'needs_optimization': 'Low efficiency indicator',
            'cost_anomaly_score': 'Cost anomaly detection score',
            'is_cost_anomaly': 'Whether query is a cost anomaly'
        }
        
        return descriptions
