"""
Cost Aggregator

This module provides comprehensive cost aggregation and analysis capabilities for
Snowflake analytics data, focusing on credit usage, cost optimization, and billing insights.
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Any
from datetime import datetime, timedelta
import warnings

logger = logging.getLogger(__name__)


class CostAggregator:
    """
    Handles cost-focused aggregation and analysis of Snowflake analytics data.
    
    This class provides methods for analyzing credit usage patterns, cost efficiency,
    budget tracking, cost allocation, and identifying optimization opportunities.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the CostAggregator with optional configuration.
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        
        # Default configuration
        self.default_config = {
            'credit_cost_per_unit': 2.0,  # Default cost per credit in USD
            'currency': 'USD',
            'cost_allocation_dimensions': [
                'USER_NAME', 'WAREHOUSE_NAME', 'DATABASE_NAME', 
                'ROLE_NAME', 'QUERY_TYPE'
            ],
            'efficiency_metrics': [
                'credits_per_query', 'credits_per_row', 'credits_per_byte',
                'cost_per_query', 'cost_per_row', 'cost_per_byte'
            ],
            'time_aggregations': ['hourly', 'daily', 'weekly', 'monthly'],
            'percentiles': [50, 75, 90, 95, 99],
            'budget_periods': {
                'daily_budget': 1000.0,
                'weekly_budget': 7000.0,
                'monthly_budget': 30000.0,
                'quarterly_budget': 90000.0
            },
            'cost_thresholds': {
                'low_cost_query': 0.01,  # Credits
                'medium_cost_query': 0.1,
                'high_cost_query': 1.0,
                'very_high_cost_query': 10.0
            },
            'anomaly_detection': {
                'enabled': True,
                'cost_spike_threshold': 3.0,  # Standard deviations
                'usage_spike_threshold': 3.0,
                'efficiency_drop_threshold': 0.5  # Proportion
            },
            'optimization_analysis': {
                'enabled': True,
                'idle_warehouse_threshold': 0.1,  # Credits per hour
                'underutilized_threshold': 0.25,  # Utilization percentage
                'oversized_query_threshold': 2.0  # Times expected cost
            },
            'include_forecasting_features': True,
            'include_cost_allocation': True,
            'include_efficiency_analysis': True,
            'include_anomaly_detection': True,
            'include_optimization_recommendations': True
        }
        
        # Merge with provided config
        self.config = {**self.default_config, **self.config}
    
    def aggregate_cost_metrics(self, 
                             data: pd.DataFrame,
                             time_column: str = 'START_TIME',
                             credit_column: str = 'CREDITS_USED') -> pd.DataFrame:
        """
        Aggregate comprehensive cost metrics from Snowflake usage data.
        
        Args:
            data: Input DataFrame with Snowflake usage data
            time_column: Name of the timestamp column
            credit_column: Name of the credits used column
        
        Returns:
            DataFrame with comprehensive cost aggregations
        """
        logger.info("Aggregating comprehensive cost metrics...")
        
        if credit_column not in data.columns:
            raise ValueError(f"Credit column {credit_column} not found in data")
        
        if time_column not in data.columns:
            raise ValueError(f"Time column {time_column} not found in data")
        
        # Prepare data
        data = data.copy()
        data[time_column] = pd.to_datetime(data[time_column])
        
        # Calculate cost in currency
        credit_cost = self.config.get('credit_cost_per_unit', 2.0)
        data['cost_amount'] = data[credit_column] * credit_cost
        
        # Basic cost aggregations
        result = self._create_basic_cost_aggregations(data, time_column, credit_column)
        
        # Add cost allocation analysis
        if self.config.get('include_cost_allocation', True):
            result = self._add_cost_allocation_analysis(result, data)
        
        # Add efficiency analysis
        if self.config.get('include_efficiency_analysis', True):
            result = self._add_efficiency_analysis(result, data)
        
        # Add anomaly detection
        if self.config.get('include_anomaly_detection', True):
            result = self._add_anomaly_detection(result)
        
        # Add optimization recommendations
        if self.config.get('include_optimization_recommendations', True):
            result = self._add_optimization_features(result, data)
        
        # Add forecasting features
        if self.config.get('include_forecasting_features', True):
            result = self._add_forecasting_features(result)
        
        logger.info(f"Generated cost aggregation with {len(result)} records and {len(result.columns)} metrics")
        return result
    
    def analyze_cost_by_dimension(self, 
                                data: pd.DataFrame,
                                dimension: str,
                                credit_column: str = 'CREDITS_USED') -> pd.DataFrame:
        """
        Analyze cost patterns by specific dimension.
        
        Args:
            data: Input DataFrame
            dimension: Dimension to analyze (e.g., USER_NAME, WAREHOUSE_NAME)
            credit_column: Name of the credits used column
        
        Returns:
            DataFrame with dimension-specific cost analysis
        """
        logger.info(f"Analyzing cost patterns by dimension: {dimension}")
        
        if dimension not in data.columns:
            raise ValueError(f"Dimension {dimension} not found in data")
        
        if credit_column not in data.columns:
            raise ValueError(f"Credit column {credit_column} not found in data")
        
        # Prepare data
        data = data.copy()
        credit_cost = self.config.get('credit_cost_per_unit', 2.0)
        data['cost_amount'] = data[credit_column] * credit_cost
        
        # Aggregate by dimension
        agg_funcs = {
            credit_column: ['sum', 'mean', 'median', 'std', 'min', 'max', 'count'],
            'cost_amount': ['sum', 'mean', 'median', 'std', 'min', 'max']
        }
        
        # Add other numeric columns if available
        numeric_cols = ['EXECUTION_TIME_MS', 'BYTES_SCANNED', 'ROWS_PRODUCED']
        for col in numeric_cols:
            if col in data.columns:
                agg_funcs[col] = ['sum', 'mean', 'median']
        
        result = data.groupby(dimension).agg(agg_funcs)
        
        # Flatten column names
        result.columns = [f"{col[0]}_{col[1]}" for col in result.columns]
        result = result.reset_index()
        
        # Add cost analysis features
        result = self._add_dimensional_cost_features(result, dimension)
        
        # Add rankings and percentiles
        result = self._add_cost_rankings(result, credit_column)
        
        # Add efficiency metrics
        result = self._add_dimensional_efficiency_metrics(result)
        
        return result
    
    def create_cost_time_series(self, 
                              data: pd.DataFrame,
                              time_column: str = 'START_TIME',
                              credit_column: str = 'CREDITS_USED',
                              frequency: str = 'daily') -> pd.DataFrame:
        """
        Create time series analysis of cost patterns.
        
        Args:
            data: Input DataFrame
            time_column: Name of the timestamp column
            credit_column: Name of the credits used column
            frequency: Time aggregation frequency ('hourly', 'daily', 'weekly', 'monthly')
        
        Returns:
            DataFrame with time series cost analysis
        """
        logger.info(f"Creating {frequency} cost time series...")
        
        data = data.copy()
        data[time_column] = pd.to_datetime(data[time_column])
        credit_cost = self.config.get('credit_cost_per_unit', 2.0)
        data['cost_amount'] = data[credit_column] * credit_cost
        
        # Create time grouping
        if frequency == 'hourly':
            data['time_group'] = data[time_column].dt.floor('H')
        elif frequency == 'daily':
            data['time_group'] = data[time_column].dt.date
        elif frequency == 'weekly':
            data['time_group'] = data[time_column].dt.to_period('W').dt.start_time
        elif frequency == 'monthly':
            data['time_group'] = data[time_column].dt.to_period('M').dt.start_time
        else:
            raise ValueError(f"Unsupported frequency: {frequency}")
        
        # Aggregate by time
        time_series = data.groupby('time_group').agg({
            credit_column: ['sum', 'mean', 'count'],
            'cost_amount': ['sum', 'mean'],
            'EXECUTION_TIME_MS': ['sum', 'mean'] if 'EXECUTION_TIME_MS' in data.columns else [],
            'BYTES_SCANNED': ['sum', 'mean'] if 'BYTES_SCANNED' in data.columns else [],
            'ROWS_PRODUCED': ['sum', 'mean'] if 'ROWS_PRODUCED' in data.columns else []
        })
        
        # Flatten columns
        time_series.columns = [f"{col[0]}_{col[1]}" for col in time_series.columns if col[1]]
        time_series = time_series.reset_index()
        
        # Add time series features
        time_series = self._add_time_series_cost_features(time_series, frequency)
        
        # Add budget tracking
        time_series = self._add_budget_tracking(time_series, frequency)
        
        # Add trend analysis
        time_series = self._add_cost_trend_analysis(time_series)
        
        return time_series
    
    def identify_cost_optimization_opportunities(self, 
                                               data: pd.DataFrame,
                                               credit_column: str = 'CREDITS_USED') -> Dict[str, Any]:
        """
        Identify specific cost optimization opportunities.
        
        Args:
            data: Input DataFrame
            credit_column: Name of the credits used column
        
        Returns:
            Dictionary with optimization recommendations
        """
        logger.info("Identifying cost optimization opportunities...")
        
        optimization_config = self.config.get('optimization_analysis', {})
        if not optimization_config.get('enabled', True):
            return {}
        
        data = data.copy()
        credit_cost = self.config.get('credit_cost_per_unit', 2.0)
        data['cost_amount'] = data[credit_column] * credit_cost
        
        opportunities = {
            'high_cost_queries': self._identify_high_cost_queries(data, credit_column),
            'inefficient_users': self._identify_inefficient_users(data, credit_column),
            'underutilized_warehouses': self._identify_underutilized_warehouses(data, credit_column),
            'cost_spikes': self._identify_cost_spikes(data, credit_column),
            'idle_time_analysis': self._analyze_idle_time(data),
            'query_optimization_candidates': self._identify_query_optimization_candidates(data, credit_column),
            'warehouse_sizing_recommendations': self._analyze_warehouse_sizing(data, credit_column)
        }
        
        return opportunities
    
    def _create_basic_cost_aggregations(self, 
                                      data: pd.DataFrame, 
                                      time_column: str, 
                                      credit_column: str) -> pd.DataFrame:
        """
        Create basic cost aggregation metrics.
        
        Args:
            data: Input DataFrame
            time_column: Time column name
            credit_column: Credit column name
        
        Returns:
            DataFrame with basic aggregations
        """
        # Overall statistics
        total_credits = data[credit_column].sum()
        total_cost = data['cost_amount'].sum()
        total_queries = len(data)
        
        result = pd.DataFrame([{
            'total_credits_used': total_credits,
            'total_cost_amount': total_cost,
            'total_queries': total_queries,
            'avg_credits_per_query': total_credits / max(total_queries, 1),
            'avg_cost_per_query': total_cost / max(total_queries, 1),
            'currency': self.config.get('currency', 'USD'),
            'analysis_period_start': data[time_column].min(),
            'analysis_period_end': data[time_column].max(),
            'analysis_duration_hours': (data[time_column].max() - data[time_column].min()).total_seconds() / 3600
        }])
        
        # Add percentile analysis
        percentiles = self.config.get('percentiles', [50, 75, 90, 95, 99])
        for p in percentiles:
            result[f'credits_p{p}'] = data[credit_column].quantile(p/100)
            result[f'cost_p{p}'] = data['cost_amount'].quantile(p/100)
        
        # Add cost distribution
        thresholds = self.config.get('cost_thresholds', {})
        for threshold_name, threshold_value in thresholds.items():
            count = (data[credit_column] >= threshold_value).sum()
            result[f'{threshold_name}_count'] = count
            result[f'{threshold_name}_percentage'] = count / max(total_queries, 1) * 100
        
        return result
    
    def _add_cost_allocation_analysis(self, result: pd.DataFrame, data: pd.DataFrame) -> pd.DataFrame:
        """
        Add cost allocation analysis across dimensions.
        
        Args:
            result: Current result DataFrame
            data: Original data
        
        Returns:
            Updated result DataFrame
        """
        allocation_dims = self.config.get('cost_allocation_dimensions', [])
        available_dims = [dim for dim in allocation_dims if dim in data.columns]
        
        for dim in available_dims:
            # Top cost consumers
            top_consumers = data.groupby(dim)['cost_amount'].sum().nlargest(10)
            
            # Add as separate columns
            for i, (consumer, cost) in enumerate(top_consumers.items()):
                result[f'top_cost_{dim}_{i+1}_name'] = consumer
                result[f'top_cost_{dim}_{i+1}_amount'] = cost
                result[f'top_cost_{dim}_{i+1}_percentage'] = cost / data['cost_amount'].sum() * 100
        
        # Calculate Gini coefficient for cost inequality
        for dim in available_dims:
            dim_costs = data.groupby(dim)['cost_amount'].sum()
            result[f'{dim}_cost_gini'] = self._calculate_gini_coefficient(dim_costs.values)
        
        return result
    
    def _add_efficiency_analysis(self, result: pd.DataFrame, data: pd.DataFrame) -> pd.DataFrame:
        """
        Add efficiency analysis metrics.
        
        Args:
            result: Current result DataFrame
            data: Original data
        
        Returns:
            Updated result DataFrame
        """
        # Overall efficiency metrics
        if 'ROWS_PRODUCED' in data.columns:
            total_rows = data['ROWS_PRODUCED'].sum()
            result['credits_per_row'] = data['CREDITS_USED'].sum() / max(total_rows, 1)
            result['cost_per_row'] = data['cost_amount'].sum() / max(total_rows, 1)
        
        if 'BYTES_SCANNED' in data.columns:
            total_bytes = data['BYTES_SCANNED'].sum()
            result['credits_per_gb'] = data['CREDITS_USED'].sum() / max(total_bytes / 1e9, 1)
            result['cost_per_gb'] = data['cost_amount'].sum() / max(total_bytes / 1e9, 1)
        
        if 'EXECUTION_TIME_MS' in data.columns:
            total_time_hours = data['EXECUTION_TIME_MS'].sum() / (1000 * 3600)
            result['credits_per_hour'] = data['CREDITS_USED'].sum() / max(total_time_hours, 1)
            result['cost_per_hour'] = data['cost_amount'].sum() / max(total_time_hours, 1)
        
        # Efficiency rankings by different metrics
        efficiency_metrics = self.config.get('efficiency_metrics', [])
        for metric in efficiency_metrics:
            if metric in ['credits_per_query', 'cost_per_query']:
                values = data['CREDITS_USED'] if 'credits' in metric else data['cost_amount']
                result[f'{metric}_median'] = values.median()
                result[f'{metric}_p90'] = values.quantile(0.9)
                result[f'{metric}_std'] = values.std()
        
        return result
    
    def _add_anomaly_detection(self, result: pd.DataFrame) -> pd.DataFrame:
        """
        Add anomaly detection features.
        
        Args:
            result: Current result DataFrame
        
        Returns:
            Updated result DataFrame
        """
        anomaly_config = self.config.get('anomaly_detection', {})
        if not anomaly_config.get('enabled', True):
            return result
        
        # Cost spike threshold
        cost_threshold = anomaly_config.get('cost_spike_threshold', 3.0)
        usage_threshold = anomaly_config.get('usage_spike_threshold', 3.0)
        
        # Add anomaly indicators (will be calculated in time series analysis)
        result['cost_spike_threshold'] = cost_threshold
        result['usage_spike_threshold'] = usage_threshold
        result['anomaly_detection_enabled'] = True
        
        return result
    
    def _add_optimization_features(self, result: pd.DataFrame, data: pd.DataFrame) -> pd.DataFrame:
        """
        Add optimization opportunity features.
        
        Args:
            result: Current result DataFrame
            data: Original data
        
        Returns:
            Updated result DataFrame
        """
        # Potential savings calculations
        total_cost = data['cost_amount'].sum()
        
        # Estimate savings from query optimization (top 10% of costly queries)
        costly_queries_threshold = data['cost_amount'].quantile(0.9)
        costly_queries_cost = data[data['cost_amount'] >= costly_queries_threshold]['cost_amount'].sum()
        result['potential_savings_query_opt'] = costly_queries_cost * 0.3  # Assume 30% savings
        result['potential_savings_query_opt_percentage'] = (costly_queries_cost * 0.3) / total_cost * 100
        
        # Estimate savings from warehouse optimization
        if 'WAREHOUSE_NAME' in data.columns:
            warehouse_costs = data.groupby('WAREHOUSE_NAME')['cost_amount'].sum()
            underutilized_threshold = warehouse_costs.median() * 0.5
            underutilized_cost = warehouse_costs[warehouse_costs < underutilized_threshold].sum()
            result['potential_savings_warehouse_opt'] = underutilized_cost * 0.2  # Assume 20% savings
            result['potential_savings_warehouse_opt_percentage'] = (underutilized_cost * 0.2) / total_cost * 100
        
        return result
    
    def _add_forecasting_features(self, result: pd.DataFrame) -> pd.DataFrame:
        """
        Add features useful for cost forecasting.
        
        Args:
            result: Current result DataFrame
        
        Returns:
            Updated result DataFrame
        """
        # Add seasonal indicators and trend features
        # These will be expanded in time series analysis
        result['forecasting_features_available'] = True
        
        # Add budget comparison features
        budgets = self.config.get('budget_periods', {})
        for period, budget in budgets.items():
            result[f'{period}_target'] = budget
        
        return result
    
    def _add_dimensional_cost_features(self, result: pd.DataFrame, dimension: str) -> pd.DataFrame:
        """
        Add dimension-specific cost features.
        
        Args:
            result: Current result DataFrame
            dimension: Dimension being analyzed
        
        Returns:
            Updated result DataFrame
        """
        # Calculate share of total cost
        total_cost = result['cost_amount_sum'].sum()
        result['cost_share_percentage'] = result['cost_amount_sum'] / total_cost * 100
        
        # Calculate cumulative cost share
        result_sorted = result.sort_values('cost_amount_sum', ascending=False)
        result_sorted['cumulative_cost_share'] = result_sorted['cost_share_percentage'].cumsum()
        
        # Merge back to original order
        result = result.merge(
            result_sorted[[dimension, 'cumulative_cost_share']], 
            on=dimension, 
            how='left'
        )
        
        # Pareto analysis (80/20 rule)
        result['is_top_20_percent'] = (result['cumulative_cost_share'] <= 20).astype(int)
        result['is_pareto_contributor'] = (result['cumulative_cost_share'] <= 80).astype(int)
        
        return result
    
    def _add_cost_rankings(self, result: pd.DataFrame, credit_column: str) -> pd.DataFrame:
        """
        Add cost-based rankings and percentiles.
        
        Args:
            result: Current result DataFrame
            credit_column: Credit column name
        
        Returns:
            Updated result DataFrame
        """
        # Cost rankings
        result['cost_rank'] = result['cost_amount_sum'].rank(method='dense', ascending=False)
        result['cost_percentile'] = result['cost_amount_sum'].rank(pct=True)
        
        # Credit rankings
        result['credits_rank'] = result[f'{credit_column}_sum'].rank(method='dense', ascending=False)
        result['credits_percentile'] = result[f'{credit_column}_sum'].rank(pct=True)
        
        # Query volume rankings
        if f'{credit_column}_count' in result.columns:
            result['query_volume_rank'] = result[f'{credit_column}_count'].rank(method='dense', ascending=False)
            result['query_volume_percentile'] = result[f'{credit_column}_count'].rank(pct=True)
        
        return result
    
    def _add_dimensional_efficiency_metrics(self, result: pd.DataFrame) -> pd.DataFrame:
        """
        Add efficiency metrics for dimensional analysis.
        
        Args:
            result: Current result DataFrame
        
        Returns:
            Updated result DataFrame
        """
        # Credits per query efficiency
        if 'CREDITS_USED_count' in result.columns:
            result['credits_per_query_efficiency'] = result['CREDITS_USED_sum'] / result['CREDITS_USED_count']
        
        # Cost per query efficiency
        if 'CREDITS_USED_count' in result.columns:
            result['cost_per_query_efficiency'] = result['cost_amount_sum'] / result['CREDITS_USED_count']
        
        # Data processing efficiency
        if 'ROWS_PRODUCED_sum' in result.columns:
            result['credits_per_row_efficiency'] = result['CREDITS_USED_sum'] / (result['ROWS_PRODUCED_sum'] + 1)
            result['cost_per_row_efficiency'] = result['cost_amount_sum'] / (result['ROWS_PRODUCED_sum'] + 1)
        
        if 'BYTES_SCANNED_sum' in result.columns:
            result['credits_per_gb_efficiency'] = result['CREDITS_USED_sum'] / (result['BYTES_SCANNED_sum'] / 1e9 + 1)
            result['cost_per_gb_efficiency'] = result['cost_amount_sum'] / (result['BYTES_SCANNED_sum'] / 1e9 + 1)
        
        return result
    
    def _add_time_series_cost_features(self, result: pd.DataFrame, frequency: str) -> pd.DataFrame:
        """
        Add time series specific cost features.
        
        Args:
            result: Current result DataFrame
            frequency: Time frequency
        
        Returns:
            Updated result DataFrame
        """
        result = result.sort_values('time_group')
        
        # Period-over-period changes
        result['cost_change'] = result['cost_amount_sum'].diff()
        result['cost_pct_change'] = result['cost_amount_sum'].pct_change()
        result['credits_change'] = result['CREDITS_USED_sum'].diff()
        result['credits_pct_change'] = result['CREDITS_USED_sum'].pct_change()
        
        # Moving averages
        windows = self._get_appropriate_windows(frequency)
        for window in windows:
            if len(result) >= window:
                result[f'cost_ma_{window}'] = result['cost_amount_sum'].rolling(window=window).mean()
                result[f'credits_ma_{window}'] = result['CREDITS_USED_sum'].rolling(window=window).mean()
        
        # Volatility measures
        if len(result) >= 5:
            result['cost_volatility'] = result['cost_amount_sum'].rolling(window=5).std()
            result['credits_volatility'] = result['CREDITS_USED_sum'].rolling(window=5).std()
        
        return result
    
    def _add_budget_tracking(self, result: pd.DataFrame, frequency: str) -> pd.DataFrame:
        """
        Add budget tracking features.
        
        Args:
            result: Current result DataFrame
            frequency: Time frequency
        
        Returns:
            Updated result DataFrame
        """
        budgets = self.config.get('budget_periods', {})
        budget_key = f'{frequency}_budget'
        
        if budget_key in budgets:
            budget_amount = budgets[budget_key]
            result['budget_target'] = budget_amount
            result['budget_actual'] = result['cost_amount_sum']
            result['budget_variance'] = result['cost_amount_sum'] - budget_amount
            result['budget_variance_pct'] = (result['cost_amount_sum'] - budget_amount) / budget_amount * 100
            result['is_over_budget'] = (result['cost_amount_sum'] > budget_amount).astype(int)
            
            # Running budget tracking
            result['cumulative_cost'] = result['cost_amount_sum'].cumsum()
            result['cumulative_budget'] = budget_amount * (result.index + 1)
            result['cumulative_variance'] = result['cumulative_cost'] - result['cumulative_budget']
        
        return result
    
    def _add_cost_trend_analysis(self, result: pd.DataFrame) -> pd.DataFrame:
        """
        Add cost trend analysis features.
        
        Args:
            result: Current result DataFrame
        
        Returns:
            Updated result DataFrame
        """
        if len(result) < 3:
            return result
        
        # Simple trend calculation
        x = np.arange(len(result))
        cost_trend = np.polyfit(x, result['cost_amount_sum'], 1)[0]
        credits_trend = np.polyfit(x, result['CREDITS_USED_sum'], 1)[0]
        
        result['cost_trend_slope'] = cost_trend
        result['credits_trend_slope'] = credits_trend
        result['cost_trend_direction'] = np.sign(cost_trend)
        result['credits_trend_direction'] = np.sign(credits_trend)
        
        # Rolling trend
        if len(result) >= 5:
            window = min(5, len(result))
            rolling_cost_trend = result['cost_amount_sum'].rolling(window=window).apply(
                lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) > 1 else 0, raw=True
            )
            result['cost_rolling_trend'] = rolling_cost_trend
        
        return result
    
    def _identify_high_cost_queries(self, data: pd.DataFrame, credit_column: str) -> Dict[str, Any]:
        """
        Identify high-cost queries for optimization.
        
        Args:
            data: Input DataFrame
            credit_column: Credit column name
        
        Returns:
            Dictionary with high-cost query analysis
        """
        threshold = data[credit_column].quantile(0.95)  # Top 5% of queries
        high_cost_queries = data[data[credit_column] >= threshold]
        
        return {
            'threshold_credits': threshold,
            'count': len(high_cost_queries),
            'total_cost': high_cost_queries['cost_amount'].sum(),
            'percentage_of_total_cost': high_cost_queries['cost_amount'].sum() / data['cost_amount'].sum() * 100,
            'avg_cost': high_cost_queries['cost_amount'].mean(),
            'top_patterns': self._identify_query_patterns(high_cost_queries)
        }
    
    def _identify_inefficient_users(self, data: pd.DataFrame, credit_column: str) -> Dict[str, Any]:
        """
        Identify users with inefficient cost patterns.
        
        Args:
            data: Input DataFrame
            credit_column: Credit column name
        
        Returns:
            Dictionary with inefficient user analysis
        """
        if 'USER_NAME' not in data.columns:
            return {}
        
        user_efficiency = data.groupby('USER_NAME').agg({
            credit_column: ['sum', 'mean', 'count'],
            'cost_amount': ['sum', 'mean'],
            'ROWS_PRODUCED': 'sum' if 'ROWS_PRODUCED' in data.columns else lambda x: 0
        })
        
        user_efficiency.columns = [f"{col[0]}_{col[1]}" for col in user_efficiency.columns]
        
        # Calculate efficiency metrics
        if f'ROWS_PRODUCED_sum' in user_efficiency.columns:
            user_efficiency['credits_per_row'] = user_efficiency[f'{credit_column}_sum'] / (user_efficiency['ROWS_PRODUCED_sum'] + 1)
        
        # Identify inefficient users (top 25% by credits per query)
        inefficient_threshold = user_efficiency[f'{credit_column}_mean'].quantile(0.75)
        inefficient_users = user_efficiency[user_efficiency[f'{credit_column}_mean'] >= inefficient_threshold]
        
        return {
            'threshold_credits_per_query': inefficient_threshold,
            'count': len(inefficient_users),
            'total_cost': inefficient_users['cost_amount_sum'].sum(),
            'users': inefficient_users.index.tolist()[:10]  # Top 10
        }
    
    def _identify_underutilized_warehouses(self, data: pd.DataFrame, credit_column: str) -> Dict[str, Any]:
        """
        Identify underutilized warehouses.
        
        Args:
            data: Input DataFrame
            credit_column: Credit column name
        
        Returns:
            Dictionary with underutilized warehouse analysis
        """
        if 'WAREHOUSE_NAME' not in data.columns:
            return {}
        
        warehouse_utilization = data.groupby('WAREHOUSE_NAME').agg({
            credit_column: ['sum', 'mean', 'count'],
            'cost_amount': 'sum',
            'START_TIME': ['min', 'max']
        })
        
        warehouse_utilization.columns = [f"{col[0]}_{col[1]}" for col in warehouse_utilization.columns]
        
        # Calculate utilization metrics
        for warehouse in warehouse_utilization.index:
            warehouse_data = data[data['WAREHOUSE_NAME'] == warehouse]
            time_span = (warehouse_data['START_TIME'].max() - warehouse_data['START_TIME'].min()).total_seconds() / 3600
            credits_per_hour = warehouse_utilization.loc[warehouse, f'{credit_column}_sum'] / max(time_span, 1)
            warehouse_utilization.loc[warehouse, 'credits_per_hour'] = credits_per_hour
        
        # Identify underutilized warehouses
        underutilized_threshold = self.config.get('optimization_analysis', {}).get('underutilized_threshold', 0.25)
        median_utilization = warehouse_utilization['credits_per_hour'].median()
        threshold = median_utilization * underutilized_threshold
        
        underutilized = warehouse_utilization[warehouse_utilization['credits_per_hour'] < threshold]
        
        return {
            'threshold_credits_per_hour': threshold,
            'count': len(underutilized),
            'warehouses': underutilized.index.tolist(),
            'potential_savings': underutilized['cost_amount_sum'].sum() * 0.3  # Assume 30% savings
        }
    
    def _identify_cost_spikes(self, data: pd.DataFrame, credit_column: str) -> Dict[str, Any]:
        """
        Identify cost spikes in the data.
        
        Args:
            data: Input DataFrame
            credit_column: Credit column name
        
        Returns:
            Dictionary with cost spike analysis
        """
        # Daily aggregation for spike detection
        data['date'] = data['START_TIME'].dt.date
        daily_costs = data.groupby('date')[credit_column].sum()
        
        # Identify spikes using statistical methods
        mean_cost = daily_costs.mean()
        std_cost = daily_costs.std()
        threshold = mean_cost + 3 * std_cost
        
        spikes = daily_costs[daily_costs > threshold]
        
        return {
            'threshold_credits': threshold,
            'spike_count': len(spikes),
            'spike_dates': spikes.index.tolist(),
            'avg_spike_magnitude': spikes.mean() if len(spikes) > 0 else 0,
            'total_spike_cost': (spikes * self.config.get('credit_cost_per_unit', 2.0)).sum()
        }
    
    def _analyze_idle_time(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze idle time patterns.
        
        Args:
            data: Input DataFrame
        
        Returns:
            Dictionary with idle time analysis
        """
        if 'WAREHOUSE_NAME' not in data.columns:
            return {}
        
        # Analyze gaps between queries for each warehouse
        idle_analysis = {}
        
        for warehouse in data['WAREHOUSE_NAME'].unique():
            warehouse_data = data[data['WAREHOUSE_NAME'] == warehouse].sort_values('START_TIME')
            
            if len(warehouse_data) > 1:
                time_gaps = warehouse_data['START_TIME'].diff().dt.total_seconds() / 60  # minutes
                long_gaps = time_gaps[time_gaps > 60]  # gaps > 1 hour
                
                idle_analysis[warehouse] = {
                    'avg_gap_minutes': time_gaps.mean(),
                    'max_gap_minutes': time_gaps.max(),
                    'idle_periods_count': len(long_gaps),
                    'total_idle_hours': long_gaps.sum() / 60
                }
        
        return idle_analysis
    
    def _identify_query_optimization_candidates(self, data: pd.DataFrame, credit_column: str) -> Dict[str, Any]:
        """
        Identify queries that are candidates for optimization.
        
        Args:
            data: Input DataFrame
            credit_column: Credit column name
        
        Returns:
            Dictionary with optimization candidates
        """
        candidates = {}
        
        # High credit usage with low data output
        if 'ROWS_PRODUCED' in data.columns:
            data['credits_per_row'] = data[credit_column] / (data['ROWS_PRODUCED'] + 1)
            high_cost_low_output = data[
                (data[credit_column] > data[credit_column].quantile(0.8)) &
                (data['credits_per_row'] > data['credits_per_row'].quantile(0.9))
            ]
            candidates['high_cost_low_output'] = len(high_cost_low_output)
        
        # Long execution time with high credits
        if 'EXECUTION_TIME_MS' in data.columns:
            long_expensive = data[
                (data['EXECUTION_TIME_MS'] > data['EXECUTION_TIME_MS'].quantile(0.9)) &
                (data[credit_column] > data[credit_column].quantile(0.9))
            ]
            candidates['long_and_expensive'] = len(long_expensive)
        
        # Repeated similar queries with high total cost
        if 'QUERY_TEXT' in data.columns:
            # Simplified pattern matching (would need more sophisticated analysis in production)
            query_patterns = data.groupby('QUERY_TEXT')[credit_column].agg(['count', 'sum'])
            repeated_expensive = query_patterns[
                (query_patterns['count'] > 5) &
                (query_patterns['sum'] > query_patterns['sum'].quantile(0.8))
            ]
            candidates['repeated_expensive_patterns'] = len(repeated_expensive)
        
        return candidates
    
    def _analyze_warehouse_sizing(self, data: pd.DataFrame, credit_column: str) -> Dict[str, Any]:
        """
        Analyze warehouse sizing for optimization recommendations.
        
        Args:
            data: Input DataFrame
            credit_column: Credit column name
        
        Returns:
            Dictionary with warehouse sizing analysis
        """
        if 'WAREHOUSE_NAME' not in data.columns:
            return {}
        
        sizing_analysis = {}
        
        for warehouse in data['WAREHOUSE_NAME'].unique():
            warehouse_data = data[data['WAREHOUSE_NAME'] == warehouse]
            
            # Calculate utilization patterns
            avg_credits = warehouse_data[credit_column].mean()
            max_credits = warehouse_data[credit_column].max()
            utilization_ratio = avg_credits / max_credits if max_credits > 0 else 0
            
            # Determine recommendation
            if utilization_ratio < 0.3:
                recommendation = 'downsize'
            elif utilization_ratio > 0.8:
                recommendation = 'upsize'
            else:
                recommendation = 'optimal'
            
            sizing_analysis[warehouse] = {
                'avg_credits_per_query': avg_credits,
                'max_credits_per_query': max_credits,
                'utilization_ratio': utilization_ratio,
                'recommendation': recommendation,
                'total_cost': warehouse_data['cost_amount'].sum()
            }
        
        return sizing_analysis
    
    def _identify_query_patterns(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Identify common patterns in high-cost queries.
        
        Args:
            data: High-cost query DataFrame
        
        Returns:
            Dictionary with pattern analysis
        """
        patterns = {}
        
        if 'QUERY_TYPE' in data.columns:
            patterns['by_query_type'] = data.groupby('QUERY_TYPE')['cost_amount'].sum().to_dict()
        
        if 'DATABASE_NAME' in data.columns:
            patterns['by_database'] = data.groupby('DATABASE_NAME')['cost_amount'].sum().to_dict()
        
        if 'USER_NAME' in data.columns:
            patterns['by_user'] = data.groupby('USER_NAME')['cost_amount'].sum().nlargest(5).to_dict()
        
        return patterns
    
    def _calculate_gini_coefficient(self, values: np.ndarray) -> float:
        """
        Calculate Gini coefficient for inequality measurement.
        
        Args:
            values: Array of values
        
        Returns:
            Gini coefficient (0-1)
        """
        if len(values) == 0:
            return 0.0
        
        values = np.sort(values)
        n = len(values)
        index = np.arange(1, n + 1)
        
        return (2 * np.sum(index * values)) / (n * np.sum(values)) - (n + 1) / n
    
    def _get_appropriate_windows(self, frequency: str) -> List[int]:
        """
        Get appropriate window sizes for moving averages based on frequency.
        
        Args:
            frequency: Time frequency
        
        Returns:
            List of window sizes
        """
        if frequency == 'hourly':
            return [3, 6, 12, 24]  # 3h, 6h, 12h, 1d
        elif frequency == 'daily':
            return [3, 7, 14, 30]  # 3d, 1w, 2w, 1m
        elif frequency == 'weekly':
            return [2, 4, 8, 12]   # 2w, 1m, 2m, 3m
        elif frequency == 'monthly':
            return [2, 3, 6, 12]   # 2m, 3m, 6m, 1y
        else:
            return [3, 7, 14]
    
    def configure_for_snowflake_data(self):
        """
        Configure the aggregator for typical Snowflake analytics data.
        """
        snowflake_config = {
            'credit_cost_per_unit': 2.0,
            'cost_allocation_dimensions': [
                'USER_NAME', 'WAREHOUSE_NAME', 'DATABASE_NAME', 
                'SCHEMA_NAME', 'ROLE_NAME', 'QUERY_TYPE'
            ],
            'cost_thresholds': {
                'low_cost_query': 0.001,     # Very small queries
                'medium_cost_query': 0.01,   # Small queries  
                'high_cost_query': 0.1,      # Medium queries
                'very_high_cost_query': 1.0, # Large queries
                'extreme_cost_query': 10.0   # Very large queries
            },
            'budget_periods': {
                'daily_budget': 500.0,
                'weekly_budget': 3500.0,
                'monthly_budget': 15000.0,
                'quarterly_budget': 45000.0
            }
        }
        
        self.config.update(snowflake_config)
        logger.info("Configured CostAggregator for Snowflake analytics data")
