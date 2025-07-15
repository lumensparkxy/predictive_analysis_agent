"""
Dimensional Aggregator

This module provides comprehensive dimensional aggregation capabilities for creating
multi-dimensional summaries and analysis from Snowflake analytics data.
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Any, Tuple
from itertools import combinations
import warnings

logger = logging.getLogger(__name__)


class DimensionalAggregator:
    """
    Handles dimensional aggregation of analytics data across various business dimensions.
    
    This class provides methods for aggregating data by user, warehouse, query type,
    database, and custom dimensional hierarchies, with support for cross-dimensional
    analysis and dimension importance ranking.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the DimensionalAggregator with optional configuration.
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        
        # Default configuration
        self.default_config = {
            'primary_dimensions': [
                'USER_NAME', 'WAREHOUSE_NAME', 'DATABASE_NAME', 
                'SCHEMA_NAME', 'QUERY_TYPE', 'ROLE_NAME'
            ],
            'hierarchical_dimensions': {
                'database_hierarchy': ['DATABASE_NAME', 'SCHEMA_NAME'],
                'user_hierarchy': ['ROLE_NAME', 'USER_NAME'],
                'execution_hierarchy': ['WAREHOUSE_NAME', 'QUERY_TYPE']
            },
            'metrics_to_aggregate': [
                'credits_used', 'execution_time_ms', 'bytes_scanned',
                'rows_produced', 'partitions_scanned', 'query_count'
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
            'include_rankings': True,
            'include_distributions': True,
            'include_cross_dimensional': True,
            'include_efficiency_metrics': True,
            'include_outlier_analysis': True,
            'max_cardinality': 1000,  # Maximum unique values per dimension
            'min_support': 5,  # Minimum observations for aggregation
            'top_n_values': 50,  # Top N values to include in detailed analysis
            'create_dimension_profiles': True,
            'calculate_dimension_importance': True
        }
        
        # Merge with provided config
        self.config = {**self.default_config, **self.config}
    
    def aggregate_by_dimension(self, 
                             data: pd.DataFrame, 
                             dimensions: Union[str, List[str]],
                             metrics: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Aggregate data by specified dimension(s).
        
        Args:
            data: Input DataFrame with dimensional data
            dimensions: Single dimension or list of dimensions to aggregate by
            metrics: Optional list of metrics to aggregate. If None, uses configured metrics.
        
        Returns:
            DataFrame with dimensional aggregations
        """
        if isinstance(dimensions, str):
            dimensions = [dimensions]
        
        logger.info(f"Performing dimensional aggregation by: {', '.join(dimensions)}")
        
        # Validate dimensions exist in data
        missing_dims = [dim for dim in dimensions if dim not in data.columns]
        if missing_dims:
            logger.warning(f"Missing dimensions in data: {missing_dims}")
            dimensions = [dim for dim in dimensions if dim in data.columns]
        
        if not dimensions:
            raise ValueError("No valid dimensions found in data")
        
        if metrics is None:
            metrics = self.config['metrics_to_aggregate']
        
        # Filter dimensions with reasonable cardinality
        dimensions = self._filter_high_cardinality_dimensions(data, dimensions)
        
        # Perform basic aggregation
        aggregated = self._perform_dimensional_aggregation(data, dimensions, metrics)
        
        # Add dimensional analysis features
        if self.config.get('include_rankings', True):
            aggregated = self._add_dimensional_rankings(aggregated, metrics)
        
        if self.config.get('include_distributions', True):
            aggregated = self._add_distribution_analysis(aggregated, data, dimensions, metrics)
        
        if self.config.get('include_efficiency_metrics', True):
            aggregated = self._add_efficiency_metrics(aggregated, metrics)
        
        if self.config.get('include_outlier_analysis', True):
            aggregated = self._add_outlier_indicators(aggregated, metrics)
        
        logger.info(f"Generated dimensional aggregation with {len(aggregated)} dimension combinations and {len(aggregated.columns)} metrics")
        return aggregated
    
    def aggregate_hierarchical_dimensions(self, 
                                        data: pd.DataFrame,
                                        hierarchy_name: Optional[str] = None,
                                        metrics: Optional[List[str]] = None) -> Dict[str, pd.DataFrame]:
        """
        Aggregate data across hierarchical dimensions.
        
        Args:
            data: Input DataFrame
            hierarchy_name: Optional specific hierarchy to process. If None, processes all.
            metrics: Optional list of metrics to aggregate
        
        Returns:
            Dictionary of aggregated DataFrames by hierarchy level
        """
        logger.info("Performing hierarchical dimensional aggregation...")
        
        if metrics is None:
            metrics = self.config['metrics_to_aggregate']
        
        hierarchies = self.config.get('hierarchical_dimensions', {})
        if hierarchy_name:
            if hierarchy_name not in hierarchies:
                raise ValueError(f"Hierarchy '{hierarchy_name}' not found in configuration")
            hierarchies = {hierarchy_name: hierarchies[hierarchy_name]}
        
        results = {}
        
        for hier_name, hier_levels in hierarchies.items():
            logger.info(f"Processing hierarchy: {hier_name}")
            
            # Filter levels that exist in data
            available_levels = [level for level in hier_levels if level in data.columns]
            if not available_levels:
                logger.warning(f"No hierarchy levels found in data for {hier_name}")
                continue
            
            hier_results = {}
            
            # Aggregate at each level of the hierarchy
            for i, level in enumerate(available_levels):
                level_dimensions = available_levels[:i+1]
                
                try:
                    level_agg = self.aggregate_by_dimension(data, level_dimensions, metrics)
                    
                    # Add hierarchy-specific features
                    level_agg = self._add_hierarchy_features(level_agg, level_dimensions, hier_name)
                    
                    hier_results[f"level_{i+1}_{level}"] = level_agg
                    
                except Exception as e:
                    logger.error(f"Error aggregating hierarchy level {level}: {str(e)}")
            
            if hier_results:
                results[hier_name] = hier_results
        
        return results
    
    def create_cross_dimensional_analysis(self, 
                                        data: pd.DataFrame,
                                        metrics: Optional[List[str]] = None,
                                        max_dimension_combinations: int = 3) -> pd.DataFrame:
        """
        Create cross-dimensional analysis by aggregating across dimension combinations.
        
        Args:
            data: Input DataFrame
            metrics: Optional list of metrics to aggregate
            max_dimension_combinations: Maximum number of dimensions to combine
        
        Returns:
            DataFrame with cross-dimensional aggregations
        """
        logger.info("Performing cross-dimensional analysis...")
        
        if not self.config.get('include_cross_dimensional', True):
            logger.info("Cross-dimensional analysis disabled in configuration")
            return pd.DataFrame()
        
        if metrics is None:
            metrics = self.config['metrics_to_aggregate']
        
        # Get available dimensions
        dimensions = [dim for dim in self.config['primary_dimensions'] if dim in data.columns]
        dimensions = self._filter_high_cardinality_dimensions(data, dimensions)
        
        if len(dimensions) < 2:
            logger.warning("Need at least 2 dimensions for cross-dimensional analysis")
            return pd.DataFrame()
        
        cross_dim_results = []
        
        # Generate dimension combinations
        for r in range(2, min(max_dimension_combinations + 1, len(dimensions) + 1)):
            for dim_combo in combinations(dimensions, r):
                try:
                    combo_agg = self.aggregate_by_dimension(data, list(dim_combo), metrics)
                    combo_agg['dimension_combination'] = '_x_'.join(dim_combo)
                    combo_agg['num_dimensions'] = len(dim_combo)
                    
                    # Add interaction analysis
                    combo_agg = self._add_interaction_analysis(combo_agg, dim_combo, metrics)
                    
                    cross_dim_results.append(combo_agg)
                    
                except Exception as e:
                    logger.error(f"Error in cross-dimensional analysis for {dim_combo}: {str(e)}")
        
        if cross_dim_results:
            return pd.concat(cross_dim_results, ignore_index=True)
        else:
            return pd.DataFrame()
    
    def create_dimension_profiles(self, data: pd.DataFrame) -> Dict[str, Dict]:
        """
        Create comprehensive profiles for each dimension.
        
        Args:
            data: Input DataFrame
        
        Returns:
            Dictionary of dimension profiles
        """
        logger.info("Creating dimension profiles...")
        
        if not self.config.get('create_dimension_profiles', True):
            return {}
        
        dimensions = [dim for dim in self.config['primary_dimensions'] if dim in data.columns]
        profiles = {}
        
        for dim in dimensions:
            try:
                profile = self._create_single_dimension_profile(data, dim)
                profiles[dim] = profile
            except Exception as e:
                logger.error(f"Error creating profile for dimension {dim}: {str(e)}")
        
        return profiles
    
    def _filter_high_cardinality_dimensions(self, data: pd.DataFrame, dimensions: List[str]) -> List[str]:
        """
        Filter out dimensions with cardinality too high for effective aggregation.
        
        Args:
            data: Input DataFrame
            dimensions: List of dimensions to check
        
        Returns:
            Filtered list of dimensions
        """
        max_cardinality = self.config.get('max_cardinality', 1000)
        filtered_dimensions = []
        
        for dim in dimensions:
            if dim in data.columns:
                cardinality = data[dim].nunique()
                if cardinality <= max_cardinality:
                    filtered_dimensions.append(dim)
                else:
                    logger.warning(f"Excluding dimension {dim} due to high cardinality ({cardinality})")
        
        return filtered_dimensions
    
    def _perform_dimensional_aggregation(self, 
                                       data: pd.DataFrame, 
                                       dimensions: List[str], 
                                       metrics: List[str]) -> pd.DataFrame:
        """
        Perform the actual dimensional aggregation.
        
        Args:
            data: Input DataFrame
            dimensions: List of dimensions to group by
            metrics: List of metrics to aggregate
        
        Returns:
            Aggregated DataFrame
        """
        # Filter metrics that exist in data
        available_metrics = [m for m in metrics if m in data.columns]
        
        if not available_metrics:
            logger.warning("No specified metrics found in data")
            # Create basic aggregation with query count
            agg_data = data.groupby(dimensions).size().reset_index(name='query_count')
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
            aggregated = data.groupby(dimensions).agg(agg_functions)
            
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
                            percentile_values = data.groupby(dimensions)[metric].quantile(p/100)
                            # Create a mapping from the multi-index to values
                            percentile_dict = percentile_values.to_dict()
                            
                            # Map percentile values to aggregated data
                            def map_percentile(row):
                                key = tuple(row[dim] for dim in dimensions)
                                return percentile_dict.get(key, 0)
                            
                            aggregated[f"{metric}_p{p}"] = aggregated.apply(map_percentile, axis=1)
            
            # Filter by minimum support
            min_support = self.config.get('min_support', 5)
            if 'query_count' in aggregated.columns:
                aggregated = aggregated[aggregated['query_count'] >= min_support]
            
        except Exception as e:
            logger.error(f"Error in dimensional aggregation: {str(e)}")
            # Fallback to simple aggregation
            aggregated = data.groupby(dimensions)[available_metrics].agg(['sum', 'mean', 'count']).reset_index()
            aggregated.columns = [f"{col[0]}_{col[1]}" if isinstance(col, tuple) else col for col in aggregated.columns]
        
        return aggregated
    
    def _add_dimensional_rankings(self, data: pd.DataFrame, metrics: List[str]) -> pd.DataFrame:
        """
        Add ranking information for dimensions based on metrics.
        
        Args:
            data: Aggregated DataFrame
            metrics: List of metrics to create rankings for
        
        Returns:
            DataFrame with ranking features
        """
        data = data.copy()
        
        for metric in metrics:
            metric_cols = [col for col in data.columns if col.startswith(f"{metric}_")]
            
            for col in metric_cols:
                if col in data.columns and data[col].dtype in ['int64', 'float64']:
                    # Rank by metric value (descending)
                    data[f"{col}_rank"] = data[col].rank(method='dense', ascending=False)
                    
                    # Percentile rank
                    data[f"{col}_percentile"] = data[col].rank(pct=True)
                    
                    # Top N indicator
                    top_n = self.config.get('top_n_values', 50)
                    data[f"{col}_top_{top_n}"] = (data[f"{col}_rank"] <= top_n).astype(int)
        
        return data
    
    def _add_distribution_analysis(self, 
                                 aggregated: pd.DataFrame, 
                                 original_data: pd.DataFrame,
                                 dimensions: List[str], 
                                 metrics: List[str]) -> pd.DataFrame:
        """
        Add distribution analysis features.
        
        Args:
            aggregated: Aggregated DataFrame
            original_data: Original data for distribution calculations
            dimensions: List of dimensions used in aggregation
            metrics: List of metrics
        
        Returns:
            DataFrame with distribution features
        """
        aggregated = aggregated.copy()
        
        for metric in metrics:
            if metric not in original_data.columns:
                continue
            
            try:
                # Calculate distribution statistics for each dimension combination
                for idx, row in aggregated.iterrows():
                    # Create filter for this dimension combination
                    dimension_filter = pd.Series([True] * len(original_data))
                    for dim in dimensions:
                        if dim in aggregated.columns:
                            dimension_filter &= (original_data[dim] == row[dim])
                    
                    # Get metric values for this dimension combination
                    metric_values = original_data[dimension_filter][metric]
                    
                    if len(metric_values) > 0:
                        # Distribution statistics
                        aggregated.loc[idx, f"{metric}_skewness"] = metric_values.skew()
                        aggregated.loc[idx, f"{metric}_kurtosis"] = metric_values.kurtosis()
                        aggregated.loc[idx, f"{metric}_cv"] = metric_values.std() / (metric_values.mean() + 1e-10)  # Coefficient of variation
                        
                        # Concentration metrics
                        q75, q25 = metric_values.quantile([0.75, 0.25])
                        aggregated.loc[idx, f"{metric}_iqr"] = q75 - q25
                        aggregated.loc[idx, f"{metric}_range"] = metric_values.max() - metric_values.min()
                        
            except Exception as e:
                logger.warning(f"Error calculating distribution for {metric}: {str(e)}")
        
        return aggregated
    
    def _add_efficiency_metrics(self, data: pd.DataFrame, metrics: List[str]) -> pd.DataFrame:
        """
        Add efficiency and performance ratio metrics.
        
        Args:
            data: Aggregated DataFrame
            metrics: List of metrics
        
        Returns:
            DataFrame with efficiency metrics
        """
        data = data.copy()
        
        # Define efficiency ratios
        efficiency_ratios = {
            'credits_per_row': ('credits_used_sum', 'rows_produced_sum'),
            'credits_per_byte': ('credits_used_sum', 'bytes_scanned_sum'),
            'time_per_row': ('execution_time_ms_sum', 'rows_produced_sum'),
            'time_per_byte': ('execution_time_ms_sum', 'bytes_scanned_sum'),
            'bytes_per_row': ('bytes_scanned_sum', 'rows_produced_sum'),
            'queries_per_credit': ('query_count', 'credits_used_sum')
        }
        
        for ratio_name, (numerator, denominator) in efficiency_ratios.items():
            if numerator in data.columns and denominator in data.columns:
                # Calculate ratio with protection against division by zero
                data[ratio_name] = data[numerator] / (data[denominator] + 1e-10)
                
                # Add ratio rankings
                data[f"{ratio_name}_rank"] = data[ratio_name].rank(method='dense')
                data[f"{ratio_name}_percentile"] = data[ratio_name].rank(pct=True)
        
        # Resource utilization intensity
        if 'credits_used_sum' in data.columns and 'execution_time_ms_sum' in data.columns:
            data['resource_intensity'] = data['credits_used_sum'] / (data['execution_time_ms_sum'] / 1000 + 1e-10)  # credits per second
        
        # Query complexity indicators
        if 'bytes_scanned_mean' in data.columns and 'execution_time_ms_mean' in data.columns:
            data['complexity_score'] = (data['bytes_scanned_mean'] * data['execution_time_ms_mean']) ** 0.5
        
        return data
    
    def _add_outlier_indicators(self, data: pd.DataFrame, metrics: List[str]) -> pd.DataFrame:
        """
        Add outlier detection indicators.
        
        Args:
            data: Aggregated DataFrame
            metrics: List of metrics
        
        Returns:
            DataFrame with outlier indicators
        """
        data = data.copy()
        
        for metric in metrics:
            metric_cols = [col for col in data.columns if col.startswith(f"{metric}_") and col.endswith(('_sum', '_mean'))]
            
            for col in metric_cols:
                if col in data.columns and data[col].dtype in ['int64', 'float64']:
                    values = data[col]
                    
                    # IQR-based outlier detection
                    Q1 = values.quantile(0.25)
                    Q3 = values.quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    
                    data[f"{col}_is_outlier"] = ((values < lower_bound) | (values > upper_bound)).astype(int)
                    
                    # Z-score based outlier detection
                    mean_val = values.mean()
                    std_val = values.std()
                    if std_val > 0:
                        z_scores = np.abs((values - mean_val) / std_val)
                        data[f"{col}_z_score"] = z_scores
                        data[f"{col}_is_extreme"] = (z_scores > 3).astype(int)
        
        return data
    
    def _add_hierarchy_features(self, 
                              data: pd.DataFrame, 
                              hierarchy_levels: List[str], 
                              hierarchy_name: str) -> pd.DataFrame:
        """
        Add hierarchy-specific features.
        
        Args:
            data: Aggregated DataFrame
            hierarchy_levels: List of hierarchy levels
            hierarchy_name: Name of the hierarchy
        
        Returns:
            DataFrame with hierarchy features
        """
        data = data.copy()
        
        # Add hierarchy metadata
        data['hierarchy_name'] = hierarchy_name
        data['hierarchy_level'] = len(hierarchy_levels)
        data['hierarchy_path'] = data[hierarchy_levels].apply(
            lambda x: ' > '.join(x.astype(str)), axis=1
        )
        
        # Calculate hierarchy-specific metrics
        if len(hierarchy_levels) > 1:
            # Concentration within parent level
            parent_levels = hierarchy_levels[:-1]
            if parent_levels:
                # Group by parent levels to calculate concentration
                parent_groups = data.groupby(parent_levels)
                
                for metric_col in data.select_dtypes(include=[np.number]).columns:
                    if not metric_col.endswith(('_rank', '_percentile', '_is_outlier')):
                        # Calculate share within parent
                        parent_totals = parent_groups[metric_col].transform('sum')
                        data[f"{metric_col}_parent_share"] = data[metric_col] / (parent_totals + 1e-10)
        
        return data
    
    def _add_interaction_analysis(self, 
                                data: pd.DataFrame, 
                                dimensions: Tuple[str], 
                                metrics: List[str]) -> pd.DataFrame:
        """
        Add interaction analysis for cross-dimensional combinations.
        
        Args:
            data: Aggregated DataFrame
            dimensions: Tuple of dimensions in the combination
            metrics: List of metrics
        
        Returns:
            DataFrame with interaction features
        """
        data = data.copy()
        
        # Add interaction strength indicators
        for metric in metrics:
            metric_cols = [col for col in data.columns if col.startswith(f"{metric}_sum")]
            
            for col in metric_cols:
                if col in data.columns and data[col].dtype in ['int64', 'float64']:
                    # Coefficient of variation as interaction strength indicator
                    cv = data[col].std() / (data[col].mean() + 1e-10)
                    data[f"{col}_interaction_strength"] = cv
        
        # Add dimension dominance indicators
        if len(dimensions) == 2:
            # For 2D combinations, analyze which dimension dominates variance
            for metric in metrics:
                sum_col = f"{metric}_sum"
                if sum_col in data.columns:
                    # Simple variance decomposition approximation
                    total_var = data[sum_col].var()
                    
                    # Group by each dimension separately to estimate individual contributions
                    for i, dim in enumerate(dimensions):
                        dim_var = data.groupby(dim)[sum_col].var().mean()
                        data[f"{sum_col}_{dim}_dominance"] = dim_var / (total_var + 1e-10)
        
        return data
    
    def _create_single_dimension_profile(self, data: pd.DataFrame, dimension: str) -> Dict:
        """
        Create a comprehensive profile for a single dimension.
        
        Args:
            data: Input DataFrame
            dimension: Dimension to profile
        
        Returns:
            Dictionary containing dimension profile
        """
        profile = {
            'dimension_name': dimension,
            'cardinality': data[dimension].nunique(),
            'total_records': len(data),
            'null_count': data[dimension].isnull().sum(),
            'null_percentage': data[dimension].isnull().mean() * 100
        }
        
        # Top values by frequency
        value_counts = data[dimension].value_counts()
        top_n = self.config.get('top_n_values', 50)
        profile['top_values'] = value_counts.head(top_n).to_dict()
        
        # Concentration metrics
        profile['concentration_ratio_top10'] = value_counts.head(10).sum() / len(data)
        profile['concentration_ratio_top20'] = value_counts.head(20).sum() / len(data)
        
        # Distribution characteristics
        profile['min_frequency'] = value_counts.min()
        profile['max_frequency'] = value_counts.max()
        profile['mean_frequency'] = value_counts.mean()
        profile['frequency_std'] = value_counts.std()
        
        # Gini coefficient for inequality measurement
        freqs = value_counts.values
        freqs_sorted = np.sort(freqs)
        n = len(freqs_sorted)
        index = np.arange(1, n + 1)
        profile['gini_coefficient'] = (2 * np.sum(index * freqs_sorted)) / (n * np.sum(freqs_sorted)) - (n + 1) / n
        
        # Calculate importance score if enabled
        if self.config.get('calculate_dimension_importance', True):
            profile['importance_score'] = self._calculate_dimension_importance(data, dimension)
        
        return profile
    
    def _calculate_dimension_importance(self, data: pd.DataFrame, dimension: str) -> float:
        """
        Calculate importance score for a dimension based on its predictive power.
        
        Args:
            data: Input DataFrame
            dimension: Dimension to calculate importance for
        
        Returns:
            Importance score (0-1)
        """
        try:
            # Simple importance based on variance explained in key metrics
            importance_scores = []
            
            key_metrics = ['credits_used', 'execution_time_ms', 'bytes_scanned']
            available_metrics = [m for m in key_metrics if m in data.columns]
            
            for metric in available_metrics:
                # Calculate between-group and within-group variance
                overall_var = data[metric].var()
                if overall_var > 0:
                    group_means = data.groupby(dimension)[metric].mean()
                    group_sizes = data.groupby(dimension)[metric].size()
                    
                    # Between-group variance
                    overall_mean = data[metric].mean()
                    between_var = np.sum(group_sizes * (group_means - overall_mean) ** 2) / len(data)
                    
                    # Importance as proportion of variance explained
                    importance = between_var / overall_var
                    importance_scores.append(min(importance, 1.0))  # Cap at 1.0
            
            return np.mean(importance_scores) if importance_scores else 0.0
            
        except Exception as e:
            logger.warning(f"Error calculating dimension importance for {dimension}: {str(e)}")
            return 0.0
    
    def configure_for_snowflake_data(self):
        """
        Configure the aggregator for typical Snowflake analytics data.
        """
        snowflake_config = {
            'primary_dimensions': [
                'USER_NAME', 'WAREHOUSE_NAME', 'DATABASE_NAME', 'SCHEMA_NAME',
                'QUERY_TYPE', 'ROLE_NAME', 'SESSION_ID', 'QUERY_TAG'
            ],
            'hierarchical_dimensions': {
                'database_hierarchy': ['DATABASE_NAME', 'SCHEMA_NAME'],
                'user_hierarchy': ['ROLE_NAME', 'USER_NAME'],
                'execution_hierarchy': ['WAREHOUSE_NAME', 'QUERY_TYPE'],
                'organizational_hierarchy': ['ROLE_NAME', 'USER_NAME', 'SESSION_ID']
            },
            'metrics_to_aggregate': [
                'CREDITS_USED', 'EXECUTION_TIME_MS', 'BYTES_SCANNED',
                'ROWS_PRODUCED', 'PARTITIONS_SCANNED', 'COMPILATION_TIME_MS',
                'query_count'
            ],
            'aggregation_functions': {
                'sum': ['CREDITS_USED', 'EXECUTION_TIME_MS', 'BYTES_SCANNED', 'ROWS_PRODUCED', 'PARTITIONS_SCANNED', 'query_count'],
                'mean': ['CREDITS_USED', 'EXECUTION_TIME_MS', 'BYTES_SCANNED', 'ROWS_PRODUCED', 'COMPILATION_TIME_MS'],
                'median': ['CREDITS_USED', 'EXECUTION_TIME_MS', 'BYTES_SCANNED', 'COMPILATION_TIME_MS'],
                'std': ['CREDITS_USED', 'EXECUTION_TIME_MS', 'BYTES_SCANNED'],
                'min': ['CREDITS_USED', 'EXECUTION_TIME_MS', 'BYTES_SCANNED'],
                'max': ['CREDITS_USED', 'EXECUTION_TIME_MS', 'BYTES_SCANNED'],
                'count': ['query_count']
            },
            'percentiles': [50, 75, 90, 95, 99],
            'max_cardinality': 500,  # Snowflake can have many users/warehouses
            'min_support': 3,  # Lower threshold for Snowflake data
            'top_n_values': 100  # More detailed analysis for top consumers
        }
        
        self.config.update(snowflake_config)
        logger.info("Configured DimensionalAggregator for Snowflake analytics data")
