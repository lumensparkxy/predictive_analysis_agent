"""
Validation Pipeline

This module provides the main orchestration for comprehensive data validation
throughout the ML pipeline processing stages.
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Any, Tuple
from datetime import datetime
import json
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import pickle

logger = logging.getLogger(__name__)


class ValidationPipeline:
    """
    Main orchestrator for comprehensive data validation throughout the ML pipeline.
    
    This class coordinates various validation components to ensure data quality,
    schema compliance, and ML readiness at different stages of the processing pipeline.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the ValidationPipeline with configuration.
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        
        # Default configuration
        self.default_config = {
            'validation_stages': [
                'raw_data_validation',
                'cleaned_data_validation', 
                'feature_engineering_validation',
                'aggregation_validation',
                'ml_readiness_validation'
            ],
            'parallel_validation': True,
            'max_workers': 4,
            'validation_thresholds': {
                'completeness': 0.95,
                'consistency': 0.98,
                'accuracy': 0.99,
                'timeliness': 0.95,
                'validity': 0.97
            },
            'data_quality_checks': {
                'check_missing_values': True,
                'check_duplicates': True,
                'check_outliers': True,
                'check_data_types': True,
                'check_constraints': True,
                'check_distributions': True,
                'check_correlations': True
            },
            'schema_validation': {
                'enforce_column_types': True,
                'enforce_constraints': True,
                'allow_additional_columns': False,
                'validate_primary_keys': True,
                'validate_foreign_keys': True
            },
            'ml_validation': {
                'min_samples': 1000,
                'max_missing_rate': 0.05,
                'min_variance': 0.001,
                'max_correlation': 0.95,
                'check_target_distribution': True,
                'check_feature_importance': True
            },
            'temporal_validation': {
                'check_time_continuity': True,
                'check_time_ordering': True,
                'check_future_dates': True,
                'max_time_gap_hours': 24
            },
            'output_options': {
                'save_validation_reports': True,
                'save_failed_records': True,
                'generate_summary': True,
                'export_metrics': True
            },
            'alert_configuration': {
                'enable_alerts': True,
                'critical_failure_threshold': 0.1,
                'warning_threshold': 0.05,
                'alert_channels': ['log', 'file']
            }
        }
        
        # Merge with provided config
        self.config = {**self.default_config, **self.config}
        
        # Initialize validation results storage
        self.validation_results = {}
        self.validation_history = []
        self.quality_metrics = {}
        
        # Initialize validator components (will be created when needed)
        self._data_validator = None
        self._quality_checker = None
        self._schema_validator = None
        self._ml_readiness_validator = None
    
    def validate_pipeline_stage(self, 
                               data: pd.DataFrame,
                               stage: str,
                               schema: Optional[Dict] = None,
                               reference_data: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        Validate data at a specific pipeline stage.
        
        Args:
            data: DataFrame to validate
            stage: Pipeline stage name
            schema: Optional schema definition for validation
            reference_data: Optional reference data for comparison
        
        Returns:
            Dictionary containing validation results
        """
        logger.info(f"Starting validation for pipeline stage: {stage}")
        
        if stage not in self.config.get('validation_stages', []):
            logger.warning(f"Stage {stage} not in configured validation stages")
        
        validation_start = datetime.now()
        
        try:
            # Initialize stage-specific configuration
            stage_config = self._get_stage_config(stage)
            
            # Perform validation checks
            results = self._perform_stage_validation(data, stage, stage_config, schema, reference_data)
            
            # Calculate quality scores
            quality_scores = self._calculate_quality_scores(results)
            
            # Generate validation summary
            summary = self._generate_validation_summary(results, quality_scores, stage)
            
            # Check for critical failures
            critical_failures = self._check_critical_failures(quality_scores)
            
            # Compile final results
            validation_results = {
                'stage': stage,
                'timestamp': validation_start,
                'duration_seconds': (datetime.now() - validation_start).total_seconds(),
                'data_shape': data.shape,
                'validation_passed': len(critical_failures) == 0,
                'critical_failures': critical_failures,
                'quality_scores': quality_scores,
                'detailed_results': results,
                'summary': summary,
                'recommendations': self._generate_recommendations(results, quality_scores)
            }
            
            # Store results
            self.validation_results[stage] = validation_results
            self.validation_history.append(validation_results)
            
            # Handle alerts if needed
            if critical_failures and self.config.get('alert_configuration', {}).get('enable_alerts', True):
                self._trigger_alerts(validation_results)
            
            # Save results if configured
            if self.config.get('output_options', {}).get('save_validation_reports', True):
                self._save_validation_report(validation_results, stage)
            
            logger.info(f"Validation completed for stage {stage}. Passed: {validation_results['validation_passed']}")
            return validation_results
            
        except Exception as e:
            logger.error(f"Error during validation for stage {stage}: {str(e)}")
            
            error_results = {
                'stage': stage,
                'timestamp': validation_start,
                'validation_passed': False,
                'error': str(e),
                'critical_failures': [f"Validation process failed: {str(e)}"]
            }
            
            self.validation_results[stage] = error_results
            return error_results
    
    def validate_complete_pipeline(self, 
                                 pipeline_data: Dict[str, pd.DataFrame],
                                 schemas: Optional[Dict[str, Dict]] = None) -> Dict[str, Any]:
        """
        Validate data across all pipeline stages.
        
        Args:
            pipeline_data: Dictionary mapping stage names to DataFrames
            schemas: Optional dictionary mapping stage names to schema definitions
        
        Returns:
            Dictionary containing comprehensive validation results
        """
        logger.info("Starting complete pipeline validation...")
        
        all_results = {}
        overall_passed = True
        
        # Validate each stage
        for stage, data in pipeline_data.items():
            schema = schemas.get(stage) if schemas else None
            stage_results = self.validate_pipeline_stage(data, stage, schema)
            all_results[stage] = stage_results
            
            if not stage_results.get('validation_passed', False):
                overall_passed = False
        
        # Cross-stage validation
        cross_stage_results = self._perform_cross_stage_validation(pipeline_data)
        all_results['cross_stage_validation'] = cross_stage_results
        
        if not cross_stage_results.get('validation_passed', False):
            overall_passed = False
        
        # Generate pipeline-wide summary
        pipeline_summary = self._generate_pipeline_summary(all_results)
        
        complete_results = {
            'pipeline_validation_passed': overall_passed,
            'validation_timestamp': datetime.now(),
            'stages_validated': list(pipeline_data.keys()),
            'pipeline_summary': pipeline_summary,
            'stage_results': all_results,
            'overall_quality_score': self._calculate_overall_quality_score(all_results)
        }
        
        logger.info(f"Complete pipeline validation finished. Overall passed: {overall_passed}")
        return complete_results
    
    def _get_stage_config(self, stage: str) -> Dict[str, Any]:
        """
        Get stage-specific validation configuration.
        
        Args:
            stage: Pipeline stage name
        
        Returns:
            Stage-specific configuration
        """
        # Base configuration
        stage_config = self.config.copy()
        
        # Stage-specific overrides
        stage_overrides = {
            'raw_data_validation': {
                'primary_checks': ['completeness', 'consistency', 'schema_compliance'],
                'skip_ml_checks': True,
                'enforce_strict_schema': False
            },
            'cleaned_data_validation': {
                'primary_checks': ['completeness', 'consistency', 'accuracy'],
                'check_cleaning_effectiveness': True,
                'compare_to_raw': True
            },
            'feature_engineering_validation': {
                'primary_checks': ['feature_validity', 'distribution_checks', 'correlation_checks'],
                'check_feature_importance': True,
                'validate_transformations': True
            },
            'aggregation_validation': {
                'primary_checks': ['aggregation_accuracy', 'temporal_consistency'],
                'check_aggregation_logic': True,
                'validate_metrics': True
            },
            'ml_readiness_validation': {
                'primary_checks': ['ml_compliance', 'feature_engineering', 'data_leakage'],
                'enforce_ml_requirements': True,
                'check_target_variable': True
            }
        }
        
        if stage in stage_overrides:
            stage_config.update(stage_overrides[stage])
        
        return stage_config
    
    def _perform_stage_validation(self, 
                                data: pd.DataFrame, 
                                stage: str, 
                                stage_config: Dict[str, Any],
                                schema: Optional[Dict] = None,
                                reference_data: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        Perform validation checks for a specific stage.
        
        Args:
            data: DataFrame to validate
            stage: Pipeline stage name
            stage_config: Stage-specific configuration
            schema: Optional schema definition
            reference_data: Optional reference data
        
        Returns:
            Dictionary containing validation check results
        """
        results = {}
        
        # Basic data validation
        if self._should_run_check('basic_validation', stage_config):
            results['basic_validation'] = self._run_basic_validation(data)
        
        # Data quality checks
        if self._should_run_check('data_quality', stage_config):
            results['data_quality'] = self._run_data_quality_checks(data, stage_config)
        
        # Schema validation
        if schema and self._should_run_check('schema_validation', stage_config):
            results['schema_validation'] = self._run_schema_validation(data, schema, stage_config)
        
        # Temporal validation
        if self._should_run_check('temporal_validation', stage_config):
            results['temporal_validation'] = self._run_temporal_validation(data, stage_config)
        
        # ML readiness validation
        if not stage_config.get('skip_ml_checks', False) and self._should_run_check('ml_validation', stage_config):
            results['ml_validation'] = self._run_ml_readiness_validation(data, stage_config)
        
        # Reference data comparison
        if reference_data is not None and self._should_run_check('comparison_validation', stage_config):
            results['comparison_validation'] = self._run_comparison_validation(data, reference_data, stage_config)
        
        # Stage-specific validation
        if hasattr(self, f'_run_{stage}_validation'):
            stage_method = getattr(self, f'_run_{stage}_validation')
            results[f'{stage}_specific'] = stage_method(data, stage_config)
        
        return results
    
    def _run_basic_validation(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Run basic data validation checks.
        
        Args:
            data: DataFrame to validate
        
        Returns:
            Basic validation results
        """
        results = {
            'data_shape': data.shape,
            'memory_usage_mb': data.memory_usage(deep=True).sum() / 1024 / 1024,
            'column_count': len(data.columns),
            'row_count': len(data),
            'duplicate_rows': data.duplicated().sum(),
            'empty_dataframe': len(data) == 0,
            'columns_with_all_null': (data.isnull().all()).sum(),
            'numeric_columns': len(data.select_dtypes(include=[np.number]).columns),
            'categorical_columns': len(data.select_dtypes(include=['object', 'category']).columns),
            'datetime_columns': len(data.select_dtypes(include=['datetime']).columns)
        }
        
        # Check for completely empty columns
        results['completely_empty_columns'] = data.columns[data.isnull().all()].tolist()
        
        # Check for constant columns
        constant_columns = []
        for col in data.columns:
            if data[col].nunique() <= 1:
                constant_columns.append(col)
        results['constant_columns'] = constant_columns
        
        # Basic statistics for numeric columns
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            results['numeric_summary'] = {
                'columns': len(numeric_cols),
                'total_missing': data[numeric_cols].isnull().sum().sum(),
                'infinite_values': np.isinf(data[numeric_cols].select_dtypes(include=[np.number])).sum().sum(),
                'negative_values': (data[numeric_cols] < 0).sum().sum()
            }
        
        return results
    
    def _run_data_quality_checks(self, data: pd.DataFrame, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run comprehensive data quality checks.
        
        Args:
            data: DataFrame to validate
            config: Configuration settings
        
        Returns:
            Data quality check results
        """
        quality_config = config.get('data_quality_checks', {})
        results = {}
        
        # Completeness checks
        if quality_config.get('check_missing_values', True):
            missing_analysis = {}
            for col in data.columns:
                missing_count = data[col].isnull().sum()
                missing_percentage = missing_count / len(data) * 100
                missing_analysis[col] = {
                    'missing_count': missing_count,
                    'missing_percentage': missing_percentage,
                    'completeness_score': 1 - (missing_percentage / 100)
                }
            results['completeness'] = missing_analysis
        
        # Uniqueness checks
        if quality_config.get('check_duplicates', True):
            uniqueness_analysis = {}
            for col in data.columns:
                unique_count = data[col].nunique()
                total_count = len(data) - data[col].isnull().sum()
                uniqueness_score = unique_count / max(total_count, 1)
                uniqueness_analysis[col] = {
                    'unique_count': unique_count,
                    'total_non_null': total_count,
                    'uniqueness_score': uniqueness_score,
                    'duplicate_count': total_count - unique_count
                }
            results['uniqueness'] = uniqueness_analysis
        
        # Validity checks (data type consistency)
        if quality_config.get('check_data_types', True):
            validity_analysis = {}
            for col in data.columns:
                col_data = data[col].dropna()
                if len(col_data) > 0:
                    # Check for mixed types
                    types_found = set(type(x).__name__ for x in col_data.head(100))
                    validity_analysis[col] = {
                        'expected_type': str(data[col].dtype),
                        'types_found': list(types_found),
                        'type_consistency': len(types_found) == 1,
                        'validity_score': 1.0 if len(types_found) == 1 else 0.5
                    }
            results['validity'] = validity_analysis
        
        # Constraint checks
        if quality_config.get('check_constraints', True):
            constraint_results = self._check_business_constraints(data)
            results['constraints'] = constraint_results
        
        # Distribution checks
        if quality_config.get('check_distributions', True):
            distribution_results = self._check_distributions(data)
            results['distributions'] = distribution_results
        
        # Correlation checks
        if quality_config.get('check_correlations', True):
            correlation_results = self._check_correlations(data)
            results['correlations'] = correlation_results
        
        return results
    
    def _run_schema_validation(self, data: pd.DataFrame, schema: Dict, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run schema validation checks.
        
        Args:
            data: DataFrame to validate
            schema: Schema definition
            config: Configuration settings
        
        Returns:
            Schema validation results
        """
        schema_config = config.get('schema_validation', {})
        results = {
            'schema_compliance': True,
            'missing_columns': [],
            'extra_columns': [],
            'type_mismatches': [],
            'constraint_violations': []
        }
        
        # Check required columns
        expected_columns = set(schema.get('columns', {}).keys())
        actual_columns = set(data.columns)
        
        missing_columns = expected_columns - actual_columns
        extra_columns = actual_columns - expected_columns
        
        if missing_columns:
            results['missing_columns'] = list(missing_columns)
            results['schema_compliance'] = False
        
        if extra_columns and not schema_config.get('allow_additional_columns', False):
            results['extra_columns'] = list(extra_columns)
            results['schema_compliance'] = False
        
        # Check column types
        if schema_config.get('enforce_column_types', True):
            type_mismatches = []
            for col, col_schema in schema.get('columns', {}).items():
                if col in data.columns:
                    expected_type = col_schema.get('type')
                    actual_type = str(data[col].dtype)
                    
                    if expected_type and not self._types_compatible(actual_type, expected_type):
                        type_mismatches.append({
                            'column': col,
                            'expected_type': expected_type,
                            'actual_type': actual_type
                        })
            
            if type_mismatches:
                results['type_mismatches'] = type_mismatches
                results['schema_compliance'] = False
        
        # Check constraints
        if schema_config.get('enforce_constraints', True):
            constraint_violations = []
            for col, col_schema in schema.get('columns', {}).items():
                if col in data.columns:
                    violations = self._check_column_constraints(data[col], col_schema, col)
                    constraint_violations.extend(violations)
            
            if constraint_violations:
                results['constraint_violations'] = constraint_violations
                results['schema_compliance'] = False
        
        return results
    
    def _run_temporal_validation(self, data: pd.DataFrame, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run temporal validation checks.
        
        Args:
            data: DataFrame to validate
            config: Configuration settings
        
        Returns:
            Temporal validation results
        """
        temporal_config = config.get('temporal_validation', {})
        results = {}
        
        # Find datetime columns
        datetime_columns = data.select_dtypes(include=['datetime']).columns.tolist()
        
        # Also check for columns that might be datetime but stored as strings
        potential_datetime_columns = []
        for col in data.select_dtypes(include=['object']).columns:
            sample_values = data[col].dropna().head(10)
            if len(sample_values) > 0:
                try:
                    pd.to_datetime(sample_values.iloc[0])
                    potential_datetime_columns.append(col)
                except:
                    pass
        
        all_datetime_columns = datetime_columns + potential_datetime_columns
        
        if not all_datetime_columns:
            results['no_datetime_columns'] = True
            return results
        
        for col in all_datetime_columns:
            col_results = {}
            
            # Convert to datetime if needed
            if col in potential_datetime_columns:
                try:
                    datetime_series = pd.to_datetime(data[col], errors='coerce')
                except:
                    col_results['conversion_failed'] = True
                    continue
            else:
                datetime_series = data[col]
            
            # Time continuity checks
            if temporal_config.get('check_time_continuity', True):
                time_gaps = datetime_series.dropna().diff().dt.total_seconds() / 3600  # hours
                max_gap_hours = temporal_config.get('max_time_gap_hours', 24)
                large_gaps = (time_gaps > max_gap_hours).sum()
                
                col_results['continuity'] = {
                    'large_gaps_count': large_gaps,
                    'max_gap_hours': time_gaps.max() if len(time_gaps) > 0 else 0,
                    'avg_gap_hours': time_gaps.mean() if len(time_gaps) > 0 else 0,
                    'continuity_score': 1 - (large_gaps / max(len(time_gaps), 1))
                }
            
            # Time ordering checks
            if temporal_config.get('check_time_ordering', True):
                is_sorted = datetime_series.dropna().is_monotonic_increasing
                col_results['ordering'] = {
                    'is_chronological': is_sorted,
                    'ordering_score': 1.0 if is_sorted else 0.0
                }
            
            # Future date checks
            if temporal_config.get('check_future_dates', True):
                current_time = datetime.now()
                future_dates = (datetime_series > current_time).sum()
                col_results['future_dates'] = {
                    'future_count': future_dates,
                    'future_percentage': future_dates / len(data) * 100,
                    'timeliness_score': 1 - (future_dates / len(data))
                }
            
            results[col] = col_results
        
        return results
    
    def _run_ml_readiness_validation(self, data: pd.DataFrame, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run ML readiness validation checks.
        
        Args:
            data: DataFrame to validate
            config: Configuration settings
        
        Returns:
            ML readiness validation results
        """
        ml_config = config.get('ml_validation', {})
        results = {}
        
        # Sample size check
        min_samples = ml_config.get('min_samples', 1000)
        results['sample_size'] = {
            'actual_samples': len(data),
            'min_required': min_samples,
            'sufficient_samples': len(data) >= min_samples,
            'sample_adequacy_score': min(len(data) / min_samples, 1.0)
        }
        
        # Missing value rate check
        max_missing_rate = ml_config.get('max_missing_rate', 0.05)
        missing_rates = data.isnull().mean()
        high_missing_columns = missing_rates[missing_rates > max_missing_rate]
        
        results['missing_values'] = {
            'max_missing_rate': missing_rates.max(),
            'columns_high_missing': high_missing_columns.to_dict(),
            'acceptable_missing_rate': missing_rates.max() <= max_missing_rate,
            'missing_score': 1 - missing_rates.mean()
        }
        
        # Feature variance check
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        min_variance = ml_config.get('min_variance', 0.001)
        
        if len(numeric_columns) > 0:
            variances = data[numeric_columns].var()
            low_variance_features = variances[variances < min_variance]
            
            results['feature_variance'] = {
                'low_variance_features': low_variance_features.to_dict(),
                'min_variance_threshold': min_variance,
                'adequate_variance': len(low_variance_features) == 0,
                'variance_score': (variances >= min_variance).mean()
            }
        
        # Feature correlation check
        if len(numeric_columns) > 1:
            max_correlation = ml_config.get('max_correlation', 0.95)
            correlation_matrix = data[numeric_columns].corr().abs()
            
            # Find high correlations (excluding diagonal)
            high_corr_pairs = []
            for i in range(len(correlation_matrix.columns)):
                for j in range(i+1, len(correlation_matrix.columns)):
                    if correlation_matrix.iloc[i, j] > max_correlation:
                        high_corr_pairs.append({
                            'feature1': correlation_matrix.columns[i],
                            'feature2': correlation_matrix.columns[j],
                            'correlation': correlation_matrix.iloc[i, j]
                        })
            
            results['feature_correlation'] = {
                'high_correlation_pairs': high_corr_pairs,
                'max_correlation_threshold': max_correlation,
                'acceptable_correlations': len(high_corr_pairs) == 0,
                'correlation_score': 1 - (len(high_corr_pairs) / max(len(numeric_columns) * (len(numeric_columns) - 1) / 2, 1))
            }
        
        # Feature importance check (simplified)
        if ml_config.get('check_feature_importance', True) and len(numeric_columns) > 0:
            feature_importance_results = self._assess_feature_importance(data, numeric_columns)
            results['feature_importance'] = feature_importance_results
        
        return results
    
    def _run_comparison_validation(self, data: pd.DataFrame, reference_data: pd.DataFrame, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run validation by comparing against reference data.
        
        Args:
            data: Current data to validate
            reference_data: Reference data for comparison
            config: Configuration settings
        
        Returns:
            Comparison validation results
        """
        results = {}
        
        # Shape comparison
        results['shape_comparison'] = {
            'current_shape': data.shape,
            'reference_shape': reference_data.shape,
            'shape_consistent': data.shape == reference_data.shape
        }
        
        # Column comparison
        current_columns = set(data.columns)
        reference_columns = set(reference_data.columns)
        
        results['column_comparison'] = {
            'columns_added': list(current_columns - reference_columns),
            'columns_removed': list(reference_columns - current_columns),
            'columns_consistent': current_columns == reference_columns
        }
        
        # Statistical comparison for common numeric columns
        common_numeric_columns = set(data.select_dtypes(include=[np.number]).columns) & set(reference_data.select_dtypes(include=[np.number]).columns)
        
        if common_numeric_columns:
            stat_comparisons = {}
            for col in common_numeric_columns:
                current_stats = data[col].describe()
                reference_stats = reference_data[col].describe()
                
                stat_comparisons[col] = {
                    'mean_change_pct': ((current_stats['mean'] - reference_stats['mean']) / reference_stats['mean'] * 100) if reference_stats['mean'] != 0 else 0,
                    'std_change_pct': ((current_stats['std'] - reference_stats['std']) / reference_stats['std'] * 100) if reference_stats['std'] != 0 else 0,
                    'median_change_pct': ((current_stats['50%'] - reference_stats['50%']) / reference_stats['50%'] * 100) if reference_stats['50%'] != 0 else 0
                }
            
            results['statistical_comparison'] = stat_comparisons
        
        return results
    
    def _check_business_constraints(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Check business logic constraints.
        
        Args:
            data: DataFrame to validate
        
        Returns:
            Constraint check results
        """
        constraint_results = {}
        
        # Common Snowflake constraints
        constraints = {
            'credits_non_negative': lambda df: (df.get('CREDITS_USED', pd.Series([0])) >= 0).all() if 'CREDITS_USED' in df.columns else True,
            'execution_time_positive': lambda df: (df.get('EXECUTION_TIME_MS', pd.Series([1])) > 0).all() if 'EXECUTION_TIME_MS' in df.columns else True,
            'bytes_scanned_non_negative': lambda df: (df.get('BYTES_SCANNED', pd.Series([0])) >= 0).all() if 'BYTES_SCANNED' in df.columns else True,
            'rows_produced_non_negative': lambda df: (df.get('ROWS_PRODUCED', pd.Series([0])) >= 0).all() if 'ROWS_PRODUCED' in df.columns else True,
            'start_before_end_time': lambda df: (df.get('START_TIME', pd.Series([pd.Timestamp.now()])) <= df.get('END_TIME', pd.Series([pd.Timestamp.now()]))).all() if 'START_TIME' in df.columns and 'END_TIME' in df.columns else True
        }
        
        for constraint_name, constraint_func in constraints.items():
            try:
                constraint_results[constraint_name] = {
                    'passed': constraint_func(data),
                    'constraint_score': 1.0 if constraint_func(data) else 0.0
                }
            except Exception as e:
                constraint_results[constraint_name] = {
                    'passed': False,
                    'error': str(e),
                    'constraint_score': 0.0
                }
        
        return constraint_results
    
    def _check_distributions(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Check data distributions for anomalies.
        
        Args:
            data: DataFrame to validate
        
        Returns:
            Distribution check results
        """
        distribution_results = {}
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        
        for col in numeric_columns:
            col_data = data[col].dropna()
            if len(col_data) > 0:
                try:
                    # Basic distribution statistics
                    skewness = col_data.skew()
                    kurtosis = col_data.kurtosis()
                    
                    # Outlier detection using IQR
                    Q1 = col_data.quantile(0.25)
                    Q3 = col_data.quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    outliers = ((col_data < lower_bound) | (col_data > upper_bound)).sum()
                    
                    distribution_results[col] = {
                        'skewness': skewness,
                        'kurtosis': kurtosis,
                        'outlier_count': outliers,
                        'outlier_percentage': outliers / len(col_data) * 100,
                        'distribution_score': 1 - min(abs(skewness) / 3, 1) - min(outliers / len(col_data), 0.1)
                    }
                except Exception as e:
                    distribution_results[col] = {
                        'error': str(e),
                        'distribution_score': 0.5
                    }
        
        return distribution_results
    
    def _check_correlations(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Check correlations between numeric variables.
        
        Args:
            data: DataFrame to validate
        
        Returns:
            Correlation check results
        """
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        
        if len(numeric_columns) < 2:
            return {'insufficient_numeric_columns': True}
        
        try:
            correlation_matrix = data[numeric_columns].corr()
            
            # Find high correlations
            high_correlations = []
            for i in range(len(correlation_matrix.columns)):
                for j in range(i+1, len(correlation_matrix.columns)):
                    corr_value = correlation_matrix.iloc[i, j]
                    if abs(corr_value) > 0.8:  # High correlation threshold
                        high_correlations.append({
                            'feature1': correlation_matrix.columns[i],
                            'feature2': correlation_matrix.columns[j],
                            'correlation': corr_value
                        })
            
            return {
                'high_correlations': high_correlations,
                'max_correlation': correlation_matrix.abs().values[np.triu_indices_from(correlation_matrix.values, k=1)].max(),
                'correlation_score': 1 - min(len(high_correlations) / 10, 1)  # Penalize many high correlations
            }
            
        except Exception as e:
            return {
                'error': str(e),
                'correlation_score': 0.5
            }
    
    def _assess_feature_importance(self, data: pd.DataFrame, numeric_columns: List[str]) -> Dict[str, Any]:
        """
        Assess feature importance (simplified approach).
        
        Args:
            data: DataFrame to validate
            numeric_columns: List of numeric column names
        
        Returns:
            Feature importance assessment results
        """
        try:
            # Calculate variance-based importance
            variances = data[numeric_columns].var()
            
            # Normalize variances
            normalized_importance = variances / variances.sum() if variances.sum() > 0 else variances
            
            # Identify potentially unimportant features
            low_importance_threshold = 0.01
            low_importance_features = normalized_importance[normalized_importance < low_importance_threshold]
            
            return {
                'feature_importance_scores': normalized_importance.to_dict(),
                'low_importance_features': low_importance_features.index.tolist(),
                'importance_score': 1 - (len(low_importance_features) / len(numeric_columns))
            }
            
        except Exception as e:
            return {
                'error': str(e),
                'importance_score': 0.5
            }
    
    def _types_compatible(self, actual_type: str, expected_type: str) -> bool:
        """
        Check if actual and expected data types are compatible.
        
        Args:
            actual_type: Actual data type string
            expected_type: Expected data type string
        
        Returns:
            Boolean indicating if types are compatible
        """
        # Type compatibility mapping
        compatible_types = {
            'int64': ['int', 'integer', 'int64', 'number'],
            'float64': ['float', 'double', 'float64', 'number'],
            'object': ['string', 'text', 'object', 'varchar'],
            'datetime64[ns]': ['datetime', 'timestamp', 'date'],
            'bool': ['boolean', 'bool'],
            'category': ['category', 'categorical']
        }
        
        expected_compatible = compatible_types.get(actual_type.lower(), [actual_type.lower()])
        return expected_type.lower() in expected_compatible
    
    def _check_column_constraints(self, column_data: pd.Series, column_schema: Dict, column_name: str) -> List[Dict]:
        """
        Check constraints for a specific column.
        
        Args:
            column_data: Column data to validate
            column_schema: Column schema definition
            column_name: Name of the column
        
        Returns:
            List of constraint violations
        """
        violations = []
        
        # Null constraints
        if not column_schema.get('nullable', True):
            null_count = column_data.isnull().sum()
            if null_count > 0:
                violations.append({
                    'column': column_name,
                    'constraint': 'not_null',
                    'violation_count': null_count,
                    'description': f"Column should not contain null values but has {null_count} null values"
                })
        
        # Range constraints
        if 'min_value' in column_schema:
            min_violations = (column_data < column_schema['min_value']).sum()
            if min_violations > 0:
                violations.append({
                    'column': column_name,
                    'constraint': 'min_value',
                    'violation_count': min_violations,
                    'description': f"Column has {min_violations} values below minimum {column_schema['min_value']}"
                })
        
        if 'max_value' in column_schema:
            max_violations = (column_data > column_schema['max_value']).sum()
            if max_violations > 0:
                violations.append({
                    'column': column_name,
                    'constraint': 'max_value',
                    'violation_count': max_violations,
                    'description': f"Column has {max_violations} values above maximum {column_schema['max_value']}"
                })
        
        # Length constraints for string columns
        if column_data.dtype == 'object' and 'max_length' in column_schema:
            long_values = column_data.str.len() > column_schema['max_length']
            long_count = long_values.sum()
            if long_count > 0:
                violations.append({
                    'column': column_name,
                    'constraint': 'max_length',
                    'violation_count': long_count,
                    'description': f"Column has {long_count} values exceeding maximum length {column_schema['max_length']}"
                })
        
        return violations
    
    def _should_run_check(self, check_name: str, config: Dict[str, Any]) -> bool:
        """
        Determine if a specific check should be run based on configuration.
        
        Args:
            check_name: Name of the check
            config: Configuration settings
        
        Returns:
            Boolean indicating if check should be run
        """
        # Default checks that should always run
        default_checks = ['basic_validation', 'data_quality']
        
        if check_name in default_checks:
            return True
        
        # Check primary_checks configuration
        primary_checks = config.get('primary_checks', [])
        if primary_checks and check_name not in primary_checks:
            return False
        
        # Check specific enable flags
        check_flags = {
            'schema_validation': config.get('schema_validation', {}).get('enforce_column_types', True),
            'temporal_validation': config.get('temporal_validation', {}).get('check_time_continuity', True),
            'ml_validation': not config.get('skip_ml_checks', False)
        }
        
        return check_flags.get(check_name, True)
    
    def _calculate_quality_scores(self, validation_results: Dict[str, Any]) -> Dict[str, float]:
        """
        Calculate overall quality scores from validation results.
        
        Args:
            validation_results: Validation check results
        
        Returns:
            Dictionary of quality scores
        """
        scores = {}
        
        # Completeness score
        if 'data_quality' in validation_results and 'completeness' in validation_results['data_quality']:
            completeness_scores = [result['completeness_score'] for result in validation_results['data_quality']['completeness'].values()]
            scores['completeness'] = np.mean(completeness_scores) if completeness_scores else 0.0
        else:
            scores['completeness'] = 1.0
        
        # Consistency score
        if 'data_quality' in validation_results and 'validity' in validation_results['data_quality']:
            validity_scores = [result['validity_score'] for result in validation_results['data_quality']['validity'].values()]
            scores['consistency'] = np.mean(validity_scores) if validity_scores else 0.0
        else:
            scores['consistency'] = 1.0
        
        # Accuracy score (based on constraints and distributions)
        accuracy_components = []
        if 'data_quality' in validation_results and 'constraints' in validation_results['data_quality']:
            constraint_scores = [result['constraint_score'] for result in validation_results['data_quality']['constraints'].values()]
            accuracy_components.extend(constraint_scores)
        
        if 'data_quality' in validation_results and 'distributions' in validation_results['data_quality']:
            distribution_scores = [result['distribution_score'] for result in validation_results['data_quality']['distributions'].values() if 'distribution_score' in result]
            accuracy_components.extend(distribution_scores)
        
        scores['accuracy'] = np.mean(accuracy_components) if accuracy_components else 1.0
        
        # Schema compliance score
        if 'schema_validation' in validation_results:
            scores['schema_compliance'] = 1.0 if validation_results['schema_validation'].get('schema_compliance', True) else 0.0
        else:
            scores['schema_compliance'] = 1.0
        
        # ML readiness score
        if 'ml_validation' in validation_results:
            ml_scores = []
            ml_results = validation_results['ml_validation']
            
            if 'sample_size' in ml_results:
                ml_scores.append(ml_results['sample_size']['sample_adequacy_score'])
            if 'missing_values' in ml_results:
                ml_scores.append(ml_results['missing_values']['missing_score'])
            if 'feature_variance' in ml_results:
                ml_scores.append(ml_results['feature_variance']['variance_score'])
            if 'feature_correlation' in ml_results:
                ml_scores.append(ml_results['feature_correlation']['correlation_score'])
            
            scores['ml_readiness'] = np.mean(ml_scores) if ml_scores else 1.0
        else:
            scores['ml_readiness'] = 1.0
        
        # Overall quality score
        scores['overall'] = np.mean(list(scores.values()))
        
        return scores
    
    def _check_critical_failures(self, quality_scores: Dict[str, float]) -> List[str]:
        """
        Check for critical validation failures.
        
        Args:
            quality_scores: Dictionary of quality scores
        
        Returns:
            List of critical failure descriptions
        """
        critical_failures = []
        thresholds = self.config.get('validation_thresholds', {})
        
        # Check each quality dimension against thresholds
        for dimension, score in quality_scores.items():
            if dimension in thresholds:
                threshold = thresholds[dimension]
                if score < threshold:
                    critical_failures.append(
                        f"{dimension.title()} score ({score:.3f}) below threshold ({threshold})"
                    )
        
        # Check overall quality
        overall_threshold = thresholds.get('overall', 0.9)
        if quality_scores.get('overall', 0) < overall_threshold:
            critical_failures.append(
                f"Overall quality score ({quality_scores.get('overall', 0):.3f}) below threshold ({overall_threshold})"
            )
        
        return critical_failures
    
    def _generate_validation_summary(self, 
                                   validation_results: Dict[str, Any], 
                                   quality_scores: Dict[str, float], 
                                   stage: str) -> Dict[str, Any]:
        """
        Generate a summary of validation results.
        
        Args:
            validation_results: Detailed validation results
            quality_scores: Quality scores
            stage: Pipeline stage name
        
        Returns:
            Validation summary
        """
        summary = {
            'stage': stage,
            'validation_passed': len(self._check_critical_failures(quality_scores)) == 0,
            'quality_scores': quality_scores,
            'checks_performed': list(validation_results.keys()),
            'issues_found': []
        }
        
        # Collect issues from validation results
        for check_name, check_results in validation_results.items():
            if check_name == 'basic_validation':
                if check_results.get('empty_dataframe', False):
                    summary['issues_found'].append("Dataset is empty")
                if check_results.get('duplicate_rows', 0) > 0:
                    summary['issues_found'].append(f"Found {check_results['duplicate_rows']} duplicate rows")
                if check_results.get('completely_empty_columns'):
                    summary['issues_found'].append(f"Found {len(check_results['completely_empty_columns'])} completely empty columns")
            
            elif check_name == 'schema_validation':
                if not check_results.get('schema_compliance', True):
                    summary['issues_found'].append("Schema compliance issues detected")
            
            elif check_name == 'ml_validation':
                if 'sample_size' in check_results and not check_results['sample_size'].get('sufficient_samples', True):
                    summary['issues_found'].append("Insufficient sample size for ML")
        
        summary['total_issues'] = len(summary['issues_found'])
        
        return summary
    
    def _generate_recommendations(self, 
                                validation_results: Dict[str, Any], 
                                quality_scores: Dict[str, float]) -> List[str]:
        """
        Generate recommendations based on validation results.
        
        Args:
            validation_results: Validation results
            quality_scores: Quality scores
        
        Returns:
            List of recommendations
        """
        recommendations = []
        
        # Completeness recommendations
        if quality_scores.get('completeness', 1.0) < 0.9:
            recommendations.append("Consider implementing data imputation strategies for missing values")
        
        # Consistency recommendations
        if quality_scores.get('consistency', 1.0) < 0.9:
            recommendations.append("Review data type consistency and implement data standardization")
        
        # ML readiness recommendations
        if quality_scores.get('ml_readiness', 1.0) < 0.8:
            if 'ml_validation' in validation_results:
                ml_results = validation_results['ml_validation']
                
                if 'sample_size' in ml_results and not ml_results['sample_size'].get('sufficient_samples', True):
                    recommendations.append("Collect more data samples to meet ML requirements")
                
                if 'feature_variance' in ml_results and ml_results['feature_variance'].get('low_variance_features'):
                    recommendations.append("Remove or transform low-variance features")
                
                if 'feature_correlation' in ml_results and ml_results['feature_correlation'].get('high_correlation_pairs'):
                    recommendations.append("Address highly correlated features through feature selection or PCA")
        
        # Schema recommendations
        if quality_scores.get('schema_compliance', 1.0) < 1.0:
            recommendations.append("Update data schema or implement data transformation to ensure compliance")
        
        # General recommendations
        if quality_scores.get('overall', 1.0) < 0.8:
            recommendations.append("Implement comprehensive data quality monitoring and alerting")
            recommendations.append("Consider data quality improvement pipeline")
        
        return recommendations
    
    def _perform_cross_stage_validation(self, pipeline_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """
        Perform validation across multiple pipeline stages.
        
        Args:
            pipeline_data: Dictionary mapping stage names to DataFrames
        
        Returns:
            Cross-stage validation results
        """
        results = {
            'validation_passed': True,
            'consistency_checks': {},
            'data_lineage_checks': {},
            'transformation_checks': {}
        }
        
        stages = list(pipeline_data.keys())
        
        # Check data consistency across stages
        for i in range(len(stages) - 1):
            current_stage = stages[i]
            next_stage = stages[i + 1]
            
            current_data = pipeline_data[current_stage]
            next_data = pipeline_data[next_stage]
            
            # Check that data transformations are logical
            consistency_result = self._check_stage_consistency(current_data, next_data, current_stage, next_stage)
            results['consistency_checks'][f"{current_stage}_to_{next_stage}"] = consistency_result
            
            if not consistency_result.get('consistent', True):
                results['validation_passed'] = False
        
        return results
    
    def _check_stage_consistency(self, 
                               current_data: pd.DataFrame, 
                               next_data: pd.DataFrame, 
                               current_stage: str, 
                               next_stage: str) -> Dict[str, Any]:
        """
        Check consistency between two pipeline stages.
        
        Args:
            current_data: Data from current stage
            next_data: Data from next stage
            current_stage: Current stage name
            next_stage: Next stage name
        
        Returns:
            Consistency check results
        """
        result = {
            'consistent': True,
            'issues': []
        }
        
        # Check for major data loss (more than 50% reduction)
        if len(next_data) < len(current_data) * 0.5:
            result['consistent'] = False
            result['issues'].append(f"Significant data loss from {current_stage} to {next_stage}: {len(current_data)} -> {len(next_data)}")
        
        # Check for unexpected column changes
        current_cols = set(current_data.columns)
        next_cols = set(next_data.columns)
        
        # For cleaning stage, we expect some columns might be removed
        if current_stage != 'cleaned_data_validation':
            removed_cols = current_cols - next_cols
            if removed_cols:
                result['issues'].append(f"Columns removed unexpectedly: {list(removed_cols)}")
        
        # For feature engineering, we expect new columns
        if 'feature' in next_stage.lower():
            added_cols = next_cols - current_cols
            if not added_cols:
                result['issues'].append("No new features created in feature engineering stage")
        
        return result
    
    def _generate_pipeline_summary(self, all_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate summary across all pipeline stages.
        
        Args:
            all_results: Results from all validation stages
        
        Returns:
            Pipeline summary
        """
        summary = {
            'total_stages_validated': len([k for k in all_results.keys() if k != 'cross_stage_validation']),
            'stages_passed': len([v for v in all_results.values() if isinstance(v, dict) and v.get('validation_passed', False)]),
            'critical_issues': [],
            'overall_recommendations': []
        }
        
        # Collect critical issues
        for stage, results in all_results.items():
            if isinstance(results, dict) and results.get('critical_failures'):
                summary['critical_issues'].extend([f"{stage}: {issue}" for issue in results['critical_failures']])
        
        # Generate overall recommendations
        failed_stages = [stage for stage, results in all_results.items() if isinstance(results, dict) and not results.get('validation_passed', True)]
        
        if failed_stages:
            summary['overall_recommendations'].append(f"Address validation failures in stages: {', '.join(failed_stages)}")
        
        if len(summary['critical_issues']) > 5:
            summary['overall_recommendations'].append("Consider implementing automated data quality monitoring")
        
        return summary
    
    def _calculate_overall_quality_score(self, all_results: Dict[str, Any]) -> float:
        """
        Calculate overall quality score across all stages.
        
        Args:
            all_results: Results from all validation stages
        
        Returns:
            Overall quality score
        """
        quality_scores = []
        
        for stage, results in all_results.items():
            if isinstance(results, dict) and 'quality_scores' in results:
                stage_scores = results['quality_scores']
                if 'overall' in stage_scores:
                    quality_scores.append(stage_scores['overall'])
        
        return np.mean(quality_scores) if quality_scores else 0.0
    
    def _trigger_alerts(self, validation_results: Dict[str, Any]):
        """
        Trigger alerts for critical validation failures.
        
        Args:
            validation_results: Validation results containing failures
        """
        alert_config = self.config.get('alert_configuration', {})
        
        if not alert_config.get('enable_alerts', True):
            return
        
        critical_failures = validation_results.get('critical_failures', [])
        stage = validation_results.get('stage', 'unknown')
        
        alert_message = f"Critical validation failures in stage '{stage}':\n"
        for failure in critical_failures:
            alert_message += f"- {failure}\n"
        
        alert_channels = alert_config.get('alert_channels', ['log'])
        
        # Log alert
        if 'log' in alert_channels:
            logger.error(f"VALIDATION ALERT: {alert_message}")
        
        # File alert
        if 'file' in alert_channels:
            try:
                alert_file = Path("validation_alerts.log")
                with open(alert_file, 'a') as f:
                    f.write(f"{datetime.now()}: {alert_message}\n")
            except Exception as e:
                logger.error(f"Failed to write alert to file: {str(e)}")
    
    def _save_validation_report(self, validation_results: Dict[str, Any], stage: str):
        """
        Save validation report to file.
        
        Args:
            validation_results: Validation results to save
            stage: Pipeline stage name
        """
        try:
            # Create reports directory if it doesn't exist
            reports_dir = Path("validation_reports")
            reports_dir.mkdir(exist_ok=True)
            
            # Save as JSON
            report_file = reports_dir / f"{stage}_validation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
            # Convert datetime objects to strings for JSON serialization
            serializable_results = self._make_json_serializable(validation_results)
            
            with open(report_file, 'w') as f:
                json.dump(serializable_results, f, indent=2)
            
            logger.info(f"Validation report saved to {report_file}")
            
        except Exception as e:
            logger.error(f"Failed to save validation report: {str(e)}")
    
    def _make_json_serializable(self, obj: Any) -> Any:
        """
        Convert object to JSON serializable format.
        
        Args:
            obj: Object to convert
        
        Returns:
            JSON serializable object
        """
        if isinstance(obj, dict):
            return {k: self._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, datetime):
            return obj.isoformat()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif pd.isna(obj):
            return None
        else:
            return obj
    
    def get_validation_history(self) -> List[Dict[str, Any]]:
        """
        Get the validation history.
        
        Returns:
            List of validation results from history
        """
        return self.validation_history.copy()
    
    def get_quality_metrics(self) -> Dict[str, Any]:
        """
        Get aggregated quality metrics across all validations.
        
        Returns:
            Dictionary of quality metrics
        """
        if not self.validation_history:
            return {}
        
        # Aggregate quality scores across all validations
        all_scores = []
        stage_scores = {}
        
        for validation in self.validation_history:
            if 'quality_scores' in validation:
                scores = validation['quality_scores']
                all_scores.append(scores.get('overall', 0))
                
                stage = validation.get('stage', 'unknown')
                if stage not in stage_scores:
                    stage_scores[stage] = []
                stage_scores[stage].append(scores.get('overall', 0))
        
        metrics = {
            'average_quality_score': np.mean(all_scores) if all_scores else 0,
            'quality_trend': 'improving' if len(all_scores) > 1 and all_scores[-1] > all_scores[0] else 'stable',
            'stage_quality_scores': {stage: np.mean(scores) for stage, scores in stage_scores.items()},
            'total_validations': len(self.validation_history),
            'successful_validations': len([v for v in self.validation_history if v.get('validation_passed', False)])
        }
        
        return metrics
    
    def configure_for_snowflake_data(self):
        """
        Configure the validation pipeline for typical Snowflake analytics data.
        """
        snowflake_config = {
            'validation_thresholds': {
                'completeness': 0.98,  # High completeness for analytics
                'consistency': 0.99,   # High consistency for analytics
                'accuracy': 0.95,      # Moderate accuracy tolerance
                'schema_compliance': 1.0  # Strict schema compliance
            },
            'temporal_validation': {
                'check_time_continuity': True,
                'check_time_ordering': True,
                'check_future_dates': True,
                'max_time_gap_hours': 1  # Expect frequent data updates
            },
            'ml_validation': {
                'min_samples': 10000,  # Large datasets expected
                'max_missing_rate': 0.02,  # Low tolerance for missing data
                'min_variance': 0.0001,  # Low variance threshold for credits/metrics
                'max_correlation': 0.9   # Allow some correlation in usage metrics
            },
            'data_quality_checks': {
                'check_missing_values': True,
                'check_duplicates': True,
                'check_outliers': True,
                'check_data_types': True,
                'check_constraints': True,
                'check_distributions': True,
                'check_correlations': True
            }
        }
        
        self.config.update(snowflake_config)
        logger.info("Configured ValidationPipeline for Snowflake analytics data")
