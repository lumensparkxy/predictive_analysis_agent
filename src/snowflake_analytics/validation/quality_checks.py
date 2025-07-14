"""
Data Quality Checker - Comprehensive data quality validation and monitoring.
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import pandas as pd
import numpy as np
from dataclasses import dataclass, field

from ..utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class QualityRule:
    """Definition of a data quality rule."""
    name: str
    description: str
    rule_type: str  # 'completeness', 'accuracy', 'consistency', 'timeliness', 'validity'
    severity: str   # 'critical', 'warning', 'info'
    threshold: Optional[float] = None
    enabled: bool = True


@dataclass
class QualityResult:
    """Result of a quality check."""
    rule_name: str
    passed: bool
    score: float  # 0.0 to 1.0
    message: str
    affected_rows: int = 0
    total_rows: int = 0
    details: Dict[str, Any] = field(default_factory=dict)


class DataQualityChecker:
    """
    Comprehensive data quality checker with configurable rules.
    
    Features:
    - Predefined quality rules for Snowflake data
    - Custom rule definitions
    - Quality scoring and reporting
    - Trend analysis and alerting
    - Performance optimization for large datasets
    """
    
    def __init__(self):
        """Initialize data quality checker with default rules."""
        self.rules = self._initialize_default_rules()
        self.quality_history = []
    
    def _initialize_default_rules(self) -> Dict[str, QualityRule]:
        """Initialize default quality rules."""
        rules = {}
        
        # Completeness rules
        rules['required_fields_complete'] = QualityRule(
            name='required_fields_complete',
            description='All required fields must have values',
            rule_type='completeness',
            severity='critical',
            threshold=0.95
        )
        
        rules['timestamp_completeness'] = QualityRule(
            name='timestamp_completeness',
            description='Timestamp fields must be complete',
            rule_type='completeness',
            severity='critical',
            threshold=0.99
        )
        
        # Accuracy rules
        rules['numeric_ranges'] = QualityRule(
            name='numeric_ranges',
            description='Numeric values must be within expected ranges',
            rule_type='accuracy',
            severity='warning',
            threshold=0.95
        )
        
        rules['positive_credits'] = QualityRule(
            name='positive_credits',
            description='Credit values must be non-negative',
            rule_type='accuracy',
            severity='critical',
            threshold=0.99
        )
        
        # Consistency rules
        rules['timestamp_order'] = QualityRule(
            name='timestamp_order',
            description='Start time must be before end time',
            rule_type='consistency',
            severity='critical',
            threshold=0.99
        )
        
        rules['user_name_format'] = QualityRule(
            name='user_name_format',
            description='User names must follow consistent format',
            rule_type='consistency',
            severity='warning',
            threshold=0.90
        )
        
        # Timeliness rules
        rules['data_freshness'] = QualityRule(
            name='data_freshness',
            description='Data must be recent (within expected timeframe)',
            rule_type='timeliness',
            severity='warning',
            threshold=0.95
        )
        
        # Validity rules
        rules['enum_values'] = QualityRule(
            name='enum_values',
            description='Enum fields must contain valid values',
            rule_type='validity',
            severity='critical',
            threshold=0.99
        )
        
        return rules
    
    def check_data_quality(self, df: pd.DataFrame, table_name: str) -> Dict[str, Any]:
        """
        Perform comprehensive data quality checks.
        
        Args:
            df: DataFrame to check
            table_name: Name of the table being checked
            
        Returns:
            Dictionary with quality results and overall score
        """
        logger.info(f"Checking data quality for {table_name} ({len(df)} rows)")
        start_time = datetime.now()
        
        results = {}
        all_results = []
        
        # Run all enabled quality rules
        for rule_name, rule in self.rules.items():
            if not rule.enabled:
                continue
                
            try:
                result = self._execute_quality_rule(df, rule, table_name)
                results[rule_name] = result
                all_results.append(result)
                
                log_level = 'error' if result.severity == 'critical' and not result.passed else 'debug'
                getattr(logger, log_level)(
                    f"Quality rule '{rule_name}': {'PASS' if result.passed else 'FAIL'} "
                    f"(score: {result.score:.3f}, affected: {result.affected_rows}/{result.total_rows})"
                )
                
            except Exception as e:
                logger.error(f"Failed to execute quality rule '{rule_name}': {e}")
                results[rule_name] = QualityResult(
                    rule_name=rule_name,
                    passed=False,
                    score=0.0,
                    message=f"Rule execution failed: {e}",
                    total_rows=len(df)
                )
        
        # Calculate overall quality score
        overall_score = self._calculate_overall_score(all_results)
        
        # Generate quality summary
        summary = self._generate_quality_summary(all_results, overall_score)
        
        check_duration = (datetime.now() - start_time).total_seconds()
        
        quality_report = {
            'table_name': table_name,
            'check_timestamp': start_time.isoformat(),
            'total_rows': len(df),
            'overall_score': overall_score,
            'summary': summary,
            'rule_results': {name: self._quality_result_to_dict(result) 
                           for name, result in results.items()},
            'check_duration_seconds': check_duration
        }
        
        # Store in history for trend analysis
        self.quality_history.append(quality_report)
        
        logger.info(
            f"Data quality check complete: {overall_score:.3f} overall score "
            f"({check_duration:.1f}s)"
        )
        
        return quality_report
    
    def _execute_quality_rule(self, df: pd.DataFrame, rule: QualityRule, table_name: str) -> QualityResult:
        """Execute a specific quality rule."""
        if rule.rule_type == 'completeness':
            return self._check_completeness(df, rule, table_name)
        elif rule.rule_type == 'accuracy':
            return self._check_accuracy(df, rule, table_name)
        elif rule.rule_type == 'consistency':
            return self._check_consistency(df, rule, table_name)
        elif rule.rule_type == 'timeliness':
            return self._check_timeliness(df, rule, table_name)
        elif rule.rule_type == 'validity':
            return self._check_validity(df, rule, table_name)
        else:
            return QualityResult(
                rule_name=rule.name,
                passed=False,
                score=0.0,
                message=f"Unknown rule type: {rule.rule_type}",
                total_rows=len(df)
            )
    
    def _check_completeness(self, df: pd.DataFrame, rule: QualityRule, table_name: str) -> QualityResult:
        """Check data completeness."""
        if rule.name == 'required_fields_complete':
            # Check required fields based on table type
            required_fields = self._get_required_fields(table_name)
            if not required_fields:
                return QualityResult(
                    rule_name=rule.name,
                    passed=True,
                    score=1.0,
                    message="No required fields defined",
                    total_rows=len(df)
                )
            
            available_fields = [f for f in required_fields if f in df.columns]
            if not available_fields:
                return QualityResult(
                    rule_name=rule.name,
                    passed=False,
                    score=0.0,
                    message=f"No required fields found in data",
                    total_rows=len(df)
                )
            
            # Calculate completeness for each field
            field_completeness = {}
            total_missing = 0
            
            for field in available_fields:
                missing_count = df[field].isna().sum()
                completeness = 1.0 - (missing_count / len(df))
                field_completeness[field] = completeness
                total_missing += missing_count
            
            overall_completeness = 1.0 - (total_missing / (len(df) * len(available_fields)))
            passed = overall_completeness >= (rule.threshold or 0.95)
            
            return QualityResult(
                rule_name=rule.name,
                passed=passed,
                score=overall_completeness,
                message=f"Overall completeness: {overall_completeness:.3f}",
                affected_rows=total_missing,
                total_rows=len(df) * len(available_fields),
                details={'field_completeness': field_completeness}
            )
        
        elif rule.name == 'timestamp_completeness':
            timestamp_fields = [col for col in df.columns if 'TIME' in col.upper()]
            if not timestamp_fields:
                return QualityResult(
                    rule_name=rule.name,
                    passed=True,
                    score=1.0,
                    message="No timestamp fields found",
                    total_rows=len(df)
                )
            
            missing_count = sum(df[field].isna().sum() for field in timestamp_fields)
            total_values = len(df) * len(timestamp_fields)
            completeness = 1.0 - (missing_count / total_values)
            passed = completeness >= (rule.threshold or 0.99)
            
            return QualityResult(
                rule_name=rule.name,
                passed=passed,
                score=completeness,
                message=f"Timestamp completeness: {completeness:.3f}",
                affected_rows=missing_count,
                total_rows=total_values
            )
        
        return QualityResult(
            rule_name=rule.name,
            passed=True,
            score=1.0,
            message="Rule not implemented",
            total_rows=len(df)
        )
    
    def _check_accuracy(self, df: pd.DataFrame, rule: QualityRule, table_name: str) -> QualityResult:
        """Check data accuracy."""
        if rule.name == 'positive_credits':
            credit_fields = [col for col in df.columns if 'CREDIT' in col.upper()]
            if not credit_fields:
                return QualityResult(
                    rule_name=rule.name,
                    passed=True,
                    score=1.0,
                    message="No credit fields found",
                    total_rows=len(df)
                )
            
            negative_count = 0
            total_values = 0
            
            for field in credit_fields:
                field_data = pd.to_numeric(df[field], errors='coerce')
                negative_count += (field_data < 0).sum()
                total_values += (~field_data.isna()).sum()
            
            if total_values == 0:
                accuracy = 1.0
            else:
                accuracy = 1.0 - (negative_count / total_values)
            
            passed = accuracy >= (rule.threshold or 0.99)
            
            return QualityResult(
                rule_name=rule.name,
                passed=passed,
                score=accuracy,
                message=f"Credit accuracy: {accuracy:.3f}",
                affected_rows=negative_count,
                total_rows=total_values
            )
        
        elif rule.name == 'numeric_ranges':
            numeric_fields = df.select_dtypes(include=[np.number]).columns
            if len(numeric_fields) == 0:
                return QualityResult(
                    rule_name=rule.name,
                    passed=True,
                    score=1.0,
                    message="No numeric fields found",
                    total_rows=len(df)
                )
            
            # Simple outlier detection using IQR
            outlier_count = 0
            total_values = 0
            
            for field in numeric_fields:
                field_data = df[field].dropna()
                if len(field_data) > 10:  # Need sufficient data for outlier detection
                    Q1 = field_data.quantile(0.25)
                    Q3 = field_data.quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    
                    outliers = ((field_data < lower_bound) | (field_data > upper_bound)).sum()
                    outlier_count += outliers
                    total_values += len(field_data)
            
            if total_values == 0:
                accuracy = 1.0
            else:
                accuracy = 1.0 - (outlier_count / total_values)
            
            passed = accuracy >= (rule.threshold or 0.95)
            
            return QualityResult(
                rule_name=rule.name,
                passed=passed,
                score=accuracy,
                message=f"Numeric range accuracy: {accuracy:.3f}",
                affected_rows=outlier_count,
                total_rows=total_values
            )
        
        return QualityResult(
            rule_name=rule.name,
            passed=True,
            score=1.0,
            message="Rule not implemented",
            total_rows=len(df)
        )
    
    def _check_consistency(self, df: pd.DataFrame, rule: QualityRule, table_name: str) -> QualityResult:
        """Check data consistency."""
        if rule.name == 'timestamp_order':
            # Check if start_time < end_time
            start_cols = [col for col in df.columns if 'START_TIME' in col.upper()]
            end_cols = [col for col in df.columns if 'END_TIME' in col.upper()]
            
            if not start_cols or not end_cols:
                return QualityResult(
                    rule_name=rule.name,
                    passed=True,
                    score=1.0,
                    message="No start/end time pairs found",
                    total_rows=len(df)
                )
            
            inconsistent_count = 0
            total_pairs = 0
            
            for start_col in start_cols:
                for end_col in end_cols:
                    if start_col.replace('START', '') == end_col.replace('END', ''):
                        start_time = pd.to_datetime(df[start_col], errors='coerce')
                        end_time = pd.to_datetime(df[end_col], errors='coerce')
                        
                        valid_pairs = ~(start_time.isna() | end_time.isna())
                        inconsistent = (start_time >= end_time) & valid_pairs
                        
                        inconsistent_count += inconsistent.sum()
                        total_pairs += valid_pairs.sum()
            
            if total_pairs == 0:
                consistency = 1.0
            else:
                consistency = 1.0 - (inconsistent_count / total_pairs)
            
            passed = consistency >= (rule.threshold or 0.99)
            
            return QualityResult(
                rule_name=rule.name,
                passed=passed,
                score=consistency,
                message=f"Timestamp order consistency: {consistency:.3f}",
                affected_rows=inconsistent_count,
                total_rows=total_pairs
            )
        
        return QualityResult(
            rule_name=rule.name,
            passed=True,
            score=1.0,
            message="Rule not implemented",
            total_rows=len(df)
        )
    
    def _check_timeliness(self, df: pd.DataFrame, rule: QualityRule, table_name: str) -> QualityResult:
        """Check data timeliness."""
        if rule.name == 'data_freshness':
            timestamp_fields = [col for col in df.columns if 'TIME' in col.upper()]
            if not timestamp_fields:
                return QualityResult(
                    rule_name=rule.name,
                    passed=True,
                    score=1.0,
                    message="No timestamp fields found",
                    total_rows=len(df)
                )
            
            # Check if data is within last 24 hours (configurable)
            cutoff_time = datetime.now() - timedelta(hours=24)
            fresh_count = 0
            total_records = 0
            
            for field in timestamp_fields:
                timestamps = pd.to_datetime(df[field], errors='coerce')
                valid_timestamps = ~timestamps.isna()
                fresh_records = (timestamps >= cutoff_time) & valid_timestamps
                
                fresh_count += fresh_records.sum()
                total_records += valid_timestamps.sum()
            
            if total_records == 0:
                freshness = 1.0
            else:
                freshness = fresh_count / total_records
            
            passed = freshness >= (rule.threshold or 0.95)
            
            return QualityResult(
                rule_name=rule.name,
                passed=passed,
                score=freshness,
                message=f"Data freshness: {freshness:.3f}",
                affected_rows=total_records - fresh_count,
                total_rows=total_records
            )
        
        return QualityResult(
            rule_name=rule.name,
            passed=True,
            score=1.0,
            message="Rule not implemented",
            total_rows=len(df)
        )
    
    def _check_validity(self, df: pd.DataFrame, rule: QualityRule, table_name: str) -> QualityResult:
        """Check data validity."""
        if rule.name == 'enum_values':
            # Check known enum fields
            enum_validations = {
                'EXECUTION_STATUS': ['SUCCESS', 'FAIL', 'CANCELLED', 'RUNNING'],
                'IS_SUCCESS': ['YES', 'NO'],
                'EVENT_TYPE': ['LOGIN', 'LOGOUT']
            }
            
            invalid_count = 0
            total_values = 0
            
            for field, valid_values in enum_validations.items():
                if field in df.columns:
                    field_data = df[field].dropna()
                    invalid = ~field_data.isin(valid_values)
                    invalid_count += invalid.sum()
                    total_values += len(field_data)
            
            if total_values == 0:
                validity = 1.0
            else:
                validity = 1.0 - (invalid_count / total_values)
            
            passed = validity >= (rule.threshold or 0.99)
            
            return QualityResult(
                rule_name=rule.name,
                passed=passed,
                score=validity,
                message=f"Enum validity: {validity:.3f}",
                affected_rows=invalid_count,
                total_rows=total_values
            )
        
        return QualityResult(
            rule_name=rule.name,
            passed=True,
            score=1.0,
            message="Rule not implemented",
            total_rows=len(df)
        )
    
    def _get_required_fields(self, table_name: str) -> List[str]:
        """Get required fields for a table."""
        required_fields_map = {
            'warehouse_usage_raw': ['START_TIME', 'WAREHOUSE_ID', 'CREDITS_USED'],
            'query_history_raw': ['QUERY_ID', 'START_TIME', 'USER_NAME'],
            'user_activity_raw': ['EVENT_ID', 'EVENT_TIMESTAMP', 'USER_NAME']
        }
        return required_fields_map.get(table_name, [])
    
    def _calculate_overall_score(self, results: List[QualityResult]) -> float:
        """Calculate overall quality score."""
        if not results:
            return 1.0
        
        # Weight scores by severity
        weights = {'critical': 3.0, 'warning': 2.0, 'info': 1.0}
        total_weighted_score = 0.0
        total_weight = 0.0
        
        for result in results:
            rule = self.rules.get(result.rule_name)
            if rule:
                weight = weights.get(rule.severity, 1.0)
                total_weighted_score += result.score * weight
                total_weight += weight
        
        return total_weighted_score / total_weight if total_weight > 0 else 1.0
    
    def _generate_quality_summary(self, results: List[QualityResult], overall_score: float) -> Dict[str, Any]:
        """Generate quality summary."""
        passed_rules = sum(1 for r in results if r.passed)
        failed_rules = len(results) - passed_rules
        
        critical_failures = sum(1 for r in results 
                              if not r.passed and self.rules.get(r.rule_name, {}).severity == 'critical')
        
        return {
            'overall_score': overall_score,
            'total_rules': len(results),
            'passed_rules': passed_rules,
            'failed_rules': failed_rules,
            'critical_failures': critical_failures,
            'quality_grade': self._get_quality_grade(overall_score),
            'needs_attention': critical_failures > 0 or overall_score < 0.8
        }
    
    def _get_quality_grade(self, score: float) -> str:
        """Convert quality score to grade."""
        if score >= 0.95:
            return 'A'
        elif score >= 0.90:
            return 'B'
        elif score >= 0.80:
            return 'C'
        elif score >= 0.70:
            return 'D'
        else:
            return 'F'
    
    def _quality_result_to_dict(self, result: QualityResult) -> Dict[str, Any]:
        """Convert QualityResult to dictionary."""
        rule = self.rules.get(result.rule_name)
        return {
            'passed': result.passed,
            'score': result.score,
            'message': result.message,
            'affected_rows': result.affected_rows,
            'total_rows': result.total_rows,
            'severity': rule.severity if rule else 'unknown',
            'rule_type': rule.rule_type if rule else 'unknown',
            'threshold': rule.threshold if rule else None,
            'details': result.details
        }
