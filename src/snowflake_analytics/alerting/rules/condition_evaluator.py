"""
Condition Evaluator

Evaluates rule conditions against metrics with support for different
aggregations, time windows, and comparison operators.
"""

import statistics
from typing import Dict, List, Any, Optional, Callable, Union
from datetime import datetime, timedelta
import logging

from .rule_engine import RuleCondition


class ConditionEvaluator:
    """
    Evaluates rule conditions against metric data
    
    Supports various aggregation functions, time windows,
    and comparison operators for flexible condition evaluation.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def evaluate_condition(self, condition: RuleCondition, 
                          metric_provider: Callable[[str, int], List[Dict[str, Any]]]) -> bool:
        """
        Evaluate a single condition against metric data
        
        Args:
            condition: The condition to evaluate
            metric_provider: Function to retrieve metric data
            
        Returns:
            bool: True if condition is met, False otherwise
        """
        try:
            # Get metric data for the time window
            metric_data = metric_provider(condition.metric_name, condition.time_window)
            
            if not metric_data:
                self.logger.warning(f"No data found for metric: {condition.metric_name}")
                return False
            
            # Apply aggregation
            aggregated_value = self._apply_aggregation(metric_data, condition.aggregation)
            
            if aggregated_value is None:
                self.logger.warning(f"Aggregation failed for metric: {condition.metric_name}")
                return False
            
            # Compare with threshold
            result = self._compare_values(aggregated_value, condition.operator, condition.value)
            
            self.logger.debug(f"Condition {condition.id}: {aggregated_value} {condition.operator} "
                            f"{condition.value} = {result}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error evaluating condition {condition.id}: {e}")
            return False
    
    def _apply_aggregation(self, metric_data: List[Dict[str, Any]], aggregation: str) -> Optional[float]:
        """
        Apply aggregation function to metric data
        
        Args:
            metric_data: List of metric data points
            aggregation: Aggregation function name
            
        Returns:
            Optional[float]: Aggregated value or None if failed
        """
        if not metric_data:
            return None
        
        try:
            values = [float(point['value']) for point in metric_data if 'value' in point]
            
            if not values:
                return None
            
            if aggregation == "latest":
                return values[-1]  # Most recent value
            elif aggregation == "avg":
                return statistics.mean(values)
            elif aggregation == "max":
                return max(values)
            elif aggregation == "min":
                return min(values)
            elif aggregation == "sum":
                return sum(values)
            elif aggregation == "count":
                return len(values)
            elif aggregation == "median":
                return statistics.median(values)
            elif aggregation == "std":
                return statistics.stdev(values) if len(values) > 1 else 0
            elif aggregation == "variance":
                return statistics.variance(values) if len(values) > 1 else 0
            elif aggregation == "p95":
                return statistics.quantiles(values, n=20)[18] if len(values) >= 20 else max(values)
            elif aggregation == "p99":
                return statistics.quantiles(values, n=100)[98] if len(values) >= 100 else max(values)
            elif aggregation == "first":
                return values[0]  # First value
            elif aggregation == "rate":
                return self._calculate_rate(metric_data)
            elif aggregation == "delta":
                return self._calculate_delta(values)
            elif aggregation == "anomaly_score":
                return self._calculate_anomaly_score(values)
            else:
                self.logger.warning(f"Unknown aggregation function: {aggregation}")
                return None
                
        except Exception as e:
            self.logger.error(f"Error applying aggregation {aggregation}: {e}")
            return None
    
    def _calculate_rate(self, metric_data: List[Dict[str, Any]]) -> float:
        """Calculate rate of change per second"""
        if len(metric_data) < 2:
            return 0.0
        
        try:
            # Sort by timestamp
            sorted_data = sorted(metric_data, key=lambda x: x.get('timestamp', ''))
            
            first_point = sorted_data[0]
            last_point = sorted_data[-1]
            
            first_time = datetime.fromisoformat(first_point['timestamp'])
            last_time = datetime.fromisoformat(last_point['timestamp'])
            
            time_diff = (last_time - first_time).total_seconds()
            
            if time_diff <= 0:
                return 0.0
            
            value_diff = float(last_point['value']) - float(first_point['value'])
            
            return value_diff / time_diff
            
        except Exception as e:
            self.logger.error(f"Error calculating rate: {e}")
            return 0.0
    
    def _calculate_delta(self, values: List[float]) -> float:
        """Calculate difference between latest and first value"""
        if len(values) < 2:
            return 0.0
        
        return values[-1] - values[0]
    
    def _calculate_anomaly_score(self, values: List[float]) -> float:
        """Calculate anomaly score using z-score"""
        if len(values) < 3:
            return 0.0
        
        try:
            mean_val = statistics.mean(values[:-1])  # Exclude latest value
            std_val = statistics.stdev(values[:-1])
            
            if std_val == 0:
                return 0.0
            
            latest_value = values[-1]
            z_score = abs(latest_value - mean_val) / std_val
            
            return z_score
            
        except Exception as e:
            self.logger.error(f"Error calculating anomaly score: {e}")
            return 0.0
    
    def _compare_values(self, left: float, operator: str, right: Union[float, str]) -> bool:
        """
        Compare two values using the specified operator
        
        Args:
            left: Left operand (actual value)
            operator: Comparison operator
            right: Right operand (threshold value)
            
        Returns:
            bool: Comparison result
        """
        try:
            # Convert right operand to float if it's a string
            if isinstance(right, str):
                try:
                    right = float(right)
                except ValueError:
                    # Handle string comparison
                    return self._compare_strings(str(left), operator, right)
            
            # Numeric comparison
            if operator == ">":
                return left > right
            elif operator == "<":
                return left < right
            elif operator == ">=":
                return left >= right
            elif operator == "<=":
                return left <= right
            elif operator == "==":
                return abs(left - right) < 1e-9  # Float equality with tolerance
            elif operator == "!=":
                return abs(left - right) >= 1e-9
            else:
                self.logger.warning(f"Unknown operator: {operator}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error comparing values: {e}")
            return False
    
    def _compare_strings(self, left: str, operator: str, right: str) -> bool:
        """Compare string values"""
        if operator == "==":
            return left == right
        elif operator == "!=":
            return left != right
        else:
            self.logger.warning(f"String comparison not supported for operator: {operator}")
            return False
    
    def evaluate_multiple_conditions(self, conditions: List[RuleCondition],
                                   metric_provider: Callable[[str, int], List[Dict[str, Any]]]) -> List[bool]:
        """
        Evaluate multiple conditions in parallel
        
        Args:
            conditions: List of conditions to evaluate
            metric_provider: Function to retrieve metric data
            
        Returns:
            List[bool]: Results for each condition
        """
        results = []
        
        for condition in conditions:
            result = self.evaluate_condition(condition, metric_provider)
            results.append(result)
        
        return results
    
    def get_condition_details(self, condition: RuleCondition,
                            metric_provider: Callable[[str, int], List[Dict[str, Any]]]) -> Dict[str, Any]:
        """
        Get detailed information about condition evaluation
        
        Args:
            condition: The condition to analyze
            metric_provider: Function to retrieve metric data
            
        Returns:
            Dict[str, Any]: Detailed condition information
        """
        try:
            # Get metric data
            metric_data = metric_provider(condition.metric_name, condition.time_window)
            
            if not metric_data:
                return {
                    'condition_id': condition.id,
                    'metric_name': condition.metric_name,
                    'status': 'no_data',
                    'data_points': 0,
                    'aggregated_value': None,
                    'threshold': condition.value,
                    'operator': condition.operator,
                    'result': False
                }
            
            # Apply aggregation
            aggregated_value = self._apply_aggregation(metric_data, condition.aggregation)
            
            # Evaluate condition
            result = self._compare_values(aggregated_value, condition.operator, condition.value) if aggregated_value is not None else False
            
            # Get additional statistics
            values = [float(point['value']) for point in metric_data if 'value' in point]
            
            stats = {}
            if values:
                stats = {
                    'min': min(values),
                    'max': max(values),
                    'mean': statistics.mean(values),
                    'median': statistics.median(values),
                    'std': statistics.stdev(values) if len(values) > 1 else 0,
                    'count': len(values)
                }
            
            return {
                'condition_id': condition.id,
                'metric_name': condition.metric_name,
                'status': 'evaluated',
                'data_points': len(metric_data),
                'aggregation': condition.aggregation,
                'aggregated_value': aggregated_value,
                'threshold': condition.value,
                'operator': condition.operator,
                'result': result,
                'time_window': condition.time_window,
                'statistics': stats,
                'raw_data': metric_data[-10:] if len(metric_data) > 10 else metric_data  # Last 10 points
            }
            
        except Exception as e:
            self.logger.error(f"Error getting condition details for {condition.id}: {e}")
            return {
                'condition_id': condition.id,
                'metric_name': condition.metric_name,
                'status': 'error',
                'error': str(e),
                'result': False
            }
    
    def validate_condition(self, condition: RuleCondition) -> List[str]:
        """
        Validate a condition configuration
        
        Args:
            condition: The condition to validate
            
        Returns:
            List[str]: List of validation errors (empty if valid)
        """
        errors = []
        
        # Validate basic fields
        if not condition.id:
            errors.append("Condition ID is required")
        
        if not condition.metric_name:
            errors.append("Metric name is required")
        
        # Validate operator
        valid_operators = [">", "<", ">=", "<=", "==", "!="]
        if condition.operator not in valid_operators:
            errors.append(f"Invalid operator: {condition.operator}. Must be one of {valid_operators}")
        
        # Validate aggregation
        valid_aggregations = [
            "latest", "avg", "max", "min", "sum", "count", "median", 
            "std", "variance", "p95", "p99", "first", "rate", "delta", "anomaly_score"
        ]
        if condition.aggregation not in valid_aggregations:
            errors.append(f"Invalid aggregation: {condition.aggregation}. Must be one of {valid_aggregations}")
        
        # Validate time window
        if condition.time_window < 0:
            errors.append("Time window must be non-negative")
        
        # Validate value type
        if not isinstance(condition.value, (int, float, str)):
            errors.append("Condition value must be numeric or string")
        
        return errors
    
    def get_supported_aggregations(self) -> List[Dict[str, str]]:
        """
        Get list of supported aggregation functions
        
        Returns:
            List[Dict[str, str]]: List of aggregation functions with descriptions
        """
        return [
            {"name": "latest", "description": "Most recent value"},
            {"name": "avg", "description": "Average (mean) of all values"},
            {"name": "max", "description": "Maximum value"},
            {"name": "min", "description": "Minimum value"},
            {"name": "sum", "description": "Sum of all values"},
            {"name": "count", "description": "Count of data points"},
            {"name": "median", "description": "Median value"},
            {"name": "std", "description": "Standard deviation"},
            {"name": "variance", "description": "Variance"},
            {"name": "p95", "description": "95th percentile"},
            {"name": "p99", "description": "99th percentile"},
            {"name": "first", "description": "First (oldest) value"},
            {"name": "rate", "description": "Rate of change per second"},
            {"name": "delta", "description": "Difference between latest and first value"},
            {"name": "anomaly_score", "description": "Anomaly score using z-score"}
        ]
    
    def get_supported_operators(self) -> List[Dict[str, str]]:
        """
        Get list of supported comparison operators
        
        Returns:
            List[Dict[str, str]]: List of operators with descriptions
        """
        return [
            {"name": ">", "description": "Greater than"},
            {"name": "<", "description": "Less than"},
            {"name": ">=", "description": "Greater than or equal to"},
            {"name": "<=", "description": "Less than or equal to"},
            {"name": "==", "description": "Equal to"},
            {"name": "!=", "description": "Not equal to"}
        ]
    
    def test_condition(self, condition: RuleCondition, 
                      test_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Test a condition against provided test data
        
        Args:
            condition: The condition to test
            test_data: Test metric data
            
        Returns:
            Dict[str, Any]: Test results
        """
        try:
            # Mock metric provider with test data
            def mock_provider(metric_name: str, time_window: int) -> List[Dict[str, Any]]:
                if metric_name == condition.metric_name:
                    return test_data
                return []
            
            # Get detailed results
            details = self.get_condition_details(condition, mock_provider)
            
            return {
                'condition_id': condition.id,
                'test_passed': details['result'],
                'details': details,
                'test_data_points': len(test_data)
            }
            
        except Exception as e:
            return {
                'condition_id': condition.id,
                'test_passed': False,
                'error': str(e),
                'test_data_points': len(test_data)
            }