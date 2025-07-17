"""
Severity Calculator

Calculates dynamic alert severity based on metric values, rule configuration,
and contextual factors with intelligent severity escalation.
"""

import statistics
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime, timedelta
from enum import Enum
import logging

from .rule_engine import AlertRule, RuleCondition


class SeverityLevel(Enum):
    """Severity levels with numeric values for comparison"""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


class SeverityCalculator:
    """
    Calculates dynamic alert severity based on multiple factors
    
    Uses rule configuration, metric values, historical data, and context
    to determine appropriate alert severity levels.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Severity calculation parameters
        self.base_severity_weights = self.config.get('base_severity_weights', {
            'low': 1.0,
            'medium': 2.0,
            'high': 3.0,
            'critical': 4.0
        })
        
        self.escalation_factors = self.config.get('escalation_factors', {
            'threshold_breach_magnitude': 0.5,
            'duration_factor': 0.3,
            'frequency_factor': 0.2,
            'trend_factor': 0.3,
            'business_impact': 0.4
        })
        
        # Severity thresholds
        self.severity_thresholds = self.config.get('severity_thresholds', {
            'low': 1.0,
            'medium': 2.0,
            'high': 3.0,
            'critical': 4.0
        })
    
    def calculate_severity(self, rule: AlertRule, 
                          metric_provider: Callable[[str, int], List[Dict[str, Any]]]) -> str:
        """
        Calculate dynamic severity for an alert rule
        
        Args:
            rule: The alert rule
            metric_provider: Function to retrieve metric data
            
        Returns:
            str: Calculated severity level
        """
        try:
            # Start with base severity
            base_severity = self._get_base_severity_score(rule.severity)
            
            # Calculate severity factors
            factors = self._calculate_severity_factors(rule, metric_provider)
            
            # Apply factors to base severity
            dynamic_severity = self._apply_severity_factors(base_severity, factors)
            
            # Convert to severity level
            severity_level = self._score_to_severity_level(dynamic_severity)
            
            self.logger.debug(f"Calculated severity for rule {rule.id}: {severity_level} "
                            f"(base: {base_severity:.2f}, final: {dynamic_severity:.2f})")
            
            return severity_level
            
        except Exception as e:
            self.logger.error(f"Error calculating severity for rule {rule.id}: {e}")
            return rule.severity  # Fallback to original severity
    
    def _get_base_severity_score(self, severity: str) -> float:
        """Get numeric score for base severity level"""
        return self.base_severity_weights.get(severity.lower(), 2.0)
    
    def _calculate_severity_factors(self, rule: AlertRule, 
                                   metric_provider: Callable[[str, int], List[Dict[str, Any]]]) -> Dict[str, float]:
        """Calculate various factors that influence severity"""
        factors = {}
        
        # Threshold breach magnitude factor
        factors['threshold_breach_magnitude'] = self._calculate_threshold_breach_factor(rule, metric_provider)
        
        # Duration factor (how long the condition has been true)
        factors['duration_factor'] = self._calculate_duration_factor(rule, metric_provider)
        
        # Frequency factor (how often this alert has fired)
        factors['frequency_factor'] = self._calculate_frequency_factor(rule)
        
        # Trend factor (is the metric getting worse?)
        factors['trend_factor'] = self._calculate_trend_factor(rule, metric_provider)
        
        # Business impact factor
        factors['business_impact'] = self._calculate_business_impact_factor(rule)
        
        return factors
    
    def _calculate_threshold_breach_factor(self, rule: AlertRule, 
                                         metric_provider: Callable[[str, int], List[Dict[str, Any]]]) -> float:
        """Calculate factor based on how much the threshold is breached"""
        try:
            # Get the primary condition (first condition in first group)
            if not rule.condition_group.conditions:
                return 1.0
            
            condition = rule.condition_group.conditions[0]
            
            # Get current metric value
            metric_data = metric_provider(condition.metric_name, condition.time_window)
            if not metric_data:
                return 1.0
            
            # Calculate aggregated value
            from .condition_evaluator import ConditionEvaluator
            evaluator = ConditionEvaluator()
            current_value = evaluator._apply_aggregation(metric_data, condition.aggregation)
            
            if current_value is None:
                return 1.0
            
            # Calculate breach magnitude
            threshold_value = condition.value
            if isinstance(threshold_value, str):
                try:
                    threshold_value = float(threshold_value)
                except:
                    return 1.0
            
            # Calculate relative breach
            if condition.operator in ['>', '>=']:
                if threshold_value > 0:
                    breach_ratio = (current_value - threshold_value) / threshold_value
                else:
                    breach_ratio = current_value - threshold_value
            elif condition.operator in ['<', '<=']:
                if threshold_value > 0:
                    breach_ratio = (threshold_value - current_value) / threshold_value
                else:
                    breach_ratio = threshold_value - current_value
            else:
                breach_ratio = 0
            
            # Convert to severity factor (0.5 to 2.0)
            return max(0.5, min(2.0, 1.0 + breach_ratio))
            
        except Exception as e:
            self.logger.error(f"Error calculating threshold breach factor: {e}")
            return 1.0
    
    def _calculate_duration_factor(self, rule: AlertRule, 
                                  metric_provider: Callable[[str, int], List[Dict[str, Any]]]) -> float:
        """Calculate factor based on how long the condition has been true"""
        try:
            # Look at recent history to see how long condition has been true
            if not rule.condition_group.conditions:
                return 1.0
            
            condition = rule.condition_group.conditions[0]
            
            # Get extended history (last hour)
            metric_data = metric_provider(condition.metric_name, 3600)
            if not metric_data:
                return 1.0
            
            # Evaluate condition over time
            from .condition_evaluator import ConditionEvaluator
            evaluator = ConditionEvaluator()
            
            # Check how many recent points violate the condition
            violating_points = 0
            total_points = min(len(metric_data), 20)  # Check last 20 points
            
            for i in range(max(0, len(metric_data) - total_points), len(metric_data)):
                point_data = [metric_data[i]]
                aggregated_value = evaluator._apply_aggregation(point_data, condition.aggregation)
                
                if aggregated_value is not None:
                    if evaluator._compare_values(aggregated_value, condition.operator, condition.value):
                        violating_points += 1
            
            # Calculate duration factor
            if total_points > 0:
                violation_ratio = violating_points / total_points
                return max(0.5, min(2.0, 1.0 + violation_ratio))
            
            return 1.0
            
        except Exception as e:
            self.logger.error(f"Error calculating duration factor: {e}")
            return 1.0
    
    def _calculate_frequency_factor(self, rule: AlertRule) -> float:
        """Calculate factor based on alert frequency"""
        try:
            # Use trigger count to determine frequency
            if rule.trigger_count == 0:
                return 1.0
            
            # Calculate triggers per day (assuming rule has been active for at least a day)
            days_active = max(1, (datetime.now() - rule.created_at).days)
            triggers_per_day = rule.trigger_count / days_active
            
            # Higher frequency increases severity
            if triggers_per_day > 10:  # Very frequent
                return 1.5
            elif triggers_per_day > 5:  # Frequent
                return 1.3
            elif triggers_per_day > 1:  # Moderate
                return 1.1
            else:  # Rare
                return 0.9
            
        except Exception as e:
            self.logger.error(f"Error calculating frequency factor: {e}")
            return 1.0
    
    def _calculate_trend_factor(self, rule: AlertRule, 
                               metric_provider: Callable[[str, int], List[Dict[str, Any]]]) -> float:
        """Calculate factor based on metric trend"""
        try:
            if not rule.condition_group.conditions:
                return 1.0
            
            condition = rule.condition_group.conditions[0]
            
            # Get recent trend data
            metric_data = metric_provider(condition.metric_name, 3600)  # 1 hour
            if len(metric_data) < 10:
                return 1.0
            
            # Calculate trend using linear regression
            values = [float(point['value']) for point in metric_data if 'value' in point]
            if len(values) < 10:
                return 1.0
            
            # Simple trend calculation
            n = len(values)
            x = list(range(n))
            y = values
            
            # Calculate slope
            sum_x = sum(x)
            sum_y = sum(y)
            sum_xy = sum(x[i] * y[i] for i in range(n))
            sum_x2 = sum(xi * xi for xi in x)
            
            denominator = n * sum_x2 - sum_x * sum_x
            if denominator == 0:
                return 1.0
            
            slope = (n * sum_xy - sum_x * sum_y) / denominator
            
            # Convert slope to trend factor
            mean_value = sum_y / n
            if mean_value > 0:
                normalized_slope = slope / mean_value
                
                # If metric is trending in alarming direction, increase severity
                if condition.operator in ['>', '>='] and normalized_slope > 0:
                    return min(1.5, 1.0 + abs(normalized_slope))
                elif condition.operator in ['<', '<='] and normalized_slope < 0:
                    return min(1.5, 1.0 + abs(normalized_slope))
            
            return 1.0
            
        except Exception as e:
            self.logger.error(f"Error calculating trend factor: {e}")
            return 1.0
    
    def _calculate_business_impact_factor(self, rule: AlertRule) -> float:
        """Calculate factor based on business impact"""
        try:
            # Use rule tags to determine business impact
            business_impact = rule.tags.get('business_impact', 'medium')
            
            impact_factors = {
                'low': 0.8,
                'medium': 1.0,
                'high': 1.3,
                'critical': 1.5
            }
            
            factor = impact_factors.get(business_impact.lower(), 1.0)
            
            # Consider team priority
            if rule.team in ['production', 'infrastructure', 'security']:
                factor *= 1.2
            
            # Consider time of day (higher impact during business hours)
            current_hour = datetime.now().hour
            if 9 <= current_hour <= 17:  # Business hours
                factor *= 1.1
            
            return factor
            
        except Exception as e:
            self.logger.error(f"Error calculating business impact factor: {e}")
            return 1.0
    
    def _apply_severity_factors(self, base_severity: float, factors: Dict[str, float]) -> float:
        """Apply severity factors to base severity"""
        dynamic_severity = base_severity
        
        for factor_name, factor_value in factors.items():
            weight = self.escalation_factors.get(factor_name, 0.1)
            
            # Apply weighted factor
            if factor_value > 1.0:
                dynamic_severity += (factor_value - 1.0) * weight
            elif factor_value < 1.0:
                dynamic_severity -= (1.0 - factor_value) * weight
        
        return max(0.5, min(5.0, dynamic_severity))  # Clamp to reasonable range
    
    def _score_to_severity_level(self, score: float) -> str:
        """Convert severity score to severity level"""
        if score >= self.severity_thresholds['critical']:
            return 'critical'
        elif score >= self.severity_thresholds['high']:
            return 'high'
        elif score >= self.severity_thresholds['medium']:
            return 'medium'
        else:
            return 'low'
    
    def get_severity_explanation(self, rule: AlertRule, 
                               metric_provider: Callable[[str, int], List[Dict[str, Any]]]) -> Dict[str, Any]:
        """Get detailed explanation of severity calculation"""
        try:
            base_severity = self._get_base_severity_score(rule.severity)
            factors = self._calculate_severity_factors(rule, metric_provider)
            dynamic_severity = self._apply_severity_factors(base_severity, factors)
            final_severity = self._score_to_severity_level(dynamic_severity)
            
            return {
                'rule_id': rule.id,
                'rule_name': rule.name,
                'base_severity': rule.severity,
                'base_severity_score': base_severity,
                'calculated_severity': final_severity,
                'calculated_severity_score': dynamic_severity,
                'factors': factors,
                'factor_weights': self.escalation_factors,
                'explanation': self._generate_severity_explanation(rule, factors, base_severity, dynamic_severity)
            }
            
        except Exception as e:
            self.logger.error(f"Error getting severity explanation: {e}")
            return {
                'rule_id': rule.id,
                'error': str(e),
                'base_severity': rule.severity,
                'calculated_severity': rule.severity
            }
    
    def _generate_severity_explanation(self, rule: AlertRule, factors: Dict[str, float], 
                                     base_severity: float, dynamic_severity: float) -> str:
        """Generate human-readable explanation of severity calculation"""
        explanation_parts = []
        
        explanation_parts.append(f"Base severity: {rule.severity} (score: {base_severity:.2f})")
        
        for factor_name, factor_value in factors.items():
            if factor_value > 1.1:
                explanation_parts.append(f"{factor_name} increases severity ({factor_value:.2f}x)")
            elif factor_value < 0.9:
                explanation_parts.append(f"{factor_name} decreases severity ({factor_value:.2f}x)")
        
        explanation_parts.append(f"Final severity score: {dynamic_severity:.2f}")
        
        return "; ".join(explanation_parts)
    
    def calculate_severity_for_multiple_rules(self, rules: List[AlertRule], 
                                            metric_provider: Callable[[str, int], List[Dict[str, Any]]]) -> Dict[str, str]:
        """Calculate severity for multiple rules"""
        results = {}
        
        for rule in rules:
            try:
                severity = self.calculate_severity(rule, metric_provider)
                results[rule.id] = severity
            except Exception as e:
                self.logger.error(f"Error calculating severity for rule {rule.id}: {e}")
                results[rule.id] = rule.severity
        
        return results
    
    def get_severity_distribution(self, rules: List[AlertRule], 
                                metric_provider: Callable[[str, int], List[Dict[str, Any]]]) -> Dict[str, int]:
        """Get distribution of severity levels across rules"""
        severity_counts = {'low': 0, 'medium': 0, 'high': 0, 'critical': 0}
        
        for rule in rules:
            try:
                severity = self.calculate_severity(rule, metric_provider)
                severity_counts[severity] = severity_counts.get(severity, 0) + 1
            except Exception as e:
                self.logger.error(f"Error calculating severity for rule {rule.id}: {e}")
                severity_counts[rule.severity] = severity_counts.get(rule.severity, 0) + 1
        
        return severity_counts
    
    def update_severity_configuration(self, config: Dict[str, Any]):
        """Update severity calculation configuration"""
        if 'base_severity_weights' in config:
            self.base_severity_weights.update(config['base_severity_weights'])
        
        if 'escalation_factors' in config:
            self.escalation_factors.update(config['escalation_factors'])
        
        if 'severity_thresholds' in config:
            self.severity_thresholds.update(config['severity_thresholds'])
        
        self.logger.info("Updated severity calculation configuration")
    
    def validate_severity_configuration(self) -> List[str]:
        """Validate severity configuration"""
        errors = []
        
        # Check base severity weights
        required_severities = ['low', 'medium', 'high', 'critical']
        for severity in required_severities:
            if severity not in self.base_severity_weights:
                errors.append(f"Missing base severity weight for: {severity}")
        
        # Check escalation factors
        if not self.escalation_factors:
            errors.append("Escalation factors configuration is empty")
        
        # Check severity thresholds
        for severity in required_severities:
            if severity not in self.severity_thresholds:
                errors.append(f"Missing severity threshold for: {severity}")
        
        return errors