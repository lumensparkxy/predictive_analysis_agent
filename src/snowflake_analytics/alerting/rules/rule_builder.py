"""
Rule Builder

Provides a fluent interface for building alert rules with complex conditions,
validation, and rule templates.
"""

from typing import Dict, List, Any, Optional, Union
from datetime import datetime
import logging
import json

from .rule_engine import (
    AlertRule, RuleCondition, RuleGroup, RuleType, RuleStatus,
    LogicalOperator
)


class RuleBuilder:
    """
    Fluent interface for building alert rules
    
    Provides methods to build complex alert rules with validation
    and rule templates.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.reset()
    
    def reset(self):
        """Reset the builder to initial state"""
        self._rule_id = None
        self._name = None
        self._description = None
        self._rule_type = RuleType.THRESHOLD
        self._severity = "medium"
        self._enabled = True
        self._cooldown_period = 300
        self._max_triggers_per_hour = 10
        self._auto_resolve = True
        self._auto_resolve_timeout = 3600
        self._tags = {}
        self._owner = ""
        self._team = ""
        self._condition_group = None
        return self
    
    def id(self, rule_id: str):
        """Set rule ID"""
        self._rule_id = rule_id
        return self
    
    def name(self, name: str):
        """Set rule name"""
        self._name = name
        return self
    
    def description(self, description: str):
        """Set rule description"""
        self._description = description
        return self
    
    def rule_type(self, rule_type: Union[RuleType, str]):
        """Set rule type"""
        if isinstance(rule_type, str):
            self._rule_type = RuleType(rule_type)
        else:
            self._rule_type = rule_type
        return self
    
    def severity(self, severity: str):
        """Set rule severity"""
        valid_severities = ["low", "medium", "high", "critical"]
        if severity not in valid_severities:
            raise ValueError(f"Invalid severity: {severity}. Must be one of {valid_severities}")
        self._severity = severity
        return self
    
    def enabled(self, enabled: bool = True):
        """Set rule enabled status"""
        self._enabled = enabled
        return self
    
    def cooldown(self, seconds: int):
        """Set cooldown period in seconds"""
        self._cooldown_period = seconds
        return self
    
    def max_triggers_per_hour(self, max_triggers: int):
        """Set maximum triggers per hour"""
        self._max_triggers_per_hour = max_triggers
        return self
    
    def auto_resolve(self, auto_resolve: bool = True, timeout: int = 3600):
        """Set auto-resolve settings"""
        self._auto_resolve = auto_resolve
        self._auto_resolve_timeout = timeout
        return self
    
    def tags(self, tags: Dict[str, Any]):
        """Set rule tags"""
        self._tags = tags
        return self
    
    def owner(self, owner: str):
        """Set rule owner"""
        self._owner = owner
        return self
    
    def team(self, team: str):
        """Set rule team"""
        self._team = team
        return self
    
    def condition_group(self, group: RuleGroup):
        """Set the main condition group"""
        self._condition_group = group
        return self
    
    def build(self) -> AlertRule:
        """Build the alert rule"""
        if not self._rule_id:
            raise ValueError("Rule ID is required")
        
        if not self._name:
            raise ValueError("Rule name is required")
        
        if not self._condition_group:
            raise ValueError("Condition group is required")
        
        rule = AlertRule(
            id=self._rule_id,
            name=self._name,
            description=self._description or "",
            rule_type=self._rule_type,
            condition_group=self._condition_group,
            severity=self._severity,
            enabled=self._enabled,
            cooldown_period=self._cooldown_period,
            max_triggers_per_hour=self._max_triggers_per_hour,
            auto_resolve=self._auto_resolve,
            auto_resolve_timeout=self._auto_resolve_timeout,
            tags=self._tags,
            owner=self._owner,
            team=self._team
        )
        
        # Validate the rule
        self._validate_rule(rule)
        
        self.logger.info(f"Built rule: {rule.name} ({rule.id})")
        return rule
    
    def _validate_rule(self, rule: AlertRule):
        """Validate the constructed rule"""
        # Validate basic fields
        if not rule.id or not rule.name:
            raise ValueError("Rule ID and name are required")
        
        # Validate condition group
        self._validate_condition_group(rule.condition_group)
        
        # Validate configuration values
        if rule.cooldown_period < 0:
            raise ValueError("Cooldown period must be non-negative")
        
        if rule.max_triggers_per_hour < 0:
            raise ValueError("Max triggers per hour must be non-negative")
        
        if rule.auto_resolve_timeout < 0:
            raise ValueError("Auto-resolve timeout must be non-negative")
    
    def _validate_condition_group(self, group: RuleGroup):
        """Validate a condition group"""
        if not group.conditions and not group.groups:
            raise ValueError("Condition group must have at least one condition or subgroup")
        
        # Validate conditions
        for condition in group.conditions:
            self._validate_condition(condition)
        
        # Validate nested groups
        for nested_group in group.groups:
            self._validate_condition_group(nested_group)
    
    def _validate_condition(self, condition: RuleCondition):
        """Validate a single condition"""
        if not condition.id or not condition.metric_name:
            raise ValueError("Condition ID and metric name are required")
        
        valid_operators = [">", "<", ">=", "<=", "==", "!="]
        if condition.operator not in valid_operators:
            raise ValueError(f"Invalid operator: {condition.operator}. Must be one of {valid_operators}")
        
        if condition.time_window < 0:
            raise ValueError("Time window must be non-negative")
        
        valid_aggregations = ["latest", "avg", "max", "min", "sum", "count"]
        if condition.aggregation not in valid_aggregations:
            raise ValueError(f"Invalid aggregation: {condition.aggregation}. Must be one of {valid_aggregations}")


class RuleTemplateBuilder:
    """
    Builder for creating rule templates
    
    Provides pre-defined rule patterns and templates for common use cases.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def cost_threshold_rule(self, rule_id: str, name: str, metric_name: str, 
                           threshold: float, severity: str = "high") -> AlertRule:
        """Create a cost threshold rule"""
        condition = RuleCondition(
            id=f"{rule_id}_condition",
            metric_name=metric_name,
            operator=">",
            value=threshold,
            time_window=300,
            aggregation="latest"
        )
        
        condition_group = RuleGroup(
            logical_operator=LogicalOperator.AND,
            conditions=[condition]
        )
        
        return (RuleBuilder()
                .id(rule_id)
                .name(name)
                .description(f"Alert when {metric_name} exceeds {threshold}")
                .rule_type(RuleType.THRESHOLD)
                .severity(severity)
                .condition_group(condition_group)
                .cooldown(300)
                .max_triggers_per_hour(5)
                .build())
    
    def performance_degradation_rule(self, rule_id: str, name: str, 
                                   performance_metric: str, baseline_multiplier: float = 1.5,
                                   severity: str = "medium") -> AlertRule:
        """Create a performance degradation rule"""
        condition = RuleCondition(
            id=f"{rule_id}_condition",
            metric_name=performance_metric,
            operator=">",
            value=baseline_multiplier,
            time_window=600,  # 10 minutes
            aggregation="avg"
        )
        
        condition_group = RuleGroup(
            logical_operator=LogicalOperator.AND,
            conditions=[condition]
        )
        
        return (RuleBuilder()
                .id(rule_id)
                .name(name)
                .description(f"Alert when {performance_metric} degrades by {baseline_multiplier}x")
                .rule_type(RuleType.THRESHOLD)
                .severity(severity)
                .condition_group(condition_group)
                .cooldown(600)
                .max_triggers_per_hour(3)
                .build())
    
    def usage_anomaly_rule(self, rule_id: str, name: str, usage_metric: str,
                          anomaly_threshold: float = 2.0, severity: str = "medium") -> AlertRule:
        """Create a usage anomaly rule"""
        condition = RuleCondition(
            id=f"{rule_id}_condition",
            metric_name=usage_metric,
            operator=">",
            value=anomaly_threshold,
            time_window=300,
            aggregation="latest"
        )
        
        condition_group = RuleGroup(
            logical_operator=LogicalOperator.AND,
            conditions=[condition]
        )
        
        return (RuleBuilder()
                .id(rule_id)
                .name(name)
                .description(f"Alert when {usage_metric} shows anomalous behavior")
                .rule_type(RuleType.ANOMALY)
                .severity(severity)
                .condition_group(condition_group)
                .cooldown(300)
                .max_triggers_per_hour(10)
                .build())
    
    def composite_cost_performance_rule(self, rule_id: str, name: str,
                                      cost_metric: str, cost_threshold: float,
                                      performance_metric: str, performance_threshold: float,
                                      severity: str = "high") -> AlertRule:
        """Create a composite rule for cost and performance"""
        cost_condition = RuleCondition(
            id=f"{rule_id}_cost_condition",
            metric_name=cost_metric,
            operator=">",
            value=cost_threshold,
            time_window=300,
            aggregation="latest"
        )
        
        performance_condition = RuleCondition(
            id=f"{rule_id}_performance_condition",
            metric_name=performance_metric,
            operator=">",
            value=performance_threshold,
            time_window=600,
            aggregation="avg"
        )
        
        condition_group = RuleGroup(
            logical_operator=LogicalOperator.AND,
            conditions=[cost_condition, performance_condition]
        )
        
        return (RuleBuilder()
                .id(rule_id)
                .name(name)
                .description(f"Alert when both cost and performance exceed thresholds")
                .rule_type(RuleType.COMPOSITE)
                .severity(severity)
                .condition_group(condition_group)
                .cooldown(600)
                .max_triggers_per_hour(3)
                .build())
    
    def escalating_threshold_rule(self, rule_id: str, name: str, metric_name: str,
                                thresholds: List[float], severities: List[str]) -> List[AlertRule]:
        """Create escalating threshold rules"""
        if len(thresholds) != len(severities):
            raise ValueError("Number of thresholds must match number of severities")
        
        rules = []
        for i, (threshold, severity) in enumerate(zip(thresholds, severities)):
            escalation_rule_id = f"{rule_id}_level_{i+1}"
            escalation_name = f"{name} - Level {i+1}"
            
            condition = RuleCondition(
                id=f"{escalation_rule_id}_condition",
                metric_name=metric_name,
                operator=">",
                value=threshold,
                time_window=300,
                aggregation="latest"
            )
            
            condition_group = RuleGroup(
                logical_operator=LogicalOperator.AND,
                conditions=[condition]
            )
            
            rule = (RuleBuilder()
                    .id(escalation_rule_id)
                    .name(escalation_name)
                    .description(f"Escalation level {i+1} for {metric_name}")
                    .rule_type(RuleType.THRESHOLD)
                    .severity(severity)
                    .condition_group(condition_group)
                    .cooldown(300)
                    .max_triggers_per_hour(5)
                    .tags({"escalation_level": i+1, "base_rule": rule_id})
                    .build())
            
            rules.append(rule)
        
        return rules
    
    def predictive_rule(self, rule_id: str, name: str, metric_name: str,
                       prediction_horizon: int = 3600, threshold: float = 100.0,
                       severity: str = "medium") -> AlertRule:
        """Create a predictive rule"""
        condition = RuleCondition(
            id=f"{rule_id}_condition",
            metric_name=f"predicted_{metric_name}",
            operator=">",
            value=threshold,
            time_window=prediction_horizon,
            aggregation="latest"
        )
        
        condition_group = RuleGroup(
            logical_operator=LogicalOperator.AND,
            conditions=[condition]
        )
        
        return (RuleBuilder()
                .id(rule_id)
                .name(name)
                .description(f"Predictive alert for {metric_name} in next {prediction_horizon} seconds")
                .rule_type(RuleType.PREDICTIVE)
                .severity(severity)
                .condition_group(condition_group)
                .cooldown(1800)  # 30 minutes
                .max_triggers_per_hour(2)
                .build())
    
    def pattern_detection_rule(self, rule_id: str, name: str, metrics: List[str],
                             pattern_type: str = "correlation", severity: str = "medium") -> AlertRule:
        """Create a pattern detection rule"""
        conditions = []
        for i, metric in enumerate(metrics):
            condition = RuleCondition(
                id=f"{rule_id}_condition_{i}",
                metric_name=metric,
                operator=">",
                value=0.7,  # Pattern strength threshold
                time_window=600,
                aggregation="latest"
            )
            conditions.append(condition)
        
        condition_group = RuleGroup(
            logical_operator=LogicalOperator.AND,
            conditions=conditions
        )
        
        return (RuleBuilder()
                .id(rule_id)
                .name(name)
                .description(f"Pattern detection rule for {pattern_type}")
                .rule_type(RuleType.PATTERN)
                .severity(severity)
                .condition_group(condition_group)
                .cooldown(600)
                .max_triggers_per_hour(5)
                .tags({"pattern_type": pattern_type})
                .build())


class RuleGroupBuilder:
    """
    Builder for creating rule condition groups
    
    Provides a fluent interface for building complex condition groups
    with logical operators.
    """
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset the builder to initial state"""
        self._logical_operator = LogicalOperator.AND
        self._conditions = []
        self._groups = []
        return self
    
    def logical_operator(self, operator: Union[LogicalOperator, str]):
        """Set logical operator"""
        if isinstance(operator, str):
            self._logical_operator = LogicalOperator(operator)
        else:
            self._logical_operator = operator
        return self
    
    def and_operator(self):
        """Set AND logical operator"""
        self._logical_operator = LogicalOperator.AND
        return self
    
    def or_operator(self):
        """Set OR logical operator"""
        self._logical_operator = LogicalOperator.OR
        return self
    
    def not_operator(self):
        """Set NOT logical operator"""
        self._logical_operator = LogicalOperator.NOT
        return self
    
    def add_condition(self, condition: RuleCondition):
        """Add a condition to the group"""
        self._conditions.append(condition)
        return self
    
    def add_group(self, group: RuleGroup):
        """Add a nested group"""
        self._groups.append(group)
        return self
    
    def condition(self, condition_id: str, metric_name: str, operator: str, 
                 value: Union[float, str], time_window: int = 300, 
                 aggregation: str = "latest"):
        """Add a condition with parameters"""
        condition = RuleCondition(
            id=condition_id,
            metric_name=metric_name,
            operator=operator,
            value=value,
            time_window=time_window,
            aggregation=aggregation
        )
        self._conditions.append(condition)
        return self
    
    def build(self) -> RuleGroup:
        """Build the condition group"""
        if not self._conditions and not self._groups:
            raise ValueError("Group must have at least one condition or nested group")
        
        group = RuleGroup(
            logical_operator=self._logical_operator,
            conditions=self._conditions.copy(),
            groups=self._groups.copy()
        )
        
        return group


def create_simple_threshold_rule(rule_id: str, name: str, metric_name: str,
                                operator: str, threshold: float, severity: str = "medium") -> AlertRule:
    """
    Convenience function to create a simple threshold rule
    
    Args:
        rule_id: Unique rule identifier
        name: Rule name
        metric_name: Metric to monitor
        operator: Comparison operator (>, <, >=, <=, ==, !=)
        threshold: Threshold value
        severity: Alert severity (low, medium, high, critical)
    
    Returns:
        AlertRule: Configured alert rule
    """
    condition_group = (RuleGroupBuilder()
                      .and_operator()
                      .condition(
                          condition_id=f"{rule_id}_condition",
                          metric_name=metric_name,
                          operator=operator,
                          value=threshold
                      )
                      .build())
    
    return (RuleBuilder()
            .id(rule_id)
            .name(name)
            .description(f"Alert when {metric_name} {operator} {threshold}")
            .rule_type(RuleType.THRESHOLD)
            .severity(severity)
            .condition_group(condition_group)
            .build())


def create_composite_rule(rule_id: str, name: str, conditions: List[Dict[str, Any]],
                         logical_operator: str = "and", severity: str = "medium") -> AlertRule:
    """
    Convenience function to create a composite rule with multiple conditions
    
    Args:
        rule_id: Unique rule identifier
        name: Rule name
        conditions: List of condition dictionaries
        logical_operator: Logical operator (and, or, not)
        severity: Alert severity
    
    Returns:
        AlertRule: Configured alert rule
    """
    group_builder = RuleGroupBuilder().logical_operator(logical_operator)
    
    for i, cond in enumerate(conditions):
        group_builder.condition(
            condition_id=f"{rule_id}_condition_{i}",
            metric_name=cond['metric_name'],
            operator=cond['operator'],
            value=cond['value'],
            time_window=cond.get('time_window', 300),
            aggregation=cond.get('aggregation', 'latest')
        )
    
    condition_group = group_builder.build()
    
    return (RuleBuilder()
            .id(rule_id)
            .name(name)
            .description(f"Composite rule with {len(conditions)} conditions")
            .rule_type(RuleType.COMPOSITE)
            .severity(severity)
            .condition_group(condition_group)
            .build())