"""
Rule Manager

Manages alert rule lifecycle, persistence, validation, and organization
with rule templates and bulk operations.
"""

import json
import time
from typing import Dict, List, Any, Optional, Set
from datetime import datetime, timedelta
from dataclasses import asdict
import logging

from .rule_engine import AlertRule, RuleType, RuleStatus, RuleGroup, RuleCondition, LogicalOperator
from .rule_builder import RuleBuilder, RuleTemplateBuilder
from .condition_evaluator import ConditionEvaluator
from .severity_calculator import SeverityCalculator


class RuleManager:
    """
    Manages alert rule lifecycle and organization
    
    Provides rule persistence, validation, templates, and bulk operations
    for comprehensive rule management.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Rule storage
        self.rules = {}  # rule_id -> AlertRule
        self.rule_templates = {}  # template_name -> template_function
        self.rule_groups = {}  # group_name -> list of rule_ids
        
        # Components
        self.condition_evaluator = ConditionEvaluator()
        self.severity_calculator = SeverityCalculator()
        self.template_builder = RuleTemplateBuilder()
        
        # Configuration
        self.auto_save = self.config.get('auto_save', True)
        self.backup_interval = self.config.get('backup_interval', 3600)  # 1 hour
        self.max_rules_per_group = self.config.get('max_rules_per_group', 100)
        
        # Load built-in templates
        self._load_builtin_templates()
        
        # Last backup timestamp
        self.last_backup = datetime.now()
        
    def create_rule(self, rule_data: Dict[str, Any]) -> Optional[AlertRule]:
        """
        Create a new alert rule from configuration data
        
        Args:
            rule_data: Rule configuration dictionary
            
        Returns:
            Optional[AlertRule]: Created rule or None if failed
        """
        try:
            # Validate rule data
            validation_errors = self.validate_rule_data(rule_data)
            if validation_errors:
                self.logger.error(f"Rule validation failed: {validation_errors}")
                return None
            
            # Create rule using builder
            rule = self._build_rule_from_data(rule_data)
            
            # Store rule
            self.rules[rule.id] = rule
            
            # Add to group if specified
            if 'group' in rule_data:
                self.add_rule_to_group(rule.id, rule_data['group'])
            
            # Auto-save if enabled
            if self.auto_save:
                self.save_rule(rule.id)
            
            self.logger.info(f"Created rule: {rule.name} ({rule.id})")
            return rule
            
        except Exception as e:
            self.logger.error(f"Error creating rule: {e}")
            return None
    
    def update_rule(self, rule_id: str, updates: Dict[str, Any]) -> bool:
        """
        Update an existing rule
        
        Args:
            rule_id: Rule identifier
            updates: Dictionary of updates
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if rule_id not in self.rules:
                self.logger.error(f"Rule not found: {rule_id}")
                return False
            
            rule = self.rules[rule_id]
            
            # Apply updates
            for key, value in updates.items():
                if hasattr(rule, key):
                    setattr(rule, key, value)
            
            rule.updated_at = datetime.now()
            
            # Validate updated rule
            validation_errors = self.validate_rule(rule)
            if validation_errors:
                self.logger.error(f"Rule validation failed after update: {validation_errors}")
                return False
            
            # Auto-save if enabled
            if self.auto_save:
                self.save_rule(rule_id)
            
            self.logger.info(f"Updated rule: {rule.name} ({rule_id})")
            return True
            
        except Exception as e:
            self.logger.error(f"Error updating rule {rule_id}: {e}")
            return False
    
    def delete_rule(self, rule_id: str) -> bool:
        """
        Delete a rule
        
        Args:
            rule_id: Rule identifier
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if rule_id not in self.rules:
                return False
            
            rule = self.rules.pop(rule_id)
            
            # Remove from groups
            self.remove_rule_from_all_groups(rule_id)
            
            # Auto-save if enabled
            if self.auto_save:
                self.delete_saved_rule(rule_id)
            
            self.logger.info(f"Deleted rule: {rule.name} ({rule_id})")
            return True
            
        except Exception as e:
            self.logger.error(f"Error deleting rule {rule_id}: {e}")
            return False
    
    def get_rule(self, rule_id: str) -> Optional[AlertRule]:
        """Get rule by ID"""
        return self.rules.get(rule_id)
    
    def get_rules(self, filters: Optional[Dict[str, Any]] = None) -> List[AlertRule]:
        """
        Get rules with optional filters
        
        Args:
            filters: Optional filters (enabled, rule_type, status, team, etc.)
            
        Returns:
            List[AlertRule]: Filtered rules
        """
        rules = list(self.rules.values())
        
        if filters:
            # Apply filters
            if 'enabled' in filters:
                rules = [r for r in rules if r.enabled == filters['enabled']]
            
            if 'rule_type' in filters:
                rules = [r for r in rules if r.rule_type.value == filters['rule_type']]
            
            if 'status' in filters:
                rules = [r for r in rules if r.status.value == filters['status']]
            
            if 'team' in filters:
                rules = [r for r in rules if r.team == filters['team']]
            
            if 'owner' in filters:
                rules = [r for r in rules if r.owner == filters['owner']]
            
            if 'severity' in filters:
                rules = [r for r in rules if r.severity == filters['severity']]
            
            if 'tags' in filters:
                tag_filters = filters['tags']
                rules = [r for r in rules if all(
                    r.tags.get(k) == v for k, v in tag_filters.items()
                )]
        
        return rules
    
    def validate_rule_data(self, rule_data: Dict[str, Any]) -> List[str]:
        """
        Validate rule configuration data
        
        Args:
            rule_data: Rule configuration dictionary
            
        Returns:
            List[str]: List of validation errors
        """
        errors = []
        
        # Required fields
        required_fields = ['id', 'name', 'condition_group']
        for field in required_fields:
            if field not in rule_data:
                errors.append(f"Missing required field: {field}")
        
        # Validate rule ID uniqueness
        if 'id' in rule_data and rule_data['id'] in self.rules:
            errors.append(f"Rule ID already exists: {rule_data['id']}")
        
        # Validate rule type
        if 'rule_type' in rule_data:
            try:
                RuleType(rule_data['rule_type'])
            except ValueError:
                errors.append(f"Invalid rule type: {rule_data['rule_type']}")
        
        # Validate severity
        if 'severity' in rule_data:
            valid_severities = ['low', 'medium', 'high', 'critical']
            if rule_data['severity'] not in valid_severities:
                errors.append(f"Invalid severity: {rule_data['severity']}")
        
        # Validate condition group
        if 'condition_group' in rule_data:
            group_errors = self._validate_condition_group_data(rule_data['condition_group'])
            errors.extend(group_errors)
        
        return errors
    
    def validate_rule(self, rule: AlertRule) -> List[str]:
        """
        Validate an alert rule
        
        Args:
            rule: Alert rule to validate
            
        Returns:
            List[str]: List of validation errors
        """
        errors = []
        
        # Basic validation
        if not rule.id:
            errors.append("Rule ID is required")
        
        if not rule.name:
            errors.append("Rule name is required")
        
        # Validate condition group
        if rule.condition_group:
            group_errors = self._validate_condition_group(rule.condition_group)
            errors.extend(group_errors)
        
        # Validate configuration values
        if rule.cooldown_period < 0:
            errors.append("Cooldown period must be non-negative")
        
        if rule.max_triggers_per_hour < 0:
            errors.append("Max triggers per hour must be non-negative")
        
        return errors
    
    def _validate_condition_group_data(self, group_data: Dict[str, Any]) -> List[str]:
        """Validate condition group data"""
        errors = []
        
        # Check required fields
        if 'logical_operator' not in group_data:
            errors.append("Missing logical_operator in condition group")
        
        # Validate logical operator
        if 'logical_operator' in group_data:
            try:
                LogicalOperator(group_data['logical_operator'])
            except ValueError:
                errors.append(f"Invalid logical operator: {group_data['logical_operator']}")
        
        # Validate conditions
        if 'conditions' in group_data:
            for i, condition_data in enumerate(group_data['conditions']):
                condition_errors = self._validate_condition_data(condition_data)
                errors.extend([f"Condition {i}: {error}" for error in condition_errors])
        
        # Validate nested groups
        if 'groups' in group_data:
            for i, nested_group_data in enumerate(group_data['groups']):
                nested_errors = self._validate_condition_group_data(nested_group_data)
                errors.extend([f"Nested group {i}: {error}" for error in nested_errors])
        
        return errors
    
    def _validate_condition_data(self, condition_data: Dict[str, Any]) -> List[str]:
        """Validate condition data"""
        errors = []
        
        # Required fields
        required_fields = ['id', 'metric_name', 'operator', 'value']
        for field in required_fields:
            if field not in condition_data:
                errors.append(f"Missing required field: {field}")
        
        # Validate operator
        if 'operator' in condition_data:
            valid_operators = ['>', '<', '>=', '<=', '==', '!=']
            if condition_data['operator'] not in valid_operators:
                errors.append(f"Invalid operator: {condition_data['operator']}")
        
        # Validate aggregation
        if 'aggregation' in condition_data:
            valid_aggregations = self.condition_evaluator.get_supported_aggregations()
            valid_agg_names = [agg['name'] for agg in valid_aggregations]
            if condition_data['aggregation'] not in valid_agg_names:
                errors.append(f"Invalid aggregation: {condition_data['aggregation']}")
        
        return errors
    
    def _validate_condition_group(self, group: RuleGroup) -> List[str]:
        """Validate condition group"""
        errors = []
        
        # Must have at least one condition or nested group
        if not group.conditions and not group.groups:
            errors.append("Condition group must have at least one condition or nested group")
        
        # Validate conditions
        for condition in group.conditions:
            condition_errors = self.condition_evaluator.validate_condition(condition)
            errors.extend(condition_errors)
        
        # Validate nested groups
        for nested_group in group.groups:
            nested_errors = self._validate_condition_group(nested_group)
            errors.extend(nested_errors)
        
        return errors
    
    def _build_rule_from_data(self, rule_data: Dict[str, Any]) -> AlertRule:
        """Build rule from configuration data"""
        # Parse condition group
        condition_group = self._parse_condition_group(rule_data['condition_group'])
        
        # Create rule using builder
        builder = RuleBuilder()
        
        rule = (builder
                .id(rule_data['id'])
                .name(rule_data['name'])
                .description(rule_data.get('description', ''))
                .rule_type(rule_data.get('rule_type', 'threshold'))
                .severity(rule_data.get('severity', 'medium'))
                .enabled(rule_data.get('enabled', True))
                .cooldown(rule_data.get('cooldown_period', 300))
                .max_triggers_per_hour(rule_data.get('max_triggers_per_hour', 10))
                .auto_resolve(rule_data.get('auto_resolve', True), 
                             rule_data.get('auto_resolve_timeout', 3600))
                .tags(rule_data.get('tags', {}))
                .owner(rule_data.get('owner', ''))
                .team(rule_data.get('team', ''))
                .condition_group(condition_group)
                .build())
        
        return rule
    
    def _parse_condition_group(self, group_data: Dict[str, Any]) -> RuleGroup:
        """Parse condition group from data"""
        # Parse logical operator
        logical_operator = LogicalOperator(group_data['logical_operator'])
        
        # Parse conditions
        conditions = []
        for condition_data in group_data.get('conditions', []):
            condition = RuleCondition(
                id=condition_data['id'],
                metric_name=condition_data['metric_name'],
                operator=condition_data['operator'],
                value=condition_data['value'],
                time_window=condition_data.get('time_window', 300),
                aggregation=condition_data.get('aggregation', 'latest')
            )
            conditions.append(condition)
        
        # Parse nested groups
        groups = []
        for nested_group_data in group_data.get('groups', []):
            nested_group = self._parse_condition_group(nested_group_data)
            groups.append(nested_group)
        
        return RuleGroup(
            logical_operator=logical_operator,
            conditions=conditions,
            groups=groups
        )
    
    def create_rule_from_template(self, template_name: str, **kwargs) -> Optional[AlertRule]:
        """
        Create rule from template
        
        Args:
            template_name: Name of the template
            **kwargs: Template parameters
            
        Returns:
            Optional[AlertRule]: Created rule or None if failed
        """
        try:
            if template_name not in self.rule_templates:
                self.logger.error(f"Template not found: {template_name}")
                return None
            
            template_func = self.rule_templates[template_name]
            rule = template_func(**kwargs)
            
            # Store rule
            self.rules[rule.id] = rule
            
            # Auto-save if enabled
            if self.auto_save:
                self.save_rule(rule.id)
            
            self.logger.info(f"Created rule from template '{template_name}': {rule.name} ({rule.id})")
            return rule
            
        except Exception as e:
            self.logger.error(f"Error creating rule from template {template_name}: {e}")
            return None
    
    def _load_builtin_templates(self):
        """Load built-in rule templates"""
        self.rule_templates = {
            'cost_threshold': self.template_builder.cost_threshold_rule,
            'performance_degradation': self.template_builder.performance_degradation_rule,
            'usage_anomaly': self.template_builder.usage_anomaly_rule,
            'composite_cost_performance': self.template_builder.composite_cost_performance_rule,
            'predictive': self.template_builder.predictive_rule,
            'pattern_detection': self.template_builder.pattern_detection_rule
        }
    
    def register_template(self, name: str, template_func: callable):
        """Register a custom rule template"""
        self.rule_templates[name] = template_func
        self.logger.info(f"Registered rule template: {name}")
    
    def get_available_templates(self) -> List[str]:
        """Get list of available rule templates"""
        return list(self.rule_templates.keys())
    
    def create_rule_group(self, group_name: str, rule_ids: List[str]) -> bool:
        """
        Create a rule group
        
        Args:
            group_name: Name of the group
            rule_ids: List of rule IDs
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if len(rule_ids) > self.max_rules_per_group:
                self.logger.error(f"Too many rules for group {group_name}: {len(rule_ids)}")
                return False
            
            # Validate all rules exist
            for rule_id in rule_ids:
                if rule_id not in self.rules:
                    self.logger.error(f"Rule not found: {rule_id}")
                    return False
            
            self.rule_groups[group_name] = rule_ids
            self.logger.info(f"Created rule group: {group_name} with {len(rule_ids)} rules")
            return True
            
        except Exception as e:
            self.logger.error(f"Error creating rule group {group_name}: {e}")
            return False
    
    def add_rule_to_group(self, rule_id: str, group_name: str) -> bool:
        """Add rule to group"""
        try:
            if rule_id not in self.rules:
                return False
            
            if group_name not in self.rule_groups:
                self.rule_groups[group_name] = []
            
            if rule_id not in self.rule_groups[group_name]:
                self.rule_groups[group_name].append(rule_id)
                self.logger.info(f"Added rule {rule_id} to group {group_name}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error adding rule {rule_id} to group {group_name}: {e}")
            return False
    
    def remove_rule_from_group(self, rule_id: str, group_name: str) -> bool:
        """Remove rule from group"""
        try:
            if group_name in self.rule_groups and rule_id in self.rule_groups[group_name]:
                self.rule_groups[group_name].remove(rule_id)
                self.logger.info(f"Removed rule {rule_id} from group {group_name}")
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error removing rule {rule_id} from group {group_name}: {e}")
            return False
    
    def remove_rule_from_all_groups(self, rule_id: str):
        """Remove rule from all groups"""
        for group_name in list(self.rule_groups.keys()):
            self.remove_rule_from_group(rule_id, group_name)
    
    def get_rules_in_group(self, group_name: str) -> List[AlertRule]:
        """Get rules in a group"""
        if group_name not in self.rule_groups:
            return []
        
        return [self.rules[rule_id] for rule_id in self.rule_groups[group_name] if rule_id in self.rules]
    
    def bulk_enable_rules(self, rule_ids: List[str]) -> int:
        """Enable multiple rules"""
        count = 0
        for rule_id in rule_ids:
            if self.update_rule(rule_id, {'enabled': True}):
                count += 1
        
        self.logger.info(f"Enabled {count} rules")
        return count
    
    def bulk_disable_rules(self, rule_ids: List[str]) -> int:
        """Disable multiple rules"""
        count = 0
        for rule_id in rule_ids:
            if self.update_rule(rule_id, {'enabled': False}):
                count += 1
        
        self.logger.info(f"Disabled {count} rules")
        return count
    
    def bulk_update_severity(self, rule_ids: List[str], severity: str) -> int:
        """Update severity for multiple rules"""
        count = 0
        for rule_id in rule_ids:
            if self.update_rule(rule_id, {'severity': severity}):
                count += 1
        
        self.logger.info(f"Updated severity for {count} rules")
        return count
    
    def save_rule(self, rule_id: str) -> bool:
        """Save rule to persistent storage"""
        try:
            # In a real implementation, this would save to database/file
            # For now, just log the action
            self.logger.debug(f"Saving rule: {rule_id}")
            return True
        except Exception as e:
            self.logger.error(f"Error saving rule {rule_id}: {e}")
            return False
    
    def load_rule(self, rule_id: str) -> Optional[AlertRule]:
        """Load rule from persistent storage"""
        try:
            # In a real implementation, this would load from database/file
            # For now, just log the action
            self.logger.debug(f"Loading rule: {rule_id}")
            return None
        except Exception as e:
            self.logger.error(f"Error loading rule {rule_id}: {e}")
            return None
    
    def delete_saved_rule(self, rule_id: str) -> bool:
        """Delete rule from persistent storage"""
        try:
            # In a real implementation, this would delete from database/file
            # For now, just log the action
            self.logger.debug(f"Deleting saved rule: {rule_id}")
            return True
        except Exception as e:
            self.logger.error(f"Error deleting saved rule {rule_id}: {e}")
            return False
    
    def export_rules(self, rule_ids: Optional[List[str]] = None) -> Dict[str, Any]:
        """Export rules to JSON-serializable format"""
        if rule_ids is None:
            rules_to_export = list(self.rules.values())
        else:
            rules_to_export = [self.rules[rule_id] for rule_id in rule_ids if rule_id in self.rules]
        
        return {
            'rules': [rule.to_dict() for rule in rules_to_export],
            'rule_groups': self.rule_groups,
            'export_timestamp': datetime.now().isoformat()
        }
    
    def import_rules(self, import_data: Dict[str, Any]) -> int:
        """Import rules from JSON data"""
        imported_count = 0
        
        for rule_data in import_data.get('rules', []):
            try:
                rule = self.create_rule(rule_data)
                if rule:
                    imported_count += 1
            except Exception as e:
                self.logger.error(f"Error importing rule {rule_data.get('id', 'unknown')}: {e}")
        
        # Import rule groups
        for group_name, rule_ids in import_data.get('rule_groups', {}).items():
            try:
                self.create_rule_group(group_name, rule_ids)
            except Exception as e:
                self.logger.error(f"Error importing rule group {group_name}: {e}")
        
        self.logger.info(f"Imported {imported_count} rules")
        return imported_count
    
    def get_rule_statistics(self) -> Dict[str, Any]:
        """Get rule management statistics"""
        total_rules = len(self.rules)
        enabled_rules = len([r for r in self.rules.values() if r.enabled])
        
        rule_types = {}
        severities = {}
        teams = {}
        
        for rule in self.rules.values():
            # Count by type
            rule_type = rule.rule_type.value
            rule_types[rule_type] = rule_types.get(rule_type, 0) + 1
            
            # Count by severity
            severities[rule.severity] = severities.get(rule.severity, 0) + 1
            
            # Count by team
            if rule.team:
                teams[rule.team] = teams.get(rule.team, 0) + 1
        
        return {
            'total_rules': total_rules,
            'enabled_rules': enabled_rules,
            'disabled_rules': total_rules - enabled_rules,
            'rule_types': rule_types,
            'severities': severities,
            'teams': teams,
            'rule_groups': len(self.rule_groups),
            'available_templates': len(self.rule_templates)
        }