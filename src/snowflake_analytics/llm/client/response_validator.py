"""
Response Validator for LLM Responses

Validates LLM responses for quality, safety, and relevance
to Snowflake analytics queries.
"""

import re
import json
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import logging

from ...utils.logger import get_logger

logger = get_logger(__name__)


class ValidationSeverity(Enum):
    """Severity levels for validation issues."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class ValidationResult:
    """Result of response validation."""
    is_valid: bool
    score: float  # 0.0 to 1.0
    issues: List[Dict[str, Any]]
    suggestions: List[str]


class ResponseValidator:
    """Validates LLM responses for quality and safety."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize response validator."""
        self.config = config or {}
        
        # Validation thresholds
        self.min_length = self.config.get('min_length', 10)
        self.max_length = self.config.get('max_length', 10000)
        self.min_score = self.config.get('min_score', 0.7)
        
        # Content filters
        self.forbidden_patterns = self.config.get('forbidden_patterns', [])
        self.required_patterns = self.config.get('required_patterns', [])
        
        # SQL injection patterns
        self.sql_injection_patterns = [
            r";\s*(drop|delete|update|insert|alter|create)\s+",
            r"union\s+select",
            r"--\s*$",
            r"/\*.*\*/",
            r"xp_cmdshell",
            r"sp_executesql"
        ]
        
        # Snowflake-specific validation
        self.snowflake_keywords = {
            'warehouses', 'databases', 'schemas', 'tables', 'views',
            'queries', 'credits', 'storage', 'compute', 'users',
            'roles', 'pipes', 'stages', 'file_formats'
        }
        
        logger.info("Response validator initialized")
    
    def validate_response(self, response: Any) -> ValidationResult:
        """Validate an LLM response.
        
        Args:
            response: LLM response object with content, usage, etc.
            
        Returns:
            ValidationResult with validation details
        """
        issues = []
        score = 1.0
        
        # Extract content
        content = self._extract_content(response)
        if not content:
            return ValidationResult(
                is_valid=False,
                score=0.0,
                issues=[{"type": "empty_response", "severity": "critical", "message": "Response is empty"}],
                suggestions=["Regenerate response with more specific prompt"]
            )
        
        # Basic validation checks
        issues.extend(self._validate_length(content))
        issues.extend(self._validate_format(content))
        issues.extend(self._validate_content_safety(content))
        issues.extend(self._validate_sql_safety(content))
        issues.extend(self._validate_snowflake_relevance(content))
        issues.extend(self._validate_coherence(content))
        
        # Calculate score based on issues
        score = self._calculate_score(issues)
        
        # Generate suggestions
        suggestions = self._generate_suggestions(issues)
        
        is_valid = score >= self.min_score and not any(
            issue["severity"] in ["error", "critical"] for issue in issues
        )
        
        return ValidationResult(
            is_valid=is_valid,
            score=score,
            issues=issues,
            suggestions=suggestions
        )
    
    def _extract_content(self, response: Any) -> str:
        """Extract content from response object."""
        if hasattr(response, 'content'):
            return response.content
        elif isinstance(response, dict):
            return response.get('content', '')
        elif isinstance(response, str):
            return response
        else:
            return str(response)
    
    def _validate_length(self, content: str) -> List[Dict[str, Any]]:
        """Validate content length."""
        issues = []
        
        if len(content) < self.min_length:
            issues.append({
                "type": "too_short",
                "severity": "warning",
                "message": f"Response is too short ({len(content)} chars, minimum {self.min_length})",
                "score_impact": -0.2
            })
        
        if len(content) > self.max_length:
            issues.append({
                "type": "too_long",
                "severity": "warning", 
                "message": f"Response is too long ({len(content)} chars, maximum {self.max_length})",
                "score_impact": -0.1
            })
        
        return issues
    
    def _validate_format(self, content: str) -> List[Dict[str, Any]]:
        """Validate content format and structure."""
        issues = []
        
        # Check for proper sentence structure
        sentences = re.split(r'[.!?]+', content)
        if len(sentences) < 2:
            issues.append({
                "type": "poor_structure",
                "severity": "info",
                "message": "Response lacks proper sentence structure",
                "score_impact": -0.05
            })
        
        # Check for balanced parentheses/brackets
        open_parens = content.count('(')
        close_parens = content.count(')')
        if open_parens != close_parens:
            issues.append({
                "type": "unbalanced_parentheses",
                "severity": "warning",
                "message": "Unbalanced parentheses in response",
                "score_impact": -0.1
            })
        
        # Check for JSON validity if content appears to be JSON
        if content.strip().startswith('{') or content.strip().startswith('['):
            try:
                json.loads(content)
            except json.JSONDecodeError:
                issues.append({
                    "type": "invalid_json",
                    "severity": "error",
                    "message": "Response appears to be JSON but is invalid",
                    "score_impact": -0.3
                })
        
        return issues
    
    def _validate_content_safety(self, content: str) -> List[Dict[str, Any]]:
        """Validate content for safety issues."""
        issues = []
        
        # Check forbidden patterns
        for pattern in self.forbidden_patterns:
            if re.search(pattern, content, re.IGNORECASE):
                issues.append({
                    "type": "forbidden_content",
                    "severity": "error",
                    "message": f"Content contains forbidden pattern: {pattern}",
                    "score_impact": -0.5
                })
        
        # Check for potential sensitive information
        sensitive_patterns = [
            r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b',  # Credit card
            r'\b\d{3}-\d{2}-\d{4}\b',  # SSN
            r'password\s*[:=]\s*[^\s]+',  # Passwords
        ]
        
        for pattern in sensitive_patterns:
            if re.search(pattern, content, re.IGNORECASE):
                issues.append({
                    "type": "sensitive_information",
                    "severity": "critical",
                    "message": "Response may contain sensitive information",
                    "score_impact": -0.8
                })
        
        return issues
    
    def _validate_sql_safety(self, content: str) -> List[Dict[str, Any]]:
        """Validate SQL content for injection attempts."""
        issues = []
        
        # Check for SQL injection patterns
        for pattern in self.sql_injection_patterns:
            if re.search(pattern, content, re.IGNORECASE):
                issues.append({
                    "type": "sql_injection_risk",
                    "severity": "critical",
                    "message": f"Potential SQL injection pattern detected: {pattern}",
                    "score_impact": -1.0
                })
        
        # Check for dangerous SQL operations
        dangerous_ops = ['drop table', 'delete from', 'truncate', 'alter table']
        for op in dangerous_ops:
            if op in content.lower():
                issues.append({
                    "type": "dangerous_sql",
                    "severity": "error",
                    "message": f"Dangerous SQL operation detected: {op}",
                    "score_impact": -0.7
                })
        
        return issues
    
    def _validate_snowflake_relevance(self, content: str) -> List[Dict[str, Any]]:
        """Validate relevance to Snowflake analytics."""
        issues = []
        
        content_lower = content.lower()
        
        # Check for Snowflake keywords
        snowflake_mentions = sum(1 for keyword in self.snowflake_keywords if keyword in content_lower)
        
        if snowflake_mentions == 0:
            issues.append({
                "type": "low_snowflake_relevance",
                "severity": "warning",
                "message": "Response does not mention Snowflake-specific concepts",
                "score_impact": -0.2
            })
        
        # Check for analytics keywords
        analytics_keywords = [
            'cost', 'usage', 'performance', 'optimization', 'queries',
            'warehouse', 'credit', 'storage', 'compute', 'metrics'
        ]
        
        analytics_mentions = sum(1 for keyword in analytics_keywords if keyword in content_lower)
        
        if analytics_mentions == 0:
            issues.append({
                "type": "low_analytics_relevance",
                "severity": "info",
                "message": "Response does not mention analytics concepts",
                "score_impact": -0.1
            })
        
        return issues
    
    def _validate_coherence(self, content: str) -> List[Dict[str, Any]]:
        """Validate content coherence and readability."""
        issues = []
        
        # Check for repeated phrases
        words = content.lower().split()
        word_count = {}
        for word in words:
            if len(word) > 3:  # Ignore short words
                word_count[word] = word_count.get(word, 0) + 1
        
        repeated_words = [word for word, count in word_count.items() if count > len(words) * 0.1]
        if repeated_words:
            issues.append({
                "type": "repetitive_content",
                "severity": "warning",
                "message": f"Repetitive words detected: {', '.join(repeated_words[:3])}",
                "score_impact": -0.15
            })
        
        # Check for contradictions (basic)
        contradiction_patterns = [
            (r'\bno\b.*\byes\b', "contradictory statements"),
            (r'\bcan\'?t\b.*\bcan\b', "contradictory ability statements"),
            (r'\bincrease\b.*\bdecrease\b', "contradictory trends"),
        ]
        
        for pattern, description in contradiction_patterns:
            if re.search(pattern, content, re.IGNORECASE):
                issues.append({
                    "type": "potential_contradiction",
                    "severity": "info",
                    "message": f"Potential contradiction: {description}",
                    "score_impact": -0.05
                })
        
        return issues
    
    def _calculate_score(self, issues: List[Dict[str, Any]]) -> float:
        """Calculate quality score based on issues."""
        score = 1.0
        
        for issue in issues:
            impact = issue.get('score_impact', 0)
            score += impact  # impacts are negative
        
        return max(0.0, min(1.0, score))
    
    def _generate_suggestions(self, issues: List[Dict[str, Any]]) -> List[str]:
        """Generate suggestions based on validation issues."""
        suggestions = []
        
        issue_types = {issue["type"] for issue in issues}
        
        if "too_short" in issue_types:
            suggestions.append("Provide more detailed explanation")
        
        if "too_long" in issue_types:
            suggestions.append("Make response more concise")
        
        if "low_snowflake_relevance" in issue_types:
            suggestions.append("Include more Snowflake-specific terminology")
        
        if "poor_structure" in issue_types:
            suggestions.append("Improve sentence structure and flow")
        
        if "sql_injection_risk" in issue_types or "dangerous_sql" in issue_types:
            suggestions.append("Review SQL content for safety")
        
        if "repetitive_content" in issue_types:
            suggestions.append("Reduce repetitive language")
        
        if not suggestions:
            suggestions.append("Response quality is good")
        
        return suggestions
    
    def get_validation_stats(self) -> Dict[str, Any]:
        """Get validation statistics."""
        # This would track validation history in a real implementation
        return {
            "total_validations": 0,
            "average_score": 0.0,
            "common_issues": [],
            "validation_rate": 0.0
        }