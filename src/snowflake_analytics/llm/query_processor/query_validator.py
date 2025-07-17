"""
Query Validator for SQL Safety and Correctness

Validates generated SQL queries for safety, performance, and correctness
before execution against Snowflake.
"""

import re
import sqlparse
from typing import Dict, List, Optional, Any, Tuple
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
class ValidationIssue:
    """Represents a validation issue."""
    type: str
    severity: ValidationSeverity
    message: str
    line_number: Optional[int] = None
    suggestion: Optional[str] = None


@dataclass
class QueryValidationResult:
    """Result of SQL query validation."""
    is_valid: bool
    is_safe: bool
    estimated_cost: float
    estimated_rows: int
    issues: List[ValidationIssue]
    sanitized_query: Optional[str]


class QueryValidator:
    """Validates SQL queries for safety and performance."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize query validator."""
        self.config = config or {}
        
        # Safety rules
        self.forbidden_keywords = [
            'DROP', 'DELETE', 'UPDATE', 'INSERT', 'ALTER', 'CREATE',
            'TRUNCATE', 'REPLACE', 'MERGE', 'COPY', 'PUT', 'GET',
            'CALL', 'EXECUTE'
        ]
        
        self.dangerous_functions = [
            'SYSTEM$', 'CURRENT_USER', 'CURRENT_ROLE', 'CURRENT_ACCOUNT',
            'GET_DDL', 'PARSE_IP', 'HASH'
        ]
        
        # Performance rules
        self.performance_thresholds = {
            'max_scan_size_gb': self.config.get('max_scan_size_gb', 100),
            'max_execution_time_minutes': self.config.get('max_execution_time_minutes', 60),
            'max_result_rows': self.config.get('max_result_rows', 100000)
        }
        
        # Known table schemas (would be loaded from Snowflake in production)
        self.table_schemas = self._load_table_schemas()
        
        logger.info("Query validator initialized")
    
    def _load_table_schemas(self) -> Dict[str, Dict[str, Any]]:
        """Load table schemas for validation."""
        # Mock schemas - in production this would query Snowflake
        return {
            'COST_METRICS': {
                'columns': ['DATE', 'WAREHOUSE', 'CREDITS_CONSUMED', 'COST_USD', 'USER_NAME'],
                'indexed_columns': ['DATE', 'WAREHOUSE'],
                'estimated_rows': 1000000,
                'avg_row_size_bytes': 256
            },
            'USAGE_METRICS': {
                'columns': ['DATE', 'WAREHOUSE', 'QUERY_COUNT', 'EXECUTION_TIME', 'USER_NAME'],
                'indexed_columns': ['DATE', 'WAREHOUSE'],
                'estimated_rows': 5000000,
                'avg_row_size_bytes': 512
            },
            'WAREHOUSE_METRICS': {
                'columns': ['DATE', 'WAREHOUSE', 'SIZE', 'UTILIZATION_PERCENT'],
                'indexed_columns': ['DATE', 'WAREHOUSE'],
                'estimated_rows': 100000,
                'avg_row_size_bytes': 128
            },
            'USER_ACTIVITY': {
                'columns': ['DATE', 'USER_NAME', 'SESSION_COUNT', 'QUERY_COUNT'],
                'indexed_columns': ['DATE', 'USER_NAME'],
                'estimated_rows': 500000,
                'avg_row_size_bytes': 200
            },
            'QUERY_HISTORY': {
                'columns': ['QUERY_ID', 'USER_NAME', 'START_TIME', 'EXECUTION_TIME_MS'],
                'indexed_columns': ['START_TIME', 'USER_NAME'],
                'estimated_rows': 10000000,
                'avg_row_size_bytes': 1024
            }
        }
    
    def validate_query(self, sql: str) -> QueryValidationResult:
        """Validate a SQL query comprehensively.
        
        Args:
            sql: SQL query string to validate
            
        Returns:
            QueryValidationResult with validation details
        """
        issues = []
        
        # Parse SQL
        try:
            parsed = sqlparse.parse(sql)[0]
        except Exception as e:
            return QueryValidationResult(
                is_valid=False,
                is_safe=False,
                estimated_cost=0.0,
                estimated_rows=0,
                issues=[ValidationIssue(
                    type="parse_error",
                    severity=ValidationSeverity.CRITICAL,
                    message=f"SQL parsing failed: {str(e)}"
                )],
                sanitized_query=None
            )
        
        # Safety validation
        safety_issues = self._validate_safety(sql, parsed)
        issues.extend(safety_issues)
        
        # Syntax validation
        syntax_issues = self._validate_syntax(sql, parsed)
        issues.extend(syntax_issues)
        
        # Performance validation
        performance_issues = self._validate_performance(sql, parsed)
        issues.extend(performance_issues)
        
        # Security validation
        security_issues = self._validate_security(sql, parsed)
        issues.extend(security_issues)
        
        # Schema validation
        schema_issues = self._validate_schema(sql, parsed)
        issues.extend(schema_issues)
        
        # Determine overall validation result
        is_safe = not any(issue.severity in [ValidationSeverity.CRITICAL, ValidationSeverity.ERROR] 
                         for issue in safety_issues + security_issues)
        
        is_valid = is_safe and not any(issue.severity == ValidationSeverity.CRITICAL 
                                      for issue in issues)
        
        # Estimate cost and rows
        estimated_cost = self._estimate_query_cost(sql, parsed)
        estimated_rows = self._estimate_result_rows(sql, parsed)
        
        # Generate sanitized query if needed
        sanitized_query = self._sanitize_query(sql) if is_valid else None
        
        return QueryValidationResult(
            is_valid=is_valid,
            is_safe=is_safe,
            estimated_cost=estimated_cost,
            estimated_rows=estimated_rows,
            issues=issues,
            sanitized_query=sanitized_query
        )
    
    def _validate_safety(self, sql: str, parsed: sqlparse.sql.Statement) -> List[ValidationIssue]:
        """Validate SQL safety - no dangerous operations."""
        issues = []
        sql_upper = sql.upper()
        
        # Check for forbidden keywords
        for keyword in self.forbidden_keywords:
            if re.search(rf'\b{keyword}\b', sql_upper):
                issues.append(ValidationIssue(
                    type="forbidden_operation",
                    severity=ValidationSeverity.CRITICAL,
                    message=f"Forbidden operation detected: {keyword}",
                    suggestion="Only SELECT queries are allowed"
                ))
        
        # Check for dangerous functions
        for func in self.dangerous_functions:
            if func in sql_upper:
                issues.append(ValidationIssue(
                    type="dangerous_function",
                    severity=ValidationSeverity.ERROR,
                    message=f"Dangerous function detected: {func}",
                    suggestion="Remove dangerous system functions"
                ))
        
        # Ensure it's a SELECT statement
        if not sql_upper.strip().startswith('SELECT'):
            issues.append(ValidationIssue(
                type="non_select_statement",
                severity=ValidationSeverity.CRITICAL,
                message="Only SELECT statements are allowed",
                suggestion="Query must start with SELECT"
            ))
        
        return issues
    
    def _validate_syntax(self, sql: str, parsed: sqlparse.sql.Statement) -> List[ValidationIssue]:
        """Validate SQL syntax correctness."""
        issues = []
        
        # Check for balanced parentheses
        if sql.count('(') != sql.count(')'):
            issues.append(ValidationIssue(
                type="unbalanced_parentheses",
                severity=ValidationSeverity.ERROR,
                message="Unbalanced parentheses in SQL",
                suggestion="Check for missing opening or closing parentheses"
            ))
        
        # Check for proper string quoting
        single_quotes = sql.count("'")
        if single_quotes % 2 != 0:
            issues.append(ValidationIssue(
                type="unmatched_quotes",
                severity=ValidationSeverity.ERROR,
                message="Unmatched single quotes in SQL",
                suggestion="Check for missing quote marks"
            ))
        
        # Check for common SQL injection patterns
        injection_patterns = [
            r";\s*--",
            r";\s*/\*",
            r"\bunion\s+select\b",
            r"'\s*or\s+'\d'\s*=\s*'\d'"
        ]
        
        for pattern in injection_patterns:
            if re.search(pattern, sql, re.IGNORECASE):
                issues.append(ValidationIssue(
                    type="potential_injection",
                    severity=ValidationSeverity.ERROR,
                    message=f"Potential SQL injection pattern: {pattern}",
                    suggestion="Remove suspicious SQL patterns"
                ))
        
        return issues
    
    def _validate_performance(self, sql: str, parsed: sqlparse.sql.Statement) -> List[ValidationIssue]:
        """Validate query performance characteristics."""
        issues = []
        sql_upper = sql.upper()
        
        # Check for missing WHERE clause on large tables
        if 'FROM' in sql_upper:
            table_matches = re.findall(r'FROM\s+(\w+)', sql_upper)
            for table in table_matches:
                if table in self.table_schemas:
                    table_info = self.table_schemas[table]
                    if table_info['estimated_rows'] > 100000 and 'WHERE' not in sql_upper:
                        issues.append(ValidationIssue(
                            type="missing_where_clause",
                            severity=ValidationSeverity.WARNING,
                            message=f"Large table {table} accessed without WHERE clause",
                            suggestion="Add WHERE clause to limit data scan"
                        ))
        
        # Check for SELECT * on large tables
        if re.search(r'SELECT\s+\*', sql_upper):
            issues.append(ValidationIssue(
                type="select_star",
                severity=ValidationSeverity.WARNING,
                message="SELECT * may return excessive data",
                suggestion="Select specific columns instead of *"
            ))
        
        # Check for LIMIT clause
        if 'LIMIT' not in sql_upper and 'ORDER BY' in sql_upper:
            issues.append(ValidationIssue(
                type="missing_limit",
                severity=ValidationSeverity.INFO,
                message="ORDER BY without LIMIT may be inefficient",
                suggestion="Consider adding LIMIT clause"
            ))
        
        # Check for potentially expensive operations
        expensive_operations = ['DISTINCT', 'GROUP BY', 'ORDER BY']
        for operation in expensive_operations:
            if operation in sql_upper:
                issues.append(ValidationIssue(
                    type="expensive_operation",
                    severity=ValidationSeverity.INFO,
                    message=f"Query contains potentially expensive operation: {operation}",
                    suggestion="Consider if this operation is necessary"
                ))
        
        return issues
    
    def _validate_security(self, sql: str, parsed: sqlparse.sql.Statement) -> List[ValidationIssue]:
        """Validate query security aspects."""
        issues = []
        
        # Check for hardcoded sensitive values
        sensitive_patterns = [
            (r'\bpassword\s*=\s*[\'"][^\'"]+[\'"]', "hardcoded password"),
            (r'\bapi_key\s*=\s*[\'"][^\'"]+[\'"]', "hardcoded API key"),
            (r'\btoken\s*=\s*[\'"][^\'"]+[\'"]', "hardcoded token")
        ]
        
        for pattern, description in sensitive_patterns:
            if re.search(pattern, sql, re.IGNORECASE):
                issues.append(ValidationIssue(
                    type="sensitive_data",
                    severity=ValidationSeverity.ERROR,
                    message=f"Potential sensitive data exposure: {description}",
                    suggestion="Remove hardcoded sensitive values"
                ))
        
        # Check for potential data exfiltration patterns
        if re.search(r'\binto\s+outfile\b', sql, re.IGNORECASE):
            issues.append(ValidationIssue(
                type="data_exfiltration",
                severity=ValidationSeverity.CRITICAL,
                message="Potential data exfiltration attempt",
                suggestion="Remove file output operations"
            ))
        
        return issues
    
    def _validate_schema(self, sql: str, parsed: sqlparse.sql.Statement) -> List[ValidationIssue]:
        """Validate against known schema."""
        issues = []
        sql_upper = sql.upper()
        
        # Extract table references
        table_matches = re.findall(r'FROM\s+(\w+)', sql_upper)
        table_matches.extend(re.findall(r'JOIN\s+(\w+)', sql_upper))
        
        for table in table_matches:
            if table not in self.table_schemas:
                issues.append(ValidationIssue(
                    type="unknown_table",
                    severity=ValidationSeverity.ERROR,
                    message=f"Unknown table referenced: {table}",
                    suggestion=f"Available tables: {', '.join(self.table_schemas.keys())}"
                ))
        
        # Extract column references (simplified)
        column_matches = re.findall(r'SELECT\s+([^FROM]+)', sql_upper)
        if column_matches:
            columns_part = column_matches[0]
            # This is a simplified column extraction - a full parser would be needed for complex queries
            if '*' not in columns_part:
                referenced_columns = [col.strip() for col in columns_part.split(',')]
                # Validate columns against schema (simplified)
                for col in referenced_columns:
                    clean_col = re.sub(r'\s+as\s+\w+', '', col.strip(), flags=re.IGNORECASE)
                    clean_col = re.sub(r'[(),]', '', clean_col).strip()
                    
                    # Skip aggregation functions and complex expressions
                    if any(func in clean_col for func in ['SUM', 'COUNT', 'AVG', 'MAX', 'MIN']):
                        continue
                    
                    # Check if column exists in any referenced table
                    column_found = False
                    for table in table_matches:
                        if table in self.table_schemas:
                            if clean_col in self.table_schemas[table]['columns']:
                                column_found = True
                                break
                    
                    if not column_found and clean_col:
                        issues.append(ValidationIssue(
                            type="unknown_column",
                            severity=ValidationSeverity.WARNING,
                            message=f"Column may not exist: {clean_col}",
                            suggestion="Verify column name against table schema"
                        ))
        
        return issues
    
    def _estimate_query_cost(self, sql: str, parsed: sqlparse.sql.Statement) -> float:
        """Estimate query execution cost in credits."""
        base_cost = 0.001  # Base cost
        sql_upper = sql.upper()
        
        # Cost based on tables accessed
        table_matches = re.findall(r'FROM\s+(\w+)', sql_upper)
        table_matches.extend(re.findall(r'JOIN\s+(\w+)', sql_upper))
        
        for table in table_matches:
            if table in self.table_schemas:
                table_info = self.table_schemas[table]
                # Cost based on table size
                base_cost += table_info['estimated_rows'] / 10000000 * 0.01
        
        # Additional cost for complex operations
        if 'GROUP BY' in sql_upper:
            base_cost *= 1.5
        if 'ORDER BY' in sql_upper:
            base_cost *= 1.3
        if 'JOIN' in sql_upper:
            base_cost *= 2.0
        if 'DISTINCT' in sql_upper:
            base_cost *= 1.4
        
        return round(base_cost, 4)
    
    def _estimate_result_rows(self, sql: str, parsed: sqlparse.sql.Statement) -> int:
        """Estimate number of result rows."""
        sql_upper = sql.upper()
        
        # Extract LIMIT if present
        limit_match = re.search(r'LIMIT\s+(\d+)', sql_upper)
        if limit_match:
            return int(limit_match.group(1))
        
        # Estimate based on table sizes and operations
        table_matches = re.findall(r'FROM\s+(\w+)', sql_upper)
        
        if not table_matches:
            return 0
        
        # Use the largest table as base
        max_rows = 0
        for table in table_matches:
            if table in self.table_schemas:
                table_rows = self.table_schemas[table]['estimated_rows']
                max_rows = max(max_rows, table_rows)
        
        # Adjust for filtering and grouping
        if 'WHERE' in sql_upper:
            max_rows = int(max_rows * 0.1)  # Assume WHERE reduces by 90%
        
        if 'GROUP BY' in sql_upper:
            max_rows = min(max_rows, 1000)  # GROUP BY typically reduces result set
        
        return max_rows
    
    def _sanitize_query(self, sql: str) -> str:
        """Sanitize SQL query by removing potentially dangerous elements."""
        # Remove comments
        sql = re.sub(r'--.*$', '', sql, flags=re.MULTILINE)
        sql = re.sub(r'/\*.*?\*/', '', sql, flags=re.DOTALL)
        
        # Normalize whitespace
        sql = re.sub(r'\s+', ' ', sql.strip())
        
        # Ensure it ends with semicolon
        if not sql.endswith(';'):
            sql += ';'
        
        return sql
    
    def get_validation_rules(self) -> Dict[str, Any]:
        """Get current validation rules and configuration."""
        return {
            "forbidden_keywords": self.forbidden_keywords,
            "dangerous_functions": self.dangerous_functions,
            "performance_thresholds": self.performance_thresholds,
            "available_tables": list(self.table_schemas.keys())
        }
    
    def update_table_schemas(self, schemas: Dict[str, Dict[str, Any]]):
        """Update table schemas for validation."""
        self.table_schemas.update(schemas)
        logger.info(f"Updated table schemas: {list(schemas.keys())}")