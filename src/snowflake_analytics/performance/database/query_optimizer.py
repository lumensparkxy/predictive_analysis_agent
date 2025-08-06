"""
SQL query optimizer for analyzing and improving query performance.
"""

import re
import hashlib
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Set
from datetime import datetime
from collections import defaultdict, Counter
import json


@dataclass
class QueryPattern:
    """SQL query pattern analysis."""
    pattern_hash: str
    pattern_template: str
    query_type: str
    tables_involved: Set[str]
    columns_accessed: Set[str]
    where_conditions: List[str]
    joins_used: List[str]
    aggregations: List[str]
    order_by_columns: List[str]
    group_by_columns: List[str]


@dataclass
class QueryOptimization:
    """Query optimization recommendation."""
    query_hash: str
    original_query: str
    optimization_type: str
    description: str
    optimized_query: Optional[str]
    impact_level: str  # 'low', 'medium', 'high'
    estimated_improvement: str
    explanation: str


class QueryOptimizer:
    """
    SQL query optimizer for analyzing query patterns and suggesting optimizations.
    Supports various database systems with focus on common optimization patterns.
    """
    
    def __init__(self):
        """Initialize query optimizer."""
        self.query_patterns: Dict[str, QueryPattern] = {}
        self.optimizations: List[QueryOptimization] = []
        
        # Common optimization rules
        self.optimization_rules = {
            'select_star': {
                'pattern': r'SELECT\s+\*\s+FROM',
                'description': 'Replace SELECT * with specific columns',
                'impact': 'medium',
                'improvement': '20-40% performance gain'
            },
            'missing_where': {
                'pattern': r'SELECT.*FROM\s+\w+(?:\s+\w+)?(?:\s*;|\s*$)',
                'description': 'Add WHERE clause to limit result set',
                'impact': 'high',
                'improvement': '50-90% performance gain'
            },
            'non_sargable': {
                'pattern': r'WHERE\s+\w+\s*\+\s*\d+\s*[=<>]|WHERE\s+SUBSTRING\s*\(',
                'description': 'Avoid functions in WHERE clause',
                'impact': 'medium',
                'improvement': '30-60% performance gain'
            },
            'inefficient_like': {
                'pattern': r'WHERE\s+\w+\s+LIKE\s+[\'"]%.*%[\'"]',
                'description': 'Leading wildcard prevents index usage',
                'impact': 'high',
                'improvement': '70-90% performance gain'
            },
            'missing_limit': {
                'pattern': r'SELECT.*(?!.*LIMIT)(?!.*TOP).*$',
                'description': 'Add LIMIT clause for large result sets',
                'impact': 'medium',
                'improvement': '40-80% performance gain'
            }
        }
        
        # Index suggestion patterns
        self.index_patterns = {
            'where_columns': r'WHERE\s+(\w+)\s*[=<>]',
            'join_columns': r'JOIN\s+\w+\s+(?:\w+\s+)?ON\s+(\w+\.\w+|\w+)\s*=\s*(\w+\.\w+|\w+)',
            'order_by_columns': r'ORDER\s+BY\s+(\w+(?:\.\w+)?)',
            'group_by_columns': r'GROUP\s+BY\s+(\w+(?:\.\w+)?)'
        }
    
    def analyze_query(self, query: str) -> QueryPattern:
        """
        Analyze SQL query to extract patterns and structure.
        
        Args:
            query: SQL query string
            
        Returns:
            Query pattern analysis
        """
        query_hash = self._generate_query_hash(query)
        
        if query_hash in self.query_patterns:
            return self.query_patterns[query_hash]
        
        # Normalize query for analysis
        normalized_query = self._normalize_query(query)
        
        # Extract query components
        query_type = self._extract_query_type(normalized_query)
        tables = self._extract_tables(normalized_query)
        columns = self._extract_columns(normalized_query)
        where_conditions = self._extract_where_conditions(normalized_query)
        joins = self._extract_joins(normalized_query)
        aggregations = self._extract_aggregations(normalized_query)
        order_by = self._extract_order_by(normalized_query)
        group_by = self._extract_group_by(normalized_query)
        
        pattern = QueryPattern(
            pattern_hash=query_hash,
            pattern_template=self._generate_pattern_template(normalized_query),
            query_type=query_type,
            tables_involved=tables,
            columns_accessed=columns,
            where_conditions=where_conditions,
            joins_used=joins,
            aggregations=aggregations,
            order_by_columns=order_by,
            group_by_columns=group_by
        )
        
        self.query_patterns[query_hash] = pattern
        return pattern
    
    def optimize_query(self, query: str) -> List[QueryOptimization]:
        """
        Generate optimization recommendations for a query.
        
        Args:
            query: SQL query string
            
        Returns:
            List of optimization recommendations
        """
        query_hash = self._generate_query_hash(query)
        pattern = self.analyze_query(query)
        optimizations = []
        
        # Apply optimization rules
        for rule_name, rule in self.optimization_rules.items():
            if re.search(rule['pattern'], query, re.IGNORECASE):
                optimized_query = self._apply_optimization_rule(query, rule_name)
                
                optimization = QueryOptimization(
                    query_hash=query_hash,
                    original_query=query,
                    optimization_type=rule_name,
                    description=rule['description'],
                    optimized_query=optimized_query,
                    impact_level=rule['impact'],
                    estimated_improvement=rule['improvement'],
                    explanation=self._generate_optimization_explanation(rule_name, pattern)
                )
                optimizations.append(optimization)
        
        # Check for missing indexes
        index_optimizations = self._suggest_indexes_for_query(query, pattern)
        optimizations.extend(index_optimizations)
        
        # Check for join optimizations
        join_optimizations = self._optimize_joins(query, pattern)
        optimizations.extend(join_optimizations)
        
        return optimizations
    
    def _generate_query_hash(self, query: str) -> str:
        """Generate hash for query identification."""
        normalized = re.sub(r'\s+', ' ', query.strip().lower())
        return hashlib.md5(normalized.encode()).hexdigest()
    
    def _normalize_query(self, query: str) -> str:
        """Normalize query for pattern analysis."""
        # Remove extra whitespace
        normalized = re.sub(r'\s+', ' ', query.strip())
        
        # Convert to uppercase for consistent analysis
        normalized = normalized.upper()
        
        return normalized
    
    def _extract_query_type(self, query: str) -> str:
        """Extract query type (SELECT, INSERT, UPDATE, DELETE)."""
        query_upper = query.strip().upper()
        for query_type in ['SELECT', 'INSERT', 'UPDATE', 'DELETE', 'CREATE', 'DROP', 'ALTER']:
            if query_upper.startswith(query_type):
                return query_type
        return 'UNKNOWN'
    
    def _extract_tables(self, query: str) -> Set[str]:
        """Extract table names from query."""
        tables = set()
        
        # FROM clause
        from_match = re.findall(r'FROM\s+(\w+)(?:\s+(?:AS\s+)?(\w+))?', query, re.IGNORECASE)
        for match in from_match:
            tables.add(match[0].lower())
        
        # JOIN clauses
        join_matches = re.findall(r'JOIN\s+(\w+)(?:\s+(?:AS\s+)?(\w+))?', query, re.IGNORECASE)
        for match in join_matches:
            tables.add(match[0].lower())
        
        # INSERT INTO, UPDATE
        insert_match = re.findall(r'(?:INSERT\s+INTO|UPDATE)\s+(\w+)', query, re.IGNORECASE)
        for match in insert_match:
            tables.add(match.lower())
        
        return tables
    
    def _extract_columns(self, query: str) -> Set[str]:
        """Extract column names from query."""
        columns = set()
        
        # SELECT columns
        select_match = re.search(r'SELECT\s+(.*?)\s+FROM', query, re.IGNORECASE | re.DOTALL)
        if select_match:
            select_part = select_match.group(1)
            if '*' not in select_part:
                # Parse individual columns (simplified)
                col_matches = re.findall(r'(\w+)(?:\.\w+)?(?:\s+AS\s+\w+)?', select_part)
                columns.update(col.lower() for col in col_matches)
        
        # WHERE columns
        where_matches = re.findall(r'WHERE.*?(\w+)\s*[=<>!]', query, re.IGNORECASE)
        columns.update(col.lower() for col in where_matches)
        
        # ORDER BY columns
        order_matches = re.findall(r'ORDER\s+BY\s+(\w+)', query, re.IGNORECASE)
        columns.update(col.lower() for col in order_matches)
        
        return columns
    
    def _extract_where_conditions(self, query: str) -> List[str]:
        """Extract WHERE clause conditions."""
        conditions = []
        where_match = re.search(r'WHERE\s+(.*?)(?:\s+(?:GROUP|ORDER|HAVING|LIMIT|$))', query, re.IGNORECASE)
        if where_match:
            where_clause = where_match.group(1).strip()
            # Split by AND/OR (simplified)
            conditions = re.split(r'\s+(?:AND|OR)\s+', where_clause, flags=re.IGNORECASE)
        return conditions
    
    def _extract_joins(self, query: str) -> List[str]:
        """Extract JOIN information."""
        joins = []
        join_matches = re.findall(r'((?:INNER|LEFT|RIGHT|FULL)?\s*JOIN\s+\w+\s+ON\s+.*?)(?:\s+(?:INNER|LEFT|RIGHT|FULL)?\s*JOIN|\s+WHERE|\s+GROUP|\s+ORDER|\s*$)', query, re.IGNORECASE)
        joins.extend(match.strip() for match in join_matches)
        return joins
    
    def _extract_aggregations(self, query: str) -> List[str]:
        """Extract aggregation functions."""
        aggregations = []
        agg_matches = re.findall(r'(COUNT|SUM|AVG|MIN|MAX|GROUP_CONCAT)\s*\([^)]*\)', query, re.IGNORECASE)
        aggregations.extend(match.upper() for match in agg_matches)
        return aggregations
    
    def _extract_order_by(self, query: str) -> List[str]:
        """Extract ORDER BY columns."""
        order_cols = []
        order_match = re.search(r'ORDER\s+BY\s+(.*?)(?:\s+(?:LIMIT|$))', query, re.IGNORECASE)
        if order_match:
            order_part = order_match.group(1).strip()
            cols = re.findall(r'(\w+(?:\.\w+)?)', order_part)
            order_cols.extend(col.lower() for col in cols)
        return order_cols
    
    def _extract_group_by(self, query: str) -> List[str]:
        """Extract GROUP BY columns."""
        group_cols = []
        group_match = re.search(r'GROUP\s+BY\s+(.*?)(?:\s+(?:HAVING|ORDER|LIMIT|$))', query, re.IGNORECASE)
        if group_match:
            group_part = group_match.group(1).strip()
            cols = re.findall(r'(\w+(?:\.\w+)?)', group_part)
            group_cols.extend(col.lower() for col in cols)
        return group_cols
    
    def _generate_pattern_template(self, query: str) -> str:
        """Generate a template pattern for similar queries."""
        template = query
        
        # Replace literal values with placeholders
        template = re.sub(r"'[^']*'", "'?'", template)  # String literals
        template = re.sub(r'"[^"]*"', '"?"', template)  # Quoted strings
        template = re.sub(r'\b\d+\b', '?', template)    # Numeric literals
        
        return template
    
    def _apply_optimization_rule(self, query: str, rule_name: str) -> Optional[str]:
        """Apply specific optimization rule to query."""
        if rule_name == 'select_star':
            # Replace SELECT * with common columns (example)
            optimized = re.sub(
                r'SELECT\s+\*\s+FROM',
                'SELECT id, name, created_at FROM',
                query,
                flags=re.IGNORECASE
            )
            return optimized
        
        elif rule_name == 'missing_where':
            # Add a sample WHERE clause
            if 'WHERE' not in query.upper():
                # Find the FROM table
                from_match = re.search(r'FROM\s+(\w+)', query, re.IGNORECASE)
                if from_match:
                    table_name = from_match.group(1)
                    # Add generic WHERE clause
                    insert_pos = query.find(';') if ';' in query else len(query)
                    optimized = query[:insert_pos] + f' WHERE {table_name}.active = 1' + query[insert_pos:]
                    return optimized
        
        elif rule_name == 'missing_limit':
            # Add LIMIT clause
            if 'LIMIT' not in query.upper() and 'TOP' not in query.upper():
                insert_pos = query.find(';') if ';' in query else len(query)
                optimized = query[:insert_pos] + ' LIMIT 100' + query[insert_pos:]
                return optimized
        
        elif rule_name == 'inefficient_like':
            # Suggest alternative for leading wildcard
            optimized = re.sub(
                r"(\w+)\s+LIKE\s+'%([^%]+)%'",
                r"\1 LIKE '\2%'  -- Consider full-text search for leading wildcards",
                query,
                flags=re.IGNORECASE
            )
            return optimized
        
        return None
    
    def _suggest_indexes_for_query(self, query: str, pattern: QueryPattern) -> List[QueryOptimization]:
        """Suggest indexes based on query patterns."""
        optimizations = []
        query_hash = self._generate_query_hash(query)
        
        # Suggest indexes for WHERE clause columns
        for condition in pattern.where_conditions:
            col_match = re.search(r'(\w+)\s*[=<>!]', condition)
            if col_match:
                column = col_match.group(1)
                optimization = QueryOptimization(
                    query_hash=query_hash,
                    original_query=query,
                    optimization_type='index_suggestion',
                    description=f"Create index on column '{column}' used in WHERE clause",
                    optimized_query=f"CREATE INDEX idx_{column} ON table_name({column});",
                    impact_level='high',
                    estimated_improvement='50-90% performance gain',
                    explanation=f"Index on '{column}' will speed up filtering operations"
                )
                optimizations.append(optimization)
        
        # Suggest composite indexes for multiple WHERE conditions
        if len(pattern.where_conditions) > 1:
            where_columns = []
            for condition in pattern.where_conditions:
                col_match = re.search(r'(\w+)\s*[=<>!]', condition)
                if col_match:
                    where_columns.append(col_match.group(1))
            
            if len(where_columns) >= 2:
                composite_index = ', '.join(where_columns[:3])  # Limit to 3 columns
                optimization = QueryOptimization(
                    query_hash=query_hash,
                    original_query=query,
                    optimization_type='composite_index',
                    description=f"Create composite index on columns: {composite_index}",
                    optimized_query=f"CREATE INDEX idx_composite ON table_name({composite_index});",
                    impact_level='high',
                    estimated_improvement='60-95% performance gain',
                    explanation="Composite index can satisfy multiple WHERE conditions efficiently"
                )
                optimizations.append(optimization)
        
        # Suggest indexes for ORDER BY columns
        for order_col in pattern.order_by_columns:
            optimization = QueryOptimization(
                query_hash=query_hash,
                original_query=query,
                optimization_type='order_index',
                description=f"Create index on ORDER BY column '{order_col}'",
                optimized_query=f"CREATE INDEX idx_order_{order_col} ON table_name({order_col});",
                impact_level='medium',
                estimated_improvement='30-70% performance gain',
                explanation=f"Index on '{order_col}' will eliminate sorting overhead"
            )
            optimizations.append(optimization)
        
        return optimizations
    
    def _optimize_joins(self, query: str, pattern: QueryPattern) -> List[QueryOptimization]:
        """Optimize JOIN operations."""
        optimizations = []
        query_hash = self._generate_query_hash(query)
        
        # Check for Cartesian products (missing JOIN conditions)
        if len(pattern.tables_involved) > 1 and not pattern.joins_used:
            optimization = QueryOptimization(
                query_hash=query_hash,
                original_query=query,
                optimization_type='join_fix',
                description='Potential Cartesian product detected - add proper JOIN conditions',
                optimized_query=None,
                impact_level='high',
                estimated_improvement='Prevents exponential result growth',
                explanation='Multiple tables without JOIN conditions create Cartesian products'
            )
            optimizations.append(optimization)
        
        # Suggest join order optimization
        if len(pattern.joins_used) > 2:
            optimization = QueryOptimization(
                query_hash=query_hash,
                original_query=query,
                optimization_type='join_order',
                description='Consider optimizing JOIN order - start with most selective table',
                optimized_query=None,
                impact_level='medium',
                estimated_improvement='20-50% performance gain',
                explanation='Proper join order reduces intermediate result sizes'
            )
            optimizations.append(optimization)
        
        return optimizations
    
    def _generate_optimization_explanation(self, rule_name: str, pattern: QueryPattern) -> str:
        """Generate detailed explanation for optimization."""
        explanations = {
            'select_star': f"SELECT * retrieves all {len(pattern.columns_accessed)} columns even if only few are needed. This increases I/O and network overhead.",
            'missing_where': f"Query scans entire table(s): {', '.join(pattern.tables_involved)}. Adding WHERE clause limits result set.",
            'missing_limit': "Query may return large result sets. LIMIT clause prevents excessive memory usage and network transfer.",
            'inefficient_like': "Leading wildcard (%) prevents index usage, causing full table scan.",
            'non_sargable': "Functions in WHERE clause prevent index usage. Move calculations outside WHERE condition."
        }
        return explanations.get(rule_name, "Optimization improves query performance")
    
    def analyze_query_patterns(self, queries: List[str]) -> Dict[str, Any]:
        """Analyze patterns across multiple queries."""
        patterns = {}
        optimization_summary = defaultdict(int)
        
        for query in queries:
            pattern = self.analyze_query(query)
            patterns[pattern.pattern_hash] = pattern
            
            optimizations = self.optimize_query(query)
            for opt in optimizations:
                optimization_summary[opt.optimization_type] += 1
        
        # Identify most common patterns
        query_types = Counter(p.query_type for p in patterns.values())
        table_usage = Counter()
        for p in patterns.values():
            table_usage.update(p.tables_involved)
        
        return {
            'total_queries_analyzed': len(queries),
            'unique_patterns': len(patterns),
            'query_type_distribution': dict(query_types),
            'most_accessed_tables': dict(table_usage.most_common(10)),
            'optimization_opportunities': dict(optimization_summary),
            'common_issues': self._identify_common_issues(patterns.values())
        }
    
    def _identify_common_issues(self, patterns) -> Dict[str, int]:
        """Identify common query issues across patterns."""
        issues = defaultdict(int)
        
        for pattern in patterns:
            if not pattern.where_conditions and pattern.query_type == 'SELECT':
                issues['missing_where_clause'] += 1
            
            if '*' in str(pattern.columns_accessed):
                issues['select_star_usage'] += 1
            
            if len(pattern.tables_involved) > 1 and not pattern.joins_used:
                issues['potential_cartesian_product'] += 1
            
            if not pattern.order_by_columns and pattern.query_type == 'SELECT':
                issues['missing_order_by'] += 1
        
        return dict(issues)
    
    def generate_optimization_report(self, queries: List[str]) -> Dict[str, Any]:
        """Generate comprehensive optimization report for queries."""
        all_optimizations = []
        
        for query in queries:
            optimizations = self.optimize_query(query)
            all_optimizations.extend(optimizations)
        
        # Group optimizations by impact level
        impact_groups = defaultdict(list)
        for opt in all_optimizations:
            impact_groups[opt.impact_level].append(opt)
        
        # Calculate potential improvements
        improvement_summary = {
            'high_impact_optimizations': len(impact_groups['high']),
            'medium_impact_optimizations': len(impact_groups['medium']),
            'low_impact_optimizations': len(impact_groups['low']),
            'total_optimizations': len(all_optimizations)
        }
        
        return {
            'generated_at': datetime.now().isoformat(),
            'queries_analyzed': len(queries),
            'improvement_summary': improvement_summary,
            'optimizations': [
                {
                    'type': opt.optimization_type,
                    'description': opt.description,
                    'impact_level': opt.impact_level,
                    'estimated_improvement': opt.estimated_improvement,
                    'explanation': opt.explanation
                }
                for opt in all_optimizations
            ],
            'pattern_analysis': self.analyze_query_patterns(queries)
        }
    
    def export_optimization_report(self, filepath: str, queries: List[str]) -> bool:
        """Export query optimization report to file."""
        try:
            report = self.generate_optimization_report(queries)
            
            with open(filepath, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            return True
        except Exception as e:
            print(f"Error exporting optimization report: {e}")
            return False