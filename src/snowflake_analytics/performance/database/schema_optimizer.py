"""
Database schema optimization for improved query performance and storage efficiency.
"""

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from collections import defaultdict
import sqlite3


@dataclass
class TableAnalysis:
    """Table analysis results."""
    table_name: str
    row_count: int
    storage_size_bytes: int
    column_count: int
    index_count: int
    unused_indexes: List[str]
    suggested_indexes: List[str]
    data_type_issues: List[str]
    normalization_opportunities: List[str]


@dataclass
class SchemaRecommendation:
    """Schema optimization recommendation."""
    table_name: str
    recommendation_type: str
    description: str
    impact_level: str  # 'low', 'medium', 'high'
    estimated_benefit: str
    sql_script: Optional[str] = None


class SchemaOptimizer:
    """
    Database schema optimizer for analyzing and improving database structure,
    data types, and storage efficiency.
    """
    
    def __init__(self, connection_string: Optional[str] = None):
        """
        Initialize schema optimizer.
        
        Args:
            connection_string: Database connection string (supports SQLite for now)
        """
        self.connection_string = connection_string
        self.analysis_cache: Dict[str, TableAnalysis] = {}
        self.recommendations: List[SchemaRecommendation] = []
        
        # Data type optimization rules
        self.data_type_rules = {
            'text_to_varchar': {
                'pattern': r'^TEXT$',
                'suggestion': 'VARCHAR with appropriate length',
                'benefit': 'Reduced storage overhead'
            },
            'integer_size_optimization': {
                'pattern': r'^INTEGER$',
                'suggestion': 'Use SMALLINT/INT based on value range',
                'benefit': 'Reduced storage and improved performance'
            },
            'decimal_precision': {
                'pattern': r'^REAL$',
                'suggestion': 'Use DECIMAL with specific precision',
                'benefit': 'Better precision control'
            }
        }
    
    def analyze_table_structure(self, table_name: str) -> TableAnalysis:
        """
        Analyze table structure for optimization opportunities.
        
        Args:
            table_name: Name of the table to analyze
            
        Returns:
            Table analysis results
        """
        if table_name in self.analysis_cache:
            return self.analysis_cache[table_name]
        
        analysis = TableAnalysis(
            table_name=table_name,
            row_count=0,
            storage_size_bytes=0,
            column_count=0,
            index_count=0,
            unused_indexes=[],
            suggested_indexes=[],
            data_type_issues=[],
            normalization_opportunities=[]
        )
        
        if self.connection_string:
            analysis = self._analyze_with_database(table_name)
        else:
            analysis = self._analyze_schema_pattern(table_name)
        
        self.analysis_cache[table_name] = analysis
        return analysis
    
    def _analyze_with_database(self, table_name: str) -> TableAnalysis:
        """Analyze table using actual database connection."""
        try:
            conn = sqlite3.connect(self.connection_string)
            cursor = conn.cursor()
            
            # Get table info
            cursor.execute(f"PRAGMA table_info('{table_name}')")
            columns = cursor.fetchall()
            
            # Get row count
            cursor.execute(f"SELECT COUNT(*) FROM '{table_name}'")
            row_count = cursor.fetchone()[0]
            
            # Get index info
            cursor.execute(f"PRAGMA index_list('{table_name}')")
            indexes = cursor.fetchall()
            
            # Analyze data types
            data_type_issues = []
            for col in columns:
                col_name, data_type = col[1], col[2]
                if self._has_data_type_issue(data_type):
                    data_type_issues.append(f"Column '{col_name}': {data_type}")
            
            # Suggest indexes based on common patterns
            suggested_indexes = self._suggest_indexes_from_schema(table_name, columns)
            
            # Check for unused indexes
            unused_indexes = self._find_unused_indexes(table_name, indexes)
            
            analysis = TableAnalysis(
                table_name=table_name,
                row_count=row_count,
                storage_size_bytes=0,  # Would need specific DB calculations
                column_count=len(columns),
                index_count=len(indexes),
                unused_indexes=unused_indexes,
                suggested_indexes=suggested_indexes,
                data_type_issues=data_type_issues,
                normalization_opportunities=self._find_normalization_opportunities(columns)
            )
            
            conn.close()
            return analysis
            
        except Exception as e:
            print(f"Error analyzing table {table_name}: {e}")
            return self._analyze_schema_pattern(table_name)
    
    def _analyze_schema_pattern(self, table_name: str) -> TableAnalysis:
        """Analyze table using pattern-based heuristics."""
        # Mock analysis when no database connection
        suggested_indexes = []
        data_type_issues = []
        normalization_opportunities = []
        
        # Common patterns for index suggestions
        if 'user' in table_name.lower():
            suggested_indexes.extend(['email', 'username', 'created_at'])
        if 'order' in table_name.lower():
            suggested_indexes.extend(['user_id', 'created_at', 'status'])
        if 'log' in table_name.lower() or 'event' in table_name.lower():
            suggested_indexes.extend(['timestamp', 'user_id', 'event_type'])
        
        # Common data type issues
        data_type_issues.append("Consider using appropriate VARCHAR lengths instead of TEXT")
        data_type_issues.append("Review REAL columns for DECIMAL precision needs")
        
        return TableAnalysis(
            table_name=table_name,
            row_count=1000,  # Estimated
            storage_size_bytes=1024 * 1024,  # 1MB estimated
            column_count=8,  # Estimated
            index_count=2,  # Estimated
            unused_indexes=[],
            suggested_indexes=suggested_indexes,
            data_type_issues=data_type_issues,
            normalization_opportunities=normalization_opportunities
        )
    
    def _has_data_type_issue(self, data_type: str) -> bool:
        """Check if data type has optimization opportunities."""
        data_type_upper = data_type.upper()
        for rule in self.data_type_rules.values():
            if re.match(rule['pattern'], data_type_upper):
                return True
        return False
    
    def _suggest_indexes_from_schema(self, table_name: str, columns: List) -> List[str]:
        """Suggest indexes based on column names and types."""
        suggestions = []
        
        for col in columns:
            col_name = col[1].lower()
            
            # Common patterns for indexing
            if col_name in ['id', 'user_id', 'customer_id', 'order_id']:
                suggestions.append(col_name)
            elif col_name in ['email', 'username', 'phone']:
                suggestions.append(col_name)
            elif col_name in ['created_at', 'updated_at', 'timestamp']:
                suggestions.append(col_name)
            elif col_name.endswith('_id'):
                suggestions.append(col_name)
            elif col_name in ['status', 'type', 'category']:
                suggestions.append(col_name)
        
        return suggestions
    
    def _find_unused_indexes(self, table_name: str, indexes: List) -> List[str]:
        """Find potentially unused indexes."""
        unused = []
        
        # This would require query log analysis in a real implementation
        # For now, return empty list
        return unused
    
    def _find_normalization_opportunities(self, columns: List) -> List[str]:
        """Find table normalization opportunities."""
        opportunities = []
        
        # Look for repeated patterns in column names
        column_groups = defaultdict(list)
        for col in columns:
            col_name = col[1].lower()
            if '_' in col_name:
                prefix = col_name.split('_')[0]
                column_groups[prefix].append(col_name)
        
        # Suggest normalization for groups with multiple columns
        for prefix, group in column_groups.items():
            if len(group) >= 3 and prefix not in ['created', 'updated']:
                opportunities.append(
                    f"Consider normalizing {prefix}_* columns into separate table"
                )
        
        return opportunities
    
    def generate_schema_recommendations(self, tables: List[str]) -> List[SchemaRecommendation]:
        """
        Generate comprehensive schema optimization recommendations.
        
        Args:
            tables: List of table names to analyze
            
        Returns:
            List of optimization recommendations
        """
        recommendations = []
        
        for table_name in tables:
            analysis = self.analyze_table_structure(table_name)
            
            # Index recommendations
            for index_suggestion in analysis.suggested_indexes:
                recommendations.append(SchemaRecommendation(
                    table_name=table_name,
                    recommendation_type='index',
                    description=f"Add index on column '{index_suggestion}'",
                    impact_level='medium',
                    estimated_benefit='20-50% query performance improvement',
                    sql_script=f"CREATE INDEX idx_{table_name}_{index_suggestion} ON {table_name}({index_suggestion});"
                ))
            
            # Data type recommendations
            for data_type_issue in analysis.data_type_issues:
                recommendations.append(SchemaRecommendation(
                    table_name=table_name,
                    recommendation_type='data_type',
                    description=f"Optimize data type: {data_type_issue}",
                    impact_level='low',
                    estimated_benefit='5-15% storage reduction',
                    sql_script=None  # Would need specific column analysis
                ))
            
            # Normalization recommendations
            for norm_opportunity in analysis.normalization_opportunities:
                recommendations.append(SchemaRecommendation(
                    table_name=table_name,
                    recommendation_type='normalization',
                    description=norm_opportunity,
                    impact_level='high',
                    estimated_benefit='Improved data integrity and reduced redundancy',
                    sql_script=None  # Complex refactoring required
                ))
            
            # Unused index cleanup
            for unused_index in analysis.unused_indexes:
                recommendations.append(SchemaRecommendation(
                    table_name=table_name,
                    recommendation_type='cleanup',
                    description=f"Remove unused index '{unused_index}'",
                    impact_level='low',
                    estimated_benefit='Reduced storage and faster writes',
                    sql_script=f"DROP INDEX {unused_index};"
                ))
        
        self.recommendations = recommendations
        return recommendations
    
    def optimize_data_types(self, table_name: str) -> Dict[str, Any]:
        """
        Generate data type optimization suggestions.
        
        Args:
            table_name: Table to optimize
            
        Returns:
            Dictionary with optimization suggestions
        """
        analysis = self.analyze_table_structure(table_name)
        
        optimizations = {
            'table_name': table_name,
            'current_issues': analysis.data_type_issues,
            'recommendations': [],
            'estimated_storage_savings': '10-30%'
        }
        
        # Generate specific recommendations
        for issue in analysis.data_type_issues:
            if 'TEXT' in issue:
                optimizations['recommendations'].append({
                    'issue': issue,
                    'suggestion': 'Replace TEXT with VARCHAR(n) where n is appropriate length',
                    'benefit': 'Reduced storage overhead and improved query performance'
                })
            elif 'REAL' in issue:
                optimizations['recommendations'].append({
                    'issue': issue,
                    'suggestion': 'Use DECIMAL(precision, scale) for exact numeric values',
                    'benefit': 'Better precision control and consistent calculations'
                })
            elif 'INTEGER' in issue:
                optimizations['recommendations'].append({
                    'issue': issue,
                    'suggestion': 'Use SMALLINT or BIGINT based on value range',
                    'benefit': 'Optimized storage and improved cache efficiency'
                })
        
        return optimizations
    
    def generate_schema_migration_script(self, 
                                       recommendations: List[SchemaRecommendation],
                                       table_name: str) -> str:
        """
        Generate SQL migration script for schema optimizations.
        
        Args:
            recommendations: List of recommendations to implement
            table_name: Target table name
            
        Returns:
            SQL migration script
        """
        script_lines = [
            f"-- Schema optimization migration for table: {table_name}",
            f"-- Generated at: {datetime.now().isoformat()}",
            "",
            "BEGIN TRANSACTION;",
            ""
        ]
        
        # Group recommendations by type
        table_recommendations = [r for r in recommendations if r.table_name == table_name]
        
        # Index creations
        index_recs = [r for r in table_recommendations if r.recommendation_type == 'index']
        if index_recs:
            script_lines.append("-- Create recommended indexes")
            for rec in index_recs:
                if rec.sql_script:
                    script_lines.append(rec.sql_script)
            script_lines.append("")
        
        # Index removals
        cleanup_recs = [r for r in table_recommendations if r.recommendation_type == 'cleanup']
        if cleanup_recs:
            script_lines.append("-- Remove unused indexes")
            for rec in cleanup_recs:
                if rec.sql_script:
                    script_lines.append(rec.sql_script)
            script_lines.append("")
        
        # Data type optimizations (would need more complex logic)
        datatype_recs = [r for r in table_recommendations if r.recommendation_type == 'data_type']
        if datatype_recs:
            script_lines.append("-- Data type optimizations")
            script_lines.append(f"-- Review and manually implement data type changes for {table_name}")
            for rec in datatype_recs:
                script_lines.append(f"-- {rec.description}")
            script_lines.append("")
        
        script_lines.extend([
            "COMMIT;",
            "",
            "-- Run ANALYZE after applying changes to update statistics",
            f"ANALYZE {table_name};"
        ])
        
        return "\n".join(script_lines)
    
    def export_schema_analysis(self, filepath: str, tables: List[str]) -> bool:
        """Export comprehensive schema analysis to file."""
        try:
            analyses = {}
            for table in tables:
                analyses[table] = self.analyze_table_structure(table)
            
            recommendations = self.generate_schema_recommendations(tables)
            
            report = {
                'generated_at': datetime.now().isoformat(),
                'tables_analyzed': len(tables),
                'total_recommendations': len(recommendations),
                'table_analyses': {},
                'recommendations_by_table': {},
                'summary': {
                    'total_suggested_indexes': 0,
                    'total_data_type_issues': 0,
                    'total_normalization_opportunities': 0
                }
            }
            
            # Process analyses
            for table, analysis in analyses.items():
                report['table_analyses'][table] = {
                    'row_count': analysis.row_count,
                    'column_count': analysis.column_count,
                    'index_count': analysis.index_count,
                    'suggested_indexes': analysis.suggested_indexes,
                    'data_type_issues': analysis.data_type_issues,
                    'normalization_opportunities': analysis.normalization_opportunities
                }
                
                report['summary']['total_suggested_indexes'] += len(analysis.suggested_indexes)
                report['summary']['total_data_type_issues'] += len(analysis.data_type_issues)
                report['summary']['total_normalization_opportunities'] += len(analysis.normalization_opportunities)
            
            # Group recommendations by table
            for rec in recommendations:
                if rec.table_name not in report['recommendations_by_table']:
                    report['recommendations_by_table'][rec.table_name] = []
                
                report['recommendations_by_table'][rec.table_name].append({
                    'type': rec.recommendation_type,
                    'description': rec.description,
                    'impact_level': rec.impact_level,
                    'estimated_benefit': rec.estimated_benefit,
                    'sql_script': rec.sql_script
                })
            
            import json
            with open(filepath, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            return True
        except Exception as e:
            print(f"Error exporting schema analysis: {e}")
            return False
    
    def reset_analysis_cache(self):
        """Reset cached analysis results."""
        self.analysis_cache.clear()
        self.recommendations.clear()