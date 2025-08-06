"""
Database index manager for creating, monitoring, and optimizing indexes.
"""

import re
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Set
from datetime import datetime, timedelta
from collections import defaultdict
from enum import Enum


class IndexType(Enum):
    """Types of database indexes."""
    PRIMARY = "primary"
    UNIQUE = "unique"
    STANDARD = "standard"
    COMPOSITE = "composite"
    PARTIAL = "partial"
    FUNCTIONAL = "functional"


@dataclass
class IndexInfo:
    """Database index information."""
    name: str
    table_name: str
    columns: List[str]
    index_type: IndexType
    size_bytes: Optional[int] = None
    cardinality: Optional[int] = None
    selectivity: Optional[float] = None
    usage_count: int = 0
    last_used: Optional[datetime] = None
    maintenance_cost: Optional[float] = None
    creation_date: Optional[datetime] = None


@dataclass
class IndexRecommendation:
    """Index creation/optimization recommendation."""
    table_name: str
    recommendation_type: str  # 'create', 'drop', 'modify'
    index_name: Optional[str]
    columns: List[str]
    index_type: IndexType
    priority: str  # 'high', 'medium', 'low'
    estimated_benefit: str
    reason: str
    sql_script: str


@dataclass
class IndexUsageStats:
    """Index usage statistics."""
    index_name: str
    seeks: int
    scans: int
    lookups: int
    updates: int
    size_mb: float
    rows_per_seek: float
    maintenance_overhead: float


class IndexManager:
    """
    Database index manager for analyzing, creating, and optimizing indexes
    to improve query performance while minimizing maintenance overhead.
    """
    
    def __init__(self, connection_string: Optional[str] = None):
        """
        Initialize index manager.
        
        Args:
            connection_string: Database connection string
        """
        self.connection_string = connection_string
        self.indexes: Dict[str, IndexInfo] = {}
        self.recommendations: List[IndexRecommendation] = []
        self.usage_statistics: Dict[str, IndexUsageStats] = {}
        
        # Index naming conventions
        self.naming_patterns = {
            IndexType.PRIMARY: "pk_{table}",
            IndexType.UNIQUE: "uq_{table}_{columns}",
            IndexType.STANDARD: "ix_{table}_{columns}",
            IndexType.COMPOSITE: "ix_{table}_{columns}",
            IndexType.PARTIAL: "ix_{table}_{columns}_partial",
            IndexType.FUNCTIONAL: "ix_{table}_{function}"
        }
        
        # Performance thresholds
        self.thresholds = {
            'unused_index_days': 30,
            'low_selectivity': 0.01,  # Less than 1% selectivity
            'high_maintenance_cost': 10.0,
            'min_usage_for_composite': 100
        }
    
    def discover_existing_indexes(self, table_name: Optional[str] = None) -> Dict[str, IndexInfo]:
        """
        Discover existing indexes in the database.
        
        Args:
            table_name: Specific table to analyze, or None for all tables
            
        Returns:
            Dictionary of discovered indexes
        """
        discovered_indexes = {}
        
        if self.connection_string:
            discovered_indexes = self._discover_indexes_from_database(table_name)
        else:
            # Mock discovery for testing
            discovered_indexes = self._mock_index_discovery(table_name)
        
        self.indexes.update(discovered_indexes)
        return discovered_indexes
    
    def _discover_indexes_from_database(self, table_name: Optional[str]) -> Dict[str, IndexInfo]:
        """Discover indexes from actual database."""
        indexes = {}
        
        try:
            import sqlite3
            conn = sqlite3.connect(self.connection_string)
            cursor = conn.cursor()
            
            # Get all tables if none specified
            if table_name is None:
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
                tables = [row[0] for row in cursor.fetchall()]
            else:
                tables = [table_name]
            
            # Get indexes for each table
            for table in tables:
                cursor.execute(f"PRAGMA index_list('{table}')")
                index_list = cursor.fetchall()
                
                for index_row in index_list:
                    index_name = index_row[1]
                    is_unique = bool(index_row[2])
                    
                    # Get index columns
                    cursor.execute(f"PRAGMA index_info('{index_name}')")
                    index_info = cursor.fetchall()
                    columns = [col[2] for col in index_info]
                    
                    # Determine index type
                    if index_name.startswith('pk_') or 'primary' in index_name.lower():
                        idx_type = IndexType.PRIMARY
                    elif is_unique:
                        idx_type = IndexType.UNIQUE
                    elif len(columns) > 1:
                        idx_type = IndexType.COMPOSITE
                    else:
                        idx_type = IndexType.STANDARD
                    
                    indexes[index_name] = IndexInfo(
                        name=index_name,
                        table_name=table,
                        columns=columns,
                        index_type=idx_type,
                        creation_date=datetime.now()  # SQLite doesn't track creation date
                    )
            
            conn.close()
            
        except Exception as e:
            print(f"Error discovering indexes: {e}")
        
        return indexes
    
    def _mock_index_discovery(self, table_name: Optional[str]) -> Dict[str, IndexInfo]:
        """Mock index discovery for testing."""
        mock_indexes = {}
        
        tables = [table_name] if table_name else ['users', 'orders', 'products']
        
        for table in tables:
            # Create mock indexes based on common patterns
            if table == 'users':
                mock_indexes[f'pk_{table}'] = IndexInfo(
                    name=f'pk_{table}',
                    table_name=table,
                    columns=['id'],
                    index_type=IndexType.PRIMARY,
                    size_bytes=1024,
                    cardinality=10000,
                    selectivity=1.0,
                    usage_count=5000,
                    last_used=datetime.now() - timedelta(minutes=5)
                )
                mock_indexes[f'uq_{table}_email'] = IndexInfo(
                    name=f'uq_{table}_email',
                    table_name=table,
                    columns=['email'],
                    index_type=IndexType.UNIQUE,
                    size_bytes=2048,
                    cardinality=10000,
                    selectivity=1.0,
                    usage_count=1500,
                    last_used=datetime.now() - timedelta(hours=1)
                )
                mock_indexes[f'ix_{table}_created_at'] = IndexInfo(
                    name=f'ix_{table}_created_at',
                    table_name=table,
                    columns=['created_at'],
                    index_type=IndexType.STANDARD,
                    size_bytes=1536,
                    cardinality=8000,
                    selectivity=0.8,
                    usage_count=300,
                    last_used=datetime.now() - timedelta(days=5)
                )
        
        return mock_indexes
    
    def analyze_index_usage(self, table_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Analyze index usage patterns and effectiveness.
        
        Args:
            table_name: Specific table to analyze
            
        Returns:
            Index usage analysis results
        """
        analysis = {
            'total_indexes': 0,
            'unused_indexes': [],
            'underutilized_indexes': [],
            'high_maintenance_indexes': [],
            'effective_indexes': [],
            'selectivity_analysis': {},
            'size_analysis': {}
        }
        
        # Filter indexes for analysis
        indexes_to_analyze = {}
        for name, info in self.indexes.items():
            if table_name is None or info.table_name == table_name:
                indexes_to_analyze[name] = info
        
        analysis['total_indexes'] = len(indexes_to_analyze)
        
        # Analyze each index
        for index_name, index_info in indexes_to_analyze.items():
            # Check for unused indexes
            if (index_info.last_used is None or 
                (datetime.now() - index_info.last_used).days > self.thresholds['unused_index_days']):
                analysis['unused_indexes'].append({
                    'name': index_name,
                    'table': index_info.table_name,
                    'last_used': index_info.last_used,
                    'size_bytes': index_info.size_bytes
                })
            
            # Check for underutilized indexes
            elif index_info.usage_count < 100:  # Less than 100 uses
                analysis['underutilized_indexes'].append({
                    'name': index_name,
                    'table': index_info.table_name,
                    'usage_count': index_info.usage_count,
                    'size_bytes': index_info.size_bytes
                })
            
            # Check for high maintenance cost
            elif (index_info.maintenance_cost and 
                  index_info.maintenance_cost > self.thresholds['high_maintenance_cost']):
                analysis['high_maintenance_indexes'].append({
                    'name': index_name,
                    'table': index_info.table_name,
                    'maintenance_cost': index_info.maintenance_cost
                })
            
            # Effective indexes
            else:
                analysis['effective_indexes'].append({
                    'name': index_name,
                    'table': index_info.table_name,
                    'usage_count': index_info.usage_count,
                    'selectivity': index_info.selectivity
                })
            
            # Selectivity analysis
            if index_info.selectivity is not None:
                if index_info.selectivity < self.thresholds['low_selectivity']:
                    analysis['selectivity_analysis'][index_name] = {
                        'selectivity': index_info.selectivity,
                        'issue': 'Low selectivity - may not be effective'
                    }
                else:
                    analysis['selectivity_analysis'][index_name] = {
                        'selectivity': index_info.selectivity,
                        'status': 'Good selectivity'
                    }
            
            # Size analysis
            if index_info.size_bytes:
                analysis['size_analysis'][index_name] = {
                    'size_mb': index_info.size_bytes / (1024 * 1024),
                    'size_category': self._categorize_index_size(index_info.size_bytes)
                }
        
        return analysis
    
    def _categorize_index_size(self, size_bytes: int) -> str:
        """Categorize index size."""
        size_mb = size_bytes / (1024 * 1024)
        if size_mb < 1:
            return 'small'
        elif size_mb < 10:
            return 'medium'
        elif size_mb < 100:
            return 'large'
        else:
            return 'very_large'
    
    def recommend_indexes(self, 
                         query_patterns: List[str],
                         table_name: Optional[str] = None) -> List[IndexRecommendation]:
        """
        Recommend indexes based on query patterns and analysis.
        
        Args:
            query_patterns: List of SQL query patterns to analyze
            table_name: Specific table to focus on
            
        Returns:
            List of index recommendations
        """
        recommendations = []
        
        # Analyze query patterns for index opportunities
        query_analysis = self._analyze_query_patterns_for_indexes(query_patterns)
        
        # Generate recommendations based on analysis
        for table, analysis in query_analysis.items():
            if table_name is None or table == table_name:
                recommendations.extend(self._generate_index_recommendations(table, analysis))
        
        # Analyze existing indexes for cleanup opportunities
        if table_name in self.indexes or table_name is None:
            cleanup_recommendations = self._generate_cleanup_recommendations(table_name)
            recommendations.extend(cleanup_recommendations)
        
        # Sort recommendations by priority
        priority_order = {'high': 3, 'medium': 2, 'low': 1}
        recommendations.sort(key=lambda x: priority_order.get(x.priority, 0), reverse=True)
        
        self.recommendations = recommendations
        return recommendations
    
    def _analyze_query_patterns_for_indexes(self, query_patterns: List[str]) -> Dict[str, Dict]:
        """Analyze query patterns to identify index opportunities."""
        analysis = defaultdict(lambda: {
            'where_columns': defaultdict(int),
            'join_columns': defaultdict(int),
            'order_by_columns': defaultdict(int),
            'group_by_columns': defaultdict(int),
            'composite_opportunities': defaultdict(int)
        })
        
        for query in query_patterns:
            query_upper = query.upper()
            
            # Extract table names
            tables = self._extract_tables_from_query(query)
            
            for table in tables:
                table_analysis = analysis[table]
                
                # Analyze WHERE clauses
                where_columns = self._extract_where_columns(query)
                for col in where_columns:
                    table_analysis['where_columns'][col] += 1
                
                # Analyze JOIN conditions
                join_columns = self._extract_join_columns(query)
                for col in join_columns:
                    table_analysis['join_columns'][col] += 1
                
                # Analyze ORDER BY
                order_columns = self._extract_order_by_columns(query)
                for col in order_columns:
                    table_analysis['order_by_columns'][col] += 1
                
                # Analyze GROUP BY
                group_columns = self._extract_group_by_columns(query)
                for col in group_columns:
                    table_analysis['group_by_columns'][col] += 1
                
                # Identify composite index opportunities
                if len(where_columns) > 1:
                    composite_key = tuple(sorted(where_columns[:3]))  # Max 3 columns
                    table_analysis['composite_opportunities'][composite_key] += 1
        
        return dict(analysis)
    
    def _extract_tables_from_query(self, query: str) -> Set[str]:
        """Extract table names from SQL query."""
        tables = set()
        
        # FROM clause
        from_matches = re.findall(r'FROM\s+(\w+)', query, re.IGNORECASE)
        tables.update(match.lower() for match in from_matches)
        
        # JOIN clauses
        join_matches = re.findall(r'JOIN\s+(\w+)', query, re.IGNORECASE)
        tables.update(match.lower() for match in join_matches)
        
        return tables
    
    def _extract_where_columns(self, query: str) -> List[str]:
        """Extract column names from WHERE clauses."""
        columns = []
        where_matches = re.findall(r'WHERE.*?(\w+)\s*[=<>!]', query, re.IGNORECASE)
        columns.extend(match.lower() for match in where_matches)
        return columns
    
    def _extract_join_columns(self, query: str) -> List[str]:
        """Extract column names from JOIN conditions."""
        columns = []
        join_matches = re.findall(r'ON\s+(\w+)\s*=\s*(\w+)', query, re.IGNORECASE)
        for match in join_matches:
            columns.extend(col.lower() for col in match)
        return columns
    
    def _extract_order_by_columns(self, query: str) -> List[str]:
        """Extract column names from ORDER BY clauses."""
        columns = []
        order_matches = re.findall(r'ORDER\s+BY\s+(\w+)', query, re.IGNORECASE)
        columns.extend(match.lower() for match in order_matches)
        return columns
    
    def _extract_group_by_columns(self, query: str) -> List[str]:
        """Extract column names from GROUP BY clauses."""
        columns = []
        group_matches = re.findall(r'GROUP\s+BY\s+(\w+)', query, re.IGNORECASE)
        columns.extend(match.lower() for match in group_matches)
        return columns
    
    def _generate_index_recommendations(self, table_name: str, analysis: Dict) -> List[IndexRecommendation]:
        """Generate index recommendations for a specific table."""
        recommendations = []
        
        # Single-column indexes for frequently used WHERE columns
        for column, usage_count in analysis['where_columns'].items():
            if usage_count >= 5:  # Used in 5+ queries
                index_name = self._generate_index_name(table_name, [column], IndexType.STANDARD)
                
                # Check if index already exists
                if not self._index_exists(table_name, [column]):
                    recommendations.append(IndexRecommendation(
                        table_name=table_name,
                        recommendation_type='create',
                        index_name=index_name,
                        columns=[column],
                        index_type=IndexType.STANDARD,
                        priority='high' if usage_count >= 10 else 'medium',
                        estimated_benefit=f"{min(usage_count * 10, 90)}% query performance improvement",
                        reason=f"Column '{column}' used in WHERE clause {usage_count} times",
                        sql_script=f"CREATE INDEX {index_name} ON {table_name}({column});"
                    ))
        
        # Composite indexes for multiple WHERE conditions
        for columns_tuple, usage_count in analysis['composite_opportunities'].items():
            if usage_count >= 3 and len(columns_tuple) >= 2:
                columns = list(columns_tuple)
                index_name = self._generate_index_name(table_name, columns, IndexType.COMPOSITE)
                
                if not self._index_exists(table_name, columns):
                    recommendations.append(IndexRecommendation(
                        table_name=table_name,
                        recommendation_type='create',
                        index_name=index_name,
                        columns=columns,
                        index_type=IndexType.COMPOSITE,
                        priority='high',
                        estimated_benefit=f"{min(usage_count * 15, 95)}% query performance improvement",
                        reason=f"Composite condition used {usage_count} times",
                        sql_script=f"CREATE INDEX {index_name} ON {table_name}({', '.join(columns)});"
                    ))
        
        # Indexes for ORDER BY columns
        for column, usage_count in analysis['order_by_columns'].items():
            if usage_count >= 3:
                index_name = self._generate_index_name(table_name, [column], IndexType.STANDARD)
                
                if not self._index_exists(table_name, [column]):
                    recommendations.append(IndexRecommendation(
                        table_name=table_name,
                        recommendation_type='create',
                        index_name=index_name,
                        columns=[column],
                        index_type=IndexType.STANDARD,
                        priority='medium',
                        estimated_benefit=f"{min(usage_count * 8, 70)}% sorting performance improvement",
                        reason=f"Column '{column}' used in ORDER BY {usage_count} times",
                        sql_script=f"CREATE INDEX {index_name} ON {table_name}({column});"
                    ))
        
        return recommendations
    
    def _generate_cleanup_recommendations(self, table_name: Optional[str]) -> List[IndexRecommendation]:
        """Generate recommendations for index cleanup."""
        recommendations = []
        analysis = self.analyze_index_usage(table_name)
        
        # Recommend dropping unused indexes
        for unused_index in analysis['unused_indexes']:
            # Don't recommend dropping primary keys
            index_info = self.indexes.get(unused_index['name'])
            if index_info and index_info.index_type != IndexType.PRIMARY:
                recommendations.append(IndexRecommendation(
                    table_name=unused_index['table'],
                    recommendation_type='drop',
                    index_name=unused_index['name'],
                    columns=[],
                    index_type=IndexType.STANDARD,
                    priority='low',
                    estimated_benefit="Reduced storage and faster writes",
                    reason=f"Index unused for {self.thresholds['unused_index_days']}+ days",
                    sql_script=f"DROP INDEX {unused_index['name']};"
                ))
        
        # Recommend reviewing underutilized indexes
        for underutil_index in analysis['underutilized_indexes']:
            index_info = self.indexes.get(underutil_index['name'])
            if index_info and index_info.index_type != IndexType.PRIMARY:
                recommendations.append(IndexRecommendation(
                    table_name=underutil_index['table'],
                    recommendation_type='drop',
                    index_name=underutil_index['name'],
                    columns=[],
                    index_type=IndexType.STANDARD,
                    priority='low',
                    estimated_benefit="Reduced maintenance overhead",
                    reason=f"Index used only {underutil_index['usage_count']} times",
                    sql_script=f"-- Consider dropping: DROP INDEX {underutil_index['name']};"
                ))
        
        return recommendations
    
    def _generate_index_name(self, table_name: str, columns: List[str], index_type: IndexType) -> str:
        """Generate standard index name."""
        columns_str = '_'.join(columns)
        pattern = self.naming_patterns.get(index_type, "ix_{table}_{columns}")
        return pattern.format(table=table_name, columns=columns_str)
    
    def _index_exists(self, table_name: str, columns: List[str]) -> bool:
        """Check if an index already exists for the given columns."""
        for index_info in self.indexes.values():
            if (index_info.table_name == table_name and 
                set(index_info.columns) == set(columns)):
                return True
        return False
    
    def create_index(self, recommendation: IndexRecommendation) -> bool:
        """
        Execute index creation based on recommendation.
        
        Args:
            recommendation: Index recommendation to implement
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if self.connection_string:
                import sqlite3
                conn = sqlite3.connect(self.connection_string)
                cursor = conn.cursor()
                
                # Execute the SQL script
                cursor.execute(recommendation.sql_script)
                conn.commit()
                conn.close()
                
                # Add to our index tracking
                self.indexes[recommendation.index_name] = IndexInfo(
                    name=recommendation.index_name,
                    table_name=recommendation.table_name,
                    columns=recommendation.columns,
                    index_type=recommendation.index_type,
                    creation_date=datetime.now(),
                    usage_count=0
                )
                
                return True
            else:
                # Mock creation for testing
                print(f"Mock creating index: {recommendation.sql_script}")
                return True
                
        except Exception as e:
            print(f"Error creating index: {e}")
            return False
    
    def drop_index(self, index_name: str) -> bool:
        """
        Drop an existing index.
        
        Args:
            index_name: Name of index to drop
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if self.connection_string:
                import sqlite3
                conn = sqlite3.connect(self.connection_string)
                cursor = conn.cursor()
                
                cursor.execute(f"DROP INDEX {index_name}")
                conn.commit()
                conn.close()
            
            # Remove from tracking
            self.indexes.pop(index_name, None)
            return True
            
        except Exception as e:
            print(f"Error dropping index {index_name}: {e}")
            return False
    
    def generate_index_maintenance_script(self, table_name: Optional[str] = None) -> str:
        """Generate SQL script for index maintenance."""
        script_lines = [
            "-- Index Maintenance Script",
            f"-- Generated at: {datetime.now().isoformat()}",
            ""
        ]
        
        # Filter recommendations
        relevant_recs = [
            rec for rec in self.recommendations
            if table_name is None or rec.table_name == table_name
        ]
        
        # Group by recommendation type
        create_recs = [r for r in relevant_recs if r.recommendation_type == 'create']
        drop_recs = [r for r in relevant_recs if r.recommendation_type == 'drop']
        
        # Add create statements
        if create_recs:
            script_lines.extend([
                "-- Create recommended indexes",
                "BEGIN TRANSACTION;",
                ""
            ])
            
            for rec in create_recs:
                script_lines.extend([
                    f"-- {rec.reason}",
                    f"-- Expected benefit: {rec.estimated_benefit}",
                    rec.sql_script,
                    ""
                ])
            
            script_lines.extend(["COMMIT;", ""])
        
        # Add drop statements
        if drop_recs:
            script_lines.extend([
                "-- Drop unused/underutilized indexes",
                "-- Review carefully before executing",
                "BEGIN TRANSACTION;",
                ""
            ])
            
            for rec in drop_recs:
                script_lines.extend([
                    f"-- {rec.reason}",
                    rec.sql_script,
                    ""
                ])
            
            script_lines.extend(["COMMIT;", ""])
        
        # Add maintenance commands
        if table_name:
            script_lines.extend([
                f"-- Update statistics for {table_name}",
                f"ANALYZE {table_name};"
            ])
        else:
            script_lines.extend([
                "-- Update database statistics",
                "ANALYZE;"
            ])
        
        return "\n".join(script_lines)
    
    def export_index_analysis(self, filepath: str, table_name: Optional[str] = None) -> bool:
        """Export comprehensive index analysis to file."""
        try:
            analysis = self.analyze_index_usage(table_name)
            recommendations = self.recommendations
            
            report = {
                'generated_at': datetime.now().isoformat(),
                'table_analyzed': table_name or 'all_tables',
                'index_analysis': analysis,
                'recommendations': [
                    {
                        'table': rec.table_name,
                        'type': rec.recommendation_type,
                        'index_name': rec.index_name,
                        'columns': rec.columns,
                        'priority': rec.priority,
                        'benefit': rec.estimated_benefit,
                        'reason': rec.reason,
                        'sql': rec.sql_script
                    }
                    for rec in recommendations
                    if table_name is None or rec.table_name == table_name
                ],
                'maintenance_script': self.generate_index_maintenance_script(table_name)
            }
            
            import json
            with open(filepath, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            return True
        except Exception as e:
            print(f"Error exporting index analysis: {e}")
            return False