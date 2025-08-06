"""
Data partition manager for implementing partitioning and archiving strategies.
"""

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
from enum import Enum
from collections import defaultdict


class PartitionType(Enum):
    """Types of data partitioning."""
    RANGE = "range"          # Range-based partitioning (dates, numbers)
    HASH = "hash"            # Hash-based partitioning
    LIST = "list"            # List-based partitioning (categories, regions)
    COMPOSITE = "composite"   # Combination of multiple partition types


class ArchiveStrategy(Enum):
    """Data archiving strategies."""
    COLD_STORAGE = "cold_storage"      # Move to slower, cheaper storage
    DELETE = "delete"                  # Delete after retention period
    COMPRESS = "compress"              # Compress old data
    SUMMARIZE = "summarize"            # Keep aggregated summaries


@dataclass
class PartitionInfo:
    """Information about a data partition."""
    partition_name: str
    table_name: str
    partition_type: PartitionType
    partition_key: str
    partition_value: str
    row_count: int
    size_bytes: int
    created_date: datetime
    last_accessed: Optional[datetime] = None
    access_frequency: int = 0
    data_range_start: Optional[str] = None
    data_range_end: Optional[str] = None


@dataclass
class PartitionRecommendation:
    """Partition strategy recommendation."""
    table_name: str
    recommendation_type: str  # 'create_partition', 'archive', 'merge', 'split'
    partition_type: PartitionType
    partition_key: str
    strategy_description: str
    estimated_benefit: str
    implementation_complexity: str  # 'low', 'medium', 'high'
    sql_script: Optional[str] = None
    prerequisites: List[str] = None


@dataclass
class ArchiveRecommendation:
    """Data archiving recommendation."""
    table_name: str
    partition_name: Optional[str]
    archive_strategy: ArchiveStrategy
    criteria: str
    estimated_data_size: int
    estimated_cost_savings: str
    retention_period: timedelta
    implementation_steps: List[str]


class PartitionManager:
    """
    Data partition manager for implementing partitioning strategies to improve
    query performance and manage data lifecycle effectively.
    """
    
    def __init__(self, connection_string: Optional[str] = None):
        """
        Initialize partition manager.
        
        Args:
            connection_string: Database connection string
        """
        self.connection_string = connection_string
        self.partitions: Dict[str, PartitionInfo] = {}
        self.partition_recommendations: List[PartitionRecommendation] = []
        self.archive_recommendations: List[ArchiveRecommendation] = []
        
        # Default partitioning strategies
        self.partition_strategies = {
            'time_series': {
                'type': PartitionType.RANGE,
                'key_patterns': ['created_at', 'timestamp', 'date', 'time'],
                'partition_interval': 'monthly',
                'benefit': 'Efficient time-range queries and data archiving'
            },
            'geographic': {
                'type': PartitionType.LIST,
                'key_patterns': ['region', 'country', 'state', 'location'],
                'partition_interval': 'by_value',
                'benefit': 'Improved query performance for location-based queries'
            },
            'categorical': {
                'type': PartitionType.LIST,
                'key_patterns': ['status', 'type', 'category', 'department'],
                'partition_interval': 'by_value',
                'benefit': 'Faster filtering on categorical data'
            },
            'high_cardinality': {
                'type': PartitionType.HASH,
                'key_patterns': ['user_id', 'customer_id', 'account_id'],
                'partition_interval': 'hash_buckets',
                'benefit': 'Distributed load across multiple partitions'
            }
        }
        
        # Archive policies
        self.archive_policies = {
            'logs': {
                'retention_days': 90,
                'strategy': ArchiveStrategy.COMPRESS,
                'partition_key': 'created_at'
            },
            'transactions': {
                'retention_days': 2555,  # 7 years
                'strategy': ArchiveStrategy.COLD_STORAGE,
                'partition_key': 'transaction_date'
            },
            'analytics': {
                'retention_days': 365,
                'strategy': ArchiveStrategy.SUMMARIZE,
                'partition_key': 'date'
            },
            'temporary': {
                'retention_days': 30,
                'strategy': ArchiveStrategy.DELETE,
                'partition_key': 'created_at'
            }
        }
    
    def analyze_table_for_partitioning(self, 
                                     table_name: str,
                                     query_patterns: List[str] = None) -> Dict[str, Any]:
        """
        Analyze table characteristics to determine optimal partitioning strategy.
        
        Args:
            table_name: Name of table to analyze
            query_patterns: Sample queries against the table
            
        Returns:
            Analysis results and recommendations
        """
        analysis = {
            'table_name': table_name,
            'analysis_date': datetime.now().isoformat(),
            'table_characteristics': {},
            'query_analysis': {},
            'partitioning_opportunities': [],
            'recommended_strategy': None
        }
        
        # Analyze table characteristics
        table_characteristics = self._analyze_table_characteristics(table_name)
        analysis['table_characteristics'] = table_characteristics
        
        # Analyze query patterns if provided
        if query_patterns:
            query_analysis = self._analyze_query_patterns_for_partitioning(
                table_name, query_patterns
            )
            analysis['query_analysis'] = query_analysis
        
        # Identify partitioning opportunities
        opportunities = self._identify_partitioning_opportunities(
            table_name, table_characteristics, query_patterns or []
        )
        analysis['partitioning_opportunities'] = opportunities
        
        # Recommend best strategy
        if opportunities:
            best_strategy = max(opportunities, key=lambda x: x.get('score', 0))
            analysis['recommended_strategy'] = best_strategy
        
        return analysis
    
    def _analyze_table_characteristics(self, table_name: str) -> Dict[str, Any]:
        """Analyze table structure and data characteristics."""
        characteristics = {
            'estimated_row_count': 0,
            'estimated_size_mb': 0,
            'columns': [],
            'potential_partition_keys': [],
            'data_distribution': {}
        }
        
        if self.connection_string:
            characteristics = self._analyze_table_from_database(table_name)
        else:
            # Mock analysis for testing
            characteristics = self._mock_table_analysis(table_name)
        
        return characteristics
    
    def _analyze_table_from_database(self, table_name: str) -> Dict[str, Any]:
        """Analyze table using database connection."""
        characteristics = {
            'estimated_row_count': 0,
            'estimated_size_mb': 0,
            'columns': [],
            'potential_partition_keys': [],
            'data_distribution': {}
        }
        
        try:
            import sqlite3
            conn = sqlite3.connect(self.connection_string)
            cursor = conn.cursor()
            
            # Get column information
            cursor.execute(f"PRAGMA table_info('{table_name}')")
            columns_info = cursor.fetchall()
            
            for col_info in columns_info:
                col_name = col_info[1]
                col_type = col_info[2]
                characteristics['columns'].append({
                    'name': col_name,
                    'type': col_type
                })
                
                # Check if column could be a good partition key
                if self._is_potential_partition_key(col_name, col_type):
                    characteristics['potential_partition_keys'].append(col_name)
            
            # Get approximate row count
            cursor.execute(f"SELECT COUNT(*) FROM '{table_name}'")
            row_count = cursor.fetchone()[0]
            characteristics['estimated_row_count'] = row_count
            
            # Estimate size (rough calculation)
            characteristics['estimated_size_mb'] = (row_count * len(columns_info) * 50) / (1024 * 1024)
            
            conn.close()
            
        except Exception as e:
            print(f"Error analyzing table {table_name}: {e}")
        
        return characteristics
    
    def _mock_table_analysis(self, table_name: str) -> Dict[str, Any]:
        """Mock table analysis for testing."""
        base_characteristics = {
            'estimated_row_count': 100000,
            'estimated_size_mb': 50,
            'columns': [
                {'name': 'id', 'type': 'INTEGER'},
                {'name': 'created_at', 'type': 'TIMESTAMP'},
                {'name': 'status', 'type': 'VARCHAR'},
                {'name': 'user_id', 'type': 'INTEGER'}
            ],
            'potential_partition_keys': ['created_at', 'status', 'user_id'],
            'data_distribution': {
                'created_at': {'min': '2023-01-01', 'max': '2024-01-01', 'distinct_values': 365},
                'status': {'distinct_values': 5, 'values': ['active', 'inactive', 'pending', 'cancelled', 'completed']},
                'user_id': {'distinct_values': 10000, 'min': 1, 'max': 50000}
            }
        }
        
        # Customize based on table name
        if 'log' in table_name.lower():
            base_characteristics['estimated_row_count'] = 1000000
            base_characteristics['estimated_size_mb'] = 200
        elif 'user' in table_name.lower():
            base_characteristics['estimated_row_count'] = 50000
            base_characteristics['potential_partition_keys'] = ['created_at', 'region']
        
        return base_characteristics
    
    def _is_potential_partition_key(self, col_name: str, col_type: str) -> bool:
        """Determine if a column could be a good partition key."""
        col_name_lower = col_name.lower()
        col_type_upper = col_type.upper()
        
        # Time-based columns
        time_patterns = ['date', 'time', 'created', 'updated', 'timestamp']
        if any(pattern in col_name_lower for pattern in time_patterns):
            return True
        
        # Categorical columns
        if col_type_upper in ['VARCHAR', 'TEXT', 'CHAR']:
            categorical_patterns = ['status', 'type', 'category', 'region', 'country', 'state']
            if any(pattern in col_name_lower for pattern in categorical_patterns):
                return True
        
        # ID columns (for hash partitioning)
        if col_name_lower.endswith('_id') and col_type_upper in ['INTEGER', 'BIGINT']:
            return True
        
        return False
    
    def _analyze_query_patterns_for_partitioning(self, 
                                               table_name: str,
                                               query_patterns: List[str]) -> Dict[str, Any]:
        """Analyze query patterns to understand access patterns."""
        analysis = {
            'frequently_filtered_columns': defaultdict(int),
            'date_range_queries': 0,
            'specific_value_queries': 0,
            'full_table_scans': 0,
            'join_patterns': defaultdict(int)
        }
        
        for query in query_patterns:
            query_upper = query.upper()
            
            # Skip queries not involving this table
            if table_name.upper() not in query_upper:
                continue
            
            # Analyze WHERE clauses
            where_columns = self._extract_where_columns_with_conditions(query)
            for col, condition_type in where_columns:
                analysis['frequently_filtered_columns'][col] += 1
                
                if condition_type == 'range':
                    analysis['date_range_queries'] += 1
                elif condition_type == 'equality':
                    analysis['specific_value_queries'] += 1
            
            # Check for full table scans (no WHERE clause)
            if 'WHERE' not in query_upper:
                analysis['full_table_scans'] += 1
            
            # Analyze JOIN patterns
            join_columns = self._extract_join_columns_for_table(query, table_name)
            for col in join_columns:
                analysis['join_patterns'][col] += 1
        
        return dict(analysis)
    
    def _extract_where_columns_with_conditions(self, query: str) -> List[tuple]:
        """Extract WHERE columns and their condition types."""
        columns_conditions = []
        
        # Range conditions (BETWEEN, >, <)
        range_matches = re.findall(r'(\w+)\s*(?:BETWEEN|[<>]=?)', query, re.IGNORECASE)
        for col in range_matches:
            columns_conditions.append((col.lower(), 'range'))
        
        # Equality conditions
        eq_matches = re.findall(r'(\w+)\s*=', query, re.IGNORECASE)
        for col in eq_matches:
            columns_conditions.append((col.lower(), 'equality'))
        
        # IN conditions
        in_matches = re.findall(r'(\w+)\s+IN\s*\(', query, re.IGNORECASE)
        for col in in_matches:
            columns_conditions.append((col.lower(), 'in_list'))
        
        return columns_conditions
    
    def _extract_join_columns_for_table(self, query: str, table_name: str) -> List[str]:
        """Extract JOIN columns for a specific table."""
        join_columns = []
        
        # Find JOIN conditions mentioning this table
        join_pattern = rf'JOIN\s+{table_name}\s+.*?ON\s+(?:{table_name}\.)?(\w+)\s*='
        matches = re.findall(join_pattern, query, re.IGNORECASE)
        join_columns.extend(match.lower() for match in matches)
        
        return join_columns
    
    def _identify_partitioning_opportunities(self,
                                           table_name: str,
                                           characteristics: Dict[str, Any],
                                           query_patterns: List[str]) -> List[Dict[str, Any]]:
        """Identify and score partitioning opportunities."""
        opportunities = []
        
        # Minimum table size threshold for partitioning
        min_size_mb = 100
        min_rows = 100000
        
        table_size = characteristics.get('estimated_size_mb', 0)
        table_rows = characteristics.get('estimated_row_count', 0)
        
        if table_size < min_size_mb or table_rows < min_rows:
            return opportunities  # Table too small to benefit from partitioning
        
        potential_keys = characteristics.get('potential_partition_keys', [])
        
        for partition_key in potential_keys:
            # Score this partitioning opportunity
            opportunity = self._score_partitioning_opportunity(
                table_name, partition_key, characteristics, query_patterns
            )
            
            if opportunity['score'] > 0:
                opportunities.append(opportunity)
        
        # Sort by score (highest first)
        opportunities.sort(key=lambda x: x['score'], reverse=True)
        
        return opportunities
    
    def _score_partitioning_opportunity(self,
                                      table_name: str,
                                      partition_key: str,
                                      characteristics: Dict[str, Any],
                                      query_patterns: List[str]) -> Dict[str, Any]:
        """Score a specific partitioning opportunity."""
        opportunity = {
            'partition_key': partition_key,
            'partition_type': None,
            'strategy': None,
            'score': 0,
            'benefits': [],
            'considerations': []
        }
        
        # Determine partition type based on key characteristics
        key_lower = partition_key.lower()
        data_dist = characteristics.get('data_distribution', {}).get(partition_key, {})
        
        if any(pattern in key_lower for pattern in ['date', 'time', 'created', 'updated']):
            opportunity['partition_type'] = PartitionType.RANGE
            opportunity['strategy'] = 'time_series'
            opportunity['score'] += 50
            opportunity['benefits'].append("Efficient time-range queries")
            opportunity['benefits'].append("Natural archiving boundaries")
        
        elif key_lower in ['status', 'type', 'category', 'region', 'country']:
            distinct_values = data_dist.get('distinct_values', 0)
            if 2 <= distinct_values <= 20:  # Good for list partitioning
                opportunity['partition_type'] = PartitionType.LIST
                opportunity['strategy'] = 'categorical'
                opportunity['score'] += 30
                opportunity['benefits'].append("Fast categorical filtering")
            else:
                opportunity['considerations'].append(f"High cardinality ({distinct_values}) may not be ideal for list partitioning")
        
        elif key_lower.endswith('_id'):
            opportunity['partition_type'] = PartitionType.HASH
            opportunity['strategy'] = 'high_cardinality'
            opportunity['score'] += 20
            opportunity['benefits'].append("Distributed load")
            opportunity['considerations'].append("May not provide query filtering benefits")
        
        # Analyze query patterns to boost score
        if query_patterns:
            query_analysis = self._analyze_query_patterns_for_partitioning(table_name, query_patterns)
            
            # Boost score if this column is frequently filtered
            filter_frequency = query_analysis['frequently_filtered_columns'].get(partition_key, 0)
            opportunity['score'] += filter_frequency * 10
            
            if filter_frequency > 0:
                opportunity['benefits'].append(f"Column used in {filter_frequency} query patterns")
        
        # Table size factor
        table_size = characteristics.get('estimated_size_mb', 0)
        if table_size > 1000:  # 1GB+
            opportunity['score'] += 20
            opportunity['benefits'].append("Large table will benefit from partition pruning")
        
        return opportunity
    
    def generate_partitioning_recommendations(self,
                                            tables: List[str],
                                            query_patterns: Dict[str, List[str]] = None) -> List[PartitionRecommendation]:
        """Generate partitioning recommendations for multiple tables."""
        recommendations = []
        
        for table_name in tables:
            table_queries = query_patterns.get(table_name, []) if query_patterns else []
            analysis = self.analyze_table_for_partitioning(table_name, table_queries)
            
            recommended_strategy = analysis.get('recommended_strategy')
            if recommended_strategy and recommended_strategy['score'] >= 30:
                rec = self._create_partition_recommendation(table_name, recommended_strategy, analysis)
                recommendations.append(rec)
        
        self.partition_recommendations = recommendations
        return recommendations
    
    def _create_partition_recommendation(self,
                                       table_name: str,
                                       strategy: Dict[str, Any],
                                       analysis: Dict[str, Any]) -> PartitionRecommendation:
        """Create a partition recommendation from analysis."""
        partition_key = strategy['partition_key']
        partition_type = PartitionType(strategy['partition_type'])
        
        # Generate implementation details
        sql_script = self._generate_partition_sql(table_name, partition_key, partition_type)
        prerequisites = self._get_partitioning_prerequisites(table_name, partition_type)
        
        return PartitionRecommendation(
            table_name=table_name,
            recommendation_type='create_partition',
            partition_type=partition_type,
            partition_key=partition_key,
            strategy_description=f"Partition {table_name} by {partition_key} using {partition_type.value} strategy",
            estimated_benefit='; '.join(strategy['benefits']),
            implementation_complexity=self._assess_implementation_complexity(partition_type),
            sql_script=sql_script,
            prerequisites=prerequisites
        )
    
    def _generate_partition_sql(self,
                              table_name: str,
                              partition_key: str,
                              partition_type: PartitionType) -> str:
        """Generate SQL for creating partitions."""
        if partition_type == PartitionType.RANGE:
            # Example for monthly partitions
            sql = f"""
-- Create partitioned table for {table_name}
-- Note: Actual syntax varies by database system

-- 1. Create new partitioned table
CREATE TABLE {table_name}_partitioned (
    -- Copy all columns from original table
    -- Add partition key constraint
) PARTITION BY RANGE ({partition_key});

-- 2. Create initial partitions (example for monthly partitions)
CREATE TABLE {table_name}_202401 PARTITION OF {table_name}_partitioned
    FOR VALUES FROM ('2024-01-01') TO ('2024-02-01');

CREATE TABLE {table_name}_202402 PARTITION OF {table_name}_partitioned
    FOR VALUES FROM ('2024-02-01') TO ('2024-03-01');

-- 3. Migrate data
INSERT INTO {table_name}_partitioned SELECT * FROM {table_name};

-- 4. Rename tables
ALTER TABLE {table_name} RENAME TO {table_name}_old;
ALTER TABLE {table_name}_partitioned RENAME TO {table_name};
"""
        elif partition_type == PartitionType.LIST:
            sql = f"""
-- Create list partitioned table for {table_name}
CREATE TABLE {table_name}_partitioned (
    -- Copy all columns from original table
) PARTITION BY LIST ({partition_key});

-- Create partitions for each value
-- Example partitions (customize based on your data):
CREATE TABLE {table_name}_active PARTITION OF {table_name}_partitioned
    FOR VALUES IN ('active');

CREATE TABLE {table_name}_inactive PARTITION OF {table_name}_partitioned
    FOR VALUES IN ('inactive', 'cancelled');
"""
        elif partition_type == PartitionType.HASH:
            sql = f"""
-- Create hash partitioned table for {table_name}
CREATE TABLE {table_name}_partitioned (
    -- Copy all columns from original table
) PARTITION BY HASH ({partition_key});

-- Create hash partitions
CREATE TABLE {table_name}_p0 PARTITION OF {table_name}_partitioned
    FOR VALUES WITH (MODULUS 4, REMAINDER 0);

CREATE TABLE {table_name}_p1 PARTITION OF {table_name}_partitioned
    FOR VALUES WITH (MODULUS 4, REMAINDER 1);

CREATE TABLE {table_name}_p2 PARTITION OF {table_name}_partitioned
    FOR VALUES WITH (MODULUS 4, REMAINDER 2);

CREATE TABLE {table_name}_p3 PARTITION OF {table_name}_partitioned
    FOR VALUES WITH (MODULUS 4, REMAINDER 3);
"""
        else:
            sql = f"-- Partitioning strategy for {table_name} not yet implemented"
        
        return sql.strip()
    
    def _get_partitioning_prerequisites(self,
                                      table_name: str,
                                      partition_type: PartitionType) -> List[str]:
        """Get prerequisites for implementing partitioning."""
        prerequisites = [
            "Backup the existing table before partitioning",
            "Analyze application queries to ensure partition key is frequently used",
            "Consider impact on existing indexes and constraints"
        ]
        
        if partition_type == PartitionType.RANGE:
            prerequisites.extend([
                "Define appropriate date/time ranges for partitions",
                "Set up automated partition maintenance for future partitions",
                "Consider partition pruning in query execution plans"
            ])
        elif partition_type == PartitionType.LIST:
            prerequisites.extend([
                "Identify all possible values for the partition key",
                "Plan for handling NULL values and new categories",
                "Consider default partition for unexpected values"
            ])
        elif partition_type == PartitionType.HASH:
            prerequisites.extend([
                "Choose appropriate number of hash partitions",
                "Ensure even data distribution across partitions",
                "Plan for partition-wise joins if needed"
            ])
        
        return prerequisites
    
    def _assess_implementation_complexity(self, partition_type: PartitionType) -> str:
        """Assess implementation complexity."""
        complexity_map = {
            PartitionType.RANGE: 'medium',
            PartitionType.LIST: 'low',
            PartitionType.HASH: 'low',
            PartitionType.COMPOSITE: 'high'
        }
        return complexity_map.get(partition_type, 'medium')
    
    def generate_archiving_recommendations(self,
                                         table_name: str,
                                         data_age_analysis: Dict[str, Any] = None) -> List[ArchiveRecommendation]:
        """Generate data archiving recommendations."""
        recommendations = []
        
        # Analyze data age if not provided
        if data_age_analysis is None:
            data_age_analysis = self._analyze_data_age(table_name)
        
        # Apply archive policies
        table_type = self._classify_table_type(table_name)
        policy = self.archive_policies.get(table_type, self.archive_policies['analytics'])
        
        # Check if table has old data that should be archived
        old_data_analysis = self._identify_old_data(table_name, policy, data_age_analysis)
        
        if old_data_analysis['should_archive']:
            recommendation = ArchiveRecommendation(
                table_name=table_name,
                partition_name=None,
                archive_strategy=policy['strategy'],
                criteria=f"Data older than {policy['retention_days']} days",
                estimated_data_size=old_data_analysis['estimated_size_bytes'],
                estimated_cost_savings=old_data_analysis['estimated_savings'],
                retention_period=timedelta(days=policy['retention_days']),
                implementation_steps=self._generate_archive_steps(table_name, policy)
            )
            recommendations.append(recommendation)
        
        return recommendations
    
    def _analyze_data_age(self, table_name: str) -> Dict[str, Any]:
        """Analyze age distribution of data in table."""
        # Mock analysis - in real implementation, this would query the database
        return {
            'oldest_record': datetime.now() - timedelta(days=365),
            'newest_record': datetime.now() - timedelta(days=1),
            'data_age_distribution': {
                'last_30_days': 0.4,
                'last_90_days': 0.7,
                'last_year': 0.9,
                'older_than_year': 0.1
            }
        }
    
    def _classify_table_type(self, table_name: str) -> str:
        """Classify table type based on name patterns."""
        name_lower = table_name.lower()
        
        if any(pattern in name_lower for pattern in ['log', 'audit', 'event']):
            return 'logs'
        elif any(pattern in name_lower for pattern in ['transaction', 'payment', 'order']):
            return 'transactions'
        elif any(pattern in name_lower for pattern in ['temp', 'tmp', 'staging']):
            return 'temporary'
        else:
            return 'analytics'
    
    def _identify_old_data(self,
                          table_name: str,
                          policy: Dict[str, Any],
                          data_age_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Identify data that should be archived based on policy."""
        retention_cutoff = datetime.now() - timedelta(days=policy['retention_days'])
        oldest_record = data_age_analysis['oldest_record']
        
        should_archive = oldest_record < retention_cutoff
        
        # Estimate size of old data (simplified calculation)
        age_distribution = data_age_analysis['data_age_distribution']
        old_data_percentage = age_distribution.get('older_than_year', 0.1)
        estimated_size = int(100 * 1024 * 1024 * old_data_percentage)  # Mock 100MB table
        
        return {
            'should_archive': should_archive,
            'estimated_size_bytes': estimated_size,
            'estimated_savings': f"{old_data_percentage*100:.1f}% storage reduction",
            'cutoff_date': retention_cutoff
        }
    
    def _generate_archive_steps(self, table_name: str, policy: Dict[str, Any]) -> List[str]:
        """Generate implementation steps for archiving."""
        strategy = policy['strategy']
        
        if strategy == ArchiveStrategy.COLD_STORAGE:
            return [
                f"Create archive table {table_name}_archive with same structure",
                f"Move old records to archive table based on {policy['partition_key']}",
                "Move archive table to cold storage system",
                f"Create view to union current and archive data if needed",
                "Set up automated archiving process"
            ]
        elif strategy == ArchiveStrategy.DELETE:
            return [
                "Backup data before deletion (if required for compliance)",
                f"Delete records older than {policy['retention_days']} days",
                "Vacuum/reindex table to reclaim space",
                "Set up automated cleanup job"
            ]
        elif strategy == ArchiveStrategy.COMPRESS:
            return [
                f"Create compressed archive of old {table_name} data",
                "Verify compressed data integrity",
                "Delete original old records",
                "Document archive location and access procedures"
            ]
        elif strategy == ArchiveStrategy.SUMMARIZE:
            return [
                f"Create summary table {table_name}_summary",
                "Generate aggregated summaries of old data",
                "Verify summary accuracy",
                "Delete detailed old records",
                "Update reporting queries to use summaries"
            ]
        
        return []
    
    def export_partition_analysis(self, filepath: str, tables: List[str]) -> bool:
        """Export comprehensive partitioning analysis."""
        try:
            report = {
                'generated_at': datetime.now().isoformat(),
                'tables_analyzed': len(tables),
                'partition_recommendations': [],
                'archive_recommendations': [],
                'table_analyses': {}
            }
            
            # Analyze each table
            for table_name in tables:
                analysis = self.analyze_table_for_partitioning(table_name)
                report['table_analyses'][table_name] = analysis
                
                # Generate recommendations
                if analysis.get('recommended_strategy'):
                    rec = self._create_partition_recommendation(
                        table_name, 
                        analysis['recommended_strategy'],
                        analysis
                    )
                    report['partition_recommendations'].append({
                        'table': rec.table_name,
                        'type': rec.recommendation_type,
                        'partition_type': rec.partition_type.value,
                        'partition_key': rec.partition_key,
                        'description': rec.strategy_description,
                        'benefit': rec.estimated_benefit,
                        'complexity': rec.implementation_complexity,
                        'sql_script': rec.sql_script,
                        'prerequisites': rec.prerequisites
                    })
                
                # Generate archive recommendations
                archive_recs = self.generate_archiving_recommendations(table_name)
                for arch_rec in archive_recs:
                    report['archive_recommendations'].append({
                        'table': arch_rec.table_name,
                        'strategy': arch_rec.archive_strategy.value,
                        'criteria': arch_rec.criteria,
                        'estimated_size_mb': arch_rec.estimated_data_size / (1024 * 1024),
                        'cost_savings': arch_rec.estimated_cost_savings,
                        'retention_days': arch_rec.retention_period.days,
                        'implementation_steps': arch_rec.implementation_steps
                    })
            
            import json
            with open(filepath, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            return True
        except Exception as e:
            print(f"Error exporting partition analysis: {e}")
            return False