"""
Database performance profiler for query analysis and optimization.
"""

import time
import threading
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
from collections import defaultdict, deque
import hashlib
import re


@dataclass
class QueryExecution:
    """Database query execution record."""
    query_hash: str
    query: str
    start_time: datetime
    end_time: Optional[datetime] = None
    duration_ms: Optional[float] = None
    rows_affected: Optional[int] = None
    rows_returned: Optional[int] = None
    connection_id: Optional[str] = None
    database_name: Optional[str] = None
    table_name: Optional[str] = None
    query_type: Optional[str] = None  # SELECT, INSERT, UPDATE, DELETE
    execution_plan: Optional[Dict] = None
    exception: Optional[str] = None


@dataclass
class QueryStats:
    """Aggregated query performance statistics."""
    query_hash: str
    query_pattern: str
    total_executions: int
    total_duration_ms: float
    avg_duration_ms: float
    min_duration_ms: float
    max_duration_ms: float
    total_rows_affected: int
    total_rows_returned: int
    last_executed: datetime
    failure_count: int = 0


class DatabaseProfiler:
    """
    Database performance profiler for analyzing query execution patterns,
    identifying slow queries, and optimizing database operations.
    """
    
    def __init__(self, max_history: int = 5000):
        """
        Initialize database profiler.
        
        Args:
            max_history: Maximum number of query executions to keep
        """
        self.max_history = max_history
        self.query_executions: deque = deque(maxlen=max_history)
        self.query_stats: Dict[str, QueryStats] = {}
        self.active_queries: Dict[str, QueryExecution] = {}
        self._lock = threading.Lock()
        self.enabled = True
        
        # Query pattern cache for similar query detection
        self._pattern_cache: Dict[str, str] = {}
        
        # Slow query threshold (milliseconds)
        self.slow_query_threshold_ms = 1000
    
    def start_query_execution(self, 
                             query: str,
                             connection_id: Optional[str] = None,
                             database_name: Optional[str] = None) -> str:
        """
        Start tracking a query execution.
        
        Args:
            query: SQL query string
            connection_id: Database connection identifier
            database_name: Database name
            
        Returns:
            Execution tracking ID
        """
        if not self.enabled:
            return ""
        
        # Generate query hash and pattern
        query_hash = self._generate_query_hash(query)
        query_pattern = self._generate_query_pattern(query)
        query_type = self._extract_query_type(query)
        table_name = self._extract_table_name(query)
        
        execution = QueryExecution(
            query_hash=query_hash,
            query=query,
            start_time=datetime.now(),
            connection_id=connection_id,
            database_name=database_name,
            table_name=table_name,
            query_type=query_type
        )
        
        execution_id = f"{threading.current_thread().ident}_{time.time()}"
        
        with self._lock:
            self.active_queries[execution_id] = execution
        
        return execution_id
    
    def end_query_execution(self,
                           execution_id: str,
                           rows_affected: Optional[int] = None,
                           rows_returned: Optional[int] = None,
                           execution_plan: Optional[Dict] = None,
                           exception: Optional[str] = None):
        """
        End tracking a query execution.
        
        Args:
            execution_id: Execution tracking ID from start_query_execution
            rows_affected: Number of rows affected (INSERT/UPDATE/DELETE)
            rows_returned: Number of rows returned (SELECT)
            execution_plan: Query execution plan if available
            exception: Exception message if query failed
        """
        if not self.enabled or not execution_id:
            return
        
        with self._lock:
            execution = self.active_queries.pop(execution_id, None)
            
            if execution is None:
                return
            
            # Update execution record
            execution.end_time = datetime.now()
            execution.duration_ms = (
                (execution.end_time - execution.start_time).total_seconds() * 1000
            )
            execution.rows_affected = rows_affected
            execution.rows_returned = rows_returned
            execution.execution_plan = execution_plan
            execution.exception = exception
            
            # Add to history
            self.query_executions.append(execution)
            
            # Update statistics
            self._update_query_stats(execution)
    
    def profile_query(self, query_func, *args, **kwargs):
        """
        Decorator/context manager to profile a query execution function.
        
        Args:
            query_func: Function that executes a query
            *args, **kwargs: Arguments for the query function
            
        Returns:
            Result of query execution
        """
        # Extract query from args or kwargs if possible
        query = ""
        if args:
            query = str(args[0]) if args[0] else ""
        elif 'query' in kwargs:
            query = str(kwargs['query'])
        
        execution_id = self.start_query_execution(query)
        
        try:
            result = query_func(*args, **kwargs)
            
            # Try to extract row information from result
            rows_returned = None
            rows_affected = None
            
            if hasattr(result, 'rowcount'):
                rows_affected = result.rowcount
            if hasattr(result, '__len__'):
                try:
                    rows_returned = len(result)
                except:
                    pass
            
            self.end_query_execution(
                execution_id,
                rows_affected=rows_affected,
                rows_returned=rows_returned
            )
            
            return result
            
        except Exception as e:
            self.end_query_execution(
                execution_id,
                exception=str(e)
            )
            raise
    
    def _generate_query_hash(self, query: str) -> str:
        """Generate hash for exact query matching."""
        normalized = re.sub(r'\s+', ' ', query.strip().lower())
        return hashlib.md5(normalized.encode()).hexdigest()
    
    def _generate_query_pattern(self, query: str) -> str:
        """Generate pattern for similar query detection."""
        if query in self._pattern_cache:
            return self._pattern_cache[query]
        
        # Normalize query for pattern matching
        pattern = query.upper().strip()
        
        # Replace literals with placeholders
        pattern = re.sub(r"'[^']*'", "'?'", pattern)  # String literals
        pattern = re.sub(r'"[^"]*"', '"?"', pattern)  # Quoted identifiers
        pattern = re.sub(r'\b\d+\b', '?', pattern)    # Numeric literals
        pattern = re.sub(r'\s+', ' ', pattern)        # Normalize whitespace
        
        # Cache the pattern
        self._pattern_cache[query] = pattern
        return pattern
    
    def _extract_query_type(self, query: str) -> Optional[str]:
        """Extract query type (SELECT, INSERT, UPDATE, DELETE, etc.)"""
        query_upper = query.strip().upper()
        for query_type in ['SELECT', 'INSERT', 'UPDATE', 'DELETE', 
                          'CREATE', 'DROP', 'ALTER', 'TRUNCATE']:
            if query_upper.startswith(query_type):
                return query_type
        return None
    
    def _extract_table_name(self, query: str) -> Optional[str]:
        """Extract primary table name from query."""
        query_upper = query.upper().strip()
        
        # Simple pattern matching for common cases
        patterns = [
            r'FROM\s+(\w+)',
            r'INSERT\s+INTO\s+(\w+)',
            r'UPDATE\s+(\w+)',
            r'DELETE\s+FROM\s+(\w+)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, query_upper)
            if match:
                return match.group(1).lower()
        
        return None
    
    def _update_query_stats(self, execution: QueryExecution):
        """Update aggregated query statistics."""
        query_pattern = self._generate_query_pattern(execution.query)
        
        if execution.query_hash in self.query_stats:
            stats = self.query_stats[execution.query_hash]
            stats.total_executions += 1
            stats.total_duration_ms += execution.duration_ms or 0
            stats.avg_duration_ms = stats.total_duration_ms / stats.total_executions
            stats.min_duration_ms = min(stats.min_duration_ms, execution.duration_ms or 0)
            stats.max_duration_ms = max(stats.max_duration_ms, execution.duration_ms or 0)
            stats.total_rows_affected += execution.rows_affected or 0
            stats.total_rows_returned += execution.rows_returned or 0
            stats.last_executed = execution.end_time or execution.start_time
            
            if execution.exception:
                stats.failure_count += 1
        else:
            self.query_stats[execution.query_hash] = QueryStats(
                query_hash=execution.query_hash,
                query_pattern=query_pattern,
                total_executions=1,
                total_duration_ms=execution.duration_ms or 0,
                avg_duration_ms=execution.duration_ms or 0,
                min_duration_ms=execution.duration_ms or 0,
                max_duration_ms=execution.duration_ms or 0,
                total_rows_affected=execution.rows_affected or 0,
                total_rows_returned=execution.rows_returned or 0,
                last_executed=execution.end_time or execution.start_time,
                failure_count=1 if execution.exception else 0
            )
    
    def get_slow_queries(self, 
                        limit: int = 10,
                        threshold_ms: Optional[float] = None) -> List[QueryStats]:
        """Get slowest queries by average execution time."""
        threshold = threshold_ms or self.slow_query_threshold_ms
        
        with self._lock:
            slow_queries = [
                stats for stats in self.query_stats.values()
                if stats.avg_duration_ms >= threshold
            ]
            
            return sorted(slow_queries, 
                         key=lambda x: x.avg_duration_ms, 
                         reverse=True)[:limit]
    
    def get_most_frequent_queries(self, limit: int = 10) -> List[QueryStats]:
        """Get most frequently executed queries."""
        with self._lock:
            return sorted(self.query_stats.values(),
                         key=lambda x: x.total_executions,
                         reverse=True)[:limit]
    
    def get_query_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive query performance summary."""
        with self._lock:
            if not self.query_stats:
                return {
                    'total_queries': 0,
                    'total_executions': 0,
                    'avg_execution_time_ms': 0,
                    'slow_query_count': 0,
                    'failed_query_count': 0
                }
            
            total_executions = sum(stats.total_executions for stats in self.query_stats.values())
            total_time = sum(stats.total_duration_ms for stats in self.query_stats.values())
            slow_queries = len([s for s in self.query_stats.values() 
                              if s.avg_duration_ms >= self.slow_query_threshold_ms])
            failed_queries = sum(stats.failure_count for stats in self.query_stats.values())
            
            return {
                'total_queries': len(self.query_stats),
                'total_executions': total_executions,
                'avg_execution_time_ms': total_time / total_executions if total_executions > 0 else 0,
                'slow_query_count': slow_queries,
                'failed_query_count': failed_queries,
                'active_queries': len(self.active_queries)
            }
    
    def analyze_query_patterns(self) -> Dict[str, Any]:
        """Analyze query execution patterns and identify optimization opportunities."""
        pattern_stats = defaultdict(lambda: {
            'count': 0, 'avg_duration': 0, 'total_duration': 0
        })
        
        table_stats = defaultdict(lambda: {
            'query_count': 0, 'avg_duration': 0, 'total_duration': 0
        })
        
        with self._lock:
            for execution in self.query_executions:
                if execution.duration_ms is None:
                    continue
                
                # Pattern analysis
                pattern = self._generate_query_pattern(execution.query)
                pattern_stats[pattern]['count'] += 1
                pattern_stats[pattern]['total_duration'] += execution.duration_ms
                pattern_stats[pattern]['avg_duration'] = (
                    pattern_stats[pattern]['total_duration'] / 
                    pattern_stats[pattern]['count']
                )
                
                # Table analysis
                if execution.table_name:
                    table_stats[execution.table_name]['query_count'] += 1
                    table_stats[execution.table_name]['total_duration'] += execution.duration_ms
                    table_stats[execution.table_name]['avg_duration'] = (
                        table_stats[execution.table_name]['total_duration'] /
                        table_stats[execution.table_name]['query_count']
                    )
        
        return {
            'pattern_analysis': dict(pattern_stats),
            'table_analysis': dict(table_stats),
            'optimization_suggestions': self._generate_optimization_suggestions(
                pattern_stats, table_stats
            )
        }
    
    def _generate_optimization_suggestions(self, 
                                         pattern_stats: Dict, 
                                         table_stats: Dict) -> List[str]:
        """Generate optimization suggestions based on query analysis."""
        suggestions = []
        
        # Check for slow patterns
        slow_patterns = [
            pattern for pattern, stats in pattern_stats.items()
            if stats['avg_duration'] > self.slow_query_threshold_ms and stats['count'] > 5
        ]
        
        if slow_patterns:
            suggestions.append(
                f"Consider optimizing {len(slow_patterns)} frequently used slow query patterns"
            )
        
        # Check for heavy table usage
        heavy_tables = [
            table for table, stats in table_stats.items()
            if stats['query_count'] > 100 and stats['avg_duration'] > 500
        ]
        
        if heavy_tables:
            suggestions.extend([
                f"Consider indexing optimization for table: {table}"
                for table in heavy_tables
            ])
        
        # Check for SELECT * patterns
        select_all_patterns = [
            pattern for pattern in pattern_stats.keys()
            if 'SELECT *' in pattern and pattern_stats[pattern]['count'] > 10
        ]
        
        if select_all_patterns:
            suggestions.append("Consider replacing SELECT * with specific columns")
        
        return suggestions
    
    def get_query_executions_by_table(self, 
                                     table_name: str,
                                     hours: int = 24) -> List[QueryExecution]:
        """Get query executions for a specific table."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        with self._lock:
            return [
                execution for execution in self.query_executions
                if (execution.table_name == table_name and 
                    execution.start_time >= cutoff_time)
            ]
    
    def export_query_analysis(self, filepath: str) -> bool:
        """Export comprehensive query analysis to file."""
        try:
            analysis = {
                'generated_at': datetime.now().isoformat(),
                'performance_summary': self.get_query_performance_summary(),
                'slow_queries': [
                    {
                        'pattern': stats.query_pattern,
                        'avg_duration_ms': stats.avg_duration_ms,
                        'total_executions': stats.total_executions,
                        'failure_count': stats.failure_count
                    }
                    for stats in self.get_slow_queries(20)
                ],
                'frequent_queries': [
                    {
                        'pattern': stats.query_pattern,
                        'total_executions': stats.total_executions,
                        'avg_duration_ms': stats.avg_duration_ms
                    }
                    for stats in self.get_most_frequent_queries(20)
                ],
                'pattern_analysis': self.analyze_query_patterns()
            }
            
            import json
            with open(filepath, 'w') as f:
                json.dump(analysis, f, indent=2, default=str)
            
            return True
        except Exception as e:
            print(f"Error exporting query analysis: {e}")
            return False
    
    def reset_statistics(self):
        """Reset all query profiling statistics."""
        with self._lock:
            self.query_executions.clear()
            self.query_stats.clear()
            self.active_queries.clear()
            self._pattern_cache.clear()
    
    def set_slow_query_threshold(self, threshold_ms: float):
        """Set threshold for identifying slow queries."""
        self.slow_query_threshold_ms = threshold_ms
    
    def enable(self):
        """Enable database profiling."""
        self.enabled = True
    
    def disable(self):
        """Disable database profiling."""
        self.enabled = False