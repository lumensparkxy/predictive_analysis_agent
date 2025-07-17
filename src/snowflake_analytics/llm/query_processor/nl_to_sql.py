"""
Natural Language to SQL Converter

Converts natural language queries about Snowflake analytics into SQL queries
using LLM capabilities and predefined templates.
"""

from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import re
import logging

from ...utils.logger import get_logger
from .intent_classifier import QueryIntent, IntentResult

logger = get_logger(__name__)


@dataclass
class SQLQuery:
    """Generated SQL query with metadata."""
    sql: str
    parameters: Dict[str, Any]
    estimated_cost: float
    estimated_duration: float
    explanation: str
    warnings: List[str]


class NaturalLanguageToSQL:
    """Converts natural language queries to SQL for Snowflake analytics."""
    
    def __init__(self, llm_client=None, config: Dict[str, Any] = None):
        """Initialize NL to SQL converter.
        
        Args:
            llm_client: LLM client for query generation
            config: Configuration dictionary
        """
        self.llm_client = llm_client
        self.config = config or {}
        
        # Database schema information
        self.schema_info = self._load_schema_info()
        
        # SQL templates for common query patterns
        self.sql_templates = self._initialize_sql_templates()
        
        # Safety rules
        self.forbidden_operations = [
            'DROP', 'DELETE', 'UPDATE', 'INSERT', 'ALTER', 'CREATE',
            'TRUNCATE', 'REPLACE'
        ]
        
        logger.info("Natural Language to SQL converter initialized")
    
    def _load_schema_info(self) -> Dict[str, Any]:
        """Load Snowflake schema information."""
        # This would typically load from the actual Snowflake schema
        # For now, return a mock schema based on the analytics tables
        return {
            "tables": {
                "COST_METRICS": {
                    "columns": [
                        "DATE", "WAREHOUSE", "CREDITS_CONSUMED", "COST_USD",
                        "STORAGE_COST", "COMPUTE_COST", "USER_NAME"
                    ],
                    "description": "Daily cost metrics by warehouse and user"
                },
                "USAGE_METRICS": {
                    "columns": [
                        "DATE", "WAREHOUSE", "QUERY_COUNT", "EXECUTION_TIME",
                        "BYTES_SCANNED", "USER_NAME", "QUERY_TYPE"
                    ],
                    "description": "Usage metrics including query performance"
                },
                "WAREHOUSE_METRICS": {
                    "columns": [
                        "DATE", "WAREHOUSE", "SIZE", "AUTO_SUSPEND_MINUTES",
                        "UTILIZATION_PERCENT", "QUEUE_TIME_SECONDS"
                    ],
                    "description": "Warehouse configuration and utilization metrics"
                },
                "USER_ACTIVITY": {
                    "columns": [
                        "DATE", "USER_NAME", "SESSION_COUNT", "QUERY_COUNT",
                        "TOTAL_EXECUTION_TIME", "ROLE"
                    ],
                    "description": "User activity and behavior metrics"
                },
                "QUERY_HISTORY": {
                    "columns": [
                        "QUERY_ID", "USER_NAME", "WAREHOUSE", "START_TIME",
                        "END_TIME", "EXECUTION_TIME_MS", "BYTES_SCANNED",
                        "QUERY_TEXT", "ERROR_MESSAGE"
                    ],
                    "description": "Detailed query execution history"
                }
            },
            "views": {
                "DAILY_COST_SUMMARY": {
                    "base_tables": ["COST_METRICS"],
                    "description": "Daily aggregated cost summary"
                },
                "WAREHOUSE_UTILIZATION": {
                    "base_tables": ["WAREHOUSE_METRICS", "USAGE_METRICS"],
                    "description": "Warehouse utilization analysis"
                }
            }
        }
    
    def _initialize_sql_templates(self) -> Dict[QueryIntent, str]:
        """Initialize SQL templates for different query intents."""
        return {
            QueryIntent.COST_OVERVIEW: """
                SELECT 
                    DATE_TRUNC('{time_period}', DATE) AS period,
                    SUM(COST_USD) AS total_cost,
                    SUM(STORAGE_COST) AS storage_cost,
                    SUM(COMPUTE_COST) AS compute_cost
                FROM COST_METRICS 
                WHERE DATE >= '{start_date}' AND DATE <= '{end_date}'
                {filters}
                GROUP BY DATE_TRUNC('{time_period}', DATE)
                ORDER BY period DESC
                {limit_clause}
            """,
            
            QueryIntent.COST_BREAKDOWN: """
                SELECT 
                    {group_by_column} AS category,
                    SUM(COST_USD) AS total_cost,
                    COUNT(*) AS record_count,
                    AVG(COST_USD) AS avg_cost
                FROM COST_METRICS 
                WHERE DATE >= '{start_date}' AND DATE <= '{end_date}'
                {filters}
                GROUP BY {group_by_column}
                ORDER BY total_cost DESC
                {limit_clause}
            """,
            
            QueryIntent.USAGE_OVERVIEW: """
                SELECT 
                    DATE_TRUNC('{time_period}', DATE) AS period,
                    COUNT(*) AS query_count,
                    SUM(EXECUTION_TIME) AS total_execution_time,
                    AVG(EXECUTION_TIME) AS avg_execution_time,
                    SUM(BYTES_SCANNED) AS total_bytes_scanned
                FROM USAGE_METRICS 
                WHERE DATE >= '{start_date}' AND DATE <= '{end_date}'
                {filters}
                GROUP BY DATE_TRUNC('{time_period}', DATE)
                ORDER BY period DESC
                {limit_clause}
            """,
            
            QueryIntent.USER_ACTIVITY: """
                SELECT 
                    USER_NAME,
                    SUM(QUERY_COUNT) AS total_queries,
                    AVG(TOTAL_EXECUTION_TIME) AS avg_execution_time,
                    SUM(SESSION_COUNT) AS total_sessions
                FROM USER_ACTIVITY 
                WHERE DATE >= '{start_date}' AND DATE <= '{end_date}'
                {filters}
                GROUP BY USER_NAME
                ORDER BY total_queries DESC
                {limit_clause}
            """,
            
            QueryIntent.WAREHOUSE_UTILIZATION: """
                SELECT 
                    WAREHOUSE,
                    AVG(UTILIZATION_PERCENT) AS avg_utilization,
                    AVG(QUEUE_TIME_SECONDS) AS avg_queue_time,
                    SIZE AS warehouse_size
                FROM WAREHOUSE_METRICS 
                WHERE DATE >= '{start_date}' AND DATE <= '{end_date}'
                {filters}
                GROUP BY WAREHOUSE, SIZE
                ORDER BY avg_utilization DESC
                {limit_clause}
            """,
            
            QueryIntent.QUERY_PERFORMANCE: """
                SELECT 
                    QUERY_ID,
                    USER_NAME,
                    WAREHOUSE,
                    EXECUTION_TIME_MS,
                    BYTES_SCANNED,
                    LEFT(QUERY_TEXT, 100) AS query_preview
                FROM QUERY_HISTORY 
                WHERE START_TIME >= '{start_date}' AND START_TIME <= '{end_date}'
                {filters}
                ORDER BY EXECUTION_TIME_MS DESC
                {limit_clause}
            """
        }
    
    async def convert_to_sql(
        self, 
        natural_query: str, 
        intent_result: IntentResult,
        context: Dict[str, Any] = None
    ) -> SQLQuery:
        """Convert natural language query to SQL.
        
        Args:
            natural_query: Natural language query
            intent_result: Result from intent classification
            context: Additional context information
            
        Returns:
            SQLQuery object with generated SQL and metadata
        """
        context = context or {}
        
        try:
            # Check if we have a template for this intent
            if intent_result.intent in self.sql_templates:
                sql_query = await self._generate_from_template(
                    natural_query, intent_result, context
                )
            else:
                # Use LLM for complex queries
                sql_query = await self._generate_with_llm(
                    natural_query, intent_result, context
                )
            
            # Validate the generated SQL
            validation_result = self._validate_sql(sql_query.sql)
            if not validation_result['is_valid']:
                sql_query.warnings.extend(validation_result['warnings'])
            
            return sql_query
            
        except Exception as e:
            logger.error(f"Error converting query to SQL: {e}")
            raise
    
    async def _generate_from_template(
        self, 
        natural_query: str, 
        intent_result: IntentResult,
        context: Dict[str, Any]
    ) -> SQLQuery:
        """Generate SQL using predefined templates."""
        
        template = self.sql_templates[intent_result.intent]
        
        # Extract parameters from intent result
        parameters = self._extract_sql_parameters(intent_result, context)
        
        # Fill template
        try:
            sql = template.format(**parameters)
            
            # Clean up SQL formatting
            sql = re.sub(r'\s+', ' ', sql.strip())
            sql = re.sub(r'\s*,\s*', ', ', sql)
            
            explanation = self._generate_sql_explanation(intent_result.intent, parameters)
            
            return SQLQuery(
                sql=sql,
                parameters=parameters,
                estimated_cost=self._estimate_query_cost(sql),
                estimated_duration=self._estimate_query_duration(sql),
                explanation=explanation,
                warnings=[]
            )
            
        except KeyError as e:
            logger.error(f"Missing parameter for SQL template: {e}")
            raise ValueError(f"Could not generate SQL: missing parameter {e}")
    
    async def _generate_with_llm(
        self, 
        natural_query: str, 
        intent_result: IntentResult,
        context: Dict[str, Any]
    ) -> SQLQuery:
        """Generate SQL using LLM for complex queries."""
        
        if not self.llm_client:
            raise ValueError("LLM client not available for complex query generation")
        
        # Prepare system prompt with schema information
        system_prompt = self._create_sql_generation_prompt()
        
        # Prepare user prompt
        user_prompt = f"""
        Convert this natural language query to SQL:
        Query: {natural_query}
        
        Intent: {intent_result.intent.value}
        Entities: {intent_result.entities}
        Time Range: {intent_result.time_range}
        Filters: {intent_result.filters}
        
        Please generate a safe, read-only SQL query that returns the requested information.
        """
        
        # Generate SQL using LLM
        response = await self.llm_client.generate_completion(
            prompt=user_prompt,
            system_prompt=system_prompt,
            temperature=0.1,  # Low temperature for precise SQL
            max_tokens=1000
        )
        
        # Extract SQL from response
        sql = self._extract_sql_from_response(response.content)
        
        return SQLQuery(
            sql=sql,
            parameters={},
            estimated_cost=self._estimate_query_cost(sql),
            estimated_duration=self._estimate_query_duration(sql),
            explanation=response.content,
            warnings=[]
        )
    
    def _extract_sql_parameters(self, intent_result: IntentResult, context: Dict[str, Any]) -> Dict[str, Any]:
        """Extract parameters for SQL template filling."""
        parameters = {}
        
        # Time range parameters
        time_range = intent_result.time_range or 'last_month'
        start_date, end_date, time_period = self._parse_time_range(time_range)
        
        parameters.update({
            'start_date': start_date,
            'end_date': end_date,
            'time_period': time_period
        })
        
        # Filter parameters
        filters = []
        if intent_result.filters.get('warehouse'):
            warehouses = "', '".join(intent_result.filters['warehouse'])
            filters.append(f"AND WAREHOUSE IN ('{warehouses}')")
        
        if intent_result.filters.get('user'):
            users = "', '".join(intent_result.filters['user'])
            filters.append(f"AND USER_NAME IN ('{users}')")
        
        parameters['filters'] = ' '.join(filters)
        
        # Group by parameter for breakdown queries
        if intent_result.intent == QueryIntent.COST_BREAKDOWN:
            if 'warehouse' in intent_result.entities:
                parameters['group_by_column'] = 'WAREHOUSE'
            elif 'user' in intent_result.entities:
                parameters['group_by_column'] = 'USER_NAME'
            else:
                parameters['group_by_column'] = 'WAREHOUSE'  # Default
        
        # Limit clause
        limit = intent_result.filters.get('limit', 20)
        parameters['limit_clause'] = f"LIMIT {limit}"
        
        # Order by
        if intent_result.filters.get('order_by'):
            order_by = intent_result.filters['order_by']
            if order_by == 'cost_desc':
                parameters['order_by_clause'] = "ORDER BY total_cost DESC"
            elif order_by == 'cost_asc':
                parameters['order_by_clause'] = "ORDER BY total_cost ASC"
        
        return parameters
    
    def _parse_time_range(self, time_range: str) -> Tuple[str, str, str]:
        """Parse time range into start date, end date, and aggregation period."""
        from datetime import datetime, timedelta
        
        now = datetime.now()
        
        if time_range == 'today':
            start_date = now.strftime('%Y-%m-%d')
            end_date = start_date
            time_period = 'DAY'
        elif time_range == 'yesterday':
            date = (now - timedelta(days=1)).strftime('%Y-%m-%d')
            start_date = end_date = date
            time_period = 'DAY'
        elif time_range == 'this_week':
            start_date = (now - timedelta(days=now.weekday())).strftime('%Y-%m-%d')
            end_date = now.strftime('%Y-%m-%d')
            time_period = 'DAY'
        elif time_range == 'last_week':
            end_date = (now - timedelta(days=now.weekday() + 1)).strftime('%Y-%m-%d')
            start_date = (now - timedelta(days=now.weekday() + 7)).strftime('%Y-%m-%d')
            time_period = 'DAY'
        elif time_range == 'this_month':
            start_date = now.replace(day=1).strftime('%Y-%m-%d')
            end_date = now.strftime('%Y-%m-%d')
            time_period = 'DAY'
        elif time_range == 'last_month':
            last_month = now.replace(day=1) - timedelta(days=1)
            start_date = last_month.replace(day=1).strftime('%Y-%m-%d')
            end_date = last_month.strftime('%Y-%m-%d')
            time_period = 'DAY'
        else:
            # Default to last 30 days
            start_date = (now - timedelta(days=30)).strftime('%Y-%m-%d')
            end_date = now.strftime('%Y-%m-%d')
            time_period = 'DAY'
        
        return start_date, end_date, time_period
    
    def _create_sql_generation_prompt(self) -> str:
        """Create system prompt for SQL generation."""
        schema_description = self._format_schema_for_prompt()
        
        return f"""
        You are a SQL expert specializing in Snowflake analytics queries. 
        
        Generate safe, read-only SQL queries based on natural language requests.
        
        IMPORTANT SAFETY RULES:
        - Only generate SELECT statements
        - Never use DROP, DELETE, UPDATE, INSERT, ALTER, CREATE, or TRUNCATE
        - Always include appropriate WHERE clauses for date filtering
        - Use proper aggregation functions
        - Include meaningful column aliases
        
        Available Schema:
        {schema_description}
        
        Best Practices:
        - Use appropriate date functions for time-based queries
        - Include proper GROUP BY clauses for aggregations
        - Use meaningful table aliases
        - Add comments for complex logic
        - Optimize for performance with appropriate filters
        
        Return only the SQL query, properly formatted.
        """
    
    def _format_schema_for_prompt(self) -> str:
        """Format schema information for LLM prompt."""
        schema_text = "Tables:\n"
        
        for table_name, table_info in self.schema_info["tables"].items():
            schema_text += f"\n{table_name}:\n"
            schema_text += f"  Description: {table_info['description']}\n"
            schema_text += f"  Columns: {', '.join(table_info['columns'])}\n"
        
        return schema_text
    
    def _extract_sql_from_response(self, response: str) -> str:
        """Extract SQL query from LLM response."""
        # Look for SQL code blocks
        sql_match = re.search(r'```sql\n(.*?)\n```', response, re.DOTALL | re.IGNORECASE)
        if sql_match:
            return sql_match.group(1).strip()
        
        # Look for SELECT statements
        sql_match = re.search(r'(SELECT.*?;?)', response, re.DOTALL | re.IGNORECASE)
        if sql_match:
            return sql_match.group(1).strip()
        
        # Return the whole response if no specific SQL found
        return response.strip()
    
    def _validate_sql(self, sql: str) -> Dict[str, Any]:
        """Validate generated SQL for safety and correctness."""
        warnings = []
        is_valid = True
        
        # Check for forbidden operations
        sql_upper = sql.upper()
        for forbidden_op in self.forbidden_operations:
            if forbidden_op in sql_upper:
                warnings.append(f"Forbidden operation detected: {forbidden_op}")
                is_valid = False
        
        # Check if it's a SELECT statement
        if not sql_upper.strip().startswith('SELECT'):
            warnings.append("Query should start with SELECT")
            is_valid = False
        
        # Check for basic SQL syntax
        if sql.count('(') != sql.count(')'):
            warnings.append("Unbalanced parentheses")
            is_valid = False
        
        return {
            'is_valid': is_valid,
            'warnings': warnings
        }
    
    def _estimate_query_cost(self, sql: str) -> float:
        """Estimate query execution cost."""
        # Simple cost estimation based on query complexity
        base_cost = 0.01  # Base cost in credits
        
        # Add cost based on table scans
        table_count = len(re.findall(r'FROM\s+(\w+)', sql, re.IGNORECASE))
        base_cost += table_count * 0.005
        
        # Add cost for joins
        join_count = len(re.findall(r'\bJOIN\b', sql, re.IGNORECASE))
        base_cost += join_count * 0.01
        
        # Add cost for aggregations
        agg_count = len(re.findall(r'\b(SUM|COUNT|AVG|MAX|MIN)\b', sql, re.IGNORECASE))
        base_cost += agg_count * 0.002
        
        return round(base_cost, 4)
    
    def _estimate_query_duration(self, sql: str) -> float:
        """Estimate query execution duration in seconds."""
        # Simple duration estimation
        base_duration = 1.0  # Base duration in seconds
        
        # Add duration based on complexity
        if 'GROUP BY' in sql.upper():
            base_duration += 2.0
        
        if 'ORDER BY' in sql.upper():
            base_duration += 1.0
        
        if 'JOIN' in sql.upper():
            base_duration += 3.0
        
        return base_duration
    
    def _generate_sql_explanation(self, intent: QueryIntent, parameters: Dict[str, Any]) -> str:
        """Generate explanation for the SQL query."""
        explanations = {
            QueryIntent.COST_OVERVIEW: "This query provides an overview of costs aggregated by time period.",
            QueryIntent.COST_BREAKDOWN: "This query breaks down costs by the specified dimension.",
            QueryIntent.USAGE_OVERVIEW: "This query summarizes usage metrics over the specified time period.",
            QueryIntent.USER_ACTIVITY: "This query analyzes user activity and behavior patterns.",
            QueryIntent.WAREHOUSE_UTILIZATION: "This query shows warehouse utilization metrics.",
            QueryIntent.QUERY_PERFORMANCE: "This query identifies query performance metrics and slow queries."
        }
        
        base_explanation = explanations.get(intent, "This query retrieves the requested information.")
        
        # Add time range info
        if parameters.get('start_date') and parameters.get('end_date'):
            base_explanation += f" Data is filtered from {parameters['start_date']} to {parameters['end_date']}."
        
        # Add filter info
        if parameters.get('filters'):
            base_explanation += f" Additional filters applied: {parameters['filters']}"
        
        return base_explanation