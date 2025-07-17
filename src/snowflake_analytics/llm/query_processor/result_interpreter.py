"""
Result Interpreter for Query Results

Interprets and explains SQL query results in natural language,
providing insights and context for Snowflake analytics data.
"""

import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging

from ...utils.logger import get_logger
from .intent_classifier import QueryIntent

logger = get_logger(__name__)


@dataclass
class InterpretationResult:
    """Result of query result interpretation."""
    summary: str
    insights: List[str]
    recommendations: List[str]
    key_metrics: Dict[str, Any]
    visualizations: List[Dict[str, Any]]
    explanation: str


class ResultInterpreter:
    """Interprets SQL query results and provides natural language explanations."""
    
    def __init__(self, llm_client=None, config: Dict[str, Any] = None):
        """Initialize result interpreter.
        
        Args:
            llm_client: LLM client for generating explanations
            config: Configuration dictionary
        """
        self.llm_client = llm_client
        self.config = config or {}
        
        # Interpretation templates for different query types
        self.interpretation_templates = self._initialize_templates()
        
        # Thresholds for insights
        self.insight_thresholds = {
            'high_cost_threshold': 1000.0,  # USD
            'high_usage_threshold': 10000,  # Query count
            'performance_threshold': 30000,  # Execution time in ms
            'utilization_threshold': 80.0,  # Percentage
            'growth_threshold': 20.0  # Percentage change
        }
        
        logger.info("Result interpreter initialized")
    
    def _initialize_templates(self) -> Dict[QueryIntent, Dict[str, str]]:
        """Initialize interpretation templates for different query intents."""
        return {
            QueryIntent.COST_OVERVIEW: {
                'summary': "Cost analysis shows total spending of ${total_cost:.2f} over {time_period}",
                'single_record': "On {date}, total cost was ${cost:.2f}",
                'trend': "Cost trend shows {trend_direction} of {trend_percent:.1f}%"
            },
            QueryIntent.COST_BREAKDOWN: {
                'summary': "Cost breakdown shows {top_category} as the highest contributor with ${top_cost:.2f} ({top_percent:.1f}%)",
                'distribution': "Cost distribution across {category_count} categories",
                'comparison': "Top category spends {multiple:.1f}x more than average"
            },
            QueryIntent.USAGE_OVERVIEW: {
                'summary': "Usage analysis shows {total_queries} queries with {avg_execution_time:.1f}ms average execution time",
                'volume': "Query volume of {query_count} over {time_period}",
                'performance': "Average query performance: {avg_time:.1f}ms"
            },
            QueryIntent.USER_ACTIVITY: {
                'summary': "User activity shows {most_active_user} as most active with {user_queries} queries",
                'distribution': "Activity across {user_count} users",
                'engagement': "{active_users} users with significant activity"
            },
            QueryIntent.WAREHOUSE_UTILIZATION: {
                'summary': "Warehouse utilization shows {avg_utilization:.1f}% average across {warehouse_count} warehouses",
                'performance': "Average queue time: {avg_queue_time:.1f} seconds",
                'efficiency': "Most efficient warehouse: {best_warehouse}"
            },
            QueryIntent.QUERY_PERFORMANCE: {
                'summary': "Performance analysis shows {slow_queries} slow queries (>{threshold}ms)",
                'distribution': "Execution time distribution from {min_time}ms to {max_time}ms",
                'bottlenecks': "Performance bottlenecks identified in {bottleneck_areas}"
            }
        }
    
    async def interpret_results(
        self, 
        query_results: Any, 
        intent: QueryIntent,
        original_query: str,
        context: Dict[str, Any] = None
    ) -> InterpretationResult:
        """Interpret query results and provide natural language explanation.
        
        Args:
            query_results: Query results (DataFrame, dict, or list)
            intent: Original query intent
            original_query: Original natural language query
            context: Additional context information
            
        Returns:
            InterpretationResult with interpretation and insights
        """
        context = context or {}
        
        # Convert results to DataFrame for consistent processing
        df = self._normalize_results(query_results)
        
        if df is None or df.empty:
            return InterpretationResult(
                summary="No data found for the specified query",
                insights=["No records match the query criteria"],
                recommendations=["Try adjusting the time range or filters"],
                key_metrics={},
                visualizations=[],
                explanation="The query executed successfully but returned no data."
            )
        
        # Generate interpretation based on intent
        if intent in self.interpretation_templates:
            result = await self._interpret_with_template(df, intent, context)
        else:
            result = await self._interpret_with_llm(df, intent, original_query, context)
        
        # Add general insights
        general_insights = self._extract_general_insights(df, intent)
        result.insights.extend(general_insights)
        
        # Add recommendations
        recommendations = self._generate_recommendations(df, intent, result.key_metrics)
        result.recommendations.extend(recommendations)
        
        # Generate visualization suggestions
        visualizations = self._suggest_visualizations(df, intent)
        result.visualizations = visualizations
        
        return result
    
    def _normalize_results(self, query_results: Any) -> Optional[pd.DataFrame]:
        """Convert query results to pandas DataFrame."""
        try:
            if isinstance(query_results, pd.DataFrame):
                return query_results
            elif isinstance(query_results, dict):
                return pd.DataFrame([query_results])
            elif isinstance(query_results, list):
                if not query_results:
                    return None
                return pd.DataFrame(query_results)
            else:
                # Try to convert to DataFrame
                return pd.DataFrame(query_results)
        except Exception as e:
            logger.warning(f"Could not normalize query results: {e}")
            return None
    
    async def _interpret_with_template(
        self, 
        df: pd.DataFrame, 
        intent: QueryIntent,
        context: Dict[str, Any]
    ) -> InterpretationResult:
        """Interpret results using predefined templates."""
        
        templates = self.interpretation_templates[intent]
        key_metrics = self._extract_key_metrics(df, intent)
        
        # Generate summary based on intent
        summary = self._generate_template_summary(df, intent, templates, key_metrics)
        
        # Extract insights
        insights = self._extract_template_insights(df, intent, key_metrics)
        
        # Generate explanation
        explanation = self._generate_template_explanation(df, intent, key_metrics)
        
        return InterpretationResult(
            summary=summary,
            insights=insights,
            recommendations=[],  # Will be filled later
            key_metrics=key_metrics,
            visualizations=[],  # Will be filled later
            explanation=explanation
        )
    
    async def _interpret_with_llm(
        self, 
        df: pd.DataFrame, 
        intent: QueryIntent,
        original_query: str,
        context: Dict[str, Any]
    ) -> InterpretationResult:
        """Interpret results using LLM for complex cases."""
        
        if not self.llm_client:
            # Fallback to basic interpretation
            return InterpretationResult(
                summary=f"Query returned {len(df)} rows of data",
                insights=[f"Data contains {df.shape[1]} columns"],
                recommendations=["Review the data for patterns"],
                key_metrics={"row_count": len(df), "column_count": df.shape[1]},
                visualizations=[],
                explanation="Basic interpretation without LLM analysis"
            )
        
        # Prepare data summary for LLM
        data_summary = self._prepare_data_summary(df)
        
        # Create prompt for LLM interpretation
        system_prompt = """
        You are a data analyst specializing in Snowflake analytics. 
        Interpret the query results and provide:
        1. A concise summary of the findings
        2. Key insights from the data
        3. Notable patterns or anomalies
        4. Business implications
        
        Be specific and focus on actionable insights.
        """
        
        user_prompt = f"""
        Original query: {original_query}
        Intent: {intent.value}
        
        Data summary:
        {data_summary}
        
        Please provide a comprehensive interpretation of these results.
        """
        
        # Generate interpretation using LLM
        response = await self.llm_client.generate_completion(
            prompt=user_prompt,
            system_prompt=system_prompt,
            temperature=0.3,
            max_tokens=800
        )
        
        # Parse LLM response into structured format
        return self._parse_llm_interpretation(response.content, df)
    
    def _extract_key_metrics(self, df: pd.DataFrame, intent: QueryIntent) -> Dict[str, Any]:
        """Extract key metrics based on query intent."""
        metrics = {}
        
        # Common metrics
        metrics['row_count'] = len(df)
        metrics['column_count'] = df.shape[1]
        
        # Intent-specific metrics
        if intent == QueryIntent.COST_OVERVIEW:
            if 'total_cost' in df.columns:
                metrics['total_cost'] = df['total_cost'].sum()
                metrics['avg_cost'] = df['total_cost'].mean()
                metrics['max_cost'] = df['total_cost'].max()
                metrics['min_cost'] = df['total_cost'].min()
            
            if 'period' in df.columns:
                metrics['time_periods'] = df['period'].nunique()
                metrics['date_range'] = f"{df['period'].min()} to {df['period'].max()}"
        
        elif intent == QueryIntent.COST_BREAKDOWN:
            if 'total_cost' in df.columns and 'category' in df.columns:
                metrics['total_categories'] = df['category'].nunique()
                metrics['top_category'] = df.loc[df['total_cost'].idxmax(), 'category']
                metrics['top_cost'] = df['total_cost'].max()
                metrics['cost_distribution'] = df.groupby('category')['total_cost'].sum().to_dict()
        
        elif intent == QueryIntent.USAGE_OVERVIEW:
            if 'query_count' in df.columns:
                metrics['total_queries'] = df['query_count'].sum()
                metrics['avg_queries'] = df['query_count'].mean()
            
            if 'avg_execution_time' in df.columns:
                metrics['avg_execution_time'] = df['avg_execution_time'].mean()
        
        elif intent == QueryIntent.USER_ACTIVITY:
            if 'user_name' in df.columns:
                metrics['total_users'] = df['user_name'].nunique()
            
            if 'total_queries' in df.columns:
                metrics['most_active_user'] = df.loc[df['total_queries'].idxmax(), 'user_name']
                metrics['max_user_queries'] = df['total_queries'].max()
        
        elif intent == QueryIntent.WAREHOUSE_UTILIZATION:
            if 'avg_utilization' in df.columns:
                metrics['avg_utilization'] = df['avg_utilization'].mean()
                metrics['max_utilization'] = df['avg_utilization'].max()
                metrics['min_utilization'] = df['avg_utilization'].min()
            
            if 'warehouse' in df.columns:
                metrics['warehouse_count'] = df['warehouse'].nunique()
        
        elif intent == QueryIntent.QUERY_PERFORMANCE:
            if 'execution_time_ms' in df.columns:
                metrics['avg_execution_time'] = df['execution_time_ms'].mean()
                metrics['max_execution_time'] = df['execution_time_ms'].max()
                metrics['min_execution_time'] = df['execution_time_ms'].min()
                
                # Count slow queries
                threshold = self.insight_thresholds['performance_threshold']
                metrics['slow_queries'] = len(df[df['execution_time_ms'] > threshold])
        
        return metrics
    
    def _generate_template_summary(
        self, 
        df: pd.DataFrame, 
        intent: QueryIntent,
        templates: Dict[str, str],
        key_metrics: Dict[str, Any]
    ) -> str:
        """Generate summary using templates."""
        
        if intent == QueryIntent.COST_OVERVIEW:
            if len(df) == 1:
                return templates['single_record'].format(
                    date=df.iloc[0].get('period', 'N/A'),
                    cost=key_metrics.get('total_cost', 0)
                )
            else:
                return templates['summary'].format(
                    total_cost=key_metrics.get('total_cost', 0),
                    time_period=key_metrics.get('date_range', 'specified period')
                )
        
        elif intent == QueryIntent.COST_BREAKDOWN:
            total_cost = key_metrics.get('total_cost', 0)
            top_cost = key_metrics.get('top_cost', 0)
            top_percent = (top_cost / total_cost * 100) if total_cost > 0 else 0
            
            return templates['summary'].format(
                top_category=key_metrics.get('top_category', 'Unknown'),
                top_cost=top_cost,
                top_percent=top_percent
            )
        
        elif intent == QueryIntent.USAGE_OVERVIEW:
            return templates['summary'].format(
                total_queries=key_metrics.get('total_queries', 0),
                avg_execution_time=key_metrics.get('avg_execution_time', 0)
            )
        
        elif intent == QueryIntent.USER_ACTIVITY:
            return templates['summary'].format(
                most_active_user=key_metrics.get('most_active_user', 'Unknown'),
                user_queries=key_metrics.get('max_user_queries', 0)
            )
        
        elif intent == QueryIntent.WAREHOUSE_UTILIZATION:
            return templates['summary'].format(
                avg_utilization=key_metrics.get('avg_utilization', 0),
                warehouse_count=key_metrics.get('warehouse_count', 0)
            )
        
        elif intent == QueryIntent.QUERY_PERFORMANCE:
            return templates['summary'].format(
                slow_queries=key_metrics.get('slow_queries', 0),
                threshold=self.insight_thresholds['performance_threshold']
            )
        
        return f"Query returned {len(df)} rows of data"
    
    def _extract_template_insights(
        self, 
        df: pd.DataFrame, 
        intent: QueryIntent,
        key_metrics: Dict[str, Any]
    ) -> List[str]:
        """Extract insights using template logic."""
        insights = []
        
        if intent == QueryIntent.COST_OVERVIEW:
            total_cost = key_metrics.get('total_cost', 0)
            if total_cost > self.insight_thresholds['high_cost_threshold']:
                insights.append(f"High cost alert: Total spending of ${total_cost:.2f} exceeds threshold")
            
            # Trend analysis if multiple periods
            if len(df) > 1 and 'total_cost' in df.columns:
                trend = self._calculate_trend(df['total_cost'])
                if abs(trend) > self.insight_thresholds['growth_threshold']:
                    direction = "increasing" if trend > 0 else "decreasing"
                    insights.append(f"Cost trend is {direction} by {abs(trend):.1f}%")
        
        elif intent == QueryIntent.COST_BREAKDOWN:
            # Cost concentration analysis
            if 'total_cost' in df.columns:
                total_cost = df['total_cost'].sum()
                top_cost = df['total_cost'].max()
                concentration = (top_cost / total_cost * 100) if total_cost > 0 else 0
                
                if concentration > 50:
                    insights.append(f"Cost concentration: Top category accounts for {concentration:.1f}% of total cost")
        
        elif intent == QueryIntent.USAGE_OVERVIEW:
            total_queries = key_metrics.get('total_queries', 0)
            if total_queries > self.insight_thresholds['high_usage_threshold']:
                insights.append(f"High usage detected: {total_queries} queries executed")
        
        elif intent == QueryIntent.WAREHOUSE_UTILIZATION:
            avg_util = key_metrics.get('avg_utilization', 0)
            if avg_util > self.insight_thresholds['utilization_threshold']:
                insights.append(f"High utilization: Average {avg_util:.1f}% may indicate capacity constraints")
            elif avg_util < 30:
                insights.append(f"Low utilization: Average {avg_util:.1f}% suggests potential for cost optimization")
        
        return insights
    
    def _generate_template_explanation(
        self, 
        df: pd.DataFrame, 
        intent: QueryIntent,
        key_metrics: Dict[str, Any]
    ) -> str:
        """Generate detailed explanation using templates."""
        
        explanations = {
            QueryIntent.COST_OVERVIEW: f"""
            The cost analysis covers {key_metrics.get('time_periods', 0)} time periods with a total cost of 
            ${key_metrics.get('total_cost', 0):.2f}. The data shows cost distribution over time, 
            helping identify spending patterns and trends.
            """,
            QueryIntent.COST_BREAKDOWN: f"""
            The cost breakdown analysis categorizes spending across {key_metrics.get('total_categories', 0)} 
            different categories. This helps identify the main cost drivers and opportunities for optimization.
            """,
            QueryIntent.USAGE_OVERVIEW: f"""
            The usage analysis shows activity patterns with {key_metrics.get('total_queries', 0)} total queries. 
            This provides insights into system utilization and performance characteristics.
            """,
            QueryIntent.USER_ACTIVITY: f"""
            The user activity analysis covers {key_metrics.get('total_users', 0)} users, showing engagement 
            patterns and identifying the most active users in the system.
            """,
            QueryIntent.WAREHOUSE_UTILIZATION: f"""
            The warehouse utilization analysis covers {key_metrics.get('warehouse_count', 0)} warehouses, 
            showing efficiency metrics and identifying optimization opportunities.
            """,
            QueryIntent.QUERY_PERFORMANCE: f"""
            The query performance analysis identifies {key_metrics.get('slow_queries', 0)} slow queries 
            that may need optimization to improve overall system performance.
            """
        }
        
        return explanations.get(intent, "Analysis of the query results provides insights into the requested data.")
    
    def _extract_general_insights(self, df: pd.DataFrame, intent: QueryIntent) -> List[str]:
        """Extract general insights that apply to any query."""
        insights = []
        
        # Data volume insights
        if len(df) > 1000:
            insights.append(f"Large result set with {len(df)} rows - consider adding filters")
        elif len(df) == 0:
            insights.append("No data found - try adjusting filters or time range")
        
        # Data quality insights
        for col in df.columns:
            null_count = df[col].isnull().sum()
            if null_count > 0:
                null_percent = (null_count / len(df)) * 100
                if null_percent > 20:
                    insights.append(f"Data quality issue: {col} has {null_percent:.1f}% missing values")
        
        return insights
    
    def _generate_recommendations(
        self, 
        df: pd.DataFrame, 
        intent: QueryIntent,
        key_metrics: Dict[str, Any]
    ) -> List[str]:
        """Generate recommendations based on the analysis."""
        recommendations = []
        
        if intent == QueryIntent.COST_OVERVIEW:
            total_cost = key_metrics.get('total_cost', 0)
            if total_cost > self.insight_thresholds['high_cost_threshold']:
                recommendations.append("Consider cost optimization strategies due to high spending")
                recommendations.append("Review warehouse auto-suspend settings")
        
        elif intent == QueryIntent.WAREHOUSE_UTILIZATION:
            avg_util = key_metrics.get('avg_utilization', 0)
            if avg_util < 30:
                recommendations.append("Consider downsizing warehouses due to low utilization")
            elif avg_util > 80:
                recommendations.append("Consider scaling up warehouses due to high utilization")
        
        elif intent == QueryIntent.QUERY_PERFORMANCE:
            slow_queries = key_metrics.get('slow_queries', 0)
            if slow_queries > 0:
                recommendations.append("Optimize slow queries to improve performance")
                recommendations.append("Review query patterns for efficiency improvements")
        
        return recommendations
    
    def _suggest_visualizations(self, df: pd.DataFrame, intent: QueryIntent) -> List[Dict[str, Any]]:
        """Suggest appropriate visualizations for the data."""
        visualizations = []
        
        if intent == QueryIntent.COST_OVERVIEW:
            if 'period' in df.columns and 'total_cost' in df.columns:
                visualizations.append({
                    'type': 'line_chart',
                    'title': 'Cost Trend Over Time',
                    'x_axis': 'period',
                    'y_axis': 'total_cost',
                    'description': 'Shows cost trends over the selected time period'
                })
        
        elif intent == QueryIntent.COST_BREAKDOWN:
            if 'category' in df.columns and 'total_cost' in df.columns:
                visualizations.append({
                    'type': 'pie_chart',
                    'title': 'Cost Distribution by Category',
                    'values': 'total_cost',
                    'labels': 'category',
                    'description': 'Shows cost distribution across categories'
                })
        
        elif intent == QueryIntent.WAREHOUSE_UTILIZATION:
            if 'warehouse' in df.columns and 'avg_utilization' in df.columns:
                visualizations.append({
                    'type': 'bar_chart',
                    'title': 'Warehouse Utilization',
                    'x_axis': 'warehouse',
                    'y_axis': 'avg_utilization',
                    'description': 'Shows utilization across warehouses'
                })
        
        return visualizations
    
    def _calculate_trend(self, values: pd.Series) -> float:
        """Calculate trend percentage change."""
        if len(values) < 2:
            return 0.0
        
        start_value = values.iloc[0]
        end_value = values.iloc[-1]
        
        if start_value == 0:
            return 0.0
        
        return ((end_value - start_value) / start_value) * 100
    
    def _prepare_data_summary(self, df: pd.DataFrame) -> str:
        """Prepare a summary of the data for LLM analysis."""
        summary_parts = [
            f"Rows: {len(df)}",
            f"Columns: {df.shape[1]}",
            f"Column names: {', '.join(df.columns[:10])}"  # Limit to first 10 columns
        ]
        
        # Add sample data
        if len(df) > 0:
            summary_parts.append(f"Sample data (first 3 rows):")
            summary_parts.append(df.head(3).to_string())
        
        # Add basic statistics for numeric columns
        numeric_cols = df.select_dtypes(include=['number']).columns
        if len(numeric_cols) > 0:
            summary_parts.append(f"Numeric columns statistics:")
            summary_parts.append(df[numeric_cols].describe().to_string())
        
        return "\n".join(summary_parts)
    
    def _parse_llm_interpretation(self, llm_response: str, df: pd.DataFrame) -> InterpretationResult:
        """Parse LLM response into structured interpretation result."""
        # This is a simplified parser - in production, you'd want more sophisticated parsing
        lines = llm_response.split('\n')
        
        summary = ""
        insights = []
        explanation = llm_response
        
        # Extract summary (usually the first substantial line)
        for line in lines:
            if len(line.strip()) > 20:  # Substantial content
                summary = line.strip()
                break
        
        # Extract insights (look for bullet points or numbered lists)
        for line in lines:
            line = line.strip()
            if line.startswith('•') or line.startswith('-') or line.startswith('*'):
                insights.append(line[1:].strip())
            elif any(line.startswith(f"{i}.") for i in range(1, 10)):
                insights.append(line[2:].strip())
        
        # Basic key metrics
        key_metrics = {
            "row_count": len(df),
            "column_count": df.shape[1]
        }
        
        return InterpretationResult(
            summary=summary or "Analysis completed",
            insights=insights,
            recommendations=[],
            key_metrics=key_metrics,
            visualizations=[],
            explanation=explanation
        )