"""
Query Interface - Main Query Processing Orchestrator

Orchestrates the complete natural language query processing pipeline,
from intent classification to result interpretation for Snowflake analytics.
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import asyncio
import logging

from ...utils.logger import get_logger
from .intent_classifier import IntentClassifier, QueryIntent, IntentResult
from .nl_to_sql import NaturalLanguageToSQL, SQLQuery
from .query_validator import QueryValidator, QueryValidationResult
from .result_interpreter import ResultInterpreter, InterpretationResult

logger = get_logger(__name__)


@dataclass
class QueryProcessingResult:
    """Complete result of query processing pipeline."""
    original_query: str
    intent: QueryIntent
    intent_confidence: float
    sql_query: str
    is_valid: bool
    is_safe: bool
    interpretation: InterpretationResult
    execution_metadata: Dict[str, Any]
    errors: List[str]
    warnings: List[str]


class QueryInterface:
    """Main interface for processing natural language queries."""
    
    def __init__(self, client=None, config: Dict[str, Any] = None):
        """Initialize query interface.
        
        Args:
            client: LLM client for advanced processing
            config: Configuration dictionary
        """
        self.client = client
        self.config = config or {}
        
        # Initialize components
        self.intent_classifier = IntentClassifier(config.get('intent_classifier', {}))
        self.nl_to_sql = NaturalLanguageToSQL(client, config.get('nl_to_sql', {}))
        self.query_validator = QueryValidator(config.get('query_validator', {}))
        self.result_interpreter = ResultInterpreter(client, config.get('result_interpreter', {}))
        
        # Query processing settings
        self.max_retries = config.get('max_retries', 3)
        self.timeout_seconds = config.get('timeout_seconds', 30)
        self.enable_caching = config.get('enable_caching', True)
        
        # Query cache (simple in-memory cache)
        self._query_cache = {}
        
        logger.info("Query interface initialized")
    
    async def process_query(
        self, 
        natural_query: str, 
        context: Dict[str, Any] = None,
        execute_query: bool = False,
        snowflake_client=None
    ) -> QueryProcessingResult:
        """Process a complete natural language query.
        
        Args:
            natural_query: Natural language query string
            context: Additional context (user info, session, etc.)
            execute_query: Whether to execute the generated SQL
            snowflake_client: Snowflake client for query execution
            
        Returns:
            QueryProcessingResult with complete processing pipeline results
        """
        context = context or {}
        errors = []
        warnings = []
        
        try:
            # Step 1: Intent Classification
            logger.info(f"Processing query: {natural_query[:100]}...")
            intent_result = self.intent_classifier.classify_intent(natural_query)
            
            if intent_result.confidence < 0.5:
                warnings.append(f"Low confidence intent classification: {intent_result.confidence:.2f}")
            
            # Step 2: Generate SQL
            sql_query = await self.nl_to_sql.convert_to_sql(
                natural_query, intent_result, context
            )
            
            # Step 3: Validate SQL
            validation_result = self.query_validator.validate_query(sql_query.sql)
            
            if not validation_result.is_valid:
                errors.extend([issue.message for issue in validation_result.issues 
                             if issue.severity.value in ['error', 'critical']])
            
            if validation_result.issues:
                warnings.extend([issue.message for issue in validation_result.issues 
                               if issue.severity.value in ['warning', 'info']])
            
            # Step 4: Execute query if requested and valid
            query_results = None
            execution_metadata = {}
            
            if execute_query and validation_result.is_safe and snowflake_client:
                try:
                    query_results, exec_meta = await self._execute_snowflake_query(
                        validation_result.sanitized_query or sql_query.sql,
                        snowflake_client
                    )
                    execution_metadata = exec_meta
                except Exception as e:
                    errors.append(f"Query execution failed: {str(e)}")
                    logger.error(f"Query execution error: {e}")
            
            # Step 5: Interpret results
            interpretation = await self.result_interpreter.interpret_results(
                query_results,
                intent_result.intent,
                natural_query,
                context
            )
            
            # Add metadata
            execution_metadata.update({
                'estimated_cost': sql_query.estimated_cost,
                'estimated_duration': sql_query.estimated_duration,
                'estimated_rows': validation_result.estimated_rows,
                'intent_confidence': intent_result.confidence,
                'query_complexity': self._assess_query_complexity(sql_query.sql)
            })
            
            return QueryProcessingResult(
                original_query=natural_query,
                intent=intent_result.intent,
                intent_confidence=intent_result.confidence,
                sql_query=validation_result.sanitized_query or sql_query.sql,
                is_valid=validation_result.is_valid,
                is_safe=validation_result.is_safe,
                interpretation=interpretation,
                execution_metadata=execution_metadata,
                errors=errors,
                warnings=warnings
            )
            
        except Exception as e:
            logger.error(f"Query processing failed: {e}")
            errors.append(f"Processing failed: {str(e)}")
            
            # Return error result
            return QueryProcessingResult(
                original_query=natural_query,
                intent=QueryIntent.UNKNOWN,
                intent_confidence=0.0,
                sql_query="",
                is_valid=False,
                is_safe=False,
                interpretation=InterpretationResult(
                    summary="Query processing failed",
                    insights=[],
                    recommendations=["Try rephrasing the query"],
                    key_metrics={},
                    visualizations=[],
                    explanation=f"Error: {str(e)}"
                ),
                execution_metadata={},
                errors=errors,
                warnings=warnings
            )
    
    async def _execute_snowflake_query(
        self, 
        sql: str, 
        snowflake_client
    ) -> tuple[Any, Dict[str, Any]]:
        """Execute SQL query against Snowflake.
        
        Args:
            sql: SQL query to execute
            snowflake_client: Snowflake client instance
            
        Returns:
            Tuple of (query_results, execution_metadata)
        """
        import time
        
        start_time = time.time()
        
        try:
            # Execute query (this would depend on the actual Snowflake client)
            if hasattr(snowflake_client, 'execute_query'):
                results = await snowflake_client.execute_query(sql)
            else:
                # Fallback for synchronous client
                results = snowflake_client.execute(sql)
            
            execution_time = time.time() - start_time
            
            # Extract metadata
            metadata = {
                'execution_time_seconds': execution_time,
                'rows_returned': len(results) if results else 0,
                'query_id': getattr(results, 'query_id', None),
                'warehouse_used': getattr(results, 'warehouse', None),
                'credits_consumed': getattr(results, 'credits_used', None)
            }
            
            return results, metadata
            
        except Exception as e:
            execution_time = time.time() - start_time
            metadata = {
                'execution_time_seconds': execution_time,
                'error': str(e)
            }
            raise Exception(f"Query execution failed: {str(e)}") from e
    
    def _assess_query_complexity(self, sql: str) -> str:
        """Assess the complexity of a SQL query."""
        sql_upper = sql.upper()
        complexity_score = 0
        
        # Basic complexity indicators
        if 'JOIN' in sql_upper:
            complexity_score += 2
        if 'GROUP BY' in sql_upper:
            complexity_score += 1
        if 'ORDER BY' in sql_upper:
            complexity_score += 1
        if 'HAVING' in sql_upper:
            complexity_score += 2
        if 'WINDOW' in sql_upper or 'OVER(' in sql_upper:
            complexity_score += 3
        if 'WITH' in sql_upper:  # CTE
            complexity_score += 2
        
        # Count subqueries
        complexity_score += sql_upper.count('SELECT') - 1
        
        # Classify complexity
        if complexity_score <= 1:
            return "simple"
        elif complexity_score <= 4:
            return "moderate"
        else:
            return "complex"
    
    async def get_query_suggestions(
        self, 
        partial_query: str = None,
        context: Dict[str, Any] = None
    ) -> List[Dict[str, Any]]:
        """Get query suggestions based on partial input or context.
        
        Args:
            partial_query: Partial query string
            context: Query context
            
        Returns:
            List of suggested queries with descriptions
        """
        suggestions = []
        
        # Intent-based suggestions
        supported_intents = self.intent_classifier.get_supported_intents()
        
        for intent_info in supported_intents[:10]:  # Limit to top 10
            intent = intent_info['intent']
            description = intent_info['description']
            examples = intent_info['examples']
            
            if partial_query:
                # Filter based on partial query match
                if any(keyword in partial_query.lower() 
                      for keyword in intent.lower().split('_')):
                    suggestions.extend([
                        {
                            "query": example,
                            "intent": intent,
                            "description": description,
                            "relevance_score": 0.8
                        }
                        for example in examples
                    ])
            else:
                # Add all examples for browsing
                suggestions.extend([
                    {
                        "query": example,
                        "intent": intent,
                        "description": description,
                        "relevance_score": 0.6
                    }
                    for example in examples
                ])
        
        # Sort by relevance score
        suggestions.sort(key=lambda x: x['relevance_score'], reverse=True)
        
        return suggestions[:20]  # Return top 20 suggestions
    
    async def explain_query(self, natural_query: str) -> Dict[str, Any]:
        """Explain what a natural language query will do without executing it.
        
        Args:
            natural_query: Natural language query to explain
            
        Returns:
            Dictionary with query explanation
        """
        try:
            # Classify intent
            intent_result = self.intent_classifier.classify_intent(natural_query)
            
            # Generate SQL
            sql_query = await self.nl_to_sql.convert_to_sql(natural_query, intent_result)
            
            # Validate SQL
            validation_result = self.query_validator.validate_query(sql_query.sql)
            
            return {
                "intent": intent_result.intent.value,
                "confidence": intent_result.confidence,
                "entities": intent_result.entities,
                "time_range": intent_result.time_range,
                "filters": intent_result.filters,
                "generated_sql": sql_query.sql,
                "sql_explanation": sql_query.explanation,
                "estimated_cost": sql_query.estimated_cost,
                "estimated_duration": sql_query.estimated_duration,
                "estimated_rows": validation_result.estimated_rows,
                "is_valid": validation_result.is_valid,
                "is_safe": validation_result.is_safe,
                "issues": [
                    {
                        "type": issue.type,
                        "severity": issue.severity.value,
                        "message": issue.message
                    }
                    for issue in validation_result.issues
                ],
                "suggested_queries": intent_result.suggested_queries
            }
            
        except Exception as e:
            logger.error(f"Query explanation failed: {e}")
            return {
                "error": str(e),
                "intent": "unknown",
                "confidence": 0.0
            }
    
    def get_interface_stats(self) -> Dict[str, Any]:
        """Get query interface statistics and health metrics."""
        return {
            "components": {
                "intent_classifier": {
                    "supported_intents": len(self.intent_classifier.intent_patterns),
                    "available": True
                },
                "nl_to_sql": {
                    "templates_available": len(self.nl_to_sql.sql_templates),
                    "llm_available": self.nl_to_sql.llm_client is not None
                },
                "query_validator": {
                    "validation_rules": len(self.query_validator.forbidden_keywords),
                    "available": True
                },
                "result_interpreter": {
                    "interpretation_templates": len(self.result_interpreter.interpretation_templates),
                    "llm_available": self.result_interpreter.llm_client is not None
                }
            },
            "configuration": {
                "max_retries": self.max_retries,
                "timeout_seconds": self.timeout_seconds,
                "caching_enabled": self.enable_caching
            },
            "cache_stats": {
                "entries": len(self._query_cache),
                "enabled": self.enable_caching
            }
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on query interface components."""
        health = {
            "overall_status": "healthy",
            "components": {},
            "timestamp": None
        }
        
        try:
            # Test intent classification
            test_result = self.intent_classifier.classify_intent("test query")
            health["components"]["intent_classifier"] = {
                "status": "healthy" if test_result else "degraded",
                "test_confidence": getattr(test_result, 'confidence', 0)
            }
            
            # Test query validation
            validation_result = self.query_validator.validate_query("SELECT 1")
            health["components"]["query_validator"] = {
                "status": "healthy" if validation_result.is_valid else "degraded",
                "test_validation": validation_result.is_valid
            }
            
            # Test LLM availability
            if self.client:
                health["components"]["llm_client"] = {
                    "status": "healthy" if self.client.is_available() else "degraded",
                    "available": self.client.is_available()
                }
            else:
                health["components"]["llm_client"] = {
                    "status": "unavailable",
                    "available": False
                }
            
            # Determine overall status
            component_statuses = [comp["status"] for comp in health["components"].values()]
            if "degraded" in component_statuses:
                health["overall_status"] = "degraded"
            elif "unavailable" in component_statuses:
                health["overall_status"] = "degraded"
            
            health["timestamp"] = "2024-01-01T00:00:00Z"  # Would use real timestamp
            
        except Exception as e:
            health["overall_status"] = "unhealthy"
            health["error"] = str(e)
        
        return health