"""
Unit tests for LLM query processor.

Tests natural language query processing and SQL generation.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import json


class TestQueryProcessor:
    """Test suite for LLM query processor functionality."""

    @pytest.fixture
    def mock_query_processor(self):
        """Create a mock query processor."""
        processor = Mock()
        processor.process_query = Mock()
        processor.generate_sql = Mock()
        processor.validate_query = Mock()
        processor.explain_query = Mock()
        processor.optimize_query = Mock()
        processor.llm_client = Mock()
        return processor

    @pytest.fixture
    def sample_natural_language_queries(self):
        """Create sample natural language queries."""
        return [
            "Show me the total cost for the last 30 days",
            "What are the top 5 most expensive warehouses this month?",
            "Find all queries that took more than 5 minutes to run",
            "Show usage trends for the analytics warehouse",
            "Which users are consuming the most credits?",
            "What was the average query execution time yesterday?",
            "Show me cost anomalies in the last week",
            "Compare this month's usage with last month"
        ]

    @pytest.fixture
    def sample_sql_queries(self):
        """Create sample SQL queries corresponding to natural language."""
        return [
            "SELECT SUM(total_cost) FROM cost_data WHERE date >= CURRENT_DATE - INTERVAL '30 days'",
            "SELECT warehouse_name, SUM(total_cost) as cost FROM cost_data WHERE MONTH(date) = MONTH(CURRENT_DATE) GROUP BY warehouse_name ORDER BY cost DESC LIMIT 5",
            "SELECT query_id, execution_time FROM query_history WHERE execution_time > 300000",
            "SELECT date, usage_hours FROM warehouse_usage WHERE warehouse_name = 'ANALYTICS' ORDER BY date",
            "SELECT user_name, SUM(credits_used) as credits FROM user_usage GROUP BY user_name ORDER BY credits DESC",
            "SELECT AVG(execution_time) FROM query_history WHERE date = CURRENT_DATE - INTERVAL '1 day'",
            "SELECT * FROM cost_anomalies WHERE detected_at >= CURRENT_DATE - INTERVAL '7 days'",
            "SELECT MONTH(date) as month, SUM(usage_hours) as usage FROM warehouse_usage WHERE date >= CURRENT_DATE - INTERVAL '60 days' GROUP BY MONTH(date)"
        ]

    def test_query_processor_initialization(self, mock_query_processor):
        """Test query processor initialization."""
        assert mock_query_processor is not None
        assert mock_query_processor.llm_client is not None

    def test_natural_language_query_processing(self, mock_query_processor, sample_natural_language_queries):
        """Test natural language query processing."""
        nl_query = sample_natural_language_queries[0]
        
        # Mock processing result
        processing_result = {
            'original_query': nl_query,
            'intent': 'cost_analysis',
            'entities': {
                'time_period': '30 days',
                'metric': 'total_cost',
                'aggregation': 'sum'
            },
            'sql_query': "SELECT SUM(total_cost) FROM cost_data WHERE date >= CURRENT_DATE - INTERVAL '30 days'",
            'confidence': 0.95
        }
        
        mock_query_processor.process_query.return_value = processing_result
        
        # Test processing
        result = mock_query_processor.process_query(nl_query)
        
        assert result['original_query'] == nl_query
        assert result['intent'] == 'cost_analysis'
        assert result['confidence'] == 0.95
        assert 'sql_query' in result
        mock_query_processor.process_query.assert_called_once_with(nl_query)

    def test_sql_generation_success(self, mock_query_processor):
        """Test successful SQL generation."""
        # Mock SQL generation
        generation_result = {
            'sql_query': "SELECT warehouse_name, SUM(total_cost) as cost FROM cost_data GROUP BY warehouse_name",
            'query_type': 'SELECT',
            'tables_used': ['cost_data'],
            'columns_used': ['warehouse_name', 'total_cost'],
            'has_aggregation': True,
            'has_joins': False,
            'complexity_score': 0.3
        }
        
        mock_query_processor.generate_sql.return_value = generation_result
        
        # Test SQL generation
        result = mock_query_processor.generate_sql()
        
        assert result['query_type'] == 'SELECT'
        assert result['has_aggregation'] is True
        assert result['complexity_score'] == 0.3
        assert 'cost_data' in result['tables_used']

    def test_sql_generation_failure(self, mock_query_processor):
        """Test SQL generation failure handling."""
        # Mock generation failure
        mock_query_processor.generate_sql.side_effect = Exception("SQL generation failed")
        
        with pytest.raises(Exception) as exc_info:
            mock_query_processor.generate_sql()
        
        assert "SQL generation failed" in str(exc_info.value)

    def test_query_validation_success(self, mock_query_processor):
        """Test successful query validation."""
        sql_query = "SELECT * FROM cost_data WHERE date >= '2024-01-01'"
        
        # Mock validation result
        validation_result = {
            'is_valid': True,
            'syntax_errors': [],
            'security_issues': [],
            'performance_warnings': [],
            'estimated_cost': 'low',
            'estimated_execution_time': '< 1 second'
        }
        
        mock_query_processor.validate_query.return_value = validation_result
        
        # Test validation
        result = mock_query_processor.validate_query(sql_query)
        
        assert result['is_valid'] is True
        assert len(result['syntax_errors']) == 0
        assert result['estimated_cost'] == 'low'
        mock_query_processor.validate_query.assert_called_once_with(sql_query)

    def test_query_validation_failure(self, mock_query_processor):
        """Test query validation with errors."""
        invalid_sql = "SELECT * FROM non_existent_table WHERE invalid_column = 'value'"
        
        # Mock validation result with errors
        validation_result = {
            'is_valid': False,
            'syntax_errors': ['Table "non_existent_table" does not exist'],
            'security_issues': [],
            'performance_warnings': ['Full table scan detected'],
            'estimated_cost': 'high',
            'estimated_execution_time': '> 10 seconds'
        }
        
        mock_query_processor.validate_query.return_value = validation_result
        
        # Test validation
        result = mock_query_processor.validate_query(invalid_sql)
        
        assert result['is_valid'] is False
        assert len(result['syntax_errors']) == 1
        assert result['estimated_cost'] == 'high'
        assert 'Full table scan detected' in result['performance_warnings']

    def test_query_explanation(self, mock_query_processor):
        """Test query explanation functionality."""
        sql_query = "SELECT warehouse_name, AVG(execution_time) FROM query_history GROUP BY warehouse_name"
        
        # Mock explanation result
        explanation_result = {
            'query_purpose': 'Calculate average query execution time by warehouse',
            'data_sources': ['query_history'],
            'operations': [
                'Group data by warehouse_name',
                'Calculate average of execution_time',
                'Return results'
            ],
            'output_description': 'Two columns: warehouse name and average execution time',
            'complexity_level': 'medium',
            'performance_notes': 'Uses aggregation function, requires GROUP BY'
        }
        
        mock_query_processor.explain_query.return_value = explanation_result
        
        # Test explanation
        result = mock_query_processor.explain_query(sql_query)
        
        assert result['query_purpose'] == 'Calculate average query execution time by warehouse'
        assert result['complexity_level'] == 'medium'
        assert len(result['operations']) == 3
        assert 'query_history' in result['data_sources']

    def test_query_optimization(self, mock_query_processor):
        """Test query optimization functionality."""
        original_query = "SELECT * FROM cost_data WHERE date >= '2024-01-01'"
        
        # Mock optimization result
        optimization_result = {
            'original_query': original_query,
            'optimized_query': "SELECT cost_id, date, total_cost FROM cost_data WHERE date >= '2024-01-01'",
            'optimization_applied': [
                'Replaced SELECT * with specific columns',
                'Added index hint for date column'
            ],
            'performance_improvement': {
                'execution_time_reduction': '40%',
                'cost_reduction': '35%',
                'memory_usage_reduction': '25%'
            },
            'optimization_score': 0.75
        }
        
        mock_query_processor.optimize_query.return_value = optimization_result
        
        # Test optimization
        result = mock_query_processor.optimize_query(original_query)
        
        assert result['optimization_score'] == 0.75
        assert len(result['optimization_applied']) == 2
        assert result['performance_improvement']['execution_time_reduction'] == '40%'
        assert result['optimized_query'] != original_query

    def test_query_intent_detection(self, mock_query_processor):
        """Test query intent detection."""
        query_intents = [
            {
                'query': 'Show me the cost breakdown',
                'intent': 'cost_analysis',
                'confidence': 0.92
            },
            {
                'query': 'Find slow queries',
                'intent': 'performance_analysis',
                'confidence': 0.88
            },
            {
                'query': 'Who is using the most resources?',
                'intent': 'usage_analysis',
                'confidence': 0.85
            }
        ]
        
        mock_query_processor.detect_intent.return_value = query_intents[0]
        
        # Test intent detection
        result = mock_query_processor.detect_intent(query_intents[0]['query'])
        
        assert result['intent'] == 'cost_analysis'
        assert result['confidence'] == 0.92

    def test_query_entity_extraction(self, mock_query_processor):
        """Test entity extraction from queries."""
        query = "Show me the cost for warehouse WH_ANALYTICS in the last 7 days"
        
        # Mock entity extraction result
        entities = {
            'time_entities': [
                {'entity': 'last 7 days', 'type': 'relative_time', 'value': '7 days'}
            ],
            'warehouse_entities': [
                {'entity': 'WH_ANALYTICS', 'type': 'warehouse_name', 'value': 'WH_ANALYTICS'}
            ],
            'metric_entities': [
                {'entity': 'cost', 'type': 'metric', 'value': 'total_cost'}
            ],
            'aggregation_entities': [
                {'entity': 'show', 'type': 'aggregation', 'value': 'sum'}
            ]
        }
        
        mock_query_processor.extract_entities.return_value = entities
        
        # Test entity extraction
        result = mock_query_processor.extract_entities(query)
        
        assert len(result['time_entities']) == 1
        assert len(result['warehouse_entities']) == 1
        assert result['warehouse_entities'][0]['value'] == 'WH_ANALYTICS'
        assert result['time_entities'][0]['value'] == '7 days'

    def test_query_context_understanding(self, mock_query_processor):
        """Test query context understanding."""
        query_context = {
            'user_role': 'analyst',
            'previous_queries': [
                'Show me warehouse usage',
                'What are the top costs?'
            ],
            'current_query': 'Show me more details about that',
            'session_context': {
                'focused_warehouse': 'WH_ANALYTICS',
                'time_period': 'last_month',
                'active_dashboards': ['cost_dashboard']
            }
        }
        
        # Mock context understanding result
        context_result = {
            'resolved_query': 'Show me more details about WH_ANALYTICS costs last month',
            'context_references': [
                {'reference': 'that', 'resolved_to': 'WH_ANALYTICS costs'}
            ],
            'inferred_entities': {
                'warehouse': 'WH_ANALYTICS',
                'time_period': 'last_month',
                'metric': 'costs'
            },
            'confidence': 0.87
        }
        
        mock_query_processor.understand_context.return_value = context_result
        
        # Test context understanding
        result = mock_query_processor.understand_context(query_context)
        
        assert result['resolved_query'] == 'Show me more details about WH_ANALYTICS costs last month'
        assert result['confidence'] == 0.87
        assert len(result['context_references']) == 1
        assert result['inferred_entities']['warehouse'] == 'WH_ANALYTICS'

    def test_query_response_generation(self, mock_query_processor):
        """Test query response generation."""
        query_results = {
            'query': 'SELECT warehouse_name, SUM(total_cost) FROM cost_data GROUP BY warehouse_name',
            'results': [
                {'warehouse_name': 'WH_ANALYTICS', 'total_cost': 1250.75},
                {'warehouse_name': 'WH_DEV', 'total_cost': 450.25},
                {'warehouse_name': 'WH_PROD', 'total_cost': 2100.50}
            ],
            'execution_time': 1.25,
            'row_count': 3
        }
        
        # Mock response generation
        response_result = {
            'natural_language_response': "Here are the total costs by warehouse: WH_ANALYTICS ($1,250.75), WH_DEV ($450.25), and WH_PROD ($2,100.50). The query executed in 1.25 seconds and returned 3 warehouses.",
            'summary_statistics': {
                'total_warehouses': 3,
                'highest_cost_warehouse': 'WH_PROD',
                'lowest_cost_warehouse': 'WH_DEV',
                'average_cost': 933.83
            },
            'visualizations': {
                'recommended_chart': 'bar_chart',
                'chart_config': {
                    'x_axis': 'warehouse_name',
                    'y_axis': 'total_cost',
                    'title': 'Total Cost by Warehouse'
                }
            }
        }
        
        mock_query_processor.generate_response.return_value = response_result
        
        # Test response generation
        result = mock_query_processor.generate_response(query_results)
        
        assert 'WH_ANALYTICS' in result['natural_language_response']
        assert result['summary_statistics']['total_warehouses'] == 3
        assert result['visualizations']['recommended_chart'] == 'bar_chart'
        assert result['summary_statistics']['highest_cost_warehouse'] == 'WH_PROD'

    @pytest.mark.parametrize("query_type,expected_intent", [
        ("Show me costs", "cost_analysis"),
        ("Find slow queries", "performance_analysis"),
        ("Who is using resources", "usage_analysis"),
        ("Show me alerts", "alert_analysis"),
        ("Optimize my warehouse", "optimization_analysis"),
    ])
    def test_query_intent_classification(self, mock_query_processor, query_type, expected_intent):
        """Test query intent classification."""
        mock_query_processor.classify_intent.return_value = expected_intent
        
        # Test intent classification
        result = mock_query_processor.classify_intent(query_type)
        
        assert result == expected_intent

    def test_query_processor_error_handling(self, mock_query_processor):
        """Test error handling in query processor."""
        # Mock various error scenarios
        error_scenarios = [
            {
                'error_type': 'parsing_error',
                'message': 'Unable to parse natural language query',
                'suggestion': 'Try rephrasing your question'
            },
            {
                'error_type': 'sql_generation_error',
                'message': 'Could not generate SQL for this query',
                'suggestion': 'Please be more specific about what you want to see'
            },
            {
                'error_type': 'validation_error',
                'message': 'Generated SQL contains security issues',
                'suggestion': 'Query has been blocked for security reasons'
            }
        ]
        
        mock_query_processor.handle_error.return_value = error_scenarios[0]
        
        # Test error handling
        result = mock_query_processor.handle_error()
        
        assert result['error_type'] == 'parsing_error'
        assert result['message'] == 'Unable to parse natural language query'
        assert 'suggestion' in result

    def test_query_processor_performance_metrics(self, mock_query_processor):
        """Test query processor performance metrics."""
        performance_metrics = {
            'average_processing_time_ms': 150.5,
            'success_rate': 0.94,
            'intent_detection_accuracy': 0.88,
            'sql_generation_accuracy': 0.92,
            'query_validation_rate': 0.96,
            'total_queries_processed': 15000,
            'cache_hit_rate': 0.35,
            'optimization_effectiveness': 0.67
        }
        
        mock_query_processor.get_performance_metrics.return_value = performance_metrics
        
        # Test performance metrics
        result = mock_query_processor.get_performance_metrics()
        
        assert result['average_processing_time_ms'] == 150.5
        assert result['success_rate'] == 0.94
        assert result['intent_detection_accuracy'] == 0.88
        assert result['total_queries_processed'] == 15000

    def test_query_processor_caching(self, mock_query_processor):
        """Test query processor caching functionality."""
        cache_results = {
            'cache_enabled': True,
            'cache_size_mb': 256,
            'cache_hit_rate': 0.35,
            'cache_entries': 1250,
            'cache_eviction_policy': 'LRU',
            'cached_queries': [
                {
                    'query_hash': 'abc123',
                    'original_query': 'Show me costs',
                    'cached_response': 'Total cost: $1,250.75',
                    'cache_timestamp': '2024-01-01T10:00:00Z'
                }
            ]
        }
        
        mock_query_processor.get_cache_status.return_value = cache_results
        
        # Test caching
        result = mock_query_processor.get_cache_status()
        
        assert result['cache_enabled'] is True
        assert result['cache_hit_rate'] == 0.35
        assert result['cache_entries'] == 1250
        assert len(result['cached_queries']) == 1

    def test_query_processor_batch_processing(self, mock_query_processor):
        """Test batch processing of queries."""
        batch_queries = [
            "Show me total costs",
            "Find expensive queries",
            "Show warehouse usage",
            "List top users"
        ]
        
        # Mock batch processing result
        batch_result = {
            'total_queries': 4,
            'processed_queries': 4,
            'failed_queries': 0,
            'processing_time_seconds': 2.5,
            'average_time_per_query': 0.625,
            'results': [
                {'query': batch_queries[0], 'status': 'success', 'response': 'Total: $5,000'},
                {'query': batch_queries[1], 'status': 'success', 'response': 'Found 10 expensive queries'},
                {'query': batch_queries[2], 'status': 'success', 'response': 'Usage: 85%'},
                {'query': batch_queries[3], 'status': 'success', 'response': 'Top 5 users listed'}
            ]
        }
        
        mock_query_processor.process_batch.return_value = batch_result
        
        # Test batch processing
        result = mock_query_processor.process_batch(batch_queries)
        
        assert result['total_queries'] == 4
        assert result['failed_queries'] == 0
        assert result['processing_time_seconds'] == 2.5
        assert len(result['results']) == 4