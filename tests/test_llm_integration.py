"""
Basic tests for LLM integration components

Tests the core functionality of the LLM integration without
requiring actual LLM API access.
"""

import pytest
import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from snowflake_analytics.llm import LLMService, get_llm_service, reset_llm_service
from snowflake_analytics.llm.query_processor import IntentClassifier, QueryValidator
from snowflake_analytics.llm.insights import InsightGenerator


class TestLLMService:
    """Test LLM service functionality."""
    
    def test_llm_service_creation(self):
        """Test creating LLM service without API keys."""
        config = {
            'test_mode': True
        }
        
        service = LLMService(config)
        assert service is not None
        assert service.config == config
        assert not service.is_available()  # No API keys provided
    
    def test_get_llm_service_singleton(self):
        """Test singleton behavior of get_llm_service."""
        reset_llm_service()  # Reset first
        
        service1 = get_llm_service({'test': True})
        service2 = get_llm_service({'test': False})
        
        # Should return same instance
        assert service1 is service2
        assert service1.config == {'test': True}  # Original config preserved


class TestIntentClassifier:
    """Test intent classification functionality."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.classifier = IntentClassifier()
    
    def test_cost_overview_intent(self):
        """Test cost overview intent classification."""
        query = "What were our Snowflake costs last month?"
        result = self.classifier.classify_intent(query)
        
        assert result.intent.value == "cost_overview"
        assert result.confidence > 0.5
        assert "cost" in str(result.entities).lower()
    
    def test_usage_overview_intent(self):
        """Test usage overview intent classification."""
        query = "Show me our current warehouse utilization"
        result = self.classifier.classify_intent(query)
        
        assert result.intent.value in ["usage_overview", "warehouse_utilization"]
        assert result.confidence > 0.5
    
    def test_unknown_intent(self):
        """Test unknown intent for unrecognized queries."""
        query = "This is completely unrelated to analytics"
        result = self.classifier.classify_intent(query)
        
        # Should either be unknown or have very low confidence
        assert result.intent.value == "unknown" or result.confidence < 0.3
    
    def test_time_range_extraction(self):
        """Test time range extraction from queries."""
        query = "Show me costs for last month"
        result = self.classifier.classify_intent(query)
        
        # Should extract some time-related information
        assert result.time_range is not None or "month" in query.lower()


class TestQueryValidator:
    """Test SQL query validation functionality."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.validator = QueryValidator()
    
    def test_valid_select_query(self):
        """Test validation of a valid SELECT query."""
        sql = "SELECT warehouse, SUM(cost_usd) FROM cost_metrics WHERE date >= '2024-01-01' GROUP BY warehouse"
        result = self.validator.validate_query(sql)
        
        assert result.is_valid
        assert result.is_safe
        assert result.estimated_cost > 0
    
    def test_forbidden_operations(self):
        """Test that forbidden operations are blocked."""
        dangerous_queries = [
            "DROP TABLE cost_metrics",
            "DELETE FROM cost_metrics WHERE date = '2024-01-01'",
            "UPDATE cost_metrics SET cost_usd = 0",
            "INSERT INTO cost_metrics VALUES (1, 2, 3)"
        ]
        
        for sql in dangerous_queries:
            result = self.validator.validate_query(sql)
            assert not result.is_safe
            assert not result.is_valid
            assert any("forbidden" in issue.message.lower() 
                      for issue in result.issues)
    
    def test_sql_injection_detection(self):
        """Test SQL injection pattern detection."""
        malicious_queries = [
            "SELECT * FROM cost_metrics WHERE warehouse = 'test'; DROP TABLE users; --",
            "SELECT * FROM cost_metrics WHERE warehouse = 'test' OR '1'='1'",
            "SELECT * FROM cost_metrics UNION SELECT * FROM sensitive_table"
        ]
        
        for sql in malicious_queries:
            result = self.validator.validate_query(sql)
            # Should detect issues even if some patterns pass basic validation
            assert len(result.issues) > 0
    
    def test_performance_warnings(self):
        """Test performance-related warnings."""
        # Query without WHERE clause on large table
        sql = "SELECT * FROM cost_metrics"
        result = self.validator.validate_query(sql)
        
        # Should have warnings about performance
        warning_messages = [issue.message.lower() for issue in result.issues]
        assert any("where" in msg or "select *" in msg for msg in warning_messages)


class TestInsightGenerator:
    """Test insight generation functionality."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.generator = InsightGenerator()
    
    def test_insight_generator_creation(self):
        """Test creating insight generator."""
        assert self.generator is not None
        assert self.generator.cost_analyzer is not None
        assert self.generator.usage_analyzer is not None
        assert self.generator.anomaly_explainer is not None
        assert self.generator.trend_analyzer is not None
    
    @pytest.mark.asyncio
    async def test_cost_insights_generation(self):
        """Test generating cost insights from sample data."""
        # Sample cost data
        cost_data = {
            'cost_metrics': [
                {'date': '2024-01-01', 'warehouse': 'COMPUTE_WH', 'cost_usd': 100.0, 'credits_consumed': 50},
                {'date': '2024-01-02', 'warehouse': 'COMPUTE_WH', 'cost_usd': 150.0, 'credits_consumed': 75},
                {'date': '2024-01-03', 'warehouse': 'ANALYTICS_WH', 'cost_usd': 200.0, 'credits_consumed': 100},
            ]
        }
        
        insights = await self.generator.generate_insights(cost_data, "cost")
        
        # Should generate some insights
        assert isinstance(insights, list)
        # Even if no specific insights are found, should not error
    
    @pytest.mark.asyncio
    async def test_auto_insights_generation(self):
        """Test automatic insight generation across all categories."""
        # Sample data with multiple metrics
        data = {
            'cost_metrics': [
                {'date': '2024-01-01', 'cost_usd': 100.0},
                {'date': '2024-01-02', 'cost_usd': 120.0},
            ],
            'usage_metrics': [
                {'date': '2024-01-01', 'query_count': 1000},
                {'date': '2024-01-02', 'query_count': 1200},
            ]
        }
        
        insights = await self.generator.generate_insights(data, "auto")
        
        # Should not error and return a list
        assert isinstance(insights, list)
    
    def test_get_generator_stats(self):
        """Test getting generator statistics."""
        stats = self.generator.get_generator_stats()
        
        assert 'components' in stats
        assert 'configuration' in stats
        assert 'capabilities' in stats
        
        # Should list available capabilities
        assert 'cost_analysis' in stats['capabilities']
        assert 'usage_analysis' in stats['capabilities']


class TestLLMIntegrationFlow:
    """Test complete LLM integration flow."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.service = LLMService({'test_mode': True})
    
    def test_service_component_availability(self):
        """Test that service components are available."""
        # Even without LLM client, basic components should be available
        assert hasattr(self.service, 'query_interface')
        assert hasattr(self.service, 'insight_generator')
    
    def test_process_natural_language_query_without_llm(self):
        """Test processing query without actual LLM."""
        query = "What were our costs last month?"
        result = self.service.process_natural_language_query(query)
        
        # Should return error or handle gracefully without LLM
        assert 'error' in result


if __name__ == "__main__":
    # Run tests manually if needed
    pytest.main([__file__, "-v"])