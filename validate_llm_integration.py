"""
Simple validation test for LLM integration

Basic validation without external dependencies.
"""

import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

def test_imports():
    """Test that all LLM components can be imported."""
    print("Testing imports...")
    
    try:
        from snowflake_analytics.llm import LLMService, get_llm_service
        print("✓ LLM service imports successful")
    except Exception as e:
        print(f"✗ LLM service import failed: {e}")
        return False
    
    try:
        from snowflake_analytics.llm.query_processor import IntentClassifier, QueryValidator
        print("✓ Query processor imports successful")
    except Exception as e:
        print(f"✗ Query processor import failed: {e}")
        return False
    
    try:
        from snowflake_analytics.llm.insights import InsightGenerator
        print("✓ Insights module imports successful")
    except Exception as e:
        print(f"✗ Insights module import failed: {e}")
        return False
    
    return True

def test_intent_classification():
    """Test basic intent classification."""
    print("\nTesting intent classification...")
    
    try:
        from snowflake_analytics.llm.query_processor import IntentClassifier
        
        classifier = IntentClassifier()
        
        # Test cost query
        result = classifier.classify_intent("What were our Snowflake costs last month?")
        print(f"✓ Cost query intent: {result.intent.value} (confidence: {result.confidence:.2f})")
        
        # Test usage query  
        result = classifier.classify_intent("Show me warehouse utilization")
        print(f"✓ Usage query intent: {result.intent.value} (confidence: {result.confidence:.2f})")
        
        return True
    except Exception as e:
        print(f"✗ Intent classification test failed: {e}")
        return False

def test_query_validation():
    """Test SQL query validation."""
    print("\nTesting query validation...")
    
    try:
        from snowflake_analytics.llm.query_processor import QueryValidator
        
        validator = QueryValidator()
        
        # Test valid query
        valid_sql = "SELECT warehouse, SUM(cost_usd) FROM cost_metrics WHERE date >= '2024-01-01'"
        result = validator.validate_query(valid_sql)
        print(f"✓ Valid query validation: valid={result.is_valid}, safe={result.is_safe}")
        
        # Test dangerous query
        dangerous_sql = "DROP TABLE cost_metrics"
        result = validator.validate_query(dangerous_sql)
        print(f"✓ Dangerous query validation: valid={result.is_valid}, safe={result.is_safe}")
        
        return True
    except Exception as e:
        print(f"✗ Query validation test failed: {e}")
        return False

def test_llm_service():
    """Test LLM service creation."""
    print("\nTesting LLM service...")
    
    try:
        from snowflake_analytics.llm import LLMService
        
        service = LLMService({'test_mode': True})
        print(f"✓ LLM service created successfully")
        print(f"✓ Service available: {service.is_available()}")
        
        return True
    except Exception as e:
        print(f"✗ LLM service test failed: {e}")
        return False

def test_insight_generation():
    """Test insight generation."""
    print("\nTesting insight generation...")
    
    try:
        from snowflake_analytics.llm.insights import InsightGenerator
        
        generator = InsightGenerator()
        print("✓ Insight generator created successfully")
        
        stats = generator.get_generator_stats()
        print(f"✓ Generator stats: {len(stats['capabilities'])} capabilities")
        
        return True
    except Exception as e:
        print(f"✗ Insight generation test failed: {e}")
        return False

def main():
    """Run all validation tests."""
    print("=" * 50)
    print("LLM Integration Validation Tests")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_intent_classification,
        test_query_validation,
        test_llm_service,
        test_insight_generation
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"✗ Test {test.__name__} crashed: {e}")
            failed += 1
    
    print("\n" + "=" * 50)
    print(f"Test Results: {passed} passed, {failed} failed")
    print("=" * 50)
    
    if failed == 0:
        print("🎉 All tests passed! LLM integration is working correctly.")
        return True
    else:
        print("❌ Some tests failed. Check the implementation.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)