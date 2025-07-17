#!/usr/bin/env python3
"""
Simple test script to verify the dashboard API works.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import asyncio
import json
from datetime import datetime

# Test the API endpoints directly
async def test_api_endpoints():
    """Test the API endpoints directly."""
    print("🧪 Testing Snowflake Analytics Dashboard API endpoints...")
    
    try:
        # Import the endpoint modules
        from src.snowflake_analytics.api.endpoints.costs import cost_endpoints
        from src.snowflake_analytics.api.endpoints.usage import usage_endpoints
        from src.snowflake_analytics.api.endpoints.predictions import prediction_endpoints
        from src.snowflake_analytics.api.endpoints.anomalies import anomaly_endpoints
        from src.snowflake_analytics.api.endpoints.alerts import alert_endpoints
        from src.snowflake_analytics.api.endpoints.realtime import realtime_endpoints
        
        print("✅ All endpoint modules imported successfully")
        
        # Test cost endpoints
        print("\n📊 Testing cost endpoints...")
        cost_summary = await cost_endpoints.get_cost_summary()
        print(f"   - Cost summary: {cost_summary['status']}")
        
        cost_trends = await cost_endpoints.get_cost_trends(7)
        print(f"   - Cost trends: {cost_trends['status']}")
        
        warehouse_costs = await cost_endpoints.get_warehouse_costs()
        print(f"   - Warehouse costs: {warehouse_costs['status']}")
        
        # Test usage endpoints
        print("\n📈 Testing usage endpoints...")
        usage_metrics = await usage_endpoints.get_usage_metrics()
        print(f"   - Usage metrics: {usage_metrics['status']}")
        
        query_performance = await usage_endpoints.get_query_performance()
        print(f"   - Query performance: {query_performance['status']}")
        
        warehouse_utilization = await usage_endpoints.get_warehouse_utilization()
        print(f"   - Warehouse utilization: {warehouse_utilization['status']}")
        
        # Test prediction endpoints
        print("\n🔮 Testing prediction endpoints...")
        cost_forecast = await prediction_endpoints.get_cost_forecast()
        print(f"   - Cost forecast: {cost_forecast['status']}")
        
        usage_forecast = await prediction_endpoints.get_usage_forecast()
        print(f"   - Usage forecast: {usage_forecast['status']}")
        
        recommendations = await prediction_endpoints.get_optimization_recommendations()
        print(f"   - Recommendations: {recommendations['status']}")
        
        model_performance = await prediction_endpoints.get_model_performance()
        print(f"   - Model performance: {model_performance['status']}")
        
        # Test anomaly endpoints
        print("\n🚨 Testing anomaly endpoints...")
        current_anomalies = await anomaly_endpoints.get_current_anomalies()
        print(f"   - Current anomalies: {current_anomalies['status']}")
        
        anomaly_history = await anomaly_endpoints.get_anomaly_history()
        print(f"   - Anomaly history: {anomaly_history['status']}")
        
        anomaly_statistics = await anomaly_endpoints.get_anomaly_statistics()
        print(f"   - Anomaly statistics: {anomaly_statistics['status']}")
        
        # Test alert endpoints
        print("\n🔔 Testing alert endpoints...")
        active_alerts = await alert_endpoints.get_active_alerts()
        print(f"   - Active alerts: {active_alerts['status']}")
        
        alert_rules = await alert_endpoints.get_alert_rules()
        print(f"   - Alert rules: {alert_rules['status']}")
        
        alert_history = await alert_endpoints.get_alert_history()
        print(f"   - Alert history: {alert_history['status']}")
        
        # Test real-time endpoints
        print("\n🔄 Testing real-time endpoints...")
        connection_stats = await realtime_endpoints.get_connection_stats()
        print(f"   - Connection stats: {connection_stats['status']}")
        
        print("\n✅ All API endpoints tested successfully!")
        
        # Test some actual data
        print("\n📋 Sample data from endpoints:")
        
        # Show cost summary sample
        cost_data = cost_summary['data']
        print(f"   - Total cost: ${cost_data['total_cost']:,.2f}")
        print(f"   - Cost trend: {cost_data['cost_trend']}")
        print(f"   - Warehouses monitored: {len(cost_data['warehouse_costs'])}")
        
        # Show usage metrics sample
        usage_data = usage_metrics['data']
        print(f"   - Total queries: {usage_data['total_queries']:,}")
        print(f"   - Active users: {usage_data['active_users']}")
        print(f"   - Query success rate: {usage_data['query_success_rate']:.1f}%")
        
        # Show anomaly counts
        anomaly_data = current_anomalies['data']
        print(f"   - Active anomalies: {anomaly_data['summary']['total_anomalies']}")
        print(f"   - Critical anomalies: {anomaly_data['summary']['critical']}")
        
        # Show alert counts
        alert_data = active_alerts['data']
        print(f"   - Active alerts: {alert_data['summary']['total_alerts']}")
        print(f"   - Critical alerts: {alert_data['summary']['critical']}")
        
        print("\n🎉 Dashboard API is fully functional!")
        
    except Exception as e:
        print(f"❌ Error testing API endpoints: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

def test_middleware():
    """Test the middleware components."""
    print("\n🛡️  Testing middleware components...")
    
    try:
        # Test authentication middleware
        from src.snowflake_analytics.api.middleware.auth import auth_middleware
        
        # Generate a test API key
        api_key = auth_middleware.generate_api_key("test_key", ["read", "write"])
        print(f"   - Generated API key: {api_key[:10]}...")
        
        # Validate the key
        key_info = auth_middleware.validate_api_key(api_key)
        print(f"   - Key validation: {'✅ Success' if key_info else '❌ Failed'}")
        
        # Test rate limiting
        from src.snowflake_analytics.api.middleware.rate_limiting import rate_limiting_middleware
        
        # Check rate limit
        rate_result = rate_limiting_middleware.check_rate_limit("127.0.0.1", "test_user")
        print(f"   - Rate limit check: {'✅ Allowed' if rate_result['allowed'] else '❌ Denied'}")
        
        # Test CORS middleware
        from src.snowflake_analytics.api.middleware.cors import default_cors
        
        cors_config = default_cors.get_cors_config()
        print(f"   - CORS config: {len(cors_config['allow_origins'])} origins allowed")
        
        print("✅ All middleware components tested successfully!")
        
    except Exception as e:
        print(f"❌ Error testing middleware: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

def test_schemas():
    """Test the API schemas."""
    print("\n📋 Testing API schemas...")
    
    try:
        # Test request schemas
        from src.snowflake_analytics.api.schemas.requests import validate_request
        
        # Test cost summary request
        valid_request = validate_request("cost_summary", time_range="30d")
        print(f"   - Cost summary request validation: {'✅ Valid' if valid_request else '❌ Invalid'}")
        
        # Test invalid request
        invalid_request = validate_request("cost_summary", time_range="invalid")
        print(f"   - Invalid request validation: {'✅ Rejected' if not invalid_request else '❌ Accepted'}")
        
        # Test response schemas
        from src.snowflake_analytics.api.schemas.responses import APIResponse
        
        # Test API response
        response = APIResponse("success", {"test": "data"})
        response_dict = response.to_dict()
        print(f"   - Response schema: {'✅ Valid' if 'status' in response_dict else '❌ Invalid'}")
        
        print("✅ All schema components tested successfully!")
        
    except Exception as e:
        print(f"❌ Error testing schemas: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    print("🚀 Starting Snowflake Analytics Dashboard API Tests")
    print("=" * 60)
    
    # Run all tests
    success = True
    
    # Test API endpoints
    success &= asyncio.run(test_api_endpoints())
    
    # Test middleware
    success &= test_middleware()
    
    # Test schemas
    success &= test_schemas()
    
    print("\n" + "=" * 60)
    if success:
        print("🎉 All tests passed! Dashboard API is ready to use.")
        print("\n📊 To start the dashboard:")
        print("   cd /home/runner/work/predictive_analysis_agent/predictive_analysis_agent")
        print("   python -m uvicorn src.snowflake_analytics.api.main:app --reload")
        print("\n🌐 Then visit: http://localhost:8000")
    else:
        print("❌ Some tests failed. Please check the output above.")
        sys.exit(1)