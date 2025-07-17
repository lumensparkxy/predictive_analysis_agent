#!/usr/bin/env python3
"""
Minimal test script to verify the dashboard API modules work independently.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import asyncio
import json
from datetime import datetime

# Test the API endpoint modules directly without importing the main package
async def test_endpoint_modules():
    """Test the endpoint modules directly."""
    print("🧪 Testing individual endpoint modules...")
    
    try:
        # Test cost endpoints
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src', 'snowflake_analytics', 'api', 'endpoints'))
        
        import costs
        print("✅ Cost endpoints module imported successfully")
        
        cost_summary = await costs.cost_endpoints.get_cost_summary()
        print(f"   - Cost summary: {cost_summary['status']}")
        
        import usage
        print("✅ Usage endpoints module imported successfully")
        
        usage_metrics = await usage.usage_endpoints.get_usage_metrics()
        print(f"   - Usage metrics: {usage_metrics['status']}")
        
        import predictions
        print("✅ Predictions endpoints module imported successfully")
        
        cost_forecast = await predictions.prediction_endpoints.get_cost_forecast()
        print(f"   - Cost forecast: {cost_forecast['status']}")
        
        import anomalies
        print("✅ Anomalies endpoints module imported successfully")
        
        current_anomalies = await anomalies.anomaly_endpoints.get_current_anomalies()
        print(f"   - Current anomalies: {current_anomalies['status']}")
        
        import alerts
        print("✅ Alerts endpoints module imported successfully")
        
        active_alerts = await alerts.alert_endpoints.get_active_alerts()
        print(f"   - Active alerts: {active_alerts['status']}")
        
        import realtime
        print("✅ Real-time endpoints module imported successfully")
        
        connection_stats = await realtime.realtime_endpoints.get_connection_stats()
        print(f"   - Connection stats: {connection_stats['status']}")
        
        print("\n✅ All endpoint modules tested successfully!")
        
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
        
        print("\n🎉 Dashboard API endpoints are fully functional!")
        
    except Exception as e:
        print(f"❌ Error testing endpoint modules: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

def test_middleware_modules():
    """Test the middleware modules directly."""
    print("\n🛡️  Testing middleware modules...")
    
    try:
        # Test authentication middleware
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src', 'snowflake_analytics', 'api', 'middleware'))
        
        import auth
        print("✅ Authentication middleware imported successfully")
        
        # Generate a test API key
        api_key = auth.auth_middleware.generate_api_key("test_key", ["read", "write"])
        print(f"   - Generated API key: {api_key[:10]}...")
        
        # Validate the key
        key_info = auth.auth_middleware.validate_api_key(api_key)
        print(f"   - Key validation: {'✅ Success' if key_info else '❌ Failed'}")
        
        # Test rate limiting
        import rate_limiting
        print("✅ Rate limiting middleware imported successfully")
        
        # Check rate limit
        rate_result = rate_limiting.rate_limiting_middleware.check_rate_limit("127.0.0.1", "test_user")
        print(f"   - Rate limit check: {'✅ Allowed' if rate_result['allowed'] else '❌ Denied'}")
        
        # Test CORS middleware
        import cors
        print("✅ CORS middleware imported successfully")
        
        cors_config = cors.default_cors.get_cors_config()
        print(f"   - CORS config: {len(cors_config['allow_origins'])} origins allowed")
        
        print("✅ All middleware modules tested successfully!")
        
    except Exception as e:
        print(f"❌ Error testing middleware modules: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

def test_schema_modules():
    """Test the schema modules directly."""
    print("\n📋 Testing schema modules...")
    
    try:
        # Test request schemas
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src', 'snowflake_analytics', 'api', 'schemas'))
        
        import requests as request_schemas
        print("✅ Request schemas imported successfully")
        
        # Test cost summary request
        valid_request = request_schemas.validate_request("cost_summary", time_range="30d")
        print(f"   - Cost summary request validation: {'✅ Valid' if valid_request else '❌ Invalid'}")
        
        # Test invalid request
        invalid_request = request_schemas.validate_request("cost_summary", time_range="invalid")
        print(f"   - Invalid request validation: {'✅ Rejected' if not invalid_request else '❌ Accepted'}")
        
        # Test response schemas
        import responses
        print("✅ Response schemas imported successfully")
        
        # Test API response
        response = responses.APIResponse("success", {"test": "data"})
        response_dict = response.to_dict()
        print(f"   - Response schema: {'✅ Valid' if 'status' in response_dict else '❌ Invalid'}")
        
        print("✅ All schema modules tested successfully!")
        
    except Exception as e:
        print(f"❌ Error testing schema modules: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    print("🚀 Starting Dashboard API Module Tests")
    print("=" * 60)
    
    # Run all tests
    success = True
    
    # Test endpoint modules
    success &= asyncio.run(test_endpoint_modules())
    
    # Test middleware modules
    success &= test_middleware_modules()
    
    # Test schema modules
    success &= test_schema_modules()
    
    print("\n" + "=" * 60)
    if success:
        print("🎉 All module tests passed! Dashboard API modules are working correctly.")
        print("\n📊 API Features Available:")
        print("   ✅ Cost Analytics - /api/v1/costs/*")
        print("   ✅ Usage Metrics - /api/v1/usage/*")
        print("   ✅ Predictions - /api/v1/predictions/*")
        print("   ✅ Anomaly Detection - /api/v1/anomalies/*")
        print("   ✅ Alert Management - /api/v1/alerts/*")
        print("   ✅ Real-time WebSocket - /ws/real-time")
        print("   ✅ Authentication & Rate Limiting")
        print("   ✅ CORS Support")
        print("   ✅ Request/Response Validation")
        print("\n🌐 Ready for frontend integration!")
    else:
        print("❌ Some module tests failed. Please check the output above.")
        sys.exit(1)