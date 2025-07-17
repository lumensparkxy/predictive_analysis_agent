#!/usr/bin/env python3
"""
Simple demonstration of the Snowflake Analytics Dashboard

This runs a basic version of the dashboard API without complex dependencies.
"""

import sys
import os
import asyncio
import json
from datetime import datetime
from typing import Dict, Any

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

try:
    from fastapi import FastAPI, Request
    from fastapi.responses import HTMLResponse
    from fastapi.middleware.cors import CORSMiddleware
    import uvicorn
    
    # Try to import our modules directly
    from snowflake_analytics.api.endpoints.costs import cost_endpoints
    from snowflake_analytics.api.endpoints.usage import usage_endpoints
    from snowflake_analytics.api.endpoints.predictions import prediction_endpoints
    from snowflake_analytics.api.endpoints.anomalies import anomaly_endpoints
    from snowflake_analytics.api.endpoints.alerts import alert_endpoints
    from snowflake_analytics.api.endpoints.realtime import realtime_endpoints
    
    # Create FastAPI app
    app = FastAPI(
        title="Snowflake Analytics Dashboard Demo",
        description="Interactive real-time analytics portal for Snowflake",
        version="1.0.0"
    )
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    @app.get("/", response_class=HTMLResponse)
    async def dashboard():
        """Enhanced dashboard with real-time features."""
        return """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Snowflake Analytics Dashboard</title>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <style>
                * { margin: 0; padding: 0; box-sizing: border-box; }
                body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #f8fafc; }
                .header { background: linear-gradient(135deg, #3b82f6, #8b5cf6); color: white; padding: 2rem 0; text-align: center; }
                .header h1 { font-size: 2.5rem; margin-bottom: 0.5rem; }
                .header p { font-size: 1.2rem; opacity: 0.9; }
                .container { max-width: 1200px; margin: 0 auto; padding: 2rem; }
                .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 1.5rem; margin-bottom: 2rem; }
                .card { background: white; border-radius: 12px; padding: 1.5rem; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1); transition: transform 0.2s; }
                .card:hover { transform: translateY(-2px); }
                .card h3 { color: #1f2937; margin-bottom: 1rem; font-size: 1.25rem; }
                .metric { font-size: 2rem; font-weight: bold; color: #3b82f6; margin-bottom: 0.5rem; }
                .metric-label { color: #6b7280; font-size: 0.875rem; }
                .status { display: inline-block; padding: 0.25rem 0.75rem; border-radius: 9999px; font-size: 0.75rem; font-weight: 500; }
                .status.online { background: #d1fae5; color: #065f46; }
                .status.warning { background: #fef3c7; color: #92400e; }
                .actions { display: flex; gap: 1rem; margin-top: 2rem; flex-wrap: wrap; }
                .btn { padding: 0.75rem 1.5rem; background: #3b82f6; color: white; text-decoration: none; border-radius: 8px; font-weight: 500; transition: background 0.2s; }
                .btn:hover { background: #2563eb; }
                .endpoints { display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 1rem; margin-top: 2rem; }
                .endpoint { background: white; padding: 1rem; border-radius: 8px; border-left: 4px solid #3b82f6; }
                .endpoint h4 { color: #1f2937; margin-bottom: 0.5rem; }
                .endpoint-url { font-family: monospace; background: #f3f4f6; padding: 0.25rem 0.5rem; border-radius: 4px; font-size: 0.8rem; }
                .live-data { background: #f0f9ff; border: 1px solid #e0f2fe; border-radius: 8px; padding: 1rem; margin-top: 1rem; }
                .live-indicator { display: inline-block; width: 8px; height: 8px; background: #10b981; border-radius: 50%; margin-right: 0.5rem; animation: pulse 2s infinite; }
                @keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.5; } }
                .footer { text-align: center; margin-top: 3rem; color: #6b7280; }
                
                @media (max-width: 768px) {
                    .header h1 { font-size: 2rem; }
                    .container { padding: 1rem; }
                    .grid { grid-template-columns: 1fr; }
                    .actions { flex-direction: column; }
                }
            </style>
        </head>
        <body>
            <div class="header">
                <h1>🚀 Snowflake Analytics Dashboard</h1>
                <p>Interactive Real-time Analytics Portal</p>
            </div>
            
            <div class="container">
                <div class="grid">
                    <div class="card">
                        <h3>💰 Cost Analytics</h3>
                        <div class="metric" id="total-cost">$15,432</div>
                        <div class="metric-label">Monthly spend</div>
                        <div class="status online">Tracking</div>
                    </div>
                    
                    <div class="card">
                        <h3>📊 Usage Metrics</h3>
                        <div class="metric" id="active-queries">12,450</div>
                        <div class="metric-label">Total queries</div>
                        <div class="status online">Monitoring</div>
                    </div>
                    
                    <div class="card">
                        <h3>🎯 Predictions</h3>
                        <div class="metric" id="forecast-accuracy">92%</div>
                        <div class="metric-label">Model accuracy</div>
                        <div class="status online">ML Active</div>
                    </div>
                    
                    <div class="card">
                        <h3>🚨 Alerts</h3>
                        <div class="metric" id="active-alerts">5</div>
                        <div class="metric-label">Active alerts</div>
                        <div class="status warning">Attention</div>
                    </div>
                    
                    <div class="card">
                        <h3>🔍 Anomalies</h3>
                        <div class="metric" id="anomalies-detected">3</div>
                        <div class="metric-label">Detected</div>
                        <div class="status warning">Investigating</div>
                    </div>
                    
                    <div class="card">
                        <h3>⚡ Real-time</h3>
                        <div class="live-data">
                            <div><span class="live-indicator"></span>Live Updates Active</div>
                            <div style="margin-top: 0.5rem; font-size: 0.875rem; color: #6b7280;" id="last-update">
                                Last update: <span id="timestamp">Just now</span>
                            </div>
                        </div>
                    </div>
                </div>
                
                <div class="actions">
                    <a href="/api/v1/costs/summary" class="btn">💰 View Cost Summary</a>
                    <a href="/api/v1/usage/metrics" class="btn">📊 Usage Metrics</a>
                    <a href="/api/v1/predictions/forecast" class="btn">🎯 Predictions</a>
                    <a href="/api/v1/anomalies/current" class="btn">🔍 Anomalies</a>
                    <a href="/api/v1/alerts/active" class="btn">🚨 Active Alerts</a>
                    <a href="/docs" class="btn">📚 API Docs</a>
                </div>
                
                <div class="endpoints">
                    <div class="endpoint">
                        <h4>💰 Cost Analytics</h4>
                        <div class="endpoint-url">GET /api/v1/costs/summary</div>
                        <p style="margin-top: 0.5rem; font-size: 0.875rem; color: #6b7280;">Real-time cost tracking</p>
                    </div>
                    
                    <div class="endpoint">
                        <h4>📊 Usage Metrics</h4>
                        <div class="endpoint-url">GET /api/v1/usage/metrics</div>
                        <p style="margin-top: 0.5rem; font-size: 0.875rem; color: #6b7280;">Query performance data</p>
                    </div>
                    
                    <div class="endpoint">
                        <h4>🎯 Predictions</h4>
                        <div class="endpoint-url">GET /api/v1/predictions/forecast</div>
                        <p style="margin-top: 0.5rem; font-size: 0.875rem; color: #6b7280;">ML-powered forecasting</p>
                    </div>
                    
                    <div class="endpoint">
                        <h4>🔍 Anomaly Detection</h4>
                        <div class="endpoint-url">GET /api/v1/anomalies/current</div>
                        <p style="margin-top: 0.5rem; font-size: 0.875rem; color: #6b7280;">Real-time anomaly detection</p>
                    </div>
                    
                    <div class="endpoint">
                        <h4>🚨 Alert Management</h4>
                        <div class="endpoint-url">GET /api/v1/alerts/active</div>
                        <p style="margin-top: 0.5rem; font-size: 0.875rem; color: #6b7280;">Active alert monitoring</p>
                    </div>
                    
                    <div class="endpoint">
                        <h4>⚡ Real-time Updates</h4>
                        <div class="endpoint-url">WebSocket /ws/real-time</div>
                        <p style="margin-top: 0.5rem; font-size: 0.875rem; color: #6b7280;">Live data streaming</p>
                    </div>
                </div>
                
                <div class="footer">
                    <p>Snowflake Analytics Dashboard v1.0.0 - Interactive Real-time Analytics Portal</p>
                    <p style="margin-top: 0.5rem;">Built with FastAPI, React, and modern web technologies</p>
                </div>
            </div>
            
            <script>
                // Update timestamp every second
                setInterval(() => {
                    document.getElementById('timestamp').textContent = new Date().toLocaleTimeString();
                }, 1000);
                
                // Simulate real-time data updates
                setInterval(async () => {
                    try {
                        // Fetch real data from API
                        const costResponse = await fetch('/api/v1/costs/summary');
                        const costData = await costResponse.json();
                        
                        const usageResponse = await fetch('/api/v1/usage/metrics');
                        const usageData = await usageResponse.json();
                        
                        const alertsResponse = await fetch('/api/v1/alerts/active');
                        const alertsData = await alertsResponse.json();
                        
                        const anomaliesResponse = await fetch('/api/v1/anomalies/current');
                        const anomaliesData = await anomaliesResponse.json();
                        
                        // Update UI with real data
                        if (costData.status === 'success') {
                            document.getElementById('total-cost').textContent = 
                                '$' + costData.data.total_cost.toLocaleString();
                        }
                        
                        if (usageData.status === 'success') {
                            document.getElementById('active-queries').textContent = 
                                usageData.data.total_queries.toLocaleString();
                        }
                        
                        if (alertsData.status === 'success') {
                            document.getElementById('active-alerts').textContent = 
                                alertsData.data.summary.total_alerts;
                        }
                        
                        if (anomaliesData.status === 'success') {
                            document.getElementById('anomalies-detected').textContent = 
                                anomaliesData.data.summary.total_anomalies;
                        }
                        
                    } catch (error) {
                        console.log('Using mock data due to API error:', error);
                    }
                }, 30000); // Update every 30 seconds
                
                // Initialize with real data
                window.addEventListener('load', () => {
                    // Trigger initial data fetch
                    setTimeout(() => {
                        const event = new Event('interval');
                        window.dispatchEvent(event);
                    }, 1000);
                });
            </script>
        </body>
        </html>
        """
    
    # API Endpoints
    @app.get("/api/v1/costs/summary")
    async def get_cost_summary():
        return await cost_endpoints.get_cost_summary()
    
    @app.get("/api/v1/usage/metrics")
    async def get_usage_metrics():
        return await usage_endpoints.get_usage_metrics()
    
    @app.get("/api/v1/predictions/forecast")
    async def get_predictions():
        return await prediction_endpoints.get_cost_forecast()
    
    @app.get("/api/v1/anomalies/current")
    async def get_anomalies():
        return await anomaly_endpoints.get_current_anomalies()
    
    @app.get("/api/v1/alerts/active")
    async def get_alerts():
        return await alert_endpoints.get_active_alerts()
    
    @app.get("/api/health")
    async def health_check():
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "version": "1.0.0",
            "message": "Dashboard API is running"
        }
    
    def run_dashboard():
        """Run the dashboard server."""
        print("🚀 Starting Snowflake Analytics Dashboard...")
        print("📊 Dashboard: http://localhost:8000")
        print("📚 API Docs: http://localhost:8000/docs")
        print("🔍 Health Check: http://localhost:8000/api/health")
        print("\nPress Ctrl+C to stop the server")
        
        try:
            uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
        except KeyboardInterrupt:
            print("\n✅ Dashboard stopped successfully!")
    
    if __name__ == "__main__":
        run_dashboard()
        
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("📦 Install required packages: pip install fastapi uvicorn")
    sys.exit(1)
    
except Exception as e:
    print(f"❌ Unexpected error: {e}")
    sys.exit(1)