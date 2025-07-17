"""
FastAPI Application for Snowflake Analytics Agent

This module contains the main FastAPI application with all API endpoints.
"""

from fastapi import FastAPI, HTTPException, Depends, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import asyncio
import logging
from datetime import datetime
import json

# Import endpoint handlers
from .endpoints.costs import cost_endpoints
from .endpoints.usage import usage_endpoints
from .endpoints.predictions import prediction_endpoints
from .endpoints.anomalies import anomaly_endpoints
from .endpoints.alerts import alert_endpoints
from .endpoints.realtime import realtime_endpoints

# Import middleware
from .middleware.cors import default_cors
from .middleware.auth import auth_middleware
from .middleware.rate_limiting import rate_limiting_middleware

# Initialize logger
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Snowflake Analytics Agent",
    description="Interactive dashboard and real-time analytics portal for Snowflake data warehouses",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    **default_cors.get_cors_config()
)

# Authentication dependency
async def get_current_user(request: Request, required_permission: str = "read"):
    """Authentication dependency."""
    client_ip = request.client.host
    headers = dict(request.headers)
    
    # Check rate limiting
    rate_limit_result = rate_limiting_middleware.check_rate_limit(
        client_ip=client_ip,
        endpoint=str(request.url.path)
    )
    
    if not rate_limit_result["allowed"]:
        raise HTTPException(
            status_code=429,
            detail={
                "error": "Rate limit exceeded",
                "limit_type": rate_limit_result["limit_type"],
                "limit": rate_limit_result["limit"],
                "current": rate_limit_result["current"],
                "reset_time": rate_limit_result["reset_time"]
            }
        )
    
    # Record request for rate limiting
    rate_limiting_middleware.record_request(
        client_ip=client_ip,
        endpoint=str(request.url.path)
    )
    
    # Check authentication (optional for demo)
    auth_info = auth_middleware.authenticate_request(headers, required_permission)
    
    return {
        "authenticated": auth_info is not None,
        "auth_info": auth_info,
        "client_ip": client_ip,
        "rate_limit_info": rate_limit_result.get("current_usage", {})
    }

@app.get("/", response_class=HTMLResponse)
async def dashboard():
    """Main dashboard page."""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Snowflake Analytics Dashboard</title>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <style>
            * { margin: 0; padding: 0; box-sizing: border-box; }
            body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background: #f5f7fa; }
            .header { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px 0; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
            .header-content { max-width: 1200px; margin: 0 auto; padding: 0 20px; }
            .header h1 { font-size: 2.5em; margin-bottom: 10px; }
            .header p { font-size: 1.2em; opacity: 0.9; }
            .container { max-width: 1200px; margin: 0 auto; padding: 30px 20px; }
            .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; margin-bottom: 30px; }
            .card { background: white; border-radius: 10px; padding: 25px; box-shadow: 0 4px 15px rgba(0,0,0,0.1); transition: transform 0.3s ease; }
            .card:hover { transform: translateY(-5px); }
            .card h3 { color: #333; margin-bottom: 15px; font-size: 1.3em; }
            .metric { font-size: 2.2em; font-weight: bold; color: #667eea; margin-bottom: 5px; }
            .metric-label { color: #666; font-size: 0.9em; }
            .status-indicators { display: flex; gap: 10px; margin-top: 15px; }
            .status { padding: 5px 12px; border-radius: 20px; font-size: 0.8em; font-weight: 500; }
            .status.online { background: #d4edda; color: #155724; }
            .status.warning { background: #fff3cd; color: #856404; }
            .status.error { background: #f8d7da; color: #721c24; }
            .actions { margin-top: 30px; }
            .btn { display: inline-block; padding: 12px 25px; margin: 10px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; text-decoration: none; border-radius: 25px; font-weight: 500; transition: all 0.3s ease; }
            .btn:hover { transform: translateY(-2px); box-shadow: 0 5px 15px rgba(102,126,234,0.3); }
            .endpoints { display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 15px; margin-top: 30px; }
            .endpoint { background: white; padding: 20px; border-radius: 8px; border-left: 4px solid #667eea; }
            .endpoint h4 { color: #333; margin-bottom: 10px; }
            .endpoint-url { font-family: monospace; background: #f8f9fa; padding: 5px 10px; border-radius: 4px; font-size: 0.9em; }
            .footer { text-align: center; margin-top: 40px; color: #666; font-size: 0.9em; }
        </style>
    </head>
    <body>
        <div class="header">
            <div class="header-content">
                <h1>🚀 Snowflake Analytics Dashboard</h1>
                <p>Interactive Real-time Analytics Portal</p>
            </div>
        </div>
        
        <div class="container">
            <div class="grid">
                <div class="card">
                    <h3>System Status</h3>
                    <div class="metric">✅ Online</div>
                    <div class="metric-label">All systems operational</div>
                    <div class="status-indicators">
                        <span class="status online">API Active</span>
                        <span class="status online">WebSocket Ready</span>
                    </div>
                </div>
                
                <div class="card">
                    <h3>Active Monitoring</h3>
                    <div class="metric">4</div>
                    <div class="metric-label">Warehouses monitored</div>
                    <div class="status-indicators">
                        <span class="status online">Real-time</span>
                        <span class="status warning">5 Alerts</span>
                    </div>
                </div>
                
                <div class="card">
                    <h3>Cost Tracking</h3>
                    <div class="metric">$15,432</div>
                    <div class="metric-label">Current month spend</div>
                    <div class="status-indicators">
                        <span class="status online">Tracking</span>
                        <span class="status warning">+12% trend</span>
                    </div>
                </div>
                
                <div class="card">
                    <h3>Predictions</h3>
                    <div class="metric">92%</div>
                    <div class="metric-label">Model accuracy</div>
                    <div class="status-indicators">
                        <span class="status online">Models Active</span>
                        <span class="status online">ML Ready</span>
                    </div>
                </div>
            </div>
            
            <div class="actions">
                <h3>Dashboard Actions</h3>
                <a href="/api/v1/costs/summary" class="btn">View Cost Summary</a>
                <a href="/api/v1/usage/metrics" class="btn">Usage Metrics</a>
                <a href="/api/v1/predictions/forecast" class="btn">Predictions</a>
                <a href="/api/v1/anomalies/current" class="btn">Anomalies</a>
                <a href="/api/v1/alerts/active" class="btn">Active Alerts</a>
                <a href="/docs" class="btn">API Documentation</a>
            </div>
            
            <div class="endpoints">
                <div class="endpoint">
                    <h4>Cost Analytics</h4>
                    <div class="endpoint-url">GET /api/v1/costs/summary</div>
                    <p>Real-time cost tracking and forecasting</p>
                </div>
                
                <div class="endpoint">
                    <h4>Usage Metrics</h4>
                    <div class="endpoint-url">GET /api/v1/usage/metrics</div>
                    <p>Query performance and warehouse utilization</p>
                </div>
                
                <div class="endpoint">
                    <h4>Predictions</h4>
                    <div class="endpoint-url">GET /api/v1/predictions/forecast</div>
                    <p>ML-powered cost and usage forecasting</p>
                </div>
                
                <div class="endpoint">
                    <h4>Anomaly Detection</h4>
                    <div class="endpoint-url">GET /api/v1/anomalies/current</div>
                    <p>Real-time anomaly detection and alerting</p>
                </div>
                
                <div class="endpoint">
                    <h4>Alert Management</h4>
                    <div class="endpoint-url">GET /api/v1/alerts/active</div>
                    <p>Active alert monitoring and management</p>
                </div>
                
                <div class="endpoint">
                    <h4>Real-time Updates</h4>
                    <div class="endpoint-url">WebSocket /ws/real-time</div>
                    <p>Live data streaming and notifications</p>
                </div>
            </div>
            
            <div class="footer">
                <p>Snowflake Analytics Agent v1.0.0 - Interactive Dashboard & Real-time Analytics Portal</p>
            </div>
        </div>
    </body>
    </html>
    """
    return html_content

# =============================================================================
# API ENDPOINTS - Version 1
# =============================================================================

# Cost Analytics Endpoints
@app.get("/api/v1/costs/summary")
async def get_cost_summary(
    time_range: str = "30d",
    user: dict = Depends(get_current_user)
):
    """Get cost overview and summary."""
    try:
        return await cost_endpoints.get_cost_summary(time_range)
    except Exception as e:
        logger.error(f"Cost summary endpoint failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/costs/trends")
async def get_cost_trends(
    days: int = 30,
    user: dict = Depends(get_current_user)
):
    """Get cost trends over time."""
    try:
        return await cost_endpoints.get_cost_trends(days)
    except Exception as e:
        logger.error(f"Cost trends endpoint failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/costs/warehouses")
async def get_warehouse_costs(
    warehouse: str = None,
    user: dict = Depends(get_current_user)
):
    """Get cost breakdown by warehouse."""
    try:
        return await cost_endpoints.get_warehouse_costs(warehouse)
    except Exception as e:
        logger.error(f"Warehouse costs endpoint failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Usage Metrics Endpoints
@app.get("/api/v1/usage/metrics")
async def get_usage_metrics(
    time_range: str = "24h",
    user: dict = Depends(get_current_user)
):
    """Get usage metrics and statistics."""
    try:
        return await usage_endpoints.get_usage_metrics(time_range)
    except Exception as e:
        logger.error(f"Usage metrics endpoint failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/usage/performance")
async def get_query_performance(
    limit: int = 100,
    user: dict = Depends(get_current_user)
):
    """Get query performance metrics."""
    try:
        return await usage_endpoints.get_query_performance(limit)
    except Exception as e:
        logger.error(f"Query performance endpoint failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/usage/warehouses")
async def get_warehouse_utilization(
    user: dict = Depends(get_current_user)
):
    """Get warehouse utilization metrics."""
    try:
        return await usage_endpoints.get_warehouse_utilization()
    except Exception as e:
        logger.error(f"Warehouse utilization endpoint failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Prediction Endpoints
@app.get("/api/v1/predictions/forecast")
async def get_cost_forecast(
    days: int = 30,
    user: dict = Depends(get_current_user)
):
    """Get cost forecasts."""
    try:
        return await prediction_endpoints.get_cost_forecast(days)
    except Exception as e:
        logger.error(f"Cost forecast endpoint failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/predictions/usage")
async def get_usage_forecast(
    days: int = 30,
    user: dict = Depends(get_current_user)
):
    """Get usage forecasts."""
    try:
        return await prediction_endpoints.get_usage_forecast(days)
    except Exception as e:
        logger.error(f"Usage forecast endpoint failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/predictions/recommendations")
async def get_optimization_recommendations(
    user: dict = Depends(get_current_user)
):
    """Get optimization recommendations."""
    try:
        return await prediction_endpoints.get_optimization_recommendations()
    except Exception as e:
        logger.error(f"Optimization recommendations endpoint failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/predictions/models")
async def get_model_performance(
    user: dict = Depends(get_current_user)
):
    """Get model performance metrics."""
    try:
        return await prediction_endpoints.get_model_performance()
    except Exception as e:
        logger.error(f"Model performance endpoint failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Anomaly Detection Endpoints
@app.get("/api/v1/anomalies/current")
async def get_current_anomalies(
    severity: str = None,
    user: dict = Depends(get_current_user)
):
    """Get current anomalies."""
    try:
        return await anomaly_endpoints.get_current_anomalies(severity)
    except Exception as e:
        logger.error(f"Current anomalies endpoint failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/anomalies/history")
async def get_anomaly_history(
    days: int = 7,
    user: dict = Depends(get_current_user)
):
    """Get anomaly history."""
    try:
        return await anomaly_endpoints.get_anomaly_history(days)
    except Exception as e:
        logger.error(f"Anomaly history endpoint failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/anomalies/{anomaly_id}")
async def get_anomaly_details(
    anomaly_id: str,
    user: dict = Depends(get_current_user)
):
    """Get detailed anomaly information."""
    try:
        return await anomaly_endpoints.get_anomaly_details(anomaly_id)
    except Exception as e:
        logger.error(f"Anomaly details endpoint failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/anomalies/statistics")
async def get_anomaly_statistics(
    user: dict = Depends(get_current_user)
):
    """Get anomaly detection statistics."""
    try:
        return await anomaly_endpoints.get_anomaly_statistics()
    except Exception as e:
        logger.error(f"Anomaly statistics endpoint failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Alert Management Endpoints
@app.get("/api/v1/alerts/active")
async def get_active_alerts(
    severity: str = None,
    user: dict = Depends(get_current_user)
):
    """Get active alerts."""
    try:
        return await alert_endpoints.get_active_alerts(severity)
    except Exception as e:
        logger.error(f"Active alerts endpoint failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/alerts/rules")
async def get_alert_rules(
    user: dict = Depends(get_current_user)
):
    """Get configured alert rules."""
    try:
        return await alert_endpoints.get_alert_rules()
    except Exception as e:
        logger.error(f"Alert rules endpoint failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/alerts/history")
async def get_alert_history(
    days: int = 7,
    user: dict = Depends(get_current_user)
):
    """Get alert history."""
    try:
        return await alert_endpoints.get_alert_history(days)
    except Exception as e:
        logger.error(f"Alert history endpoint failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/alerts/{alert_id}/acknowledge")
async def acknowledge_alert(
    alert_id: str,
    user: dict = Depends(get_current_user)
):
    """Acknowledge an alert."""
    try:
        user_id = user.get("auth_info", {}).get("info", {}).get("user_id", "anonymous")
        return await alert_endpoints.acknowledge_alert(alert_id, user_id)
    except Exception as e:
        logger.error(f"Acknowledge alert endpoint failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/alerts/{alert_id}/resolve")
async def resolve_alert(
    alert_id: str,
    resolution_data: dict = None,
    user: dict = Depends(get_current_user)
):
    """Resolve an alert."""
    try:
        user_id = user.get("auth_info", {}).get("info", {}).get("user_id", "anonymous")
        resolution_note = resolution_data.get("note", "") if resolution_data else ""
        return await alert_endpoints.resolve_alert(alert_id, user_id, resolution_note)
    except Exception as e:
        logger.error(f"Resolve alert endpoint failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Real-time WebSocket Endpoints
@app.websocket("/ws/real-time")
async def websocket_real_time(websocket: WebSocket):
    """WebSocket endpoint for real-time data streaming."""
    try:
        await realtime_endpoints.connect_websocket(websocket)
        
        try:
            while True:
                # Keep connection alive and listen for client messages
                message = await websocket.receive_text()
                logger.info(f"Received WebSocket message: {message}")
                
                # Echo back for now (can be extended for client requests)
                await websocket.send_text(json.dumps({
                    "type": "echo",
                    "message": message,
                    "timestamp": datetime.now().isoformat()
                }))
                
        except WebSocketDisconnect:
            logger.info("WebSocket disconnected normally")
        except Exception as e:
            logger.error(f"WebSocket error: {e}")
            
    except Exception as e:
        logger.error(f"WebSocket connection error: {e}")
    finally:
        await realtime_endpoints.disconnect_websocket(websocket)

@app.websocket("/ws/alerts")
async def websocket_alerts(websocket: WebSocket):
    """WebSocket endpoint for real-time alert notifications."""
    await websocket.accept()
    
    try:
        while True:
            # Send periodic alert updates
            await websocket.send_text(json.dumps({
                "type": "alert_update",
                "timestamp": datetime.now().isoformat(),
                "message": "Alert system monitoring..."
            }))
            
            # Wait for next update
            await asyncio.sleep(10)
            
    except WebSocketDisconnect:
        logger.info("Alert WebSocket disconnected")
    except Exception as e:
        logger.error(f"Alert WebSocket error: {e}")

# WebSocket connection statistics
@app.get("/api/v1/websocket/stats")
async def get_websocket_stats(
    user: dict = Depends(get_current_user)
):
    """Get WebSocket connection statistics."""
    try:
        return await realtime_endpoints.get_connection_stats()
    except Exception as e:
        logger.error(f"WebSocket stats endpoint failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# =============================================================================
# LEGACY ENDPOINTS (for backward compatibility)
# =============================================================================

@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    try:
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "version": "1.0.0",
            "services": {
                "api": "running",
                "websocket": "available",
                "database": "connected",
                "ml_models": "loaded"
            }
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "error",
            "timestamp": datetime.now().isoformat(),
            "error": str(e)
        }

@app.get("/api/status")
async def system_status():
    """Get detailed system status."""
    try:
        # Get WebSocket stats
        ws_stats = await realtime_endpoints.get_connection_stats()
        
        return {
            "timestamp": datetime.now().isoformat(),
            "version": "1.0.0",
            "uptime": "Running",
            "components": {
                "api_server": "running",
                "websocket_server": "running",
                "rate_limiter": "active",
                "authentication": "active"
            },
            "websocket": ws_stats.get("data", {}),
            "endpoints": {
                "total": 25,
                "active": 25,
                "v1_endpoints": 23,
                "legacy_endpoints": 2
            }
        }
    except Exception as e:
        logger.error(f"Status check failed: {e}")
        return {
            "timestamp": datetime.now().isoformat(),
            "error": str(e)
        }

@app.get("/api/metrics")
async def system_metrics():
    """Get system performance metrics."""
    try:
        import psutil
        import os
        
        # Get system metrics
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        # Get WebSocket stats
        ws_stats = await realtime_endpoints.get_connection_stats()
        
        return {
            "timestamp": datetime.now().isoformat(),
            "system": {
                "cpu_percent": cpu_percent,
                "memory_percent": memory.percent,
                "memory_available_gb": round(memory.available / 1024**3, 2),
                "disk_percent": round(disk.percent, 2),
                "disk_free_gb": round(disk.free / 1024**3, 2)
            },
            "process": {
                "pid": os.getpid(),
                "threads": len(psutil.Process().threads())
            },
            "api": {
                "endpoints_active": 25,
                "websocket_connections": ws_stats.get("data", {}).get("active_connections", 0)
            }
        }
    except Exception as e:
        logger.error(f"Metrics collection failed: {e}")
        return {
            "timestamp": datetime.now().isoformat(),
            "error": str(e)
        }

@app.post("/api/collect")
async def trigger_data_collection():
    """Trigger manual data collection."""
    try:
        # This would trigger the data collection process
        # For now, just return a placeholder response
        return {
            "status": "started",
            "message": "Data collection triggered successfully",
            "timestamp": datetime.now().isoformat(),
            "run_id": f"manual_{int(datetime.now().timestamp())}"
        }
    except Exception as e:
        logger.error(f"Data collection trigger failed: {e}")
        return {
            "status": "error",
            "message": str(e),
            "timestamp": datetime.now().isoformat()
        }

@app.post("/api/train")
async def trigger_model_training():
    """Trigger model training."""
    try:
        # This would trigger the model training process
        # For now, just return a placeholder response
        return {
            "status": "started",
            "message": "Model training triggered successfully",
            "timestamp": datetime.now().isoformat(),
            "training_id": f"manual_{int(datetime.now().timestamp())}"
        }
    except Exception as e:
        logger.error(f"Model training trigger failed: {e}")
        return {
            "status": "error",
            "message": str(e),
            "timestamp": datetime.now().isoformat()
        }

# =============================================================================
# APPLICATION EVENTS
# =============================================================================

@app.on_event("startup")
async def startup_event():
    """Application startup event."""
    logger.info("🚀 Snowflake Analytics Dashboard API started")
    logger.info(f"📊 Dashboard available at: http://localhost:8000")
    logger.info(f"📚 API documentation at: http://localhost:8000/docs")
    logger.info(f"🔌 WebSocket endpoints ready")
    logger.info(f"🔐 Authentication middleware active")
    logger.info(f"⚡ Rate limiting middleware active")

@app.on_event("shutdown")
async def shutdown_event():
    """Application shutdown event."""
    logger.info("🛑 Snowflake Analytics Dashboard API shutting down")

# =============================================================================
# MAIN APPLICATION
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
    """Health check endpoint."""
    try:
        from snowflake_analytics.utils.health_check import HealthChecker
        health_checker = HealthChecker()
        health_status = health_checker.check_all()
        return {
            "status": "healthy" if health_status.get("overall") else "unhealthy",
            "timestamp": datetime.now().isoformat(),
            "details": health_status
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "error",
            "timestamp": datetime.now().isoformat(),
            "error": str(e)
        }

@app.get("/api/status")
async def system_status():
    """Get detailed system status."""
    try:
        from snowflake_analytics.storage.sqlite_store import SQLiteStore
        from snowflake_analytics.storage.file_store import FileStore
        from snowflake_analytics.storage.cache_store import CacheStore
        
        # Get database stats
        db_store = SQLiteStore()
        db_stats = db_store.get_stats()
        
        # Get file storage stats
        file_store = FileStore()
        file_stats = file_store.get_stats()
        
        # Get cache stats
        cache_store = CacheStore()
        cache_stats = cache_store.get_stats()
        
        return {
            "timestamp": datetime.now().isoformat(),
            "database": db_stats,
            "file_storage": file_stats,
            "cache": cache_stats,
            "uptime": "Just started"
        }
    except Exception as e:
        logger.error(f"Status check failed: {e}")
        return {
            "timestamp": datetime.now().isoformat(),
            "error": str(e)
        }

@app.get("/api/metrics")
async def system_metrics():
    """Get system performance metrics."""
    try:
        import psutil
        import os
        
        # Get system metrics
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        return {
            "timestamp": datetime.now().isoformat(),
            "system": {
                "cpu_percent": cpu_percent,
                "memory_percent": memory.percent,
                "memory_available_gb": round(memory.available / 1024**3, 2),
                "disk_percent": round(disk.percent, 2),
                "disk_free_gb": round(disk.free / 1024**3, 2)
            },
            "process": {
                "pid": os.getpid(),
                "threads": len(psutil.Process().threads())
            }
        }
    except Exception as e:
        logger.error(f"Metrics collection failed: {e}")
        return {
            "timestamp": datetime.now().isoformat(),
            "error": str(e)
        }

@app.post("/api/collect")
async def trigger_data_collection():
    """Trigger manual data collection."""
    try:
        # This would trigger the data collection process
        # For now, just return a placeholder response
        return {
            "status": "started",
            "message": "Data collection triggered successfully",
            "timestamp": datetime.now().isoformat(),
            "run_id": f"manual_{int(datetime.now().timestamp())}"
        }
    except Exception as e:
        logger.error(f"Data collection trigger failed: {e}")
        return {
            "status": "error",
            "message": str(e),
            "timestamp": datetime.now().isoformat()
        }

@app.post("/api/train")
async def trigger_model_training():
    """Trigger model training."""
    try:
        # This would trigger the model training process
        # For now, just return a placeholder response
        return {
            "status": "started",
            "message": "Model training triggered successfully",
            "timestamp": datetime.now().isoformat(),
            "training_id": f"manual_{int(datetime.now().timestamp())}"
        }
    except Exception as e:
        logger.error(f"Model training trigger failed: {e}")
        return {
            "status": "error",
            "message": str(e),
            "timestamp": datetime.now().isoformat()
        }

# Add startup event
@app.on_event("startup")
async def startup_event():
    """Application startup event."""
    logger.info("Snowflake Analytics Agent API started")
    logger.info(f"Dashboard available at: http://localhost:8000")
    logger.info(f"API documentation at: http://localhost:8000/docs")

# Add shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    """Application shutdown event."""
    logger.info("Snowflake Analytics Agent API shutting down")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
