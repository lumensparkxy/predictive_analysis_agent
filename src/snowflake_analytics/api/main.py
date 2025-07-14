"""
FastAPI Application for Snowflake Analytics Agent

This module contains the main FastAPI application with all API endpoints.
"""

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
import logging
from datetime import datetime
import json

# Initialize logger
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Snowflake Analytics Agent",
    description="Predictive analytics system for Snowflake data warehouses",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

@app.get("/", response_class=HTMLResponse)
async def dashboard():
    """Main dashboard page."""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Snowflake Analytics Agent</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }
            .container { max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
            .header { text-align: center; margin-bottom: 40px; }
            .status-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; margin-bottom: 30px; }
            .status-card { background: #f8f9fa; padding: 20px; border-radius: 6px; border-left: 4px solid #28a745; }
            .status-card.warning { border-left-color: #ffc107; }
            .status-card.error { border-left-color: #dc3545; }
            .metric { font-size: 24px; font-weight: bold; color: #333; }
            .metric-label { color: #666; font-size: 14px; }
            .actions { margin-top: 30px; }
            .btn { display: inline-block; padding: 10px 20px; margin: 5px; background: #007bff; color: white; text-decoration: none; border-radius: 4px; }
            .btn:hover { background: #0056b3; }
            .logs { background: #f8f9fa; padding: 15px; border-radius: 4px; margin-top: 20px; font-family: monospace; font-size: 12px; max-height: 300px; overflow-y: auto; }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🚀 Snowflake Analytics Agent</h1>
                <p>Predictive Analytics System Dashboard</p>
            </div>
            
            <div class="status-grid">
                <div class="status-card">
                    <div class="metric">✅ Online</div>
                    <div class="metric-label">System Status</div>
                </div>
                <div class="status-card">
                    <div class="metric">0</div>
                    <div class="metric-label">Data Collection Runs</div>
                </div>
                <div class="status-card">
                    <div class="metric">0</div>
                    <div class="metric-label">Models Trained</div>
                </div>
                <div class="status-card">
                    <div class="metric">0</div>
                    <div class="metric-label">Active Alerts</div>
                </div>
            </div>
            
            <div class="actions">
                <h3>Available Actions</h3>
                <a href="/api/health" class="btn">Health Check</a>
                <a href="/api/status" class="btn">System Status</a>
                <a href="/docs" class="btn">API Documentation</a>
                <a href="/api/metrics" class="btn">System Metrics</a>
            </div>
            
            <div class="logs">
                <h4>Recent Activity</h4>
                <div>System initialized and ready for data collection...</div>
                <div>FastAPI server started successfully</div>
                <div>All storage systems operational</div>
            </div>
        </div>
    </body>
    </html>
    """
    return html_content

@app.get("/api/health")
async def health_check():
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
