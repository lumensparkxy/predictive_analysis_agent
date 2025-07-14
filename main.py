#!/usr/bin/env python3
"""
Snowflake Predictive Analytics Agent

Main application entry point for the Snowflake analytics system.
Provides CLI interface and FastAPI server startup.
"""

import logging
import sys
from pathlib import Path

import click
import uvicorn
from dotenv import load_dotenv

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from snowflake_analytics.utils.logger import setup_logging
from snowflake_analytics.config.settings import get_settings


def setup_environment():
    """Initialize environment and logging."""
    # Load environment variables
    load_dotenv()
    
    # Setup logging
    setup_logging()
    
    # Create necessary directories
    settings = get_settings()
    for directory in ["data/raw", "data/processed", "data/models", "data/exports", "cache", "logs"]:
        Path(directory).mkdir(parents=True, exist_ok=True)
    
    logging.info("Environment initialized successfully")


@click.group()
def cli():
    """Snowflake Predictive Analytics Agent CLI."""
    setup_environment()


@cli.command()
@click.option("--host", default="0.0.0.0", help="Host to bind the server to")
@click.option("--port", default=8000, help="Port to bind the server to")
@click.option("--reload", is_flag=True, help="Enable auto-reload for development")
def serve(host: str, port: int, reload: bool):
    """Start the FastAPI web server."""
    click.echo(f"Starting Snowflake Analytics Server on {host}:{port}")
    
    # Import here to avoid circular imports
    from snowflake_analytics.api.main import app
    
    uvicorn.run(
        "snowflake_analytics.api.main:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info"
    )


@cli.command()
def collect():
    """Manually trigger data collection from Snowflake."""
    click.echo("Starting data collection...")
    
    from snowflake_analytics.data_collection.collector import SnowflakeCollector
    
    collector = SnowflakeCollector()
    try:
        result = collector.collect_all()
        click.echo(f"Data collection completed: {result}")
    except Exception as e:
        click.echo(f"Data collection failed: {e}", err=True)
        sys.exit(1)


@cli.command()
def train():
    """Train ML models on collected data."""
    click.echo("Starting model training...")
    
    from snowflake_analytics.models.trainer import ModelTrainer
    
    trainer = ModelTrainer()
    try:
        result = trainer.train_all_models()
        click.echo(f"Model training completed: {result}")
    except Exception as e:
        click.echo(f"Model training failed: {e}", err=True)
        sys.exit(1)


@cli.command()
def status():
    """Show system status and health check."""
    click.echo("Checking system status...")
    
    from snowflake_analytics.utils.health_check import HealthChecker
    
    health_checker = HealthChecker()
    status = health_checker.check_all()
    
    click.echo("System Status:")
    for component, result in status.items():
        status_icon = "✅" if result["healthy"] else "❌"
        click.echo(f"  {status_icon} {component}: {result['message']}")
    
    overall_healthy = all(result["healthy"] for result in status.values())
    if not overall_healthy:
        sys.exit(1)


@cli.command()
def init():
    """Initialize the application (create database, setup configs)."""
    click.echo("Initializing Snowflake Analytics Agent...")
    
    from snowflake_analytics.storage.sqlite_store import SQLiteStore
    from snowflake_analytics.config.settings import get_settings
    
    try:
        # Initialize database
        store = SQLiteStore()
        store.initialize_database()
        click.echo("✅ Database initialized")
        
        # Validate configuration
        settings = get_settings()
        click.echo("✅ Configuration loaded")
        
        # Test directory structure
        for directory in ["data/raw", "data/processed", "data/models", "data/exports", "cache", "logs"]:
            Path(directory).mkdir(parents=True, exist_ok=True)
        click.echo("✅ Directory structure created")
        
        click.echo("\n🎉 Initialization completed successfully!")
        click.echo("\nNext steps:")
        click.echo("1. Copy .env.example to .env and configure your Snowflake credentials")
        click.echo("2. Run 'python main.py status' to verify everything is working")
        click.echo("3. Run 'python main.py serve' to start the web interface")
        
    except Exception as e:
        click.echo(f"❌ Initialization failed: {e}", err=True)
        sys.exit(1)


if __name__ == "__main__":
    cli()
