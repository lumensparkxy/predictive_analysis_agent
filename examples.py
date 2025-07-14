"""
Example usage and quick start for the Snowflake Data Collection System.

This module demonstrates how to use the data collection system and provides
sample code for common operations.
"""

import time
from datetime import datetime

from src.snowflake_analytics import (
    create_service, start_service, stop_service, get_service
)
from src.snowflake_analytics.utils.logger import get_logger

logger = get_logger(__name__)


def quick_start_example():
    """
    Quick start example showing basic service operations.
    
    This example demonstrates:
    1. Starting the data collection service
    2. Checking service status
    3. Running a manual collection
    4. Viewing collection results
    5. Stopping the service
    """
    print("🚀 Snowflake Data Collection System - Quick Start")
    print("=" * 50)
    
    # 1. Start the service
    print("\n1. Starting data collection service...")
    success = start_service()
    
    if not success:
        print("❌ Failed to start service. Check configuration and logs.")
        return False
    
    print("✅ Service started successfully!")
    
    # 2. Get service instance and check status
    print("\n2. Checking service status...")
    service = get_service()
    status = service.get_service_status()
    
    print(f"   Service Status: {status['service_status']}")
    print(f"   Running: {status['is_running']}")
    print(f"   Start Time: {status['start_time']}")
    
    # 3. Check system health
    print("\n3. Checking system health...")
    health = service.get_health_status()
    print(f"   Overall Health: {health.get('overall_status', 'unknown')}")
    print(f"   Active Alerts: {health.get('active_alerts', 0)}")
    
    # 4. Run a manual collection
    print("\n4. Running manual data collection...")
    result = service.run_job_now('usage_collection')
    
    if result.get('success'):
        print("✅ Manual collection completed successfully!")
    else:
        print(f"❌ Manual collection failed: {result.get('error')}")
    
    # 5. Get collection summary
    print("\n5. Collection summary (last 1 hour)...")
    summary = service.get_collection_summary(hours_back=1)
    
    if 'error' not in summary:
        print(f"   Total Jobs: {summary.get('total_jobs', 0)}")
        print(f"   Active Jobs: {summary.get('active_jobs', 0)}")
        print(f"   Recent Runs: {summary.get('recent_runs', 0)}")
        print(f"   Recent Errors: {summary.get('recent_errors', 0)}")
        print(f"   Collection Health: {summary.get('collection_health', 'unknown')}")
    
    # 6. Show scheduled jobs
    print("\n6. Scheduled jobs status...")
    job_status = service.get_service_status().get('components', {}).get('scheduler', {})
    
    if isinstance(job_status, dict) and 'jobs' in job_status:
        for job_name, details in job_status['jobs'].items():
            enabled = "✅" if details.get('enabled') else "❌"
            print(f"   {enabled} {job_name}: {details.get('schedule_pattern', 'N/A')}")
    
    print("\n7. Service is now running automated collections every 15 minutes!")
    print("   Use Ctrl+C to stop or call stop_service() programmatically.")
    
    return True


def monitoring_example():
    """
    Example showing monitoring and dashboard capabilities.
    """
    print("\n📊 Monitoring Dashboard Example")
    print("=" * 40)
    
    service = get_service()
    
    if not service.is_running:
        print("Service is not running. Please start it first.")
        return
    
    # Get comprehensive dashboard data
    dashboard_data = service.get_dashboard_data()
    
    # System Health Overview
    print("\n🏥 System Health:")
    health = dashboard_data.get('system_health', {})
    print(f"   Overall Status: {health.get('overall_status', 'unknown').upper()}")
    
    component_statuses = health.get('component_statuses', {})
    for component, status in component_statuses.items():
        status_emoji = "✅" if status == "operational" else "⚠️" if status == "degraded" else "❌"
        print(f"   {status_emoji} {component.title()}: {status}")
    
    # Active Alerts
    print(f"\n🚨 Active Alerts: {health.get('active_alerts', 0)}")
    active_alerts = dashboard_data.get('active_alerts', [])
    for alert in active_alerts[:5]:  # Show first 5 alerts
        severity_emoji = "🔴" if alert['severity'] == 'critical' else "🟡" if alert['severity'] == 'warning' else "🔵"
        print(f"   {severity_emoji} [{alert['severity'].upper()}] {alert['component']}: {alert['message']}")
    
    # Collection Summary
    print("\n📈 Collection Activity:")
    collection_summary = dashboard_data.get('collection_summary', {})
    if 'error' not in collection_summary:
        print(f"   Recent Runs: {collection_summary.get('recent_runs', 0)}")
        print(f"   Recent Errors: {collection_summary.get('recent_errors', 0)}")
        print(f"   Jobs with Errors: {collection_summary.get('jobs_with_errors', 0)}")
        
        health_emoji = "✅" if collection_summary.get('collection_health') == 'healthy' else "⚠️"
        print(f"   {health_emoji} Collection Health: {collection_summary.get('collection_health', 'unknown')}")
    
    # Recent Metrics
    print("\n📊 Recent Metrics:")
    recent_metrics = dashboard_data.get('recent_metrics', [])
    for metric in recent_metrics[-10:]:  # Show last 10 metrics
        print(f"   {metric['component']}.{metric['name']}: {metric['value']} {metric['unit']}")


def job_management_example():
    """
    Example showing job management capabilities.
    """
    print("\n⚙️ Job Management Example")
    print("=" * 35)
    
    service = get_service()
    
    if not service.is_running:
        print("Service is not running. Please start it first.")
        return
    
    # List all jobs
    print("\n📋 Available Jobs:")
    status = service.get_service_status()
    scheduler_status = status.get('components', {}).get('scheduler', {})
    
    if isinstance(scheduler_status, dict) and 'jobs' in scheduler_status:
        jobs = scheduler_status['jobs']
        
        for job_name, details in jobs.items():
            enabled = "✅ Enabled" if details.get('enabled') else "❌ Disabled"
            last_run = details.get('last_run', 'Never')
            if last_run != 'Never':
                last_run = datetime.fromisoformat(last_run).strftime('%Y-%m-%d %H:%M:%S')
            
            print(f"\n   📊 {job_name}")
            print(f"      Status: {enabled}")
            print(f"      Schedule: {details.get('schedule_pattern', 'N/A')}")
            print(f"      Last Run: {last_run}")
            print(f"      Run Count: {details.get('run_count', 0)}")
            print(f"      Error Count: {details.get('error_count', 0)}")
    
    # Example: Disable and re-enable a job
    print("\n🔧 Job Control Example:")
    test_job = 'user_activity_collection'
    
    print(f"   Disabling job: {test_job}")
    result = service.disable_job(test_job)
    print(f"   Result: {result.get('message', result)}")
    
    time.sleep(2)
    
    print(f"   Re-enabling job: {test_job}")
    result = service.enable_job(test_job)
    print(f"   Result: {result.get('message', result)}")
    
    # Example: Run a job manually
    print(f"\n▶️ Running job manually: {test_job}")
    result = service.run_job_now(test_job)
    
    if result.get('success'):
        print("   ✅ Job completed successfully!")
    else:
        print(f"   ❌ Job failed: {result.get('error')}")


def continuous_monitoring_example():
    """
    Example showing continuous monitoring.
    """
    print("\n🔄 Continuous Monitoring Example")
    print("=" * 40)
    print("   This will monitor the service for 60 seconds...")
    print("   Press Ctrl+C to stop early")
    
    service = get_service()
    
    try:
        for i in range(12):  # Monitor for 60 seconds (12 * 5 second intervals)
            print(f"\n⏰ Check #{i+1} ({datetime.now().strftime('%H:%M:%S')})")
            
            # Quick health check
            health = service.get_health_status()
            overall_status = health.get('overall_status', 'unknown')
            active_alerts = health.get('active_alerts', 0)
            
            status_emoji = "✅" if overall_status == 'healthy' else "⚠️" if overall_status == 'warning' else "❌"
            print(f"   {status_emoji} Health: {overall_status.upper()}")
            
            if active_alerts > 0:
                print(f"   🚨 Active Alerts: {active_alerts}")
            
            # Show active jobs
            status = service.get_service_status()
            job_queue_status = status.get('components', {}).get('job_queue', {})
            if isinstance(job_queue_status, dict):
                active_jobs = job_queue_status.get('active_jobs', 0)
                queue_size = job_queue_status.get('queue_size', 0)
                
                if active_jobs > 0 or queue_size > 0:
                    print(f"   🏃 Active Jobs: {active_jobs}, Queued: {queue_size}")
            
            time.sleep(5)
            
    except KeyboardInterrupt:
        print("\n   Monitoring stopped by user")


def main():
    """
    Main example runner.
    """
    print("🎯 Snowflake Data Collection System Examples")
    print("=" * 50)
    
    try:
        # Run quick start
        if quick_start_example():
            print("\n" + "=" * 50)
            print("✅ Quick start completed successfully!")
            
            # Wait a moment for initial collections
            print("\n⏳ Waiting for initial collections to start...")
            time.sleep(10)
            
            # Run monitoring examples
            monitoring_example()
            job_management_example()
            
            # Ask user if they want continuous monitoring
            try:
                response = input("\nWould you like to run continuous monitoring? (y/N): ")
                if response.lower().startswith('y'):
                    continuous_monitoring_example()
            except KeyboardInterrupt:
                pass
            
            print("\n🎉 Examples completed!")
            print("\nThe service is still running in the background.")
            print("Use stop_service() to stop it, or let it run for automated collections.")
            
    except KeyboardInterrupt:
        print("\n\n👋 Examples interrupted by user")
    except Exception as e:
        print(f"\n❌ Error running examples: {e}")
        logger.error(f"Example error: {e}")
    finally:
        # Option to stop service
        try:
            response = input("\nWould you like to stop the service? (y/N): ")
            if response.lower().startswith('y'):
                print("\n🛑 Stopping service...")
                stop_service()
                print("✅ Service stopped")
        except (KeyboardInterrupt, EOFError):
            print("\n👋 Goodbye!")


if __name__ == '__main__':
    main()
