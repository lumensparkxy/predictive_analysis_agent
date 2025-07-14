#!/usr/bin/env python3
"""
Main entry point for the Snowflake Data Collection System.

This script provides the main entry point for starting and managing
the automated Snowflake data collection service.

Usage:
    python main.py                          # Start interactive mode
    python main.py --daemon                 # Run as daemon
    python main.py --config ./config       # Custom config path
    python main.py --examples              # Run examples
    python main.py --status                # Check service status
    python main.py --stop                  # Stop running service
"""

import argparse
import sys
import os
from pathlib import Path

# Add src to path for imports
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from src.snowflake_analytics import (
    DataCollectionService, 
    start_service, 
    stop_service, 
    get_service,
    main as service_main
)
from src.snowflake_analytics.utils.logger import get_logger, setup_logging

logger = get_logger(__name__)


def interactive_mode():
    """Run in interactive mode with user prompts."""
    print("🚀 Snowflake Data Collection System")
    print("=" * 40)
    
    try:
        # Start service
        print("\n1. Starting data collection service...")
        success = start_service()
        
        if not success:
            print("❌ Failed to start service. Check configuration and logs.")
            return 1
        
        print("✅ Service started successfully!")
        
        service = get_service()
        
        # Show initial status
        print("\n2. Service Status:")
        status = service.get_service_status()
        print(f"   Running: {status['is_running']}")
        print(f"   Start Time: {status.get('start_time', 'N/A')}")
        
        # Main menu loop
        while True:
            print("\n" + "=" * 40)
            print("📋 Available Commands:")
            print("   1. Show service status")
            print("   2. Show system health")
            print("   3. Show collection summary")
            print("   4. Show job status")
            print("   5. Run job manually")
            print("   6. Enable/disable job")
            print("   7. Show dashboard data")
            print("   8. Run examples")
            print("   9. Stop service and exit")
            print("   0. Exit (keep service running)")
            
            try:
                choice = input("\nEnter choice (0-9): ").strip()
                
                if choice == '0':
                    print("Service will continue running in background.")
                    break
                elif choice == '1':
                    show_service_status(service)
                elif choice == '2':
                    show_system_health(service)
                elif choice == '3':
                    show_collection_summary(service)
                elif choice == '4':
                    show_job_status(service)
                elif choice == '5':
                    run_manual_job(service)
                elif choice == '6':
                    manage_job(service)
                elif choice == '7':
                    show_dashboard_data(service)
                elif choice == '8':
                    run_examples()
                elif choice == '9':
                    print("\n🛑 Stopping service...")
                    stop_service()
                    print("✅ Service stopped")
                    break
                else:
                    print("❌ Invalid choice. Please try again.")
                    
            except KeyboardInterrupt:
                print("\n\n👋 Exiting...")
                break
            except EOFError:
                print("\n\n👋 Exiting...")
                break
        
        return 0
        
    except Exception as e:
        logger.error(f"Error in interactive mode: {e}")
        print(f"❌ Error: {e}")
        return 1


def show_service_status(service):
    """Show detailed service status."""
    print("\n📊 Service Status:")
    print("-" * 20)
    
    status = service.get_service_status()
    
    print(f"Status: {status['service_status']}")
    print(f"Running: {status['is_running']}")
    print(f"Start Time: {status.get('start_time', 'N/A')}")
    
    uptime = status.get('uptime')
    if uptime:
        print(f"Uptime: {uptime['days']}d {uptime['hours']}h {uptime['minutes']}m")
    
    # Component status
    components = status.get('components', {})
    if components:
        print("\nComponent Status:")
        for component, comp_status in components.items():
            if isinstance(comp_status, dict):
                running = comp_status.get('is_running', comp_status.get('running', 'unknown'))
                print(f"  {component}: {running}")


def show_system_health(service):
    """Show system health information."""
    print("\n🏥 System Health:")
    print("-" * 20)
    
    health = service.get_health_status()
    
    if 'error' in health:
        print(f"❌ Error: {health['error']}")
        return
    
    overall = health.get('overall_status', 'unknown')
    status_emoji = "✅" if overall == 'healthy' else "⚠️" if overall == 'warning' else "❌"
    print(f"{status_emoji} Overall Status: {overall.upper()}")
    
    print(f"🚨 Active Alerts: {health.get('active_alerts', 0)}")
    
    # Component health
    components = health.get('component_statuses', {})
    if components:
        print("\nComponent Health:")
        for component, status in components.items():
            status_emoji = "✅" if status == "operational" else "⚠️" if status == "degraded" else "❌"
            print(f"  {status_emoji} {component}: {status}")


def show_collection_summary(service):
    """Show collection activity summary."""
    print("\n📈 Collection Summary:")
    print("-" * 25)
    
    summary = service.get_collection_summary(hours_back=6)
    
    if 'error' in summary:
        print(f"❌ Error: {summary['error']}")
        return
    
    print(f"Time Window: {summary.get('time_window_hours', 0)} hours")
    print(f"Total Jobs: {summary.get('total_jobs', 0)}")
    print(f"Active Jobs: {summary.get('active_jobs', 0)}")
    print(f"Recent Runs: {summary.get('recent_runs', 0)}")
    print(f"Recent Errors: {summary.get('recent_errors', 0)}")
    
    health = summary.get('collection_health', 'unknown')
    health_emoji = "✅" if health == 'healthy' else "⚠️" if health == 'warning' else "❌"
    print(f"{health_emoji} Collection Health: {health}")


def show_job_status(service):
    """Show detailed job status."""
    print("\n⚙️ Job Status:")
    print("-" * 15)
    
    status = service.get_service_status()
    scheduler_status = status.get('components', {}).get('scheduler', {})
    
    if not isinstance(scheduler_status, dict) or 'jobs' not in scheduler_status:
        print("❌ Job status not available")
        return
    
    jobs = scheduler_status['jobs']
    
    for job_name, details in jobs.items():
        enabled = "✅" if details.get('enabled') else "❌"
        print(f"\n{enabled} {job_name}")
        print(f"   Schedule: {details.get('schedule_pattern', 'N/A')}")
        print(f"   Runs: {details.get('run_count', 0)}")
        print(f"   Errors: {details.get('error_count', 0)}")
        
        last_run = details.get('last_run')
        if last_run:
            from datetime import datetime
            last_run_dt = datetime.fromisoformat(last_run)
            print(f"   Last Run: {last_run_dt.strftime('%Y-%m-%d %H:%M:%S')}")


def run_manual_job(service):
    """Run a job manually."""
    print("\n▶️ Manual Job Execution:")
    print("-" * 25)
    
    # Get available jobs
    status = service.get_service_status()
    scheduler_status = status.get('components', {}).get('scheduler', {})
    
    if not isinstance(scheduler_status, dict) or 'jobs' not in scheduler_status:
        print("❌ No jobs available")
        return
    
    jobs = list(scheduler_status['jobs'].keys())
    
    print("Available jobs:")
    for i, job_name in enumerate(jobs, 1):
        print(f"   {i}. {job_name}")
    
    try:
        choice = input(f"\nSelect job (1-{len(jobs)}): ").strip()
        job_index = int(choice) - 1
        
        if 0 <= job_index < len(jobs):
            job_name = jobs[job_index]
            print(f"\n🚀 Running job: {job_name}")
            
            result = service.run_job_now(job_name)
            
            if result.get('success'):
                print("✅ Job completed successfully!")
            else:
                print(f"❌ Job failed: {result.get('error')}")
        else:
            print("❌ Invalid selection")
            
    except (ValueError, KeyboardInterrupt):
        print("❌ Operation cancelled")


def manage_job(service):
    """Enable or disable a job."""
    print("\n🔧 Job Management:")
    print("-" * 20)
    
    # Get available jobs
    status = service.get_service_status()
    scheduler_status = status.get('components', {}).get('scheduler', {})
    
    if not isinstance(scheduler_status, dict) or 'jobs' not in scheduler_status:
        print("❌ No jobs available")
        return
    
    jobs = scheduler_status['jobs']
    job_names = list(jobs.keys())
    
    print("Available jobs:")
    for i, job_name in enumerate(job_names, 1):
        enabled = "✅" if jobs[job_name].get('enabled') else "❌"
        print(f"   {i}. {enabled} {job_name}")
    
    try:
        choice = input(f"\nSelect job (1-{len(job_names)}): ").strip()
        job_index = int(choice) - 1
        
        if 0 <= job_index < len(job_names):
            job_name = job_names[job_index]
            current_enabled = jobs[job_name].get('enabled', False)
            
            action = "disable" if current_enabled else "enable"
            confirm = input(f"\n{action.title()} job '{job_name}'? (y/N): ").strip()
            
            if confirm.lower().startswith('y'):
                if current_enabled:
                    result = service.disable_job(job_name)
                else:
                    result = service.enable_job(job_name)
                
                if result.get('success'):
                    print(f"✅ Job {action}d successfully!")
                else:
                    print(f"❌ Failed to {action} job: {result.get('error')}")
        else:
            print("❌ Invalid selection")
            
    except (ValueError, KeyboardInterrupt):
        print("❌ Operation cancelled")


def show_dashboard_data(service):
    """Show comprehensive dashboard data."""
    print("\n📊 Dashboard Data:")
    print("-" * 20)
    
    dashboard = service.get_dashboard_data()
    
    if 'error' in dashboard:
        print(f"❌ Error: {dashboard['error']}")
        return
    
    # System health overview
    health = dashboard.get('system_health', {})
    overall = health.get('overall_status', 'unknown')
    print(f"🏥 System Health: {overall.upper()}")
    
    # Active alerts
    alerts = dashboard.get('active_alerts', [])
    print(f"🚨 Active Alerts: {len(alerts)}")
    for alert in alerts[:3]:  # Show first 3 alerts
        print(f"   - [{alert['severity']}] {alert['component']}: {alert['message'][:50]}...")
    
    # Recent metrics
    metrics = dashboard.get('recent_metrics', [])
    print(f"\n📈 Recent Metrics ({len(metrics)} total):")
    for metric in metrics[-5:]:  # Show last 5 metrics
        print(f"   - {metric['component']}.{metric['name']}: {metric['value']} {metric['unit']}")


def run_examples():
    """Run the examples module."""
    print("\n🎯 Running Examples...")
    print("-" * 20)
    
    try:
        import examples
        examples.monitoring_example()
        examples.job_management_example()
    except ImportError:
        print("❌ Examples module not found")
    except Exception as e:
        print(f"❌ Error running examples: {e}")


def daemon_mode(config_path: str):
    """Run in daemon mode."""
    print("🔄 Starting in daemon mode...")
    
    # Setup logging for daemon
    setup_logging(log_level="INFO")
    
    # Create and run service
    service = DataCollectionService(config_path=config_path)
    
    try:
        service.run_continuous()
        return 0
    except Exception as e:
        logger.error(f"Daemon mode error: {e}")
        return 1


def check_status():
    """Check if service is running and show status."""
    print("📊 Service Status Check")
    print("-" * 25)
    
    try:
        service = get_service()
        
        if service.is_running:
            status = service.get_service_status()
            print("✅ Service is RUNNING")
            print(f"   Start Time: {status.get('start_time', 'N/A')}")
            
            uptime = status.get('uptime')
            if uptime:
                print(f"   Uptime: {uptime['days']}d {uptime['hours']}h {uptime['minutes']}m")
        else:
            print("❌ Service is NOT RUNNING")
            
        return 0
        
    except Exception as e:
        print(f"❌ Error checking status: {e}")
        return 1


def stop_running_service():
    """Stop a running service."""
    print("🛑 Stopping Service")
    print("-" * 20)
    
    try:
        service = get_service()
        
        if service.is_running:
            print("Stopping service...")
            stop_service()
            print("✅ Service stopped successfully")
        else:
            print("❌ Service is not running")
            
        return 0
        
    except Exception as e:
        print(f"❌ Error stopping service: {e}")
        return 1


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Snowflake Data Collection System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                     # Interactive mode
  python main.py --daemon            # Run as daemon
  python main.py --config ./config  # Custom config
  python main.py --examples         # Run examples
  python main.py --status           # Check status
  python main.py --stop             # Stop service
        """
    )
    
    parser.add_argument(
        '--config', 
        default='config',
        help='Configuration directory path (default: config)'
    )
    parser.add_argument(
        '--daemon', 
        action='store_true',
        help='Run as daemon process'
    )
    parser.add_argument(
        '--examples', 
        action='store_true',
        help='Run examples and demonstrations'
    )
    parser.add_argument(
        '--status', 
        action='store_true',
        help='Check service status'
    )
    parser.add_argument(
        '--stop', 
        action='store_true',
        help='Stop running service'
    )
    parser.add_argument(
        '--log-level', 
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Logging level (default: INFO)'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(log_level=args.log_level)
    
    try:
        if args.status:
            return check_status()
        elif args.stop:
            return stop_running_service()
        elif args.examples:
            import examples
            examples.main()
            return 0
        elif args.daemon:
            return daemon_mode(args.config)
        else:
            return interactive_mode()
            
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
        return 0
    except Exception as e:
        logger.error(f"Main error: {e}")
        print(f"❌ Error: {e}")
        return 1


if __name__ == '__main__':
    sys.exit(main())
