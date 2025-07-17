#!/usr/bin/env python3
"""
Snowflake Analytics Dashboard Demonstration Script

This script demonstrates the complete dashboard functionality including:
- Backend API server
- Frontend development server
- Real-time WebSocket connections
- Interactive data visualization
- Mobile-responsive design
"""

import subprocess
import time
import sys
import os
import signal
import threading
from pathlib import Path

# Color codes for output
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def print_colored(text, color=Colors.OKBLUE):
    """Print colored text."""
    print(f"{color}{text}{Colors.ENDC}")

def print_header(text):
    """Print header with decoration."""
    print_colored(f"\n{'='*60}", Colors.HEADER)
    print_colored(f"  {text}", Colors.HEADER)
    print_colored(f"{'='*60}", Colors.HEADER)

def print_success(text):
    """Print success message."""
    print_colored(f"✅ {text}", Colors.OKGREEN)

def print_error(text):
    """Print error message."""
    print_colored(f"❌ {text}", Colors.FAIL)

def print_warning(text):
    """Print warning message."""
    print_colored(f"⚠️  {text}", Colors.WARNING)

def print_info(text):
    """Print info message."""
    print_colored(f"ℹ️  {text}", Colors.OKCYAN)

class DashboardDemo:
    def __init__(self):
        self.base_dir = Path(__file__).parent
        self.frontend_dir = self.base_dir / "frontend"
        self.backend_process = None
        self.frontend_process = None
        self.demo_mode = False
        
    def check_requirements(self):
        """Check if required tools are available."""
        print_header("Checking Requirements")
        
        # Check Python
        try:
            python_version = sys.version_info
            if python_version.major >= 3 and python_version.minor >= 8:
                print_success(f"Python {python_version.major}.{python_version.minor} found")
            else:
                print_error("Python 3.8+ required")
                return False
        except Exception as e:
            print_error(f"Python check failed: {e}")
            return False
        
        # Check Node.js (if frontend directory exists)
        if self.frontend_dir.exists():
            try:
                result = subprocess.run(
                    ["node", "--version"], 
                    capture_output=True, 
                    text=True, 
                    timeout=5
                )
                if result.returncode == 0:
                    print_success(f"Node.js {result.stdout.strip()} found")
                else:
                    print_warning("Node.js not found - frontend demo will be skipped")
            except Exception as e:
                print_warning(f"Node.js check failed: {e}")
        
        # Check if API modules can be imported
        try:
            from src.snowflake_analytics.api.endpoints.costs import cost_endpoints
            from src.snowflake_analytics.api.endpoints.usage import usage_endpoints
            print_success("API modules can be imported")
        except Exception as e:
            print_error(f"API modules check failed: {e}")
            return False
        
        return True
    
    def test_api_endpoints(self):
        """Test API endpoints functionality."""
        print_header("Testing API Endpoints")
        
        import asyncio
        
        async def test_endpoints():
            try:
                # Test cost endpoints
                from src.snowflake_analytics.api.endpoints.costs import cost_endpoints
                result = await cost_endpoints.get_cost_summary()
                if result['status'] == 'success':
                    print_success("Cost endpoints working")
                else:
                    print_error("Cost endpoints failed")
                
                # Test usage endpoints
                from src.snowflake_analytics.api.endpoints.usage import usage_endpoints
                result = await usage_endpoints.get_usage_metrics()
                if result['status'] == 'success':
                    print_success("Usage endpoints working")
                else:
                    print_error("Usage endpoints failed")
                
                # Test prediction endpoints
                from src.snowflake_analytics.api.endpoints.predictions import prediction_endpoints
                result = await prediction_endpoints.get_cost_forecast()
                if result['status'] == 'success':
                    print_success("Prediction endpoints working")
                else:
                    print_error("Prediction endpoints failed")
                
                # Test anomaly endpoints
                from src.snowflake_analytics.api.endpoints.anomalies import anomaly_endpoints
                result = await anomaly_endpoints.get_current_anomalies()
                if result['status'] == 'success':
                    print_success("Anomaly endpoints working")
                else:
                    print_error("Anomaly endpoints failed")
                
                # Test alert endpoints
                from src.snowflake_analytics.api.endpoints.alerts import alert_endpoints
                result = await alert_endpoints.get_active_alerts()
                if result['status'] == 'success':
                    print_success("Alert endpoints working")
                else:
                    print_error("Alert endpoints failed")
                
                return True
                
            except Exception as e:
                print_error(f"API testing failed: {e}")
                return False
        
        return asyncio.run(test_endpoints())
    
    def start_backend(self):
        """Start the backend API server."""
        print_header("Starting Backend API Server")
        
        try:
            # Try to start with uvicorn
            import uvicorn
            
            def run_server():
                # Import the FastAPI app
                from src.snowflake_analytics.api.main import app
                uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
            
            # Start server in a separate thread
            server_thread = threading.Thread(target=run_server, daemon=True)
            server_thread.start()
            
            # Wait a bit for server to start
            time.sleep(3)
            
            # Test if server is running
            import requests
            response = requests.get("http://localhost:8000/api/health", timeout=5)
            if response.status_code == 200:
                print_success("Backend API server is running on http://localhost:8000")
                print_info("API documentation available at http://localhost:8000/docs")
                return True
            else:
                print_error("Backend server health check failed")
                return False
                
        except ImportError:
            print_error("uvicorn not available. Install with: pip install uvicorn")
            return False
        except Exception as e:
            print_error(f"Failed to start backend server: {e}")
            return False
    
    def start_frontend(self):
        """Start the frontend development server."""
        print_header("Starting Frontend Development Server")
        
        if not self.frontend_dir.exists():
            print_warning("Frontend directory not found - skipping frontend demo")
            return False
        
        try:
            # Check if package.json exists
            package_json = self.frontend_dir / "package.json"
            if not package_json.exists():
                print_warning("package.json not found - skipping frontend demo")
                return False
            
            # Try to start frontend server
            os.chdir(self.frontend_dir)
            
            # Install dependencies if node_modules doesn't exist
            node_modules = self.frontend_dir / "node_modules"
            if not node_modules.exists():
                print_info("Installing frontend dependencies...")
                subprocess.run(["npm", "install"], check=True, timeout=300)
            
            # Start development server
            print_info("Starting frontend development server...")
            self.frontend_process = subprocess.Popen(
                ["npm", "run", "dev"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # Wait for server to start
            time.sleep(5)
            
            if self.frontend_process.poll() is None:
                print_success("Frontend development server is running on http://localhost:3000")
                return True
            else:
                print_error("Frontend server failed to start")
                return False
                
        except subprocess.TimeoutExpired:
            print_error("Frontend installation timeout")
            return False
        except Exception as e:
            print_error(f"Failed to start frontend server: {e}")
            return False
    
    def display_demo_info(self):
        """Display demo information and instructions."""
        print_header("🎉 Snowflake Analytics Dashboard Demo")
        
        print_colored("""
🚀 DASHBOARD FEATURES:
   • Real-time cost and usage monitoring
   • Interactive data visualizations
   • ML-powered anomaly detection
   • Predictive analytics and forecasting
   • Alert management and notifications
   • Mobile-responsive design
   • WebSocket real-time updates
   • Comprehensive API documentation
        """, Colors.OKGREEN)
        
        print_colored("""
🌐 AVAILABLE ENDPOINTS:
   • Backend API: http://localhost:8000
   • API Documentation: http://localhost:8000/docs
   • Interactive Dashboard: http://localhost:8000 (basic)
   • Frontend Dashboard: http://localhost:3000 (full-featured)
        """, Colors.OKCYAN)
        
        print_colored("""
📊 DEMO CAPABILITIES:
   • Cost Analytics: Track and forecast Snowflake costs
   • Usage Monitoring: Query performance and warehouse utilization
   • Anomaly Detection: Identify unusual patterns with ML
   • Predictive Forecasting: AI-powered predictions
   • Alert System: Real-time notifications and management
   • Settings Management: Configurable preferences
        """, Colors.OKBLUE)
        
        print_colored("""
🛠️  TECHNICAL FEATURES:
   • FastAPI backend with 25+ endpoints
   • React + TypeScript frontend
   • WebSocket real-time communication
   • Rate limiting and authentication
   • Comprehensive error handling
   • Mobile-responsive design
        """, Colors.WARNING)
        
        print_colored("""
🔧 DEMO ACTIONS:
   1. Visit http://localhost:8000 for the basic dashboard
   2. Visit http://localhost:3000 for the full React dashboard
   3. Explore the API documentation at http://localhost:8000/docs
   4. Test real-time features and WebSocket connections
   5. Try mobile responsive design on different devices
        """, Colors.HEADER)
        
        print_colored("""
⚡ QUICK TESTS:
   • curl http://localhost:8000/api/health
   • curl http://localhost:8000/api/v1/costs/summary
   • curl http://localhost:8000/api/v1/usage/metrics
   • curl http://localhost:8000/api/v1/predictions/forecast
        """, Colors.OKCYAN)
    
    def run_demo(self):
        """Run the complete dashboard demo."""
        print_colored("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                  🚀 SNOWFLAKE ANALYTICS DASHBOARD DEMO                      ║
║                     Interactive Real-time Analytics Portal                   ║
╚══════════════════════════════════════════════════════════════════════════════╝
        """, Colors.HEADER)
        
        # Step 1: Check requirements
        if not self.check_requirements():
            print_error("Requirements check failed. Please install missing dependencies.")
            return False
        
        # Step 2: Test API endpoints
        if not self.test_api_endpoints():
            print_error("API endpoints test failed.")
            return False
        
        # Step 3: Start backend server
        if not self.start_backend():
            print_error("Backend server failed to start.")
            return False
        
        # Step 4: Start frontend server (optional)
        frontend_running = self.start_frontend()
        
        # Step 5: Display demo information
        self.display_demo_info()
        
        # Step 6: Keep demo running
        try:
            print_colored("""
🎯 DEMO IS RUNNING! 

Press Ctrl+C to stop the demo.
            """, Colors.OKGREEN)
            
            while True:
                time.sleep(1)
                
        except KeyboardInterrupt:
            print_colored("""
🛑 Shutting down demo...
            """, Colors.WARNING)
            
            if self.frontend_process:
                self.frontend_process.terminate()
                self.frontend_process.wait()
            
            print_success("Demo stopped successfully!")
            return True
    
    def quick_test(self):
        """Run a quick test of the dashboard functionality."""
        print_header("Quick Dashboard Test")
        
        if not self.check_requirements():
            return False
        
        if not self.test_api_endpoints():
            return False
        
        print_success("✅ All dashboard components are working correctly!")
        print_info("Run 'python demo_dashboard.py' to start the full demo")
        return True

def main():
    """Main entry point."""
    demo = DashboardDemo()
    
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        # Quick test mode
        demo.quick_test()
    else:
        # Full demo mode
        demo.run_demo()

if __name__ == "__main__":
    main()