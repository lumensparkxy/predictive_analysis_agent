#!/bin/bash

# Snowflake Analytics - Health Check Script
# Comprehensive health monitoring for production system

set -euo pipefail

# Configuration
APP_ROOT="/opt/analytics"
API_BASE_URL="http://localhost:8000"
DASHBOARD_URL="http://localhost:8501"
LOG_FILE="/var/log/analytics/health.log"
TIMEOUT=30

# Exit codes
EXIT_OK=0
EXIT_WARNING=1
EXIT_CRITICAL=2
EXIT_UNKNOWN=3

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Logging function
log() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') - $1" | tee -a "$LOG_FILE"
}

status() {
    local level="$1"
    local message="$2"
    case "$level" in
        "OK")
            echo -e "${GREEN}OK${NC}: $message"
            ;;
        "WARNING")
            echo -e "${YELLOW}WARNING${NC}: $message"
            ;;
        "CRITICAL")
            echo -e "${RED}CRITICAL${NC}: $message"
            ;;
        *)
            echo "UNKNOWN: $message"
            ;;
    esac
    log "$level: $message"
}

# Check API health endpoint
check_api_health() {
    local health_url="$API_BASE_URL/health"
    
    if curl -s --max-time "$TIMEOUT" "$health_url" >/dev/null 2>&1; then
        local response=$(curl -s --max-time "$TIMEOUT" "$health_url")
        if echo "$response" | grep -q "healthy"; then
            status "OK" "API health endpoint responding"
            return 0
        else
            status "WARNING" "API health endpoint returned unexpected response: $response"
            return 1
        fi
    else
        status "CRITICAL" "API health endpoint not responding"
        return 2
    fi
}

# Check API functionality
check_api_functionality() {
    local metrics_url="$API_BASE_URL/api/v1/metrics"
    
    if curl -s --max-time "$TIMEOUT" -H "Content-Type: application/json" "$metrics_url" >/dev/null 2>&1; then
        status "OK" "API metrics endpoint functional"
        return 0
    else
        status "WARNING" "API metrics endpoint not responding"
        return 1
    fi
}

# Check database connectivity
check_database() {
    # Check PostgreSQL
    if command -v psql &> /dev/null; then
        if PGPASSWORD="${DATABASE_PASSWORD:-}" psql -h localhost -U analytics -d analytics_prod -c "SELECT 1;" >/dev/null 2>&1; then
            status "OK" "PostgreSQL database connectivity"
        else
            status "CRITICAL" "PostgreSQL database connection failed"
            return 2
        fi
    fi
    
    # Check Snowflake connectivity
    if [[ -f "$APP_ROOT/venv/bin/activate" ]]; then
        source "$APP_ROOT/venv/bin/activate"
        cd "$APP_ROOT"
        
        local snowflake_check=$(python3 -c "
import sys
sys.path.append('src')
try:
    from snowflake_analytics.database.snowflake_connector import SnowflakeConnector
    conn = SnowflakeConnector()
    result = conn.execute_query('SELECT 1 as test')
    print('OK')
except Exception as e:
    print(f'ERROR: {e}')
    sys.exit(1)
" 2>/dev/null)
        
        if [[ "$snowflake_check" == "OK" ]]; then
            status "OK" "Snowflake database connectivity"
        else
            status "CRITICAL" "Snowflake database connection failed: $snowflake_check"
            return 2
        fi
    fi
    
    return 0
}

# Check Redis connectivity
check_redis() {
    if command -v redis-cli &> /dev/null; then
        if redis-cli ping >/dev/null 2>&1; then
            status "OK" "Redis connectivity"
            return 0
        else
            status "WARNING" "Redis connection failed"
            return 1
        fi
    else
        status "WARNING" "Redis CLI not available"
        return 1
    fi
}

# Check system services
check_services() {
    local services=("analytics-api" "analytics-worker" "analytics-scheduler" "nginx")
    local failed_services=()
    
    for service in "${services[@]}"; do
        if systemctl is-active --quiet "$service.service"; then
            status "OK" "Service $service is active"
        else
            status "CRITICAL" "Service $service is not active"
            failed_services+=("$service")
        fi
    done
    
    if [[ ${#failed_services[@]} -gt 0 ]]; then
        return 2
    fi
    
    return 0
}

# Check system resources
check_resources() {
    local warnings=0
    local critical=0
    
    # Check disk space
    local disk_usage=$(df "$APP_ROOT" | tail -1 | awk '{print $5}' | sed 's/%//')
    if [[ $disk_usage -gt 90 ]]; then
        status "CRITICAL" "Disk usage is ${disk_usage}% (>90%)"
        ((critical++))
    elif [[ $disk_usage -gt 80 ]]; then
        status "WARNING" "Disk usage is ${disk_usage}% (>80%)"
        ((warnings++))
    else
        status "OK" "Disk usage is ${disk_usage}%"
    fi
    
    # Check memory usage
    local mem_usage=$(free | grep Mem | awk '{printf("%.0f", ($3/$2) * 100.0)}')
    if [[ $mem_usage -gt 90 ]]; then
        status "CRITICAL" "Memory usage is ${mem_usage}% (>90%)"
        ((critical++))
    elif [[ $mem_usage -gt 80 ]]; then
        status "WARNING" "Memory usage is ${mem_usage}% (>80%)"
        ((warnings++))
    else
        status "OK" "Memory usage is ${mem_usage}%"
    fi
    
    # Check CPU load
    local cpu_load=$(uptime | awk -F'load average:' '{print $2}' | awk '{print $1}' | sed 's/,//')
    local cpu_cores=$(nproc)
    local load_percentage=$(python3 -c "print(int(float('$cpu_load') / int('$cpu_cores') * 100))")
    
    if [[ $load_percentage -gt 90 ]]; then
        status "CRITICAL" "CPU load is ${load_percentage}% (>90%)"
        ((critical++))
    elif [[ $load_percentage -gt 80 ]]; then
        status "WARNING" "CPU load is ${load_percentage}% (>80%)"
        ((warnings++))
    else
        status "OK" "CPU load is ${load_percentage}%"
    fi
    
    # Return appropriate code
    if [[ $critical -gt 0 ]]; then
        return 2
    elif [[ $warnings -gt 0 ]]; then
        return 1
    fi
    
    return 0
}

# Check log errors
check_logs() {
    local error_count=0
    local warning_count=0
    
    # Check application logs for recent errors
    if [[ -f "$APP_ROOT/logs/application.log" ]]; then
        error_count=$(tail -n 1000 "$APP_ROOT/logs/application.log" | grep -i "error" | wc -l)
        warning_count=$(tail -n 1000 "$APP_ROOT/logs/application.log" | grep -i "warning" | wc -l)
    fi
    
    # Check nginx error logs
    if [[ -f "/var/log/nginx/analytics_error.log" ]]; then
        local nginx_errors=$(tail -n 100 /var/log/nginx/analytics_error.log 2>/dev/null | wc -l)
        error_count=$((error_count + nginx_errors))
    fi
    
    if [[ $error_count -gt 10 ]]; then
        status "WARNING" "Found $error_count recent errors in logs"
        return 1
    elif [[ $error_count -gt 0 ]]; then
        status "OK" "Found $error_count recent errors (acceptable level)"
    else
        status "OK" "No recent errors in logs"
    fi
    
    return 0
}

# Check SSL certificate
check_ssl() {
    local cert_path="/etc/nginx/ssl/certificates/analytics.crt"
    
    if [[ -f "$cert_path" ]]; then
        local expiry_date=$(openssl x509 -in "$cert_path" -enddate -noout | cut -d= -f2)
        local expiry_epoch=$(date -d "$expiry_date" +%s)
        local current_epoch=$(date +%s)
        local days_until_expiry=$(( (expiry_epoch - current_epoch) / 86400 ))
        
        if [[ $days_until_expiry -lt 7 ]]; then
            status "CRITICAL" "SSL certificate expires in $days_until_expiry days"
            return 2
        elif [[ $days_until_expiry -lt 30 ]]; then
            status "WARNING" "SSL certificate expires in $days_until_expiry days"
            return 1
        else
            status "OK" "SSL certificate valid for $days_until_expiry days"
            return 0
        fi
    else
        status "WARNING" "SSL certificate not found"
        return 1
    fi
}

# Check data freshness
check_data_freshness() {
    if [[ -f "$APP_ROOT/venv/bin/activate" ]]; then
        source "$APP_ROOT/venv/bin/activate"
        cd "$APP_ROOT"
        
        local last_collection=$(python3 -c "
import sys
sys.path.append('src')
try:
    from datetime import datetime, timedelta
    from snowflake_analytics.storage import SQLiteStore
    store = SQLiteStore()
    last_run = store.get_last_collection_time()
    if last_run:
        hours_ago = (datetime.utcnow() - last_run).total_seconds() / 3600
        print(int(hours_ago))
    else:
        print('999')
except Exception:
    print('999')
" 2>/dev/null)
        
        if [[ $last_collection -gt 24 ]]; then
            status "CRITICAL" "Data collection last ran ${last_collection}h ago (>24h)"
            return 2
        elif [[ $last_collection -gt 6 ]]; then
            status "WARNING" "Data collection last ran ${last_collection}h ago (>6h)"
            return 1
        else
            status "OK" "Data collection last ran ${last_collection}h ago"
            return 0
        fi
    fi
    
    return 0
}

# Performance check
check_performance() {
    # Test API response time
    local start_time=$(date +%s%N)
    if curl -s --max-time "$TIMEOUT" "$API_BASE_URL/health" >/dev/null; then
        local end_time=$(date +%s%N)
        local response_time=$(( (end_time - start_time) / 1000000 )) # Convert to milliseconds
        
        if [[ $response_time -gt 5000 ]]; then
            status "CRITICAL" "API response time is ${response_time}ms (>5s)"
            return 2
        elif [[ $response_time -gt 2000 ]]; then
            status "WARNING" "API response time is ${response_time}ms (>2s)"
            return 1
        else
            status "OK" "API response time is ${response_time}ms"
            return 0
        fi
    else
        status "CRITICAL" "API performance check failed"
        return 2
    fi
}

# Main health check function
main() {
    local overall_status=0
    local checks_run=0
    local checks_passed=0
    local checks_warning=0
    local checks_critical=0
    
    log "Starting health check..."
    echo "Snowflake Analytics System Health Check"
    echo "======================================="
    
    # Run all health checks
    local checks=(
        "check_api_health"
        "check_api_functionality"
        "check_database"
        "check_redis"
        "check_services"
        "check_resources"
        "check_logs"
        "check_ssl"
        "check_data_freshness"
        "check_performance"
    )
    
    for check in "${checks[@]}"; do
        echo ""
        echo "Running $check..."
        ((checks_run++))
        
        if $check; then
            local check_result=$?
            case $check_result in
                0)
                    ((checks_passed++))
                    ;;
                1)
                    ((checks_warning++))
                    if [[ $overall_status -lt 1 ]]; then overall_status=1; fi
                    ;;
                2)
                    ((checks_critical++))
                    overall_status=2
                    ;;
            esac
        else
            local check_result=$?
            case $check_result in
                1)
                    ((checks_warning++))
                    if [[ $overall_status -lt 1 ]]; then overall_status=1; fi
                    ;;
                2)
                    ((checks_critical++))
                    overall_status=2
                    ;;
                *)
                    ((checks_critical++))
                    overall_status=2
                    ;;
            esac
        fi
    done
    
    # Summary
    echo ""
    echo "Health Check Summary"
    echo "==================="
    echo "Total checks: $checks_run"
    echo "Passed: $checks_passed"
    echo "Warnings: $checks_warning"
    echo "Critical: $checks_critical"
    echo ""
    
    case $overall_status in
        0)
            status "OK" "System is healthy"
            ;;
        1)
            status "WARNING" "System has warnings"
            ;;
        2)
            status "CRITICAL" "System has critical issues"
            ;;
    esac
    
    log "Health check completed with status: $overall_status"
    exit $overall_status
}

# Script entry point
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi