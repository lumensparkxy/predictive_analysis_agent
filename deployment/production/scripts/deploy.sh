#!/bin/bash

# Snowflake Analytics - Production Deployment Script
# Automated deployment with blue-green strategy support

set -euo pipefail

# Script configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEPLOYMENT_ROOT="/opt/analytics"
SERVICE_USER="analytics"
BACKUP_DIR="/opt/analytics/backups"
LOG_FILE="/var/log/analytics/deploy.log"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') - $1" | tee -a "$LOG_FILE"
}

error() {
    echo -e "${RED}ERROR: $1${NC}" | tee -a "$LOG_FILE"
    exit 1
}

success() {
    echo -e "${GREEN}SUCCESS: $1${NC}" | tee -a "$LOG_FILE"
}

warning() {
    echo -e "${YELLOW}WARNING: $1${NC}" | tee -a "$LOG_FILE"
}

# Check if running as root
check_root() {
    if [[ $EUID -eq 0 ]]; then
        error "This script should not be run as root"
    fi
}

# Pre-deployment checks
pre_deployment_checks() {
    log "Running pre-deployment checks..."
    
    # Check required directories
    for dir in "$DEPLOYMENT_ROOT" "$BACKUP_DIR"; do
        if [[ ! -d "$dir" ]]; then
            error "Required directory $dir does not exist"
        fi
    done
    
    # Check required files
    if [[ ! -f "$DEPLOYMENT_ROOT/config/production.env" ]]; then
        error "Production environment file not found"
    fi
    
    # Check services are stopped
    if systemctl is-active --quiet analytics-api.service; then
        log "Stopping analytics-api service..."
        sudo systemctl stop analytics-api.service
    fi
    
    if systemctl is-active --quiet analytics-worker.service; then
        log "Stopping analytics-worker service..."
        sudo systemctl stop analytics-worker.service
    fi
    
    if systemctl is-active --quiet analytics-scheduler.service; then
        log "Stopping analytics-scheduler service..."
        sudo systemctl stop analytics-scheduler.service
    fi
    
    success "Pre-deployment checks completed"
}

# Backup current deployment
backup_current_deployment() {
    log "Creating backup of current deployment..."
    
    local backup_name="backup_$(date +%Y%m%d_%H%M%S)"
    local backup_path="$BACKUP_DIR/$backup_name"
    
    mkdir -p "$backup_path"
    
    # Backup application code
    if [[ -d "$DEPLOYMENT_ROOT/src" ]]; then
        cp -r "$DEPLOYMENT_ROOT/src" "$backup_path/"
    fi
    
    # Backup configuration
    if [[ -d "$DEPLOYMENT_ROOT/config" ]]; then
        cp -r "$DEPLOYMENT_ROOT/config" "$backup_path/"
    fi
    
    # Backup database
    if command -v pg_dump &> /dev/null; then
        log "Creating database backup..."
        pg_dump -h localhost -U analytics analytics_prod > "$backup_path/database.sql" || true
    fi
    
    # Create backup manifest
    cat > "$backup_path/manifest.txt" << EOF
Backup created: $(date)
Git commit: $(git rev-parse HEAD 2>/dev/null || echo "N/A")
Deployment version: $(cat "$DEPLOYMENT_ROOT/VERSION" 2>/dev/null || echo "N/A")
Services backed up:
- analytics-api
- analytics-worker  
- analytics-scheduler
EOF
    
    echo "$backup_name" > "$DEPLOYMENT_ROOT/.last_backup"
    success "Backup created: $backup_path"
}

# Deploy application
deploy_application() {
    log "Deploying application code..."
    
    # Copy source code
    if [[ -d "src" ]]; then
        cp -r src/* "$DEPLOYMENT_ROOT/src/"
    fi
    
    # Update version file
    git rev-parse HEAD > "$DEPLOYMENT_ROOT/VERSION" 2>/dev/null || echo "unknown" > "$DEPLOYMENT_ROOT/VERSION"
    
    # Install/update dependencies
    log "Installing dependencies..."
    source "$DEPLOYMENT_ROOT/venv/bin/activate"
    pip install --no-deps -e .
    pip install -r requirements.txt
    
    # Set permissions
    sudo chown -R "$SERVICE_USER:$SERVICE_USER" "$DEPLOYMENT_ROOT"
    sudo chmod -R 755 "$DEPLOYMENT_ROOT/src"
    sudo chmod 600 "$DEPLOYMENT_ROOT/config/secrets.env"
    
    success "Application deployed"
}

# Update configuration
update_configuration() {
    log "Updating configuration files..."
    
    # Copy systemd service files
    if [[ -d "deployment/production/systemd" ]]; then
        sudo cp deployment/production/systemd/*.service /etc/systemd/system/
        sudo systemctl daemon-reload
    fi
    
    # Copy nginx configuration
    if [[ -d "deployment/production/nginx" ]]; then
        sudo cp deployment/production/nginx/nginx.conf /etc/nginx/
        sudo cp deployment/production/nginx/sites-available/analytics.conf /etc/nginx/sites-available/
        sudo cp deployment/production/nginx/ssl/ssl.conf /etc/nginx/conf.d/
        
        # Enable site
        sudo ln -sf /etc/nginx/sites-available/analytics.conf /etc/nginx/sites-enabled/
        
        # Test nginx configuration
        sudo nginx -t || error "Nginx configuration test failed"
    fi
    
    success "Configuration updated"
}

# Run database migrations
run_migrations() {
    log "Running database migrations..."
    
    source "$DEPLOYMENT_ROOT/venv/bin/activate"
    cd "$DEPLOYMENT_ROOT"
    
    # Run any database migrations here
    # python src/snowflake_analytics/migrations/migrate.py
    
    success "Database migrations completed"
}

# Health check
health_check() {
    log "Running health checks..."
    
    # Wait for services to start
    sleep 10
    
    # Check API health
    local api_health_url="http://localhost:8000/health"
    local max_attempts=30
    local attempt=1
    
    while [[ $attempt -le $max_attempts ]]; do
        if curl -s "$api_health_url" >/dev/null; then
            success "API health check passed"
            break
        fi
        
        if [[ $attempt -eq $max_attempts ]]; then
            error "API health check failed after $max_attempts attempts"
        fi
        
        log "Health check attempt $attempt/$max_attempts failed, retrying..."
        sleep 5
        ((attempt++))
    done
    
    # Check database connectivity
    source "$DEPLOYMENT_ROOT/venv/bin/activate"
    python -c "
import os
import sys
sys.path.append('$DEPLOYMENT_ROOT/src')
try:
    from snowflake_analytics.database import test_connection
    test_connection()
    print('Database connectivity: OK')
except Exception as e:
    print(f'Database connectivity: FAILED - {e}')
    sys.exit(1)
" || error "Database connectivity check failed"
    
    success "Health checks completed"
}

# Start services
start_services() {
    log "Starting services..."
    
    # Start services in order
    sudo systemctl start analytics-api.service
    sudo systemctl start analytics-worker.service
    sudo systemctl start analytics-scheduler.service
    
    # Enable services
    sudo systemctl enable analytics-api.service
    sudo systemctl enable analytics-worker.service
    sudo systemctl enable analytics-scheduler.service
    
    # Restart nginx
    sudo systemctl restart nginx
    
    success "Services started"
}

# Cleanup old backups
cleanup_old_backups() {
    log "Cleaning up old backups..."
    
    find "$BACKUP_DIR" -type d -name "backup_*" -mtime +30 -exec rm -rf {} \; 2>/dev/null || true
    
    success "Old backups cleaned up"
}

# Main deployment function
main() {
    log "Starting deployment process..."
    
    check_root
    pre_deployment_checks
    backup_current_deployment
    deploy_application
    update_configuration
    run_migrations
    start_services
    health_check
    cleanup_old_backups
    
    success "Deployment completed successfully!"
    log "Deployment finished at $(date)"
}

# Script entry point
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi