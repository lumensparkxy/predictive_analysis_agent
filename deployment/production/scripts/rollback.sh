#!/bin/bash

# Snowflake Analytics - Rollback Script
# Automated rollback to previous deployment version

set -euo pipefail

# Configuration
APP_ROOT="/opt/analytics"
BACKUP_ROOT="/opt/analytics/backups"
LOG_FILE="/var/log/analytics/rollback.log"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

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
check_permissions() {
    if [[ $EUID -eq 0 ]]; then
        error "This script should not be run as root"
    fi
    
    # Check if user can sudo
    if ! sudo -n true 2>/dev/null; then
        error "User needs sudo privileges for rollback"
    fi
}

# List available backups
list_backups() {
    log "Available backups:"
    echo ""
    echo "Available backups for rollback:"
    echo "==============================="
    
    local backups=()
    if [[ -d "$BACKUP_ROOT" ]]; then
        while IFS= read -r -d '' backup; do
            backups+=("$backup")
        done < <(find "$BACKUP_ROOT" -name "backup_*.tar.gz*" -type f -print0 | sort -z -r)
        
        if [[ ${#backups[@]} -eq 0 ]]; then
            error "No backups found in $BACKUP_ROOT"
        fi
        
        for i in "${!backups[@]}"; do
            local backup_file="${backups[$i]}"
            local backup_name=$(basename "$backup_file" | sed 's/\(\.tar\.gz\)\(\.enc\)\?$//')
            local backup_date=$(echo "$backup_name" | sed 's/backup_//' | sed 's/_/ /')
            local backup_size=$(du -sh "$backup_file" | cut -f1)
            
            echo "$((i+1)). $backup_name"
            echo "   Date: $backup_date"
            echo "   Size: $backup_size"
            echo "   File: $backup_file"
            
            # Show manifest if available
            if [[ -f "${backup_file%.*}/manifest.txt" ]]; then
                echo "   Version: $(grep "Git commit:" "${backup_file%.*}/manifest.txt" 2>/dev/null | cut -d: -f2 | xargs)"
            fi
            echo ""
        done
    else
        error "Backup directory $BACKUP_ROOT does not exist"
    fi
    
    echo "${#backups[@]}"
}

# Select backup for rollback
select_backup() {
    local backup_count=$(list_backups)
    
    echo "Select backup to rollback to (1-$backup_count), or 0 to cancel:"
    read -r selection
    
    if [[ "$selection" == "0" ]]; then
        log "Rollback cancelled by user"
        exit 0
    fi
    
    if [[ ! "$selection" =~ ^[0-9]+$ ]] || [[ $selection -lt 1 ]] || [[ $selection -gt $backup_count ]]; then
        error "Invalid selection: $selection"
    fi
    
    # Get the selected backup file
    local backups=()
    while IFS= read -r -d '' backup; do
        backups+=("$backup")
    done < <(find "$BACKUP_ROOT" -name "backup_*.tar.gz*" -type f -print0 | sort -z -r)
    
    SELECTED_BACKUP="${backups[$((selection-1))]}"
    local backup_name=$(basename "$SELECTED_BACKUP" | sed 's/\(\.tar\.gz\)\(\.enc\)\?$//')
    
    log "Selected backup: $backup_name"
    
    # Confirm rollback
    echo ""
    echo "WARNING: This will rollback the system to: $backup_name"
    echo "Current data and configuration will be lost!"
    echo ""
    read -p "Are you sure you want to proceed? (yes/no): " confirm
    
    if [[ "$confirm" != "yes" ]]; then
        log "Rollback cancelled by user"
        exit 0
    fi
}

# Stop services
stop_services() {
    log "Stopping services before rollback..."
    
    local services=("analytics-api" "analytics-worker" "analytics-scheduler")
    
    for service in "${services[@]}"; do
        if systemctl is-active --quiet "$service.service"; then
            log "Stopping $service service..."
            sudo systemctl stop "$service.service" || warning "Failed to stop $service"
        fi
    done
    
    success "Services stopped"
}

# Create pre-rollback backup
create_pre_rollback_backup() {
    log "Creating pre-rollback backup of current state..."
    
    local pre_rollback_backup="$BACKUP_ROOT/pre_rollback_$(date +%Y%m%d_%H%M%S)"
    mkdir -p "$pre_rollback_backup"
    
    # Backup current state
    if [[ -d "$APP_ROOT/src" ]]; then
        cp -r "$APP_ROOT/src" "$pre_rollback_backup/"
    fi
    
    if [[ -d "$APP_ROOT/config" ]]; then
        cp -r "$APP_ROOT/config" "$pre_rollback_backup/"
    fi
    
    # Create manifest
    cat > "$pre_rollback_backup/manifest.txt" << EOF
Pre-Rollback Backup
==================
Created: $(date)
Reason: Automatic backup before rollback
Original Version: $(cat "$APP_ROOT/VERSION" 2>/dev/null || echo "Unknown")
Git Commit: $(git -C "$APP_ROOT" rev-parse HEAD 2>/dev/null || echo "Unknown")
EOF
    
    success "Pre-rollback backup created: $pre_rollback_backup"
}

# Extract and restore backup
restore_backup() {
    log "Restoring backup: $SELECTED_BACKUP"
    
    local temp_dir="/tmp/analytics_rollback_$$"
    mkdir -p "$temp_dir"
    
    cd "$temp_dir"
    
    # Extract backup
    if [[ "$SELECTED_BACKUP" == *.enc ]]; then
        log "Decrypting backup..."
        if [[ -z "${BACKUP_ENCRYPTION_KEY:-}" ]]; then
            error "Backup is encrypted but BACKUP_ENCRYPTION_KEY is not set"
        fi
        openssl enc -aes-256-cbc -d -in "$SELECTED_BACKUP" -pass "pass:$BACKUP_ENCRYPTION_KEY" | tar -xzf -
    else
        log "Extracting backup..."
        tar -xzf "$SELECTED_BACKUP"
    fi
    
    # Find extracted backup directory
    local backup_dir=$(find . -maxdepth 1 -type d -name "backup_*" | head -1)
    if [[ -z "$backup_dir" ]]; then
        error "Could not find extracted backup directory"
    fi
    
    log "Backup extracted to: $temp_dir/$backup_dir"
    
    # Restore application code
    if [[ -d "$backup_dir/src" ]]; then
        log "Restoring application code..."
        rm -rf "$APP_ROOT/src"
        cp -r "$backup_dir/src" "$APP_ROOT/"
        success "Application code restored"
    fi
    
    # Restore configuration (excluding secrets)
    if [[ -d "$backup_dir/config" ]]; then
        log "Restoring configuration..."
        # Backup current secrets before restoration
        local secrets_backup=""
        if [[ -f "$APP_ROOT/config/secrets.env" ]]; then
            secrets_backup=$(mktemp)
            cp "$APP_ROOT/config/secrets.env" "$secrets_backup"
        fi
        
        cp -r "$backup_dir/config"/* "$APP_ROOT/config/"
        
        # Restore secrets if they were backed up
        if [[ -n "$secrets_backup" && -f "$secrets_backup" ]]; then
            cp "$secrets_backup" "$APP_ROOT/config/secrets.env"
            rm "$secrets_backup"
        fi
        
        success "Configuration restored"
    fi
    
    # Restore data
    if [[ -d "$backup_dir/data" ]]; then
        log "Restoring application data..."
        rm -rf "$APP_ROOT/data"
        cp -r "$backup_dir/data" "$APP_ROOT/"
        success "Application data restored"
    fi
    
    # Set proper permissions
    sudo chown -R analytics:analytics "$APP_ROOT"
    chmod 600 "$APP_ROOT/config/secrets.env" 2>/dev/null || true
    
    # Cleanup
    rm -rf "$temp_dir"
    
    success "Backup restoration completed"
}

# Restore database
restore_database() {
    log "Restoring database..."
    
    local temp_dir="/tmp/analytics_db_rollback_$$"
    mkdir -p "$temp_dir"
    
    # Extract backup to get database files
    if [[ "$SELECTED_BACKUP" == *.enc ]]; then
        openssl enc -aes-256-cbc -d -in "$SELECTED_BACKUP" -pass "pass:$BACKUP_ENCRYPTION_KEY" | tar -xzf - -C "$temp_dir"
    else
        tar -xzf "$SELECTED_BACKUP" -C "$temp_dir"
    fi
    
    local backup_dir=$(find "$temp_dir" -maxdepth 1 -type d -name "backup_*" | head -1)
    
    # Restore PostgreSQL database
    if [[ -f "$backup_dir/database/postgresql_analytics_prod.sql.gz" ]]; then
        log "Restoring PostgreSQL database..."
        gunzip -c "$backup_dir/database/postgresql_analytics_prod.sql.gz" | PGPASSWORD="${DATABASE_PASSWORD:-}" psql -h localhost -U analytics
        success "PostgreSQL database restored"
    fi
    
    # Restore SQLite database
    if [[ -f "$backup_dir/database/analytics_sqlite.db.gz" ]]; then
        log "Restoring SQLite database..."
        gunzip -c "$backup_dir/database/analytics_sqlite.db.gz" > "$APP_ROOT/data/analytics.db"
        success "SQLite database restored"
    fi
    
    # Restore Redis data
    if [[ -f "$backup_dir/database/redis_dump.rdb.gz" ]]; then
        log "Restoring Redis data..."
        sudo systemctl stop redis-server || true
        gunzip -c "$backup_dir/database/redis_dump.rdb.gz" > /var/lib/redis/dump.rdb
        sudo chown redis:redis /var/lib/redis/dump.rdb
        sudo systemctl start redis-server
        success "Redis data restored"
    fi
    
    # Cleanup
    rm -rf "$temp_dir"
}

# Start services
start_services() {
    log "Starting services after rollback..."
    
    local services=("analytics-api" "analytics-worker" "analytics-scheduler")
    
    for service in "${services[@]}"; do
        log "Starting $service service..."
        sudo systemctl start "$service.service" || error "Failed to start $service"
    done
    
    # Restart nginx
    sudo systemctl restart nginx || error "Failed to restart nginx"
    
    success "Services started"
}

# Verify rollback
verify_rollback() {
    log "Verifying rollback..."
    
    # Wait for services to fully start
    sleep 15
    
    # Check if services are running
    local services=("analytics-api" "analytics-worker" "analytics-scheduler" "nginx")
    for service in "${services[@]}"; do
        if ! systemctl is-active --quiet "$service.service"; then
            error "Service $service failed to start after rollback"
        fi
    done
    
    # Check API health
    local health_url="http://localhost:8000/health"
    local max_attempts=10
    local attempt=1
    
    while [[ $attempt -le $max_attempts ]]; do
        if curl -s --max-time 10 "$health_url" >/dev/null 2>&1; then
            success "API health check passed after rollback"
            break
        fi
        
        if [[ $attempt -eq $max_attempts ]]; then
            error "API health check failed after rollback"
        fi
        
        log "Health check attempt $attempt/$max_attempts failed, retrying..."
        sleep 5
        ((attempt++))
    done
    
    # Update version file
    if [[ -f "$APP_ROOT/VERSION" ]]; then
        log "Restored version: $(cat "$APP_ROOT/VERSION")"
    fi
    
    success "Rollback verification completed"
}

# Send rollback notification
send_rollback_notification() {
    local message="System rollback completed successfully to backup: $(basename "$SELECTED_BACKUP")"
    
    # Email notification
    if command -v mail &> /dev/null && [[ -n "${ROLLBACK_EMAIL_TO:-}" ]]; then
        echo "$message" | mail -s "Analytics System Rollback Completed" "$ROLLBACK_EMAIL_TO"
    fi
    
    # Slack notification
    if [[ -n "${SLACK_WEBHOOK_URL:-}" ]]; then
        curl -X POST -H 'Content-type: application/json' \
            --data "{\"text\":\"🔄 $message\"}" \
            "$SLACK_WEBHOOK_URL" >/dev/null 2>&1 || true
    fi
}

# Main rollback function
main() {
    log "Starting rollback process..."
    
    check_permissions
    
    # Source environment if available
    if [[ -f "$APP_ROOT/config/production.env" ]]; then
        set -a
        source "$APP_ROOT/config/production.env"
        set +a
    fi
    
    select_backup
    stop_services
    create_pre_rollback_backup
    restore_backup
    restore_database
    start_services
    verify_rollback
    send_rollback_notification
    
    success "Rollback completed successfully!"
    log "Rollback finished at $(date)"
}

# Error handling
trap 'error "Rollback failed unexpectedly"' ERR

# Script entry point
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi