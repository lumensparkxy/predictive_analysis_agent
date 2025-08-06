#!/bin/bash

# Snowflake Analytics - Backup Script
# Automated backup for database, configuration, and application data

set -euo pipefail

# Configuration
BACKUP_ROOT="/opt/analytics/backups"
APP_ROOT="/opt/analytics"
DATABASE_NAME="analytics_prod"
DATABASE_USER="analytics"
RETENTION_DAYS=90
LOG_FILE="/var/log/analytics/backup.log"

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

# Create backup directory structure
create_backup_structure() {
    local backup_date=$(date +%Y%m%d_%H%M%S)
    BACKUP_DIR="$BACKUP_ROOT/backup_$backup_date"
    
    mkdir -p "$BACKUP_DIR"/{database,config,data,logs}
    
    log "Created backup directory: $BACKUP_DIR"
}

# Backup database
backup_database() {
    log "Starting database backup..."
    
    # PostgreSQL backup
    if command -v pg_dump &> /dev/null; then
        log "Backing up PostgreSQL database..."
        pg_dump -h localhost -U "$DATABASE_USER" -W "$DATABASE_NAME" \
            --verbose --clean --if-exists --create \
            > "$BACKUP_DIR/database/postgresql_${DATABASE_NAME}.sql" 2>> "$LOG_FILE"
        
        # Compress the backup
        gzip "$BACKUP_DIR/database/postgresql_${DATABASE_NAME}.sql"
        success "PostgreSQL database backup completed"
    else
        warning "pg_dump not found, skipping PostgreSQL backup"
    fi
    
    # SQLite backup (if applicable)
    if [[ -f "$APP_ROOT/data/analytics.db" ]]; then
        log "Backing up SQLite database..."
        sqlite3 "$APP_ROOT/data/analytics.db" ".backup '$BACKUP_DIR/database/analytics_sqlite.db'"
        gzip "$BACKUP_DIR/database/analytics_sqlite.db"
        success "SQLite database backup completed"
    fi
    
    # Redis backup (if applicable)
    if command -v redis-cli &> /dev/null && redis-cli ping &> /dev/null; then
        log "Backing up Redis data..."
        redis-cli --rdb "$BACKUP_DIR/database/redis_dump.rdb" &> /dev/null
        gzip "$BACKUP_DIR/database/redis_dump.rdb"
        success "Redis backup completed"
    fi
}

# Backup configuration files
backup_configuration() {
    log "Backing up configuration files..."
    
    # Application configuration
    if [[ -d "$APP_ROOT/config" ]]; then
        cp -r "$APP_ROOT/config" "$BACKUP_DIR/"
        # Remove sensitive files from backup (they should be handled separately)
        rm -f "$BACKUP_DIR/config/secrets.env" 2>/dev/null || true
        success "Configuration files backed up"
    fi
    
    # Nginx configuration
    if [[ -d "/etc/nginx" ]]; then
        mkdir -p "$BACKUP_DIR/config/nginx"
        cp -r /etc/nginx/sites-available "$BACKUP_DIR/config/nginx/"
        cp -r /etc/nginx/sites-enabled "$BACKUP_DIR/config/nginx/"
        cp /etc/nginx/nginx.conf "$BACKUP_DIR/config/nginx/" 2>/dev/null || true
        success "Nginx configuration backed up"
    fi
    
    # Systemd service files
    if [[ -f "/etc/systemd/system/analytics-api.service" ]]; then
        mkdir -p "$BACKUP_DIR/config/systemd"
        cp /etc/systemd/system/analytics-*.service "$BACKUP_DIR/config/systemd/" 2>/dev/null || true
        success "Systemd configuration backed up"
    fi
}

# Backup application data
backup_application_data() {
    log "Backing up application data..."
    
    # Data files
    if [[ -d "$APP_ROOT/data" ]]; then
        cp -r "$APP_ROOT/data" "$BACKUP_DIR/"
        success "Application data backed up"
    fi
    
    # Model files
    if [[ -d "$APP_ROOT/models" ]]; then
        cp -r "$APP_ROOT/models" "$BACKUP_DIR/"
        success "Model files backed up"
    fi
    
    # Cache data (selective backup)
    if [[ -d "$APP_ROOT/cache" ]]; then
        mkdir -p "$BACKUP_DIR/cache"
        find "$APP_ROOT/cache" -name "*.pkl" -o -name "*.json" | head -100 | xargs -I {} cp {} "$BACKUP_DIR/cache/" 2>/dev/null || true
        log "Cache data selectively backed up"
    fi
}

# Backup logs
backup_logs() {
    log "Backing up recent logs..."
    
    # Application logs
    if [[ -d "$APP_ROOT/logs" ]]; then
        # Only backup logs from the last 7 days
        find "$APP_ROOT/logs" -name "*.log" -mtime -7 -exec cp {} "$BACKUP_DIR/logs/" \;
        success "Recent application logs backed up"
    fi
    
    # System logs related to our services
    if [[ -d "/var/log" ]]; then
        mkdir -p "$BACKUP_DIR/logs/system"
        # Backup nginx logs
        find /var/log/nginx -name "*analytics*" -mtime -7 -exec cp {} "$BACKUP_DIR/logs/system/" \; 2>/dev/null || true
        # Backup systemd journal for our services
        journalctl -u analytics-api.service --since="7 days ago" > "$BACKUP_DIR/logs/system/analytics-api.journal" 2>/dev/null || true
        journalctl -u analytics-worker.service --since="7 days ago" > "$BACKUP_DIR/logs/system/analytics-worker.journal" 2>/dev/null || true
        journalctl -u analytics-scheduler.service --since="7 days ago" > "$BACKUP_DIR/logs/system/analytics-scheduler.journal" 2>/dev/null || true
        success "System logs backed up"
    fi
}

# Create backup manifest
create_backup_manifest() {
    log "Creating backup manifest..."
    
    cat > "$BACKUP_DIR/manifest.txt" << EOF
Snowflake Analytics Backup
========================

Backup Information:
- Backup Date: $(date)
- Backup Type: Full System Backup
- Retention Policy: $RETENTION_DAYS days
- Created by: backup.sh script

System Information:
- Hostname: $(hostname)
- OS: $(uname -a)
- Disk Usage: $(df -h "$APP_ROOT")

Application Information:
- Version: $(cat "$APP_ROOT/VERSION" 2>/dev/null || echo "Unknown")
- Git Commit: $(git -C "$APP_ROOT" rev-parse HEAD 2>/dev/null || echo "Unknown")
- Last Deployment: $(stat -c %y "$APP_ROOT/src" 2>/dev/null || echo "Unknown")

Service Status:
- analytics-api: $(systemctl is-active analytics-api.service 2>/dev/null || echo "unknown")
- analytics-worker: $(systemctl is-active analytics-worker.service 2>/dev/null || echo "unknown")
- analytics-scheduler: $(systemctl is-active analytics-scheduler.service 2>/dev/null || echo "unknown")

Backup Contents:
$(find "$BACKUP_DIR" -type f -exec ls -lh {} \; | head -20)

Total Backup Size: $(du -sh "$BACKUP_DIR" | cut -f1)
EOF
    
    success "Backup manifest created"
}

# Compress and encrypt backup
compress_backup() {
    log "Compressing backup..."
    
    cd "$BACKUP_ROOT"
    local backup_name=$(basename "$BACKUP_DIR")
    
    # Create compressed archive
    tar -czf "${backup_name}.tar.gz" "$backup_name"
    
    # Encrypt if encryption key is available
    if [[ -n "${BACKUP_ENCRYPTION_KEY:-}" ]]; then
        log "Encrypting backup..."
        openssl enc -aes-256-cbc -salt -in "${backup_name}.tar.gz" -out "${backup_name}.tar.gz.enc" -pass "pass:$BACKUP_ENCRYPTION_KEY"
        rm "${backup_name}.tar.gz"
        success "Backup compressed and encrypted"
    else
        success "Backup compressed"
    fi
    
    # Remove uncompressed backup
    rm -rf "$BACKUP_DIR"
}

# Cleanup old backups
cleanup_old_backups() {
    log "Cleaning up old backups..."
    
    # Remove backups older than retention period
    find "$BACKUP_ROOT" -name "backup_*.tar.gz*" -mtime +$RETENTION_DAYS -delete
    
    # Keep at least 3 backups even if they exceed retention period
    local backup_count=$(find "$BACKUP_ROOT" -name "backup_*.tar.gz*" | wc -l)
    if [[ $backup_count -lt 3 ]]; then
        log "Keeping all $backup_count backups (minimum retention)"
    else
        # Remove excess backups beyond retention + minimum
        find "$BACKUP_ROOT" -name "backup_*.tar.gz*" -type f -printf '%T@ %p\n' | sort -n | head -n -10 | cut -d' ' -f2- | xargs -r rm
        success "Old backups cleaned up"
    fi
}

# Verify backup integrity
verify_backup() {
    log "Verifying backup integrity..."
    
    cd "$BACKUP_ROOT"
    local backup_file=$(find . -name "backup_*.tar.gz*" -type f -newer "$LOG_FILE" | head -1)
    
    if [[ -n "$backup_file" ]]; then
        if [[ "$backup_file" == *.enc ]]; then
            # Verify encrypted backup
            if [[ -n "${BACKUP_ENCRYPTION_KEY:-}" ]]; then
                openssl enc -aes-256-cbc -d -in "$backup_file" -pass "pass:$BACKUP_ENCRYPTION_KEY" | tar -tzf - >/dev/null
                success "Encrypted backup integrity verified"
            else
                warning "Cannot verify encrypted backup - no decryption key"
            fi
        else
            # Verify regular backup
            tar -tzf "$backup_file" >/dev/null
            success "Backup integrity verified"
        fi
    else
        warning "No recent backup file found for verification"
    fi
}

# Send backup notification
send_notification() {
    local status="$1"
    local message="$2"
    
    # Send email notification if configured
    if [[ "${BACKUP_EMAIL_ENABLED:-false}" == "true" && -n "${BACKUP_EMAIL_TO:-}" ]]; then
        echo "$message" | mail -s "Analytics Backup $status" "$BACKUP_EMAIL_TO"
    fi
    
    # Send Slack notification if configured
    if [[ "${BACKUP_SLACK_ENABLED:-false}" == "true" && -n "${BACKUP_SLACK_WEBHOOK:-}" ]]; then
        curl -X POST -H 'Content-type: application/json' \
            --data "{\"text\":\"Analytics Backup $status: $message\"}" \
            "$BACKUP_SLACK_WEBHOOK" >/dev/null 2>&1 || true
    fi
}

# Main backup function
main() {
    local start_time=$(date +%s)
    
    log "Starting backup process..."
    
    # Source environment if available
    if [[ -f "$APP_ROOT/config/production.env" ]]; then
        set -a
        source "$APP_ROOT/config/production.env"
        set +a
    fi
    
    # Create backup
    create_backup_structure
    backup_database
    backup_configuration
    backup_application_data
    backup_logs
    create_backup_manifest
    compress_backup
    verify_backup
    cleanup_old_backups
    
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    
    success "Backup completed successfully in ${duration}s"
    send_notification "SUCCESS" "Backup completed successfully in ${duration}s"
    
    log "Backup finished at $(date)"
}

# Error handling
trap 'error "Backup failed unexpectedly"' ERR

# Script entry point
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi