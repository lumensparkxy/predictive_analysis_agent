# Production Deployment Guide

## Overview
This guide covers the deployment and operation of the Snowflake Analytics system in production environments.

## Prerequisites
- Ubuntu 20.04+ server with 8GB+ RAM, 50GB+ disk
- PostgreSQL 13+ database
- Redis 6+ cache
- Nginx web server
- SSL certificate for HTTPS
- Domain name and DNS configuration

## Initial Setup

### 1. Server Preparation
```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install required packages
sudo apt install -y postgresql redis-server nginx python3.9 python3-pip git supervisor

# Create application user
sudo useradd -r -s /bin/bash -d /opt/analytics analytics
sudo mkdir -p /opt/analytics
sudo chown analytics:analytics /opt/analytics
```

### 2. Application Deployment
```bash
# Switch to analytics user
sudo su - analytics

# Clone repository
git clone https://github.com/your-org/predictive_analysis_agent.git
cd predictive_analysis_agent

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
pip install -e .
```

### 3. Configuration Setup
```bash
# Copy environment files
cp deployment/production/environment/production.env config/
cp deployment/production/environment/config.yaml config/

# Edit configuration with your settings
nano config/production.env
nano config/config.yaml

# Set secure file permissions
chmod 600 config/secrets.env
```

### 4. Database Setup
```bash
# Create database and user
sudo -u postgres createdb analytics_prod
sudo -u postgres createuser analytics
sudo -u postgres psql -c "GRANT ALL PRIVILEGES ON DATABASE analytics_prod TO analytics;"

# Run migrations (if any)
python scripts/setup.py
```

### 5. Service Installation
```bash
# Install systemd services
sudo cp deployment/production/systemd/*.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable analytics-api analytics-worker analytics-scheduler
```

### 6. Nginx Configuration
```bash
# Install nginx configuration
sudo cp deployment/production/nginx/nginx.conf /etc/nginx/
sudo cp deployment/production/nginx/sites-available/analytics.conf /etc/nginx/sites-available/
sudo ln -s /etc/nginx/sites-available/analytics.conf /etc/nginx/sites-enabled/

# Install SSL certificate
# (Use Let's Encrypt or your certificate provider)

# Test and restart nginx
sudo nginx -t
sudo systemctl restart nginx
```

## Starting Services

### Start All Services
```bash
# Start database services
sudo systemctl start postgresql redis-server

# Start analytics services
sudo systemctl start analytics-api analytics-worker analytics-scheduler

# Start web server
sudo systemctl start nginx
```

### Verify Deployment
```bash
# Run health check
./deployment/production/scripts/health-check.sh

# Check service status
systemctl status analytics-api analytics-worker analytics-scheduler nginx

# Test API endpoint
curl https://your-domain.com/health
```

## Monitoring and Maintenance

### Daily Operations
- Monitor system health via `/health` endpoint
- Check service logs: `journalctl -u analytics-api -f`
- Review application logs in `/opt/analytics/logs/`
- Automated backups run daily at 2 AM

### Weekly Operations
- Review security alerts and monitoring dashboards
- Check disk space usage
- Review backup integrity
- Update system packages if needed

### Monthly Operations
- Review and test disaster recovery procedures
- Analyze performance metrics and optimize if needed
- Update SSL certificates if expiring
- Conduct security vulnerability scans

## Backup and Recovery

### Automated Backups
Backups run automatically daily via cron job:
```bash
0 2 * * * /opt/analytics/deployment/production/scripts/backup.sh
```

### Manual Backup
```bash
cd /opt/analytics
./deployment/production/scripts/backup.sh
```

### Recovery Procedure
```bash
# Stop services
sudo systemctl stop analytics-api analytics-worker analytics-scheduler

# Run recovery
./deployment/production/scripts/rollback.sh

# Select backup to restore when prompted
# Services will be automatically restarted after recovery
```

## Troubleshooting

### Common Issues

#### API Service Won't Start
1. Check logs: `journalctl -u analytics-api -f`
2. Verify configuration files
3. Check database connectivity
4. Ensure ports are available

#### High CPU/Memory Usage
1. Check resource usage: `htop`
2. Review application logs for errors
3. Scale worker processes if needed
4. Optimize database queries

#### Database Connection Issues
1. Verify PostgreSQL is running: `systemctl status postgresql`
2. Check connection settings in config
3. Verify database user permissions
4. Check firewall rules

### Log Locations
- Application logs: `/opt/analytics/logs/`
- System logs: `journalctl -u analytics-*`
- Nginx logs: `/var/log/nginx/analytics_*.log`
- Database logs: `/var/log/postgresql/`

### Performance Tuning
- Adjust worker process counts based on CPU cores
- Optimize database connection pool sizes
- Configure Redis memory limits
- Tune Nginx worker processes

## Security Considerations

### Regular Security Tasks
- Keep system packages updated
- Monitor security alerts
- Review access logs for suspicious activity
- Rotate API keys and certificates regularly
- Run vulnerability scans monthly

### Access Control
- Use SSH key authentication only
- Limit sudo access to necessary personnel
- Enable firewall with minimal required ports
- Regular access reviews and user deprovisioning

## Scaling

### Vertical Scaling
- Increase server CPU/RAM as needed
- Monitor resource utilization trends
- Scale database connections accordingly

### Horizontal Scaling
- Deploy multiple API instances behind load balancer
- Use separate database server for high load
- Implement Redis clustering for cache
- Consider container orchestration (Kubernetes)

## Support Contacts
- Operations Team: ops@yourcompany.com
- Security Team: security@yourcompany.com
- Development Team: dev@yourcompany.com