"""
Backup Manager for Snowflake Analytics
Automated backup management with encryption, compression, and retention policies.
"""

import os
import shutil
import subprocess
import gzip
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import structlog

from ..security.production.encryption_manager import EncryptionManager

logger = structlog.get_logger(__name__)


class BackupManager:
    """Automated backup management system."""
    
    def __init__(self, encryption_manager: Optional[EncryptionManager] = None):
        self.encryption_manager = encryption_manager or EncryptionManager()
        self.backup_root = os.getenv('BACKUP_ROOT', '/opt/analytics/backups')
        self.app_root = os.getenv('APP_ROOT', '/opt/analytics')
        
        # Ensure backup directory exists
        os.makedirs(self.backup_root, exist_ok=True)
        
    def create_full_backup(self) -> Dict[str, Any]:
        """Create a full system backup."""
        backup_id = f"full_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        backup_path = os.path.join(self.backup_root, backup_id)
        os.makedirs(backup_path, exist_ok=True)
        
        result = {
            'backup_id': backup_id,
            'backup_type': 'full',
            'timestamp': datetime.utcnow().isoformat(),
            'components': {},
            'success': True,
            'errors': []
        }
        
        # Backup components
        components = {
            'database': self._backup_database,
            'application': self._backup_application_data,
            'configuration': self._backup_configuration,
            'logs': self._backup_logs
        }
        
        for component, backup_func in components.items():
            try:
                component_result = backup_func(backup_path)
                result['components'][component] = component_result
                if not component_result.get('success', False):
                    result['success'] = False
            except Exception as e:
                error_msg = f"Failed to backup {component}: {str(e)}"
                result['errors'].append(error_msg)
                result['success'] = False
                logger.error(error_msg)
        
        # Create backup manifest
        self._create_backup_manifest(backup_path, result)
        
        # Compress and encrypt backup
        if result['success']:
            self._compress_and_encrypt_backup(backup_path)
        
        return result
    
    def _backup_database(self, backup_path: str) -> Dict[str, Any]:
        """Backup PostgreSQL database."""
        try:
            db_backup_path = os.path.join(backup_path, 'database')
            os.makedirs(db_backup_path, exist_ok=True)
            
            # PostgreSQL backup
            pg_dump_cmd = [
                'pg_dump',
                '-h', os.getenv('DB_HOST', 'localhost'),
                '-p', os.getenv('DB_PORT', '5432'),
                '-U', os.getenv('DB_USER', 'analytics'),
                '-d', os.getenv('DB_NAME', 'analytics_prod'),
                '--verbose', '--clean', '--if-exists', '--create',
                '-f', os.path.join(db_backup_path, 'postgresql.sql')
            ]
            
            env = os.environ.copy()
            env['PGPASSWORD'] = os.getenv('DB_PASSWORD', '')
            
            result = subprocess.run(pg_dump_cmd, env=env, capture_output=True, text=True)
            
            if result.returncode == 0:
                # Compress the backup
                with open(os.path.join(db_backup_path, 'postgresql.sql'), 'rb') as f_in:
                    with gzip.open(os.path.join(db_backup_path, 'postgresql.sql.gz'), 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)
                os.remove(os.path.join(db_backup_path, 'postgresql.sql'))
                
                return {'success': True, 'size_bytes': os.path.getsize(os.path.join(db_backup_path, 'postgresql.sql.gz'))}
            else:
                return {'success': False, 'error': result.stderr}
                
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _backup_application_data(self, backup_path: str) -> Dict[str, Any]:
        """Backup application data and models."""
        try:
            app_backup_path = os.path.join(backup_path, 'application')
            os.makedirs(app_backup_path, exist_ok=True)
            
            # Backup data directory
            data_path = os.path.join(self.app_root, 'data')
            if os.path.exists(data_path):
                shutil.copytree(data_path, os.path.join(app_backup_path, 'data'))
            
            # Backup models directory
            models_path = os.path.join(self.app_root, 'models')
            if os.path.exists(models_path):
                shutil.copytree(models_path, os.path.join(app_backup_path, 'models'))
            
            # Calculate total size
            total_size = sum(
                os.path.getsize(os.path.join(dirpath, filename))
                for dirpath, dirnames, filenames in os.walk(app_backup_path)
                for filename in filenames
            )
            
            return {'success': True, 'size_bytes': total_size}
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _backup_configuration(self, backup_path: str) -> Dict[str, Any]:
        """Backup configuration files."""
        try:
            config_backup_path = os.path.join(backup_path, 'configuration')
            os.makedirs(config_backup_path, exist_ok=True)
            
            # Backup application configuration
            config_path = os.path.join(self.app_root, 'config')
            if os.path.exists(config_path):
                shutil.copytree(config_path, os.path.join(config_backup_path, 'app'))
            
            # Backup nginx configuration
            nginx_config = '/etc/nginx/sites-available/analytics.conf'
            if os.path.exists(nginx_config):
                os.makedirs(os.path.join(config_backup_path, 'nginx'), exist_ok=True)
                shutil.copy2(nginx_config, os.path.join(config_backup_path, 'nginx'))
            
            # Backup systemd services
            systemd_services = [
                '/etc/systemd/system/analytics-api.service',
                '/etc/systemd/system/analytics-worker.service', 
                '/etc/systemd/system/analytics-scheduler.service'
            ]
            
            systemd_backup_path = os.path.join(config_backup_path, 'systemd')
            os.makedirs(systemd_backup_path, exist_ok=True)
            
            for service in systemd_services:
                if os.path.exists(service):
                    shutil.copy2(service, systemd_backup_path)
            
            total_size = sum(
                os.path.getsize(os.path.join(dirpath, filename))
                for dirpath, dirnames, filenames in os.walk(config_backup_path)
                for filename in filenames
            )
            
            return {'success': True, 'size_bytes': total_size}
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _backup_logs(self, backup_path: str) -> Dict[str, Any]:
        """Backup recent log files."""
        try:
            logs_backup_path = os.path.join(backup_path, 'logs')
            os.makedirs(logs_backup_path, exist_ok=True)
            
            # Backup application logs (last 7 days)
            app_logs_path = os.path.join(self.app_root, 'logs')
            if os.path.exists(app_logs_path):
                cutoff_time = datetime.now() - timedelta(days=7)
                for log_file in os.listdir(app_logs_path):
                    log_file_path = os.path.join(app_logs_path, log_file)
                    if os.path.isfile(log_file_path):
                        mod_time = datetime.fromtimestamp(os.path.getmtime(log_file_path))
                        if mod_time > cutoff_time:
                            shutil.copy2(log_file_path, logs_backup_path)
            
            total_size = sum(
                os.path.getsize(os.path.join(dirpath, filename))
                for dirpath, dirnames, filenames in os.walk(logs_backup_path)
                for filename in filenames
            )
            
            return {'success': True, 'size_bytes': total_size}
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _create_backup_manifest(self, backup_path: str, backup_result: Dict[str, Any]):
        """Create backup manifest file."""
        manifest = {
            'backup_id': backup_result['backup_id'],
            'backup_type': backup_result['backup_type'],
            'timestamp': backup_result['timestamp'],
            'hostname': os.uname().nodename,
            'components': backup_result['components'],
            'total_size_bytes': sum(
                comp.get('size_bytes', 0) 
                for comp in backup_result['components'].values()
                if comp.get('success', False)
            ),
            'success': backup_result['success'],
            'errors': backup_result['errors']
        }
        
        manifest_path = os.path.join(backup_path, 'manifest.json')
        import json
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)
    
    def _compress_and_encrypt_backup(self, backup_path: str):
        """Compress and encrypt the backup."""
        try:
            # Create compressed archive
            archive_path = f"{backup_path}.tar.gz"
            subprocess.run([
                'tar', '-czf', archive_path,
                '-C', os.path.dirname(backup_path),
                os.path.basename(backup_path)
            ], check=True)
            
            # Encrypt if encryption key available
            encryption_key = os.getenv('BACKUP_ENCRYPTION_KEY')
            if encryption_key:
                encrypted_path = f"{archive_path}.enc"
                with open(archive_path, 'rb') as f_in:
                    encrypted_data = self.encryption_manager.encrypt_backup(f_in.read(), encryption_key)
                    with open(encrypted_path, 'wb') as f_out:
                        f_out.write(encrypted_data)
                
                os.remove(archive_path)
                logger.info("Backup encrypted", backup_path=encrypted_path)
            
            # Remove uncompressed backup directory
            shutil.rmtree(backup_path)
            
        except Exception as e:
            logger.error("Failed to compress/encrypt backup", error=str(e))
    
    def list_backups(self) -> List[Dict[str, Any]]:
        """List all available backups."""
        backups = []
        
        for item in os.listdir(self.backup_root):
            item_path = os.path.join(self.backup_root, item)
            
            if os.path.isfile(item_path) and (item.endswith('.tar.gz') or item.endswith('.tar.gz.enc')):
                stat_info = os.stat(item_path)
                backups.append({
                    'backup_id': item,
                    'path': item_path,
                    'size_bytes': stat_info.st_size,
                    'created_at': datetime.fromtimestamp(stat_info.st_ctime).isoformat(),
                    'encrypted': item.endswith('.enc')
                })
        
        return sorted(backups, key=lambda x: x['created_at'], reverse=True)
    
    def delete_backup(self, backup_id: str) -> bool:
        """Delete a specific backup."""
        try:
            backup_path = os.path.join(self.backup_root, backup_id)
            if os.path.exists(backup_path):
                os.remove(backup_path)
                logger.info("Backup deleted", backup_id=backup_id)
                return True
            return False
        except Exception as e:
            logger.error("Failed to delete backup", backup_id=backup_id, error=str(e))
            return False