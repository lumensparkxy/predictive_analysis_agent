"""
Encryption Management System for Snowflake Analytics
Handles data encryption at rest and in transit, key management, and secure storage.
"""

import os
import base64
import secrets
import hashlib
from typing import Optional, Dict, Any, Union
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import structlog

logger = structlog.get_logger(__name__)


class EncryptionManager:
    """
    Comprehensive encryption management for production security.
    Handles symmetric and asymmetric encryption, key derivation, and secure storage.
    """
    
    def __init__(self, master_key: Optional[str] = None):
        """Initialize encryption manager with master key."""
        self.master_key = master_key or os.getenv('ENCRYPTION_KEY')
        if not self.master_key:
            raise ValueError("Master encryption key not provided")
        
        if len(self.master_key) != 32:
            # Derive a 32-byte key from the provided key
            self.master_key = self._derive_key(self.master_key.encode(), b'salt')[:32]
        
        self.fernet = Fernet(base64.urlsafe_b64encode(self.master_key.encode()[:32]))
        self._field_keys: Dict[str, Fernet] = {}
        
        logger.info("EncryptionManager initialized")
    
    def _derive_key(self, password: bytes, salt: bytes, iterations: int = 100000) -> bytes:
        """Derive encryption key using PBKDF2."""
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=iterations,
        )
        return kdf.derive(password)
    
    def encrypt_field(self, data: Union[str, bytes], field_name: str) -> str:
        """
        Encrypt a specific field with field-specific encryption.
        Each field gets its own derived key for additional security.
        """
        try:
            if isinstance(data, str):
                data = data.encode('utf-8')
            
            # Get or create field-specific key
            if field_name not in self._field_keys:
                field_salt = hashlib.sha256(field_name.encode()).digest()[:16]
                field_key = self._derive_key(self.master_key.encode(), field_salt)
                self._field_keys[field_name] = Fernet(base64.urlsafe_b64encode(field_key))
            
            encrypted_data = self._field_keys[field_name].encrypt(data)
            
            logger.debug("Field encrypted", field=field_name)
            return base64.urlsafe_b64encode(encrypted_data).decode('ascii')
            
        except Exception as e:
            logger.error("Field encryption failed", field=field_name, error=str(e))
            raise
    
    def decrypt_field(self, encrypted_data: str, field_name: str) -> str:
        """Decrypt a field-encrypted value."""
        try:
            # Get field-specific key
            if field_name not in self._field_keys:
                field_salt = hashlib.sha256(field_name.encode()).digest()[:16]
                field_key = self._derive_key(self.master_key.encode(), field_salt)
                self._field_keys[field_name] = Fernet(base64.urlsafe_b64encode(field_key))
            
            encrypted_bytes = base64.urlsafe_b64decode(encrypted_data.encode('ascii'))
            decrypted_data = self._field_keys[field_name].decrypt(encrypted_bytes)
            
            logger.debug("Field decrypted", field=field_name)
            return decrypted_data.decode('utf-8')
            
        except Exception as e:
            logger.error("Field decryption failed", field=field_name, error=str(e))
            raise
    
    def encrypt_sensitive_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Encrypt sensitive fields in a data dictionary.
        Automatically encrypts fields marked as sensitive.
        """
        sensitive_fields = {
            'password', 'secret', 'key', 'token', 'credential',
            'private_key', 'api_key', 'auth_token', 'session_key'
        }
        
        encrypted_data = {}
        for key, value in data.items():
            if any(sensitive_field in key.lower() for sensitive_field in sensitive_fields):
                if value is not None:
                    encrypted_data[key] = self.encrypt_field(str(value), key)
                    logger.debug("Sensitive field encrypted", field=key)
                else:
                    encrypted_data[key] = None
            else:
                encrypted_data[key] = value
        
        return encrypted_data
    
    def decrypt_sensitive_data(self, encrypted_data: Dict[str, Any]) -> Dict[str, Any]:
        """Decrypt sensitive fields in a data dictionary."""
        sensitive_fields = {
            'password', 'secret', 'key', 'token', 'credential',
            'private_key', 'api_key', 'auth_token', 'session_key'
        }
        
        decrypted_data = {}
        for key, value in encrypted_data.items():
            if any(sensitive_field in key.lower() for sensitive_field in sensitive_fields):
                if value is not None:
                    decrypted_data[key] = self.decrypt_field(value, key)
                else:
                    decrypted_data[key] = None
            else:
                decrypted_data[key] = value
        
        return decrypted_data
    
    def encrypt_file(self, file_path: str, output_path: Optional[str] = None) -> str:
        """Encrypt a file using the master key."""
        try:
            with open(file_path, 'rb') as f:
                file_data = f.read()
            
            encrypted_data = self.fernet.encrypt(file_data)
            
            if output_path is None:
                output_path = file_path + '.enc'
            
            with open(output_path, 'wb') as f:
                f.write(encrypted_data)
            
            logger.info("File encrypted", input=file_path, output=output_path)
            return output_path
            
        except Exception as e:
            logger.error("File encryption failed", file=file_path, error=str(e))
            raise
    
    def decrypt_file(self, encrypted_file_path: str, output_path: Optional[str] = None) -> str:
        """Decrypt a file using the master key."""
        try:
            with open(encrypted_file_path, 'rb') as f:
                encrypted_data = f.read()
            
            decrypted_data = self.fernet.decrypt(encrypted_data)
            
            if output_path is None:
                output_path = encrypted_file_path.replace('.enc', '')
            
            with open(output_path, 'wb') as f:
                f.write(decrypted_data)
            
            logger.info("File decrypted", input=encrypted_file_path, output=output_path)
            return output_path
            
        except Exception as e:
            logger.error("File decryption failed", file=encrypted_file_path, error=str(e))
            raise
    
    def generate_api_key(self, length: int = 32) -> str:
        """Generate a secure API key."""
        return secrets.token_urlsafe(length)
    
    def generate_secret_key(self, length: int = 32) -> str:
        """Generate a secure secret key."""
        return secrets.token_hex(length)
    
    def hash_password(self, password: str, salt: Optional[bytes] = None) -> Dict[str, str]:
        """Hash a password with salt using PBKDF2."""
        if salt is None:
            salt = os.urandom(32)
        
        pwdhash = hashlib.pbkdf2_hmac('sha256',
                                     password.encode('utf-8'),
                                     salt,
                                     100000)
        
        return {
            'hash': base64.b64encode(pwdhash).decode('ascii'),
            'salt': base64.b64encode(salt).decode('ascii')
        }
    
    def verify_password(self, password: str, stored_hash: str, stored_salt: str) -> bool:
        """Verify a password against stored hash and salt."""
        try:
            salt = base64.b64decode(stored_salt.encode('ascii'))
            stored_hash_bytes = base64.b64decode(stored_hash.encode('ascii'))
            
            pwdhash = hashlib.pbkdf2_hmac('sha256',
                                         password.encode('utf-8'),
                                         salt,
                                         100000)
            
            return pwdhash == stored_hash_bytes
        except Exception:
            return False
    
    def generate_rsa_keypair(self, key_size: int = 2048) -> Dict[str, bytes]:
        """Generate RSA public/private keypair."""
        private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=key_size
        )
        
        public_key = private_key.public_key()
        
        private_pem = private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption()
        )
        
        public_pem = public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        )
        
        logger.info("RSA keypair generated", key_size=key_size)
        
        return {
            'private_key': private_pem,
            'public_key': public_pem
        }
    
    def encrypt_backup(self, data: bytes, additional_key: Optional[str] = None) -> bytes:
        """Encrypt backup data with additional key if provided."""
        try:
            if additional_key:
                # Use additional key for backup encryption
                backup_key = self._derive_key(additional_key.encode(), b'backup_salt')
                backup_fernet = Fernet(base64.urlsafe_b64encode(backup_key))
                return backup_fernet.encrypt(data)
            else:
                # Use master key
                return self.fernet.encrypt(data)
        except Exception as e:
            logger.error("Backup encryption failed", error=str(e))
            raise
    
    def decrypt_backup(self, encrypted_data: bytes, additional_key: Optional[str] = None) -> bytes:
        """Decrypt backup data with additional key if provided."""
        try:
            if additional_key:
                # Use additional key for backup decryption
                backup_key = self._derive_key(additional_key.encode(), b'backup_salt')
                backup_fernet = Fernet(base64.urlsafe_b64encode(backup_key))
                return backup_fernet.decrypt(encrypted_data)
            else:
                # Use master key
                return self.fernet.decrypt(encrypted_data)
        except Exception as e:
            logger.error("Backup decryption failed", error=str(e))
            raise
    
    def rotate_key(self, new_master_key: str) -> bool:
        """
        Rotate the master encryption key.
        Note: This requires re-encrypting all existing encrypted data.
        """
        try:
            old_fernet = self.fernet
            
            # Create new fernet with new key
            if len(new_master_key) != 32:
                new_master_key = self._derive_key(new_master_key.encode(), b'salt')[:32]
            
            new_fernet = Fernet(base64.urlsafe_b64encode(new_master_key.encode()[:32]))
            
            # Update instance
            self.master_key = new_master_key
            self.fernet = new_fernet
            self._field_keys = {}  # Clear field keys to force regeneration
            
            logger.warning("Encryption key rotated - re-encrypt existing data")
            return True
            
        except Exception as e:
            logger.error("Key rotation failed", error=str(e))
            return False