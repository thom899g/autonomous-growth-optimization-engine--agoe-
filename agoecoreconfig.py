"""
AGOE Configuration Management
Centralized configuration with environment variable validation and type-safe settings.
Uses Firestore for distributed configuration management across ecosystem components.
"""
import os
import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum
import firebase_admin
from firebase_admin import credentials, firestore
from pydantic import BaseSettings, ValidationError

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GrowthPhase(Enum):
    """Growth phases for the ecosystem"""
    BOOTSTRAP = "bootstrap"
    ACCELERATION = "acceleration"
    MATURATION = "maturation"
    OPTIMIZATION = "optimization"

class MetricCategory(Enum):
    """Categories of growth metrics"""
    USER_ACQUISITION = "user_acquisition"
    ENGAGEMENT = "engagement"
    RETENTION = "retention"
    MONETIZATION = "monetization"
    TECHNICAL = "technical"

@dataclass
class FirestoreConfig:
    """Firestore configuration with validation"""
    project_id: str
    credential_path: Optional[str] = None
    collection_prefix: str = "agoe"
    
    def __post_init__(self):
        if not self.project_id:
            raise ValueError("Firestore project_id must be provided")
        
        # Check for environment variable fallback
        if not self.credential_path:
            self.credential_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
            
        if not self.credential_path or not os.path.exists(self.credential_path):
            logger.warning("Firestore credentials not found. Some features may be limited.")

class AGOESettings(BaseSettings):
    """Main AGOE settings with environment variable loading"""
    # Core settings
    environment: str = "development"
    growth_phase: GrowthPhase = GrowthPhase.BOOTSTRAP
    monitoring_interval_minutes: int = 15
    
    # Firestore settings
    firestore_project_id: str
    firestore_credential_path: Optional[str] = None
    
    # ML settings
    anomaly_detection_threshold: float = 0.85
    min_training_samples: int = 100
    
    # Safety settings
    max_actions_per_cycle: int = 5
    require_human_approval: bool = True
    
    class Config:
        env_file = ".env"
        env_prefix = "agoe_"
    
    def get_firestore_config(self) -> FirestoreConfig:
        """Get validated Firestore configuration"""
        return FirestoreConfig(
            project_id=self.firestore_project_id,
            credential_path=self.firestore_credential_path
        )

# Global configuration instance with lazy loading
_config_instance: Optional[AGOESettings] = None

def get_config() -> AGOESettings:
    """Singleton pattern for configuration access with validation"""
    global _config_instance
    
    if _config_instance is None:
        try:
            _config_instance = AGOESettings()
            logger.info(f"Loaded AGOE configuration for {_config_instance.environment} environment")
        except ValidationError as e:
            logger.error(f"Configuration validation failed: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected configuration error: {e}")
            raise
    
    return _config_instance

def initialize_firebase() -> firestore.Client:
    """Initialize Firebase Admin SDK with error handling"""
    config = get_config()
    firestore_config = config.get_firestore_config()
    
    try:
        # Check if Firebase app already exists
        if not firebase_admin._apps:
            if firestore_config.credential_path and os.path.exists(firestore_config.credential_path):
                cred = credentials.Certificate(firestore_config.credential_path)
                firebase_admin.initialize_app(cred, {
                    'projectId': firestore_config.project_id
                })
            else:
                # Use Application Default Credentials
                firebase_admin.initialize_app()
        
        db = firestore.client()
        logger.info(f"Firebase initialized successfully for project: {firestore_config.project_id}")
        return db
        
    except Exception as e:
        logger.error(f"Firebase initialization failed: {e}")
        # Return a mock client for development if Firebase fails
        if config.environment == "development":
            logger.warning("Using mock Firestore client for development")
            from unittest.mock import Mock
            return Mock()
        raise

# Export public interface
__all__ = [
    "get_config",
    "initialize_firebase",
    "GrowthPhase",
    "MetricCategory",
    "AGOESettings",
    "FirestoreConfig"
]