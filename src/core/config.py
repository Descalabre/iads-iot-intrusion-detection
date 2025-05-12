"""
Configuration module for the Intrusion and Anomaly Detection System (IADS).
Handles all configuration settings, paths, and logging setup.
"""

import logging
import os
from typing import Dict, Any
import json

# Base paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC_DIR = os.path.join(BASE_DIR, "src")
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")

# Ensure required directories exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# File paths
MODEL_PATH = os.path.join(MODELS_DIR, "iads_model.h5")
CONFIG_FILE = os.path.join(BASE_DIR, "config.json")

# Default configuration
DEFAULT_CONFIG: Dict[str, Any] = {
    "log_level": "DEBUG",
    "model": {
        "batch_size": 32,
        "epochs": 10,
        "validation_split": 0.2
    },
    "detection": {
        "threshold": 0.8,
        "window_size": 100
    },
    "dashboard": {
        "host": "localhost",
        "port": 5000,
        "debug": True
    }
}

def load_config() -> Dict[str, Any]:
    """
    Load configuration from config.json file, creating it with defaults if it doesn't exist.
    
    Returns:
        Dict[str, Any]: Configuration dictionary
    """
    try:
        if os.path.exists(CONFIG_FILE):
            with open(CONFIG_FILE, 'r') as f:
                config = json.load(f)
                # Update with any missing default values
                for key, value in DEFAULT_CONFIG.items():
                    if key not in config:
                        config[key] = value
        else:
            config = DEFAULT_CONFIG
            # Create config file with defaults
            with open(CONFIG_FILE, 'w') as f:
                json.dump(DEFAULT_CONFIG, f, indent=4)
        
        return config
    
    except json.JSONDecodeError as e:
        logger.error(f"Error parsing config file: {e}")
        logger.warning("Using default configuration")
        return DEFAULT_CONFIG
    except Exception as e:
        logger.error(f"Unexpected error loading config: {e}")
        logger.warning("Using default configuration")
        return DEFAULT_CONFIG

# Load configuration
config = load_config()

# Configure logging
LOG_LEVELS = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL
}

def setup_logging() -> logging.Logger:
    """
    Configure logging with proper formatting and handlers.
    
    Returns:
        logging.Logger: Configured logger instance
    """
    try:
        # Create logs directory if it doesn't exist
        logs_dir = os.path.join(BASE_DIR, "logs")
        os.makedirs(logs_dir, exist_ok=True)
        
        # Determine log level from config
        log_level = LOG_LEVELS.get(config["log_level"].upper(), logging.DEBUG)
        
        # Configure logging
        logging.basicConfig(
            level=log_level,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(os.path.join(logs_dir, "iads.log"))
            ]
        )
        
        logger = logging.getLogger("IADS")
        logger.setLevel(log_level)
        
        return logger
    
    except Exception as e:
        # Fallback to basic logging if setup fails
        logging.basicConfig(
            level=logging.DEBUG,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
        )
        logger = logging.getLogger("IADS")
        logger.error(f"Error setting up logging: {e}")
        logger.warning("Using basic logging configuration")
        return logger

# Initialize logger
logger = setup_logging()

# Validate critical paths
def validate_paths() -> None:
    """Validate that all required paths and directories exist."""
    try:
        for path in [DATA_DIR, MODELS_DIR]:
            if not os.path.exists(path):
                logger.warning(f"Directory not found: {path}")
                os.makedirs(path)
                logger.info(f"Created directory: {path}")
    except Exception as e:
        logger.error(f"Error validating paths: {e}")
        raise

# Run validation
validate_paths()

# Export all configuration variables
__all__ = [
    'BASE_DIR',
    'SRC_DIR',
    'DATA_DIR',
    'MODELS_DIR',
    'MODEL_PATH',
    'config',
    'logger'
]
