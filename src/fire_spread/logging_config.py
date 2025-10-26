"""
Logging configuration for the fire spread simulation.

This module provides centralized logging setup for the entire project.
"""

import logging
import sys
from pathlib import Path


def setup_logging(level=logging.INFO, log_to_file=False, log_dir="logs"):
    """
    Configure logging for the fire spread simulation.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_to_file: Whether to log to a file in addition to console
        log_dir: Directory for log files (only used if log_to_file=True)
    
    Example:
        # Basic setup (console only, INFO level)
        setup_logging()
        
        # Debug mode (console only, DEBUG level)
        setup_logging(level=logging.DEBUG)
        
        # Production mode (console + file, WARNING level)
        setup_logging(level=logging.WARNING, log_to_file=True)
    """
    # Create formatter
    formatter = logging.Formatter(
        fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    # Clear existing handlers
    root_logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # File handler (optional)
    if log_to_file:
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        
        from datetime import datetime
        log_file = log_path / f"fire_simulation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
        
        root_logger.info("Logging to file: %s", log_file)
    
    # Set levels for third-party libraries to reduce noise
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('requests').setLevel(logging.WARNING)
    
    return root_logger


# Quick setup functions for common scenarios
def setup_debug_logging():
    """Setup logging for development/debugging (DEBUG level, console only)."""
    return setup_logging(level=logging.DEBUG, log_to_file=False)


def setup_production_logging():
    """Setup logging for production (INFO level, console + file)."""
    return setup_logging(level=logging.INFO, log_to_file=True)


def setup_quiet_logging():
    """Setup minimal logging (WARNING level and above only)."""
    return setup_logging(level=logging.WARNING, log_to_file=False)
