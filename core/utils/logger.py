import logging
import os
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent.parent
STORAGE_DIR = BASE_DIR / "storage"
LOG_FILE_PATH = STORAGE_DIR / "logs.txt"

# Ensure storage directory exists
os.makedirs(STORAGE_DIR, exist_ok=True)

def get_logger(module_name: str) -> logging.Logger:
    """
    Returns a logger configured to write to storage/logs.txt
    and standard output, uniformly formatted.
    """
    logger = logging.getLogger(module_name)
    
    # Only configure if the logger doesn't already have handlers
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        
        # Create file handler
        file_handler = logging.FileHandler(LOG_FILE_PATH, mode='a', encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        
        # Create console handler (optional, good for terminal debugging)
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter('[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # Add handlers to the logger
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
    return logger
