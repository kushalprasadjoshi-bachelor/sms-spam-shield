import logging
import sys
from typing import Optional


def setup_logger(name: Optional[str] = None) -> logging.Logger:
    """Setup application logger"""
    
    logger = logging.getLogger(name or "sms_spam_shield")
    
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(formatter)
        
        logger.addHandler(console_handler)
        logger.propagate = False
    
    return logger


logger = setup_logger()