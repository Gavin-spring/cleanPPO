# src/utils/logger.py
import logging
import sys
import os
from datetime import datetime

def setup_logger(run_name: str, log_dir: str, level=logging.INFO):
    """
    Configures the root logger for the entire application.
    This should be called only ONCE at the application's entry point.
    """
    logger = logging.getLogger() # Get the root logger

    # prevent reconfiguration of the logger
    if logger.hasHandlers():
        return

    logger.setLevel(level)

    # 1. File handler to log messages to a file
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"{run_name}_{timestamp}.log"
    log_filepath = os.path.join(log_dir, log_filename)
    
    file_handler = logging.FileHandler(log_filepath, mode='w')
    file_handler.setLevel(logging.DEBUG) # debug level for file logging
    file_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s"
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    # 2. Console handler to log messages to stdout
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO) # info level for console logging
    console_formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    logger.info(f"Logger initialized. All subsequent logs will be saved to: {log_filepath}")

# setup_logger should be called at the entry point of the application
def get_logger(name: str) -> logging.Logger:
    """A helper to get a logger instance."""
    return logging.getLogger(name)