"""
Main entry point for Lightweight IDS application.
Run this script to launch the PyQt5 GUI.

Usage:
    python main.py
"""

import sys
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ids_training.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# ── CRITICAL: import torch BEFORE PyQt5 ──────────────────────────────────────
# On Windows, PyQt5 loads DLLs that conflict with CUDA's DLL initialization.
# Importing torch first lets it claim the CUDA runtime before PyQt5 interferes.
import torch
logger.info(f"PyTorch {torch.__version__} loaded | "
            f"CUDA available: {torch.cuda.is_available()} | "
            f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
# ─────────────────────────────────────────────────────────────────────────────

# Safe to import PyQt5-based UI now
from ui import main

if __name__ == '__main__':
    logger.info("Starting Lightweight IDS Application...")
    try:
        main()
    except Exception as e:
        logger.error(f"Application error: {e}", exc_info=True)
        sys.exit(1)