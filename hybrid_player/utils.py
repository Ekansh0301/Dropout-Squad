"""
Utility functions for hybrid player data processing.
File I/O, data loading, and basic operations.
"""
import os
import json
import pickle
import pandas as pd
from typing import List, Dict, Any, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def ensure_dir(path: str) -> None:
    """Create directory if it doesn't exist."""
    os.makedirs(path, exist_ok=True)

def load_json_files(directory: str) -> List[Dict]:
    """Load all JSON files from a directory."""
    data = []
    if not os.path.exists(directory):
        logger.warning(f"Directory not found: {directory}")
        return data
        
    for filename in os.listdir(directory):
        if filename.endswith('.json'):
            filepath = os.path.join(directory, filename)
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    data.append(json.load(f))
                logger.info(f"Loaded {filename}")
            except Exception as e:
                logger.error(f"Error loading {filename}: {e}")
    return data

def save_pickle(obj: Any, filepath: str) -> None:
    """Save object as pickle file."""
    ensure_dir(os.path.dirname(filepath))
    with open(filepath, 'wb') as f:
        pickle.dump(obj, f)

def load_pickle(filepath: str) -> Any:
    """Load object from pickle file."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Pickle file not found: {filepath}")
    
    with open(filepath, 'rb') as f:
        return pickle.load(f)