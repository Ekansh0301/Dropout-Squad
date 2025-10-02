import os
from dataclasses import dataclass, field
from typing import List, Dict, Any

def get_base_dir():

    current_file_dir = os.path.dirname(os.path.abspath(__file__))
    hybrid_player_dir = current_file_dir
    project_root = os.path.dirname(hybrid_player_dir)
    return project_root

BASE_DIR = get_base_dir()

@dataclass
class DataConfig:
    crd3_base_path: str = field(default_factory=lambda: os.path.join(BASE_DIR, "data", "crd3"))
    light_base_path: str = field(default_factory=lambda: os.path.join(BASE_DIR, "data", "light"))
    output_path: str = field(default_factory=lambda: os.path.join(BASE_DIR, "data", "processed"))
    
    # CRD3 settings
    crd3_chunk_sizes: List[int] = field(default_factory=lambda: [2, 3, 4])
    dm_names: List[str] = field(default_factory=lambda: ["MATT", "DM", "DUNGEON MASTER", "GAME MASTER"])
    
    # LIGHT settings
    light_player_categories: List[str] = field(default_factory=lambda: ["player", "user", "human"])
    
    # Training data settings
    train_split: float = 0.8
    val_split: float = 0.1
    test_split: float = 0.1
    max_sequence_length: int = 128

@dataclass
class ModelConfig:
    # Language Model (distilgpt2)
    lm_model_name: str = "distilgpt2"
    lm_learning_rate: float = 5e-5
    lm_batch_size: int = 16
    lm_epochs: int = 3
    lm_max_length: int = 128
    
    # Intent Classifier (distilbert-base-uncased)
    classifier_model_name: str = "distilbert-base-uncased"
    classifier_learning_rate: float = 2e-5
    classifier_batch_size: int = 32
    classifier_epochs: int = 5
    classifier_num_labels: int = 3  # EXPLORE, ACTION, DIALOGUE
    
    # Common
    seed: int = 42

@dataclass
class HybridPlayerConfig:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)

    def __post_init__(self):
        """Log the paths for debugging"""
        import logging
        logger = logging.getLogger(__name__)
        logger.info(f"BASE_DIR: {BASE_DIR}")
        logger.info(f"CRD3 path: {self.data.crd3_base_path}")
        logger.info(f"LIGHT path: {self.data.light_base_path}")
        logger.info(f"Output path: {self.data.output_path}")