"""
Configuration settings for the Causal Responsiveness Critic

This module contains configuration parameters and settings for the
Director LLM's Causal Responsiveness Critic component.
"""

from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class CausalCriticConfig:
    """Configuration class for Causal Responsiveness Critic"""
    
    # Model settings
    model_name: str = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
    device: Optional[str] = None  # Auto-detect if None
    max_length: int = 512
    
    # Scoring parameters
    entailment_weight: float = 1.0      # Weight for entailment probability
    neutral_weight: float = 0.5         # Weight for neutral probability  
    contradiction_weight: float = 0.0   # Weight for contradiction probability
    
    # Batch processing
    batch_size: int = 8
    
    # Thresholds for explanations
    strong_causality_threshold: float = 0.7
    moderate_causality_threshold: float = 0.4
    contradiction_threshold: float = 0.5
    
    # Cache settings (for potential future caching implementation)
    enable_caching: bool = False
    cache_size: int = 1000
    
    # Logging
    log_level: str = "INFO"
    log_predictions: bool = False


# Predefined configurations for different use cases
CONFIGS = {
    "default": CausalCriticConfig(),
    
    "strict": CausalCriticConfig(
        entailment_weight=1.0,
        neutral_weight=0.2,
        contradiction_weight=0.0,
        strong_causality_threshold=0.8,
        moderate_causality_threshold=0.6
    ),
    
    "lenient": CausalCriticConfig(
        entailment_weight=1.0,
        neutral_weight=0.7,
        contradiction_weight=0.1,
        strong_causality_threshold=0.5,
        moderate_causality_threshold=0.3
    ),
    
    "balanced": CausalCriticConfig(
        entailment_weight=1.0,
        neutral_weight=0.5,
        contradiction_weight=0.0,
        strong_causality_threshold=0.6,
        moderate_causality_threshold=0.4
    ),
    
    # For development/testing with faster processing
    "fast": CausalCriticConfig(
        batch_size=16,
        max_length=256,
        enable_caching=True,
        log_predictions=True
    )
}


def get_config(config_name: str = "default") -> CausalCriticConfig:
    """
    Get a predefined configuration
    
    Args:
        config_name: Name of the configuration to retrieve
        
    Returns:
        CausalCriticConfig object
        
    Raises:
        KeyError: If config_name is not found
    """
    if config_name not in CONFIGS:
        available = ", ".join(CONFIGS.keys())
        raise KeyError(f"Config '{config_name}' not found. Available: {available}")
    
    return CONFIGS[config_name]


# MCRL Training Configuration
@dataclass
class MCRLIntegrationConfig:
    """Configuration for integrating causal critic into MCRL training"""
    
    # Reward scaling
    reward_scale: float = 1.0
    reward_clip_min: float = 0.0
    reward_clip_max: float = 1.0
    
    # Dynamic weighting parameters
    enable_dynamic_weighting: bool = True
    intent_weight_mapping: Dict[str, float] = None
    
    # Episode configuration  
    max_episode_length: int = 10
    evaluation_frequency: int = 100  # Evaluate every N episodes
    
    # Integration with other critics
    causality_weight_in_ensemble: float = 0.25  # 1/4 of total reward by default
    
    def __post_init__(self):
        """Set default intent weight mapping if not provided"""
        if self.intent_weight_mapping is None:
            self.intent_weight_mapping = {
                "EXPLORE": 0.6,    # Exploration actions need moderate causality  
                "ACTION": 1.0,     # Direct actions need strong causality
                "DIALOGUE": 0.8,   # Dialogue needs good causality
                "DEFAULT": 0.7     # Default weight for unlabeled intents
            }


# Example usage configurations for different domains
DOMAIN_CONFIGS = {
    "fantasy_rpg": {
        "causality_importance": 0.8,
        "context_templates": [
            "In a fantasy world where magic exists...",
            "You are adventuring in a medieval setting...",
            "The realm is filled with dragons, wizards, and ancient magic..."
        ],
        "common_actions": [
            "cast spell", "swing sword", "pick lock", "search room",
            "talk to NPC", "examine object", "drink potion"
        ]
    },
    
    "sci_fi": {
        "causality_importance": 0.9,  # Sci-fi often requires logical consistency
        "context_templates": [
            "In a futuristic space station...",
            "You are exploring an alien planet...", 
            "Advanced technology surrounds you..."
        ],
        "common_actions": [
            "scan with tricorder", "fire phaser", "hack computer",
            "contact ship", "analyze sample", "activate shield"
        ]
    },
    
    "mystery": {
        "causality_importance": 0.95,  # Mystery requires very strong causality
        "context_templates": [
            "You are investigating a crime scene...",
            "The mystery deepens as you gather clues...",
            "Every detail could be important to solving the case..."
        ],
        "common_actions": [
            "examine evidence", "question witness", "take notes",
            "follow lead", "analyze clues", "make deduction"
        ]
    }
}