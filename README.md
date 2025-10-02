# Director LLM: Multi-Critic Reinforcement Learning for D&D Narrative Generation

**Dropout Squad** - Advanced NLP project implementing Multi-Critic Reinforcement Learning (MCRL) for domain-aware Dungeon Master response generation in tabletop RPG scenarios.

## Project Overview

This project implements a multi-component system for generating contextually appropriate and narratively engaging Dungeon Master responses in D&D scenarios. The approach combines supervised fine-tuning with multi-critic reinforcement learning, utilizing dynamic reward weighting based on player intent classification.

### System Architecture

The Director LLM system consists of several interconnected components:

1. **Supervised Fine-Tuning (SFT) Baseline**: Creates initial DM model using QLoRA optimization on Llama-2-7B
2. **Multi-Critic System**:
   - Narrative Critic: DeBERTa-based regression for quality assessment
   - Causal Critic: NLI-based evaluation for logical consistency
3. **Hybrid Player Simulation**:
   - Language Model: DistilGPT-2 for generating player utterances
   - Intent Classifier: DistilBERT for categorizing player actions (EXPLORE/ACTION/DIALOGUE)
4. **PPO Training**: Reinforcement learning with dynamic reward weighting
5. **Evaluation Pipeline**: Comprehensive assessment and analysis tools

### Key Features

- **Dynamic Reward Weighting**: Adjusts critic importance based on player intent
- **Memory-Efficient Training**: QLoRA optimization enables training on consumer GPUs
- **Modular Design**: Each component can be trained and evaluated independently
- **Comprehensive Evaluation**: Statistical analysis and domain-specific metrics
- **Research-Grade Implementation**: Reproducible experiments with detailed configuration

### Dynamic Reward Weighting System

The core innovation of the Director LLM is intent-aware reward weighting:

**Player Intent Classification**:
- **EXPLORE**: Investigation, movement, environmental interaction
- **ACTION**: Combat, physical actions, skill usage
- **DIALOGUE**: Conversation, social interaction, roleplay

**Adaptive Reward Weights**:
```python
intent_weights = {
    "EXPLORE": {"narrative": 0.8, "causal": 0.2},  # Prioritize world-building
    "ACTION": {"narrative": 0.3, "causal": 0.7},   # Focus on logical consequences  
    "DIALOGUE": {"narrative": 0.6, "causal": 0.4}  # Balance storytelling and logic
}
```

**Benefits**:
- Context-appropriate response optimization
- Reduces reward signal conflicts between critics
- Enables domain-specific behavior adaptation

## Technical Stack

- **Base Language Model**: Llama-2-7B (via microsoft/phi-2) with QLoRA adapters
- **Critic Models**: DeBERTa-v3-base for both narrative and causal evaluation
- **Player Simulation**: DistilGPT-2 (generation) + DistilBERT (classification)
- **Training Framework**: PyTorch, Transformers, TRL, PEFT
- **Optimization**: 4-bit quantization, LoRA adapters, mixed precision training

## Project Structure

```
Dropout-Squad/
├── README.md                 # Project overview and setup guide
├── DM-SFT/                  # Supervised Fine-Tuning baseline
│   ├── train_sft.py         # Main SFT training script
│   ├── sft_config.yaml      # Training configuration
│   └── sft_model/           # Model artifacts and configs
├── PPO/                     # Multi-Critic Reinforcement Learning
│   ├── ppo.py              # PPO training orchestrator
│   ├── critics.py          # Multi-critic reward system
│   └── ppo_config.yaml     # PPO training configuration
├── causal critic/           # Causal Consistency Evaluation
│   ├── causal_critic.py    # NLI-based causal evaluation
│   ├── data_prep.py        # Data extraction and preprocessing
│   └── train.py            # Critic training script
├── narrative critic/        # Narrative Quality Assessment
│   ├── narrative_critic.py # DeBERTa-based quality scoring
│   ├── train.py            # Narrative critic training
│   └── critic_config.yaml  # Training configuration
├── hybrid_player/           # Player Behavior Simulation
│   ├── __init__.py         # Module initialization
│   ├── config.py           # Configuration classes
│   ├── models.py           # Player models (generator + classifier)
│   ├── data_loader.py      # Data loading utilities
│   ├── trainer.py          # Training utilities
│   ├── utils.py            # Helper functions
│   └── train_hybrid_player.py # Main training script
└── Evaluation/             # Comprehensive Evaluation Pipeline
    ├── full_eval.py        # Complete evaluation with metrics
    ├── evalc.py            # Basic critic evaluation
    └── eval_config.yaml    # Evaluation configuration
```

## Module Descriptions

### DM-SFT (Supervised Fine-Tuning)

**Purpose**: Creates baseline Dungeon Master model using supervised learning

- `train_sft.py`: Main training script with QLoRA optimization for Llama-2-7B
- `sft_config.yaml`: Configuration for training parameters, model settings, and hardware optimization
- `sft_model/`: Directory containing model artifacts and training configurations

### PPO (Proximal Policy Optimization)

**Purpose**: Multi-critic reinforcement learning with dynamic reward weighting

- `ppo.py`: Complete PPO training orchestrator with multi-critic integration
- `critics.py`: Multi-critic reward system implementation
- `ppo_config.yaml`: Configuration for PPO hyperparameters and dynamic weighting

### causal critic

**Purpose**: Evaluates causal consistency between player actions and DM responses

- `causal_critic.py`: NLI-based causal consistency evaluator using DeBERTa
- `data_prep.py`: Extracts and preprocesses data from CRD3 dataset
- `train.py`: Training script for causal critic model

### narrative critic

**Purpose**: Assesses narrative quality of generated responses

- `narrative_critic.py`: DeBERTa-based regression model for quality scoring
- `train.py`: Training pipeline for narrative quality assessment
- `critic_config.yaml`: Configuration for critic training parameters

### hybrid_player

**Purpose**: Simulates realistic player behavior for RL training using dual-component architecture

- `__init__.py`: Module exports and initialization
- `config.py`: Configuration classes for data and model settings
- `models.py`: Player language model and intent classifier implementations
- `data_loader.py`: Data loading from CRD3 and LIGHT datasets with intent labeling
- `trainer.py`: Training utilities for both components
- `utils.py`: File I/O and data processing utilities
- `train_hybrid_player.py`: Main training script for hybrid player system
- `test_trained_model.py`: Quick testing and validation of trained models
- `evaluate_trained_models.py`: Comprehensive model evaluation with metrics

**Key Features**:
- **Dual Architecture**: DistilGPT-2 for utterance generation + DistilBERT for intent classification
- **Intent Categories**: EXPLORE, ACTION, DIALOGUE with automatic keyword-based labeling
- **Data Integration**: Combines CRD3 and LIGHT datasets for comprehensive training
- **PPO Integration**: Provides intent-classified prompts for dynamic reward weighting

### Evaluation

**Purpose**: Comprehensive evaluation pipeline with advanced metrics

- `full_eval.py`: Complete evaluation system with statistical analysis and visualization
- `evalc.py`: Basic critic evaluation utilities
- `eval_config.yaml`: Configuration for evaluation parameters and model paths

## Technical Stack

- **Base Model**: Llama-2-7B with QLoRA optimization
- **Critics**: DeBERTa-v3-base for both narrative and causal evaluation
- **Player Simulation**: DistilGPT-2 (generator) + DistilBERT (classifier)
- **Training**: PyTorch, Transformers, TRL, PEFT
- **Optimization**: 4-bit quantization, LoRA adapters, mixed precision

## Datasets

The project utilizes multiple datasets for comprehensive training:

### Dataset Download

🔗 **[Dataset](https://1drv.ms/f/c/bdcf3b74ef9b6129/Ep8Im9Kl-SNOspd2NAYqJ4MBzBsoeKe3uRlr6IhZiDkyGg?e=hrZgDd)**

### Primary Datasets
- **CRD3**: Critical Role D&D transcripts (~200 episodes, 2 campaigns)
- **LIGHT**: Fantasy dialogue and action data (~20K training samples)
- **ROCStories/TinyStories**: Narrative coherence training (~1.9GB stories)

### Processed Training Data
- **Data Splits**: Combined LIGHT+CRD3 for DM-SFT baseline training
- **Critic Training**: 40,906 examples (30K ROCStories + 10.9K DM pairs)
- **Causal Critic Training**: Premise-hypothesis pairs for NLI evaluation

### Dataset Usage by Component
- **DM-SFT**: Data Splits (instruction-tuned LIGHT+CRD3 combination)
- **Narrative Critic**: Critic Training dataset with quality labels
- **Causal Critic**: Causal critic training data for consistency evaluation
- **Hybrid Player**: LIGHT dialogue data for player simulation
- **PPO Training**: All datasets integrated with critic feedback

*See `Data/README.md` for detailed dataset documentation and setup instructions.*

## Getting Started

1. **Environment Setup**

   ```bash
   pip install torch transformers peft trl datasets
   pip install accelerate bitsandbytes wandb
   ```

2. **Dataset Setup**

   ```bash
   cd Data/
   # Link datasets from main data directory
   ln -s ../../data/crd3 ./crd3
   ln -s ../../data/light_dialogue_processed ./light_dialogue_processed
   ln -s ../../data/critic_training ./critic_training
   ln -s ../../data/rocstories ./rocstories
   ln -s ../../data/splits ./splits
   ln -s ../../data/causal_critic_training ./causal_critic_training
   ``` 

## Key Features

- **Memory Efficient**: QLoRA optimization enables training on consumer GPUs
- **Modular Design**: Each component can be trained and evaluated independently
- **Comprehensive Evaluation**: Statistical analysis, visualization, and domain-specific metrics
- **Intent-Aware**: Dynamic reward weighting based on player action classification
- **Research-Grade**: Reproducible experiments with detailed configuration management

## Configuration

Each module includes detailed YAML configuration files for:

- Model hyperparameters
- Training schedules
- Hardware optimization
- Evaluation parameters

See individual module READMEs for specific configuration options and usage instructions.

---
