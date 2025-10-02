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

**Purpose**: Simulates realistic player behavior for RL training

- `__init__.py`: Module exports and initialization
- `config.py`: Configuration classes for data and model settings
- `models.py`: Player language model and intent classifier implementations
- `data_loader.py`: Data loading from CRD3 and LIGHT datasets
- `trainer.py`: Training utilities for both components
- `utils.py`: File I/O and data processing utilities
- `train_hybrid_player.py`: Main training script for hybrid player system

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


## Getting Started

1. **Environment Setup**

   ```bash
   pip install torch transformers peft trl datasets
   pip install accelerate bitsandbytes wandb
   ```

## Dataset Link :  
## Narrative Model : 

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
