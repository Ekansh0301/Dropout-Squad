# Hybrid Player: Player Behavior Simulation

This module implements realistic player behavior simulation for reinforcement learning training, combining a language model for generating player utterances with an intent classifier for action type prediction.

## Purpose

Simulates realistic player behavior in D&D sessions to support multi-critic reinforcement learning training. The hybrid approach combines text generation (what players say) with intent classification (what players intend to do) to provide rich, contextual prompts for training the Director LLM.

## Files Overview

### Core Files

#### `__init__.py`

Module initialization and exports for the hybrid player system.

**Exports:**

- `HybridPlayerConfig`: Main configuration class
- `DataConfig`: Data loading and preprocessing configuration
- `ModelConfig`: Model architecture and training configuration
- `HybridPlayerDataProcessor`: Data loading and processing utilities
- `HybridPlayerModel`: Combined language model and intent classifier
- `HybridPlayerTrainer`: Training orchestrator for both components

#### `config.py`

Configuration classes for data processing, model setup, and training parameters.

**Main Classes:**

**`DataConfig`**

- CRD3 dataset paths and processing parameters
- LIGHT dataset integration settings
- DM name identification for speaker classification
- Data filtering and quality control parameters

**`ModelConfig`**

- Language model configuration (DistilGPT-2)
- Intent classifier setup (DistilBERT)
- Training hyperparameters and optimization settings
- Model architecture and output specifications

**`HybridPlayerConfig`**

- Combined configuration orchestrating both components
- Training pipeline settings and integration parameters
- Output paths and model saving configurations

#### `models.py`

Core model implementations for player simulation components.

**Main Classes:**

**`PlayerLanguageModel`**

- DistilGPT-2-based text generator for player utterances
- Fine-tuned on actual player dialogue from CRD3 and LIGHT datasets
- Generates contextually appropriate player responses and actions
- Optimized for realistic D&D player speech patterns

**Key Methods:**

- `load_model()`: Loads pre-trained DistilGPT-2 with custom tokenizer
- `get_model_and_tokenizer()`: Returns configured model and tokenizer
- `generate()`: Produces player utterances with controllable parameters

**`IntentClassifier`**

- DistilBERT-based classifier for player intent prediction
- Three-class classification: EXPLORE, ACTION, DIALOGUE
- Trained on labeled player utterances with intent annotations
- Enables dynamic reward weighting in PPO training

**Key Methods:**

- `load_model()`: Loads DistilBERT with custom label mapping
- `predict_intent()`: Classifies player utterance into intent category
- `get_probabilities()`: Returns confidence scores for all intent classes

**`HybridPlayerModel`**

- Combined system integrating both language model and classifier
- Provides unified interface for player simulation
- Handles model coordination and batch processing
- Optimized for integration with PPO training pipeline

**Intent Categories:**

```python
EXPLORE: "I look around the room"          # Environmental investigation
ACTION:  "I attack the orc with my sword"  # Direct combat/interaction
DIALOGUE: "I ask the NPC about the quest"  # Conversation and roleplay
```

#### `data_loader.py`

Data loading utilities for training both model components.

**Main Classes:**

**`CRD3DataLoader`**

- Extracts player utterances from Critical Role D&D dataset
- Identifies player vs. DM speakers using name recognition
- Filters and processes dialogue for training data
- Handles multi-file dataset processing with progress tracking

**`LIGHTDataLoader`**

- Loads player dialogue from LIGHT fantasy conversation dataset
- Processes structured dialogue data for training
- Handles both main and unseen data splits
- Removes duplicates and ensures data quality

**`HybridPlayerDataProcessor`**

- Unified data processing orchestrator
- Combines CRD3 and LIGHT datasets for comprehensive training
- Implements data splitting and validation
- Creates intent-labeled training data for classifier

**Key Functions:**

- `extract_player_utterances()`: Extracts all player dialogue
- `create_intent_labels()`: Generates intent annotations
- `prepare_training_data()`: Creates final training datasets
- `validate_data_quality()`: Ensures data meets quality standards

#### `trainer.py`

Training utilities and dataset classes for both model components.

**Dataset Classes:**

**`PlayerTextDataset`**

- PyTorch dataset for language model training
- Handles tokenization and sequence formatting
- Supports variable-length sequences with padding
- Optimized for causal language modeling objective

**`IntentDataset`**

- PyTorch dataset for intent classifier training
- Pairs text with intent labels for supervised learning
- Implements proper tokenization for classification
- Supports batch processing with attention masks

**Training Functions:**

- Model-specific training loops for each component
- Integration with HuggingFace Trainer API
- Comprehensive evaluation and metric tracking
- Checkpoint management and model saving

#### `utils.py`

Utility functions for file I/O, data processing, and helper operations.

**Key Functions:**

- `ensure_dir()`: Creates directories if they don't exist
- `load_json_files()`: Loads and processes JSON dataset files
- `save_pickle()` / `load_pickle()`: Serialization utilities
- Data validation and quality control helpers

#### `train_hybrid_player.py`

Main training script orchestrating the complete hybrid player training pipeline.

**Training Pipeline:**

1. **Configuration Loading**: Loads and validates all training parameters
2. **Data Validation**: Checks dataset paths and availability
3. **Data Processing**: Extracts and prepares training data from CRD3/LIGHT
4. **Language Model Training**: Fine-tunes DistilGPT-2 on player utterances
5. **Intent Data Creation**: Generates labeled data for classification
6. **Classifier Training**: Trains DistilBERT on intent classification
7. **Model Integration**: Combines both components into unified system
8. **Validation**: Tests complete hybrid player functionality

**Key Functions:**

- `validate_data_paths()`: Ensures all required datasets are available
- `train_language_model()`: Orchestrates player text generation training
- `train_intent_classifier()`: Manages intent classification training
- `integrate_models()`: Combines components into hybrid system

## Technical Implementation

### Dual-Component Architecture

- **Language Generation**: DistilGPT-2 fine-tuned on player dialogue
- **Intent Classification**: DistilBERT trained on annotated player actions
- **Integration Layer**: Unified interface for both components
- **Batch Processing**: Efficient handling of multiple inputs

### Data Processing Pipeline

- **Multi-Source Integration**: Combines CRD3 and LIGHT datasets
- **Speaker Identification**: Separates player from DM utterances
- **Quality Filtering**: Removes low-quality or irrelevant data
- **Intent Annotation**: Creates training labels for classification

### Training Strategy

- **Sequential Training**: Language model first, then intent classifier
- **Transfer Learning**: Leverages pre-trained models for efficiency
- **Fine-Tuning**: Domain adaptation for D&D-specific language
- **Validation**: Comprehensive testing throughout pipeline

## Usage

### Prerequisites

```bash
pip install torch transformers datasets
pip install pandas numpy tqdm
pip install scikit-learn matplotlib
```

### Configuration

1. Edit configuration files to specify dataset paths
2. Adjust model parameters for your hardware setup
3. Configure training hyperparameters

### Data Preparation

```bash
# Ensure CRD3 and LIGHT datasets are available
# Update paths in configuration files
```

### Training

```bash
# Run complete training pipeline
python train_hybrid_player.py

# Or train components separately
python trainer.py --component language_model
python trainer.py --component intent_classifier
```

### Usage in PPO Training

```python
from hybrid_player import HybridPlayerModel

# Initialize hybrid player
player = HybridPlayerModel(config)

# Generate player prompts with intent classification
prompts, intents = player.generate_prompts(batch_size=16)

# Use intents for dynamic reward weighting in PPO
```

## Integration with Main System

### PPO Training Integration

- **Prompt Generation**: Provides realistic player prompts for RL training
- **Intent Classification**: Enables dynamic reward weighting
- **Batch Processing**: Efficient generation for large training batches
- **Consistent Interface**: Standardized API for easy integration

### Dynamic Reward Weighting

```python
# Intent-based weight adjustment in PPO training
intent_weights = {
    "EXPLORE": {"narrative": 0.8, "causal": 0.2},
    "ACTION": {"narrative": 0.3, "causal": 0.7},
    "DIALOGUE": {"narrative": 0.6, "causal": 0.4}
}
```

## Model Outputs

### Language Model Outputs

```
Input Context: "The party enters a dark dungeon"
Generated: "I light my torch and look around for any signs of danger"

Input Context: "A goblin appears from behind a rock"
Generated: "I draw my sword and attack the goblin"
```

### Intent Classification

```
Utterance: "I search the room for treasure"
Intent: EXPLORE (confidence: 0.89)

Utterance: "I cast fireball at the enemies"
Intent: ACTION (confidence: 0.94)

Utterance: "I ask the shopkeeper about magic items"
Intent: DIALOGUE (confidence: 0.82)
```

## Advanced Features

### Customization Options

- **Model Variants**: Support for different language model sizes
- **Intent Categories**: Extensible intent classification system
- **Training Parameters**: Flexible hyperparameter configuration
- **Data Sources**: Easy integration of additional datasets

### Quality Controls

- **Data Validation**: Comprehensive dataset quality checks
- **Model Evaluation**: Automated testing during training
- **Performance Monitoring**: Real-time training metrics
- **Error Handling**: Robust training pipeline with recovery

### Research Applications

- **Player Behavior Study**: Analysis of player action patterns
- **Intent Distribution**: Understanding player motivation patterns
- **Language Modeling**: D&D-specific language generation research
- **Multi-Modal Learning**: Integration of text and intent prediction
