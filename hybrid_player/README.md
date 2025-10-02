# Hybrid Player: Player Behavior Simulation

This module implements realistic player behavior simulation for reinforcement learning training, combining a language model for generating player utterances with an intent classifier for action type prediction.

## Purpose

Simulates realistic player behavior in D&D sessions to support multi-critic reinforcement learning training. The hybrid approach combines text generation (what players say) with intent classification (what players intend to do) to provide rich, contextual prompts for training the Director LLM.

## Dataset

**Primary Dataset**: LIGHT Dialogue Processed

- **Location**: `../Data/light_dialogue_processed/`
- **Format**: Text files with structured fantasy dialogue and action sequences
- **Total Samples**: ~20,000+ training examples across different action types

**Data Types and Usage**:

- **Action Files**: `action_{split}.txt` - Character action prediction training
  - Purpose: Train intent classifier to recognize player action types
  - Content: Setting descriptions, character interactions, action labels
- **Speech Files**: `speech_{split}.txt` - Dialogue generation training
  - Purpose: Train language model to generate realistic player utterances
  - Content: Fantasy dialogue with context and character personas
- **Emote Files**: `emote_{split}.txt` - Emotional expression training
  - Content: Character emotional responses and expressions
- **Which Files**: `which_{split}.txt` - Action selection training
  - Content: Decision-making scenarios for action type classification

**Secondary Dataset**: CRD3 (D&D-Specific Examples)

- **Source**: Critical Role session transcripts
- **Purpose**: Domain-specific player behavior patterns
- **Usage**: Supplement LIGHT data with authentic D&D player interactions
- **Processing**: Extract player utterances and classify intent types

**Intent Classification Categories**:

- **EXPLORE**: Investigation, exploration, information-seeking actions
- **ACTION**: Combat, physical actions, skill-based activities
- **DIALOGUE**: Social interaction, conversation, roleplay scenarios

**Dual Training Architecture**:

1. **Language Model Component** (DistilGPT-2):

   - Trained on speech/emote data for utterance generation
   - Learns fantasy dialogue patterns and character voice
   - Generates contextually appropriate player responses

2. **Intent Classifier Component** (DistilBERT):
   - Trained on action/which data for intent prediction
   - Classifies player utterances into action categories
   - Enables dynamic reward weighting in PPO training

**Data Integration for PPO**:

- Generates diverse player prompts during RL training
- Provides intent classification for dynamic reward weighting
- Maintains realistic player behavior patterns for training context

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
- Data split ratios (train/val/test) configuration
- DM name variations for filtering out narrator turns

**`ModelConfig`**

- Language model configuration (DistilGPT-2)
- Intent classifier setup (DistilBERT)
- Training hyperparameters and optimization settings
- Model architecture and output specifications
- Hyperparameters for both language model and intent classifier training

**`HybridPlayerConfig`**

- Combined configuration orchestrating both components
- Training pipeline settings and integration parameters
- Output paths and model saving configurations
- Main configuration container for the entire system

**Key Functions:**

- `get_base_dir()`: Automatic path resolution for project structure
- `__post_init__()`: Configuration validation and setup

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

Data loading utilities for training both model components with comprehensive processing pipeline.

**Main Classes:**

**`CRD3DataLoader`**

- Extracts player utterances from Critical Role D&D dataset
- Identifies player vs. DM speakers using name recognition
- Filters out DM/narrator turns using configurable DM names
- Handles multi-file dataset processing with progress tracking
- Processes chunked dialogue structure from JSON files

**`LIGHTDataLoader`**

- Loads player dialogue from LIGHT fantasy conversation dataset
- Processes structured dialogue data for training
- Handles both seen and unseen data splits
- Extracts player speech, actions, and emotes
- Removes duplicates and ensures data quality

**`IntentLabeler`**

- Automatically labels utterances with intents using keyword matching
- Three intent categories: EXPLORE, ACTION, DIALOGUE
- Uses configurable keyword sets for each intent type
- Fallback to DIALOGUE when no keywords match
- Supports manual annotation validation

**`HybridPlayerDataProcessor`**

- Unified data processing orchestrator
- Combines CRD3 and LIGHT datasets for comprehensive training
- Implements data splitting and validation with stratification
- Creates intent-labeled training data for classifier
- Handles train/val/test splitting with proper balance
- Saves processed data to CSV format

**Key Functions:**

- `extract_player_utterances()`: Extracts all player dialogue from datasets
- `create_intent_labels()`: Generates intent annotations using keyword matching
- `prepare_training_data()`: Creates final training datasets with proper formatting
- `validate_data_quality()`: Ensures data meets quality standards and completeness
- `combine_datasets()`: Merges multiple data sources with deduplication

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

**Additional Features:**

- Comprehensive error handling and logging
- Environment setup and dependency validation
- Progress tracking throughout the training pipeline

#### `test_trained_model.py`

Quick testing script for immediate validation of trained models.

**Testing Features:**

- **Model Loading**: Loads saved models from disk for testing
- **Language Model Testing**: Tests generation on sample prompts with various contexts
- **Intent Classification Testing**: Validates classifier on sample utterances
- **Performance Feedback**: Provides immediate feedback on model performance
- **Quick Validation**: Rapid testing without full evaluation pipeline

#### `evaluate_trained_models.py`

Comprehensive evaluation script for detailed model performance analysis.

**Evaluation Components:**

- **Test Data Loading**: Loads held-out test data for unbiased evaluation
- **Classification Metrics**: Calculates accuracy, precision, recall, F1-score for intent classifier
- **Language Model Metrics**: Computes perplexity and generation quality metrics
- **Diversity Testing**: Evaluates generation diversity with different temperature settings
- **Confidence Analysis**: Analyzes classification confidence distributions
- **Detailed Reports**: Generates comprehensive classification reports and performance summaries

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

## How the Hybrid Player Works

### Data Flow Pipeline

1. **Data Collection Phase**

   - **CRD3 Source**: Extracts player utterances from D&D gameplay transcripts
     - Filters out DM/narrator turns using configurable DM names
     - Processes multi-file JSON structure with progress tracking
   - **LIGHT Source**: Gathers player commands from text adventure games
     - Extracts speech, actions, and emotes from structured data
     - Handles both seen and unseen game scenarios

2. **Intent Labeling Process**

   - **Automatic Labeling**: Uses keyword matching for intent classification
     - **EXPLORE**: Movement, examination, navigation keywords
     - **ACTION**: Combat, item interaction, spell casting keywords
     - **DIALOGUE**: Conversation, questions, social interaction keywords
   - **Fallback Strategy**: Defaults to DIALOGUE when no keywords match
   - **Quality Control**: Manual validation and refinement of labels

3. **Model Training Pipeline**
   - **Language Model**: Fine-tuned DistilGPT-2 on player utterances for response generation
   - **Intent Classifier**: Trained DistilBERT on labeled data for intent recognition
   - **Sequential Training**: Language model training followed by classifier training
   - **Validation**: Comprehensive testing with held-out data

### Inference Process

When generating a player response during game simulation:

1. **Context Input**: System receives game context (e.g., "You see a dragon blocking the path")
2. **Response Generation**: Language model generates plausible player response based on learned patterns
3. **Intent Classification**: Generated response is automatically classified into one of three intents
4. **Confidence Scoring**: System provides confidence scores for intent predictions
5. **Output**: Returns both response text and detected intent with confidence

**Example Inference Flow**:

```python
# Input context
context = "The party enters a dark dungeon with strange sounds echoing"

# Language model generates response
response = "I light my torch and carefully examine the walls for traps"

# Intent classifier analyzes response
intent = "EXPLORE"  # confidence: 0.89
probabilities = {"EXPLORE": 0.89, "ACTION": 0.08, "DIALOGUE": 0.03}
```

### Integration with Director System

The Hybrid Player serves multiple roles in the Director LLM framework:

- **Automated Testing**: Provides consistent player responses during RL training
- **Dynamic Reward Weighting**: Classifies responses to enable intent-aware reward adjustment
- **Controlled Experimentation**: Allows systematic testing without human player variability
- **Training Data Generation**: Creates diverse scenarios for policy improvement

**Intent-Based Reward Weighting Example**:

```python
intent_weights = {
    "EXPLORE": {"narrative": 0.8, "causal": 0.2},  # Prioritize world-building
    "ACTION": {"narrative": 0.3, "causal": 0.7},   # Focus on logical consequences
    "DIALOGUE": {"narrative": 0.6, "causal": 0.4}  # Balance storytelling and logic
}
```

## Intent Classification System

### Automatic Intent Labeling

The hybrid player uses a sophisticated keyword-based system for automatically labeling player utterances:

**EXPLORE Intent Keywords:**

- Movement: "go", "move", "walk", "travel", "head", "enter", "exit"
- Investigation: "look", "examine", "search", "check", "inspect", "scan"
- Navigation: "find", "locate", "explore", "investigate", "follow"

**ACTION Intent Keywords:**

- Combat: "attack", "hit", "strike", "fight", "shoot", "stab", "slash"
- Magic: "cast", "spell", "magic", "enchant", "summon", "conjure"
- Items: "use", "activate", "open", "close", "pick", "take", "grab"

**DIALOGUE Intent Keywords:**

- Communication: "say", "tell", "ask", "speak", "talk", "discuss"
- Social: "persuade", "convince", "negotiate", "question", "greet"
- Information: "inquire", "request", "demand", "explain"

**Labeling Process:**

1. **Keyword Matching**: Utterances are scanned for intent-specific keywords
2. **Priority System**: ACTION keywords take precedence over EXPLORE, EXPLORE over DIALOGUE
3. **Fallback Strategy**: Utterances with no matches default to DIALOGUE intent
4. **Manual Validation**: System supports manual review and correction of labels

**Example Classifications:**

```python
"I search the room for hidden doors" → EXPLORE
"I cast fireball at the goblin" → ACTION
"I ask the innkeeper about local rumors" → DIALOGUE
"Let's head north to the mountains" → EXPLORE
"I draw my sword and attack" → ACTION
```

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
