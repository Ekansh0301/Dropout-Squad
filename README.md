# Director LLM: Multi-Critic Reinforcement Learning for D&D Narrative Generation

**Dropout Squad** - Advanced NLP project implementing Multi-Critic Reinforcement Learning (MCRL) for domain-aware Dungeon Master response generation in tabletop RPG scenarios.

## Project Overview

This project implements a multi-component system for generating contextually appropriate and narratively engaging Dungeon Master responses in D&D scenarios. The approach combines supervised fine-tuning with multi-critic reinforcement learning, utilizing dynamic reward weighting based on player intent classification.

### System Architecture

The Director LLM system consists of several interconnected components:

1. **Supervised Fine-Tuning (SFT) Baseline**: Phi-2 (2.7B) model fine-tuned with LoRA on CRD3 D&D transcripts
2. **Multi-Critic System**:
   - **Narrative Critic**: DeBERTa-v3-base regression for quality assessment (40,906 training examples)
   - **Causal Critic**: RoBERTa-base 3-class NLI model for logical consistency (88.09% accuracy, 382K examples)
   - **World Consistency Critic**: RoBERTa-large for D&D rules validation (98.3% accuracy)
   - **Character Voice Critic**: DeBERTa-v3-base with learned embeddings for NPC consistency (87.6% accuracy)
3. **Hybrid Player Simulation**:
   - Intent Classifier: BERT-based model for categorizing player actions (EXPLORE/ACTION/DIALOGUE)
4. **PPO Training**: Multi-critic reinforcement learning with dynamic reward weighting
5. **Evaluation Pipeline**: Comprehensive assessment and analysis tools

### Key Features

- **Dynamic Reward Weighting**: Adjusts critic importance based on player intent
- **Memory-Efficient Training**: LoRA adapters enable training on consumer GPUs (RTX 4080 Super 16GB)
- **Modular Design**: Each component can be trained and evaluated independently
- **Comprehensive Evaluation**: Statistical analysis and domain-specific metrics
- **Research-Grade Implementation**: Reproducible experiments with detailed configuration
- **Production-Ready Models**: All critics trained and validated with high accuracy

### Dynamic Reward Weighting System

The core innovation of the Director LLM is intent-aware reward weighting:

**Player Intent Classification**:

- **EXPLORE**: Investigation, movement, environmental interaction
- **ACTION**: Combat, physical actions, skill usage
- **DIALOGUE**: Conversation, social interaction, roleplay

**Adaptive Reward Weights**:

```python
intent_weights = {
    "EXPLORE": {"narrative": 0.40, "causal": 0.20, "world": 0.30, "character": 0.10},
    "ACTION":  {"narrative": 0.20, "causal": 0.40, "world": 0.30, "character": 0.10},
    "DIALOGUE": {"narrative": 0.20, "causal": 0.20, "world": 0.20, "character": 0.40}
}
```

**Benefits**:

- Context-appropriate response optimization
- Reduces reward signal conflicts between critics
- Enables domain-specific behavior adaptation

## Technical Stack

- **Base Language Model**: Phi-2 (2.7B parameters) with LoRA adapters (r=32, α=64)
- **Critic Models**:
  - Narrative: DeBERTa-v3-base (regression)
  - Causal: RoBERTa-base (3-class NLI, 88.09% accuracy)
  - World: RoBERTa-large (98.3% accuracy)
  - Character: DeBERTa-v3-base with embeddings (87.6% accuracy)
- **Player Simulation**: BERT-based intent classifier
- **Training Framework**: PyTorch, Transformers, TRL, PEFT
- **Optimization**: LoRA adapters, mixed precision training, gradient checkpointing

## Project Structure

```
Dropout-Squad/
├── README.md                    # Project overview and setup guide
├── DM-SFT/                     # Supervised Fine-Tuning (Phi-2)
│   ├── train_sft_phi2_optimized.py  # Main SFT training script
│   ├── sft_config.yaml         # Training configuration
│   ├── evaluate_sft_model.py   # Model evaluation
│   └── models/                 # Trained model artifacts
├── PPO/                        # Multi-Critic Reinforcement Learning
│   ├── train_complete_ppo.py   # Production PPO training script
│   ├── ppo_training_complete.py # Complete PPO implementation
│   ├── integrated_critics.py   # Multi-critic reward system
│   └── ppo_config.yaml         # PPO training configuration
├── causal critic/              # Causal Consistency (3-class NLI)
│   ├── causal_critic.py        # RoBERTa-based causal evaluator
│   ├── train_3class.py         # 3-class critic training (production)
│   ├── data_prep_3class.py     # Data preparation
│   └── test_causal_accuracy.py # Model validation
├── narrative critic/           # Narrative Quality Assessment
│   ├── narrative_critic.py     # DeBERTa-based quality scoring
│   ├── critic_config.yaml      # Training configuration
│   └── model/                  # Trained model artifacts
├── world_consistency critic/   # World/Rules Consistency
│   ├── README.md              # 98.3% accuracy documentation
│   └── [implementation files]
├── character-voice critic/     # Character Voice Consistency
│   ├── README.md              # 87.6% accuracy documentation
│   └── [implementation files]
├── hybrid_player/              # Player Intent Classification
│   ├── models.py              # BERT-based intent classifier
│   ├── train_hybrid_player.py # Training script
│   └── models/                # Trained models
└── Evaluation/                # Comprehensive Evaluation
    ├── full_eval.py           # Complete evaluation pipeline
    └── eval_config.yaml       # Evaluation configuration
```

## Module Descriptions

### DM-SFT (Supervised Fine-Tuning)

**Purpose**: Creates baseline Dungeon Master model using supervised learning on Phi-2

- `train_sft_phi2_optimized.py`: Main training script with LoRA optimization for Phi-2 (2.7B)
- `sft_config.yaml`: Configuration for training parameters and optimization settings
- `evaluate_sft_model.py`: Comprehensive model evaluation
- `models/`: Directory containing trained model artifacts (75K training examples, 1 epoch)

**Implementation Details**:

- **Base Model**: microsoft/phi-2 (2.7B parameters)
- **Training Method**: Parameter-Efficient Fine-Tuning (PEFT) using LoRA
- **LoRA Configuration**:
  - Rank (r): 32
  - Alpha: 64
  - Target modules: All linear layers in transformer
  - Dropout: 0.05
- **Training Data**: 75,000 CRD3 dialogue turns (Player action → DM response pairs)
- **Training Time**: ~3.87 hours on RTX 4080 Super (16GB)
- **Loss**: Train: 1.987, Eval: 1.660
- **Response Quality**: 122-word average, coherent narrative structure
- **Prompt Format**: Phi-2 specific format (not Llama-2 style)

  ```
  You are a Dungeon Master in a fantasy RPG game.

  Player: {player_action}
  Dungeon Master: {dm_response}
  ```

**Key Improvements Over Base Model**:

- D&D-specific vocabulary and terminology
- Proper game mechanics understanding (dice rolls, abilities, spells)
- Narrative coherence and descriptive quality
- Context-aware responses to player actions

### PPO (Proximal Policy Optimization)

**Purpose**: Multi-critic reinforcement learning with dynamic reward weighting

- `train_complete_ppo.py`: Production PPO training script (1771 lines, full integration)
- `ppo_training_complete.py`: Complete PPO implementation with all 4 critics
- `integrated_critics.py`: Multi-critic reward system implementation
- `ppo_config.yaml`: Configuration for PPO hyperparameters and dynamic weighting

**Implementation Details**:

- **Algorithm**: Proximal Policy Optimization (PPO) with clipping
- **Policy Network**: Phi-2 + LoRA adapter (initialized from SFT model)
- **Value Network**: Integrated value head for advantage estimation
- **Training Configuration**:
  - Batch size: 32
  - Mini-batch size: 8
  - PPO epochs: 4
  - Learning rate: 1e-5
  - Clip range: 0.2
  - GAE lambda: 0.95
  - Total steps: 1,000 (converged)
  - Training time: ~53 minutes on RTX 4080 Super

**Multi-Critic Reward System**:

- **Four Specialized Critics**:

  1. **Narrative Quality** (DeBERTa-v3): Scores descriptiveness, atmosphere, engagement
  2. **Causal Consistency** (RoBERTa): Validates logical action-response relationships
  3. **World Consistency** (RoBERTa-large): Checks D&D rules and physics adherence
  4. **Character Voice** (DeBERTa-v3): Ensures NPC personality consistency

- **Dynamic Weighting by Intent**:

  - **EXPLORE**: Narrative-focused (N:0.40, C:0.20, W:0.30, Ch:0.10)
  - **ACTION**: Causal-focused (N:0.20, C:0.40, W:0.30, Ch:0.10)
  - **DIALOGUE**: Character-focused (N:0.20, C:0.20, W:0.20, Ch:0.40)

- **Reward Calculation**:
  ```python
  reward = (narrative_score * weight_n +
            causal_score * weight_c +
            world_score * weight_w +
            character_score * weight_ch)
  ```

**Training Results**:

- **Overall Performance**: 0.412 → 0.611 mean reward (+48.3%)
- **Causal Consistency**: 0.189 → 0.567 (+200% improvement)
- **Narrative Quality**: 0.523 → 0.694 (+32.7%)
- **World Consistency**: 0.618 → 0.821 (+32.8%)
- **Character Voice**: 0.138 → 0.268 (+94.2%)

**Key Features**:

- Checkpoint/resume capability for interrupted training
- Gradient accumulation for memory efficiency
- Real-time logging with wandb integration
- Validation every 50 steps
- Example generation for qualitative assessment
- No reward hacking or mode collapse observed

**Performance**: Mean reward improvement from 0.412 (SFT baseline) to 0.611 (PPO trained)

### causal critic

**Purpose**: Evaluates causal consistency between player actions and DM responses using 3-class NLI

- `causal_critic.py`: RoBERTa-base causal consistency evaluator
- `train_3class.py`: Production training script for 3-class model (entailment/neutral/contradiction)
- `data_prep_3class.py`: Extracts and preprocesses data from CRD3 dataset
- `test_causal_accuracy.py`: Model validation and accuracy testing

**Implementation Details**:

- **Model Architecture**: RoBERTa-base (125M parameters)
- **Task**: 3-class Natural Language Inference
  - **Entailment**: Response logically follows from player action
  - **Neutral**: Response is possible but not necessarily implied
  - **Contradiction**: Response conflicts with player action
- **Training Data**: 382,530 examples
  - Positive pairs: Sequential player-DM turns from CRD3
  - Negative pairs: Mismatched player actions and DM responses
  - Balanced across all 3 classes
- **Training Configuration**:
  - Learning rate: 2e-5
  - Batch size: 16
  - Epochs: 3
  - Max sequence length: 512 tokens
  - Training time: ~2-3 hours
- **Input Format**:
  ```
  [CLS] Player: {action} [SEP] DM: {response} [SEP]
  ```

**Performance Metrics**:

- **Overall Accuracy**: 88.09%
- **Per-Class Performance**:
  - Entailment: 89.2% F1
  - Neutral: 85.7% F1
  - Contradiction: 89.4% F1
- **Model Size**: 476MB

**Performance**: 88.09% accuracy on 382,530 training examples

### narrative critic

**Purpose**: Assesses narrative quality of generated DM responses

- `narrative_critic.py`: DeBERTa-v3-base regression model for quality scoring
- `critic_config.yaml`: Configuration for critic training parameters
- `model/`: Trained model artifacts

**Implementation Details**:

- **Model Architecture**: DeBERTa-v3-base (184M parameters)
- **Task**: Regression (quality score prediction, 0.0 to 1.0)
- **Training Data**: 40,906 quality-labeled examples
  - 30,000 from ROCStories corpus
  - 10,906 from human-annotated DM responses
- **Quality Dimensions Evaluated**:
  - **Descriptiveness**: Sensory details, vivid imagery, atmospheric elements
  - **Engagement**: Narrative hooks, dramatic tension, player agency
  - **Coherence**: Logical flow, consistent world-building, clear communication
  - **Structure**: Proper pacing, response length appropriateness
- **Training Configuration**:
  - Learning rate: 2e-5
  - Batch size: 8
  - Epochs: 5
  - Loss function: MSE (Mean Squared Error)
  - Training time: ~4-5 hours
- **Output**: Continuous score between 0.0 (poor quality) and 1.0 (excellent quality)

**Scoring Rubric**:

- **0.0-0.3**: Poor (generic, minimal detail, unclear)
- **0.3-0.6**: Moderate (adequate but lacking engagement)
- **0.6-0.8**: Good (descriptive, engaging, coherent)
- **0.8-1.0**: Excellent (vivid, immersive, professionally crafted)

**Training Data**: 40,906 quality-labeled examples

### world_consistency critic

**Purpose**: Validates adherence to D&D 5e rules, game mechanics, and physical consistency

**Implementation Details**:

- **Model Architecture**: RoBERTa-large (355M parameters)
- **Task**: Binary classification (consistent vs. inconsistent)
- **Validation Accuracy**: 98.3%
- **Knowledge Base**:
  - D&D 5e System Reference Document (SRD)
  - Player's Handbook rules and mechanics
  - Spell descriptions and effects
  - Character abilities and resource limitations
  - Physics and world logic constraints
- **Consistency Types Checked**:
  1. **Rule Violations**: Incorrect spell usage, ability mechanics, dice rolls
  2. **Resource Tracking**: Spell slots, hit points, ability uses
  3. **Physics Consistency**: Impossible actions, contradictory states
  4. **Lore Adherence**: World-building consistency, established facts
- **Input Format**: Context + DM response → Consistency score

**Key Features**:

- Hybrid architecture (symbolic + neural)
- Explicit world state tracking
- Detects contradictions, hallucinations, and amnesia
- Provides interpretable error explanations

### character-voice critic

**Purpose**: Ensures NPC personality and dialogue consistency across interactions

**Implementation Details**:

- **Model Architecture**: DeBERTa-v3-base with learned character embeddings
- **Task**: Character voice matching (regression scoring)
- **Validation Accuracy**: 87.6%
- **Training Approach**:
  - Learns 128-dimensional embeddings per character
  - Captures personality traits, speech patterns, vocabulary preferences
  - Trained on Critical Role character dialogues (professional voice actors)
- **Evaluation Dimensions**:
  - **Personality Consistency**: Brave/cowardly, formal/casual, optimistic/cynical
  - **Speech Patterns**: Vocabulary choice, sentence structure, verbal tics
  - **Behavioral Traits**: Character-specific quirks and mannerisms
  - **Emotional Tone**: Consistent emotional expression
- **Scoring**:
  - 0.0-0.3: Poor match (out of character)
  - 0.3-0.6: Moderate (generic dialogue)
  - 0.6-0.8: Good (captures essence)
  - 0.8-1.0: Excellent (authentic voice)

**Key Innovation**: Character-specific embeddings enable distinguishing subtle differences between similar character types (e.g., different warrior personalities)

### hybrid_player

**Purpose**: Provides player intent classification for dynamic reward weighting in PPO training

- `models.py`: BERT-based intent classifier for EXPLORE/ACTION/DIALOGUE categorization
- `train_hybrid_player.py`: Training script for intent classification
- `models/`: Trained classifier models

**Key Feature**: Enables context-aware critic weighting based on player action type

### Evaluation

**Purpose**: Comprehensive evaluation pipeline with advanced metrics

- `full_eval.py`: Complete evaluation system with statistical analysis and visualization
- `evalc.py`: Basic critic evaluation utilities
- `eval_config.yaml`: Configuration for evaluation parameters and model paths

## Technical Stack

- **Base Model**: Phi-2 (2.7B) with LoRA adapters (r=32, α=64)
- **Critics**:
  - Narrative: DeBERTa-v3-base (regression)
  - Causal: RoBERTa-base (3-class, 88.09% accuracy)
  - World: RoBERTa-large (98.3% accuracy)
  - Character: DeBERTa-v3-base (87.6% accuracy)
- **Player Simulation**: BERT-based intent classifier
- **Training**: PyTorch, Transformers, TRL, PEFT
- **Optimization**: LoRA adapters, mixed precision, gradient checkpointing

## Datasets

The project utilizes multiple datasets for comprehensive training:

### Dataset Download

🔗 **[Dataset](https://1drv.ms/f/c/bdcf3b74ef9b6129/Ep8Im9Kl-SNOspd2NAYqJ4MBzBsoeKe3uRlr6IhZiDkyGg?e=hrZgDd)**
🔗 **[Model](https://iiithydresearch-my.sharepoint.com/my?id=%2Fpersonal%2Faman%5Fsrivastava%5Fresearch%5Fiiit%5Fac%5Fin%2FDocuments%2FANLPProjectModels&viewid=645125c6%2Dfd29%2D494e%2D9af6%2Ddc9d91243e02&source=waffle)**

### Primary Datasets

- **CRD3**: Critical Role D&D transcripts (~200 episodes, 2 campaigns)
- **LIGHT**: Fantasy dialogue and action data (~20K training samples)
- **ROCStories/TinyStories**: Narrative coherence training (~1.9GB stories)

### Processed Training Data

- **DM-SFT Dataset**: 75,000 examples (CRD3 D&D transcripts)
- **Critic Training**: 40,906 examples (30K ROCStories + 10.9K DM pairs)
- **Causal Critic Training**: 382,530 premise-hypothesis pairs (3-class NLI)
- **World Consistency**: D&D 5e rules corpus and validation dataset
- **Character Voice**: Critical Role character dialogue with embeddings

### Dataset Usage by Component

- **DM-SFT**: Data Splits (instruction-tuned LIGHT+CRD3 combination)
- **Narrative Critic**: Critic Training dataset with quality labels
- **Causal Critic**: Causal critic training data for consistency evaluation
- **Hybrid Player**: LIGHT dialogue data for player simulation
- **PPO Training**: All datasets integrated with critic feedback

_See `Data/README.md` for detailed dataset documentation and setup instructions._

## Getting Started

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (16GB+ VRAM recommended)
- PyTorch 2.0+

### Environment Setup

```bash
# Install core dependencies
pip install torch transformers peft trl datasets
pip install accelerate bitsandbytes wandb

# Install additional requirements
pip install PyYAML numpy pandas scikit-learn
pip install sentencepiece protobuf
```

### Quick Start Training

1. **Train SFT Baseline (Phi-2)**

   ```bash
   cd DM-SFT/
   python train_sft_phi2_optimized.py --config sft_config.yaml
   ```

2. **Train Critics** (if not using pre-trained)

   ```bash
   # Causal Critic
   cd "causal critic/"
   python train_3class.py

   # Narrative Critic
   cd ../narrative\ critic/
   python narrative_critic.py --train
   ```

3. **Run PPO Training**
   ```bash
   cd PPO/
   python train_complete_ppo.py \
     --steps 1000 \
     --batch-size 32 \
     --val-interval 50 \
     --output-dir checkpoints/
   ```

### Model Inference

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Load base model
model = AutoModelForCausalLM.from_pretrained("microsoft/phi-2")
tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2")

# Load LoRA adapter
model = PeftModel.from_pretrained(model, "DM-SFT/models/sft_phi2_improved")

# Generate DM response
prompt = "Player: I examine the ancient door.\nDungeon Master:"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_length=200)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
```

## Training Details

### Hardware Requirements

- **GPU**: NVIDIA RTX 4080 Super (16GB VRAM) or equivalent
- **RAM**: 32GB+ system memory recommended
- **Storage**: ~50GB for models and datasets

### Training Times (on RTX 4080 Super)

- **SFT Training**: ~3.87 hours (1 epoch, 75K examples)
- **Causal Critic**: ~2-3 hours (382K examples)
- **PPO Training**: ~53 minutes (1000 steps, batch size 32)
- **Narrative Critic**: ~4-5 hours (40.9K examples)

### Hyperparameters

**SFT (Phi-2)**:

- Learning rate: 2e-4
- LoRA rank (r): 32
- LoRA alpha: 64
- Batch size: 4 (gradient accumulation: 4)
- Max length: 512 tokens

**PPO**:

- Learning rate: 1e-5
- Batch size: 32
- PPO epochs: 4
- Clip range: 0.2
- Value function coefficient: 0.1

**Causal Critic**:

- Model: RoBERTa-base
- Learning rate: 2e-5
- Batch size: 16
- Training examples: 382,530 (3-class balanced)

## Key Features

- **Memory Efficient**: LoRA adapters enable training on consumer GPUs (16GB VRAM)
- **Modular Design**: Each component can be trained and evaluated independently
- **Comprehensive Evaluation**: Statistical analysis, visualization, and domain-specific metrics
- **Intent-Aware**: Dynamic reward weighting based on player action classification
- **Research-Grade**: Reproducible experiments with detailed configuration management
- **Production-Ready**: All critics validated with >85% accuracy

## Performance Summary

### Model Accuracies

| Component         | Model         | Accuracy   | Training Examples   |
| ----------------- | ------------- | ---------- | ------------------- |
| Causal Critic     | RoBERTa-base  | 88.09%     | 382,530             |
| World Consistency | RoBERTa-large | 98.3%      | D&D rules corpus    |
| Character Voice   | DeBERTa-v3    | 87.6%      | Character dialogues |
| Narrative Quality | DeBERTa-v3    | Regression | 40,906              |

### PPO Training Results

- **SFT Baseline**: Mean reward 0.412
- **PPO Trained**: Mean reward 0.611 (+48.3% improvement)
- **Causal Consistency**: 0.189 → 0.567 (+200% improvement)
- **Training Steps**: 1,000 steps (converged, no overfitting)

## Configuration

Each module includes detailed YAML configuration files for:

- Model hyperparameters
- Training schedules
- Hardware optimization
- Evaluation parameters

See individual module READMEs for specific configuration options and usage instructions.

---
