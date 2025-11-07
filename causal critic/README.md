# Causal Critic: Causal Consistency Evaluation

This module implements causal consistency evaluation between player actions and DM responses using a fine-tuned 3-class Natural Language Inference (NLI) model trained on D&D gameplay data.

## Purpose

Evaluates the logical consistency and causal responsiveness of DM responses by using NLI entailment probability to measure how well a DM response follows from a player action. This critic ensures that generated responses maintain logical coherence with the established narrative context.

## Current Production Model

**Model**: RoBERTa-base Fine-tuned on 3-Class Causal NLI

- **Location**: `../model_causalcritic_3class/`
- **Architecture**: FacebookAI/roberta-base (125M parameters)
- **Task**: 3-class NLI (contradiction, neutral, entailment)
- **Test Accuracy**: **88.09%**
- **Training Date**: November 7, 2025
- **Model Size**: 476MB

### Training Results

**Overall Performance:**

- Test Accuracy: 88.09%
- Macro F1: 88.15%
- Macro Precision: 88.23%
- Macro Recall: 88.09%

**Per-Class Metrics:**
| Class | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| Contradiction (0) | 82.83% | 83.49% | 83.16% |
| Neutral (1) | 96.11% | 92.54% | 94.29% |
| Entailment (2) | 85.76% | 88.25% | 86.99% |

**Training Configuration:**

- Base Model: FacebookAI/roberta-base
- Training Examples: 306,024
- Epochs: 3
- Batch Size: 16
- Learning Rate: 2e-5
- Optimizer: AdamW

## Dataset

**Production Dataset**: 3-Class Causal Critic Training Data

- **Location**: `../data/causal_critic_training_3class/`
- **Total Size**: 382,530 examples (balanced across 3 classes)
  - **Entailment**: 127,510 (33.3%) - Direct Player→DM causal pairs
  - **Contradiction**: 127,510 (33.3%) - Mismatched action-response pairs
  - **Neutral**: 127,510 (33.3%) - Related but non-causal statements
- **Split**: Train 80% (306,024) / Val 10% (38,253) / Test 10% (38,253)

**Legacy Dataset**: 2-Class Causal Critic Training Data

- **Location**: `../data/causal_critic_training/`
- **Note**: Used for older 2-class model (deprecated)

**Data Source**: CRD3 Dataset Processing

- **Raw Data**: Critical Role D&D session transcripts (324 episodes)
- **Extraction**: 127,510 Player action → DM response pairs
- **Processing**: 3-class pair generation via `data_prep_3class.py`

## Files Overview

### Core Files

#### `causal_critic.py`

Main causal consistency evaluator using pre-trained NLI models.

**Main Class: `CausalCritic`**

- Zero-shot NLI-based evaluation (no training required)
  **Main Class: `CausalCritic`**

- Loads fine-tuned 3-class model from `../model_causalcritic_3class/`
- Uses RoBERTa-base trained on 306K D&D examples
- Converts NLI probabilities into causal consistency scores
- Optimized for both single evaluation and batch processing

**Key Methods:**

- `__init__(model_path)`: Loads trained 3-class NLI model (RoBERTa-base)
- `score(premise, hypothesis)`: Evaluates single premise-hypothesis pair
- `batch_score()`: Efficient batch evaluation for multiple pairs
- `evaluate_pairs()`: Comprehensive evaluation with detailed output
- `_prepare_nli_input()`: Formats input for NLI model
- `_extract_entailment_score()`: Converts NLI logits to consistency scores

**NLI Framework:**

- **Premise**: Player action or established context
- **Hypothesis**: DM response to evaluate
- **Entailment Score**: Probability that response logically follows from action
- **Label Mapping**: 0=contradiction, 1=neutral, 2=entailment

**Scoring Logic:**

```python
# Convert NLI probabilities to causal consistency scores
logits = model(input_ids, attention_mask).logits
probs = F.softmax(logits, dim=-1)
entailment_prob = probs[:, 2]  # Index 2 = entailment
score = entailment_prob.item()  # Range: 0.0 to 1.0
```

#### `train_3class.py` ⭐ **[PRODUCTION MODEL TRAINING]**

Training script for the current production 3-class causal critic.

**Training Process:**

- Loads RoBERTa-base as foundation model
- Fine-tunes on 306K balanced 3-class examples from CRD3
- Implements proper 3-class classification (contradiction, neutral, entailment)
- Achieves 88.09% test accuracy
- Saves domain-adapted model to `../model_causalcritic_3class/`

**Key Features:**

- **Balanced Training**: Equal examples per class prevents bias
- **Comprehensive Metrics**: Per-class precision, recall, F1 tracking
- **Early Stopping**: Monitors validation loss to prevent overfitting
- **Model Checkpointing**: Saves best performing checkpoint
- **Training Summary**: Generates detailed metrics JSON

**Training Configuration:**

- 3 epochs, batch size 16
- Learning rate: 2e-5 with AdamW optimizer
- Evaluation every epoch
- Saves final model + training summary

#### `data_prep_3class.py` ⭐ **[PRODUCTION DATA PIPELINE]**

Data preparation for the production 3-class model.

**Key Functions:**

- `extract_causal_pairs_from_crd3()`: Extracts 127K Player→DM pairs
- `create_negative_pairs()`: Generates contradiction examples
- `create_neutral_pairs()`: Creates neutral relationship examples
- `balance_dataset()`: Ensures equal class distribution
- `save_to_datasets_format()`: Exports in HuggingFace format

**Data Processing Pipeline:**

1. **Entailment Pairs**: Direct Player action → DM response (causal)
2. **Contradiction Pairs**: Mismatched player actions with unrelated DM responses
3. **Neutral Pairs**: Related statements from same speaker type, different contexts
4. **Balancing**: Equal samples per class (127,510 each)
5. **Train/Val/Test Split**: 80/10/10

**Output Location**: `../data/causal_critic_training_3class/`

#### `data_prep.py` (Legacy)

Data extraction and preprocessing for the older 2-class version.

**Note**: This generates data for the deprecated 2-class model. Use `data_prep_3class.py` for current production pipeline.

**Key Functions:**

- `extract_causal_pairs_from_crd3()`: Extracts player→DM interaction sequences
- `create_negative_pairs()`: Generates negative examples for training
- `save_to_datasets_format()`: Saves processed data in HuggingFace format
- `validate_pairs()`: Quality control for extracted pairs

**Speaker Identification:**

- **DM Detection**: Identifies "MATT"/"MATTHEW" as Dungeon Master
- **Player Detection**: All other speakers classified as players
- **Sequential Logic**: Extracts Player→DM response patterns

**Quality Filters:**

- Minimum text length (20 characters)
- Maximum text length (500 characters)
- Content quality heuristics
- Speaker consistency checks

#### `train.py` (Legacy)

Training script for the older 2-class version.

**Note**: This trains the deprecated 2-class model. Use `train_3class.py` for current production training.

**Training Process:**

- Loads pre-trained NLI model as starting point
- Fine-tunes on extracted CRD3 causal pairs
- Implements contrastive learning with positive/negative pairs
- Saves domain-adapted model for improved D&D consistency evaluation

**Key Components:**

- **Data Loading**: Loads processed causal pairs from data_prep.py
- **Model Setup**: Configures pre-trained NLI model for fine-tuning
- **Training Loop**: Implements supervised learning on causal pairs
- **Evaluation**: Validates performance on held-out test set
- **Model Saving**: Saves fine-tuned model for production use

## Technical Implementation

### Fine-Tuned 3-Class NLI Approach

- **Base Model**: FacebookAI/roberta-base (125M parameters)
- **Training**: Fine-tuned on 306K D&D-specific causal pairs
- **Domain Adaptation**: Specialized for D&D narrative consistency
- **Robust Performance**: 88.09% accuracy on held-out test set
- **Balanced Classes**: Equal training examples prevent prediction bias

### Input Format

```python
# NLI input format for causal evaluation
premise = "I cast Fireball at the goblin horde"
hypothesis = "The goblins scatter as flames engulf them"
nli_input = f"{premise} [SEP] {hypothesis}"
```

### Scoring Mechanism

- **Entailment Probability**: Primary metric for causal consistency
- **Contradiction Detection**: Identifies logically inconsistent responses
- **Neutral Handling**: Manages ambiguous or unrelated responses
- **Score Normalization**: Maps probabilities to 0.0-1.0 range

### Batch Processing

- **Efficient Tokenization**: Handles multiple pairs simultaneously
- **GPU Optimization**: Leverages CUDA acceleration when available
- **Memory Management**: Processes large batches without OOM errors
- **Progress Tracking**: Provides detailed evaluation progress

## Usage

### Prerequisites

```bash
pip install torch transformers
pip install datasets tqdm pathlib
pip install numpy pandas scikit-learn
```

### Training the Model

```bash
# 1. Prepare 3-class dataset (if not already done)
python data_prep_3class.py

# 2. Train the model
python train_3class.py

# This creates model at: ../model_causalcritic_3class/
# Training takes ~2-3 hours on GPU
```

### Using Trained Model for Evaluation

```python
from causal_critic import CausalCritic

# Initialize critic with trained model
critic = CausalCritic(model_path="../model_causalcritic_3class")

# Evaluate single pair
score = critic.score(
    premise="I search the room for traps",
    hypothesis="You find a pressure plate hidden under the carpet"
)
print(f"Causal consistency: {score:.3f}")

# Batch evaluation
pairs = [
    ("I attack with my sword", "You strike the orc for 8 damage"),
    ("I cast healing spell", "The dragon breathes fire at you")
]
scores = critic.batch_score(pairs)
print(f"Scores: {scores}")
```

## Integration with PPO Training

### Reward Computation

```python
# Used in PPO/train_complete_ppo.py for RL training
class CausalConsistencyCritic:
    def __init__(self, model_path, device):
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

    def get_reward(self, contexts, responses):
        # Convert player context + DM response to NLI format
        # Return causal consistency scores for PPO optimization
        # Uses entailment probability as reward signal
```

### Dynamic Weighting in PPO

```yaml
# PPO configuration for different player intents
ACTION:
  narrative: 0.3
  causal: 0.7 # High weight for action sequences

EXPLORE:
  narrative: 0.8
  causal: 0.2 # Lower weight for exploration

DIALOGUE:
  narrative: 0.6
  causal: 0.4 # Balanced for conversations
```

### PPO Configuration

```yaml
# In PPO/ppo_config.yaml
model_paths:
  causal_critic_path: "/path/to/model_causalcritic_3class"
```

## Output Examples

### High Consistency Examples (Entailment)

```
Premise: "I cast Fireball at the goblin horde"
Hypothesis: "The goblins scatter as flames engulf them"
Score: 0.91 (Strong causal relationship)

Premise: "I search the room for secret doors"
Hypothesis: "Rolling perception, you notice a crack in the wall"
Score: 0.84 (Good logical follow-through)

Premise: "I ask the innkeeper about the missing villagers"
Hypothesis: "He nervously glances around and lowers his voice"
Score: 0.78 (Appropriate causal response)
```

### Low Consistency Examples (Contradiction)

```
Premise: "I ask the bartender about rumors"
Hypothesis: "A dragon suddenly appears and attacks"
Score: 0.09 (Poor causal relationship)

Premise: "I heal my wounded companion"
Hypothesis: "You discover a hidden treasure chest"
Score: 0.12 (No logical connection)

Premise: "I stealth past the guards"
Hypothesis: "The guards congratulate you on your arrival"
Score: 0.15 (Contradictory response)
```

### Neutral Examples

```
Premise: "I enter the tavern"
Hypothesis: "The weather outside is getting worse"
Score: 0.45 (Related but not causally linked)

Premise: "I examine my equipment"
Hypothesis: "Other adventurers are gathering at the quest board"
Score: 0.42 (Contextually related, not causal)
```

## Training Pipeline

### Complete Training Workflow

```bash
# Option 1: Use automated pipeline
bash train_3class_pipeline.sh

# Option 2: Manual step-by-step
python data_prep_3class.py   # Prepare dataset
python train_3class.py        # Train model
```

### Output Files

After training, the following are generated:

- `../model_causalcritic_3class/model.safetensors` - Trained model weights
- `../model_causalcritic_3class/config.json` - Model configuration
- `../model_causalcritic_3class/tokenizer.json` - Tokenizer
- `../model_causalcritic_3class/training_summary.json` - Metrics and results

## Model Comparison

### Current Model (3-Class Fine-tuned)

- **Architecture**: RoBERTa-base
- **Training**: 306K D&D examples
- **Accuracy**: 88.09%
- **Classes**: 3 (balanced)
- **Status**: ✅ **Production**

### Legacy Model (Deprecated)

- **Architecture**: DeBERTa-v2
- **Training**: 2-class only
- **Issue**: Failed on 3-class data
- **Status**: ❌ Removed

## Research Applications

### Academic Use Cases

- **Narrative Consistency**: Measuring logical coherence in generated stories
- **Dialogue Systems**: Evaluating response appropriateness in conversational AI
- **Content Generation**: Quality control for automated D&D content
- **Ablation Studies**: Component analysis in multi-critic RL systems

### Evaluation Framework

- **Baseline Comparison**: Outperforms rule-based heuristics
- **Human Correlation**: Strong alignment with human judgments of consistency
- **Cross-Domain**: Applicable to story generation beyond D&D
- **Reproducible**: Consistent results across runs (deterministic inference)

### Customization Options

- **Threshold Tuning**: Adjustable consistency thresholds for different use cases
- **Context Length**: Variable input sequence lengths (up to 512 tokens)
- **Batch Sizes**: Configurable for different hardware constraints
- **Output Format**: Multiple score formats and detailed explanations
