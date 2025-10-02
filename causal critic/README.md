# Causal Critic: Causal Consistency Evaluation

This module implements causal consistency evaluation between player actions and DM responses using a pre-trained Natural Language Inference (NLI) model for zero-shot logical coherence assessment.

## Purpose

Evaluates the logical consistency and causal responsiveness of DM responses by using NLI entailment probability to measure how well a DM response follows from a player action. This critic ensures that generated responses maintain logical coherence with the established narrative context.

## Dataset

**Primary Dataset**: Causal Critic Training Data
- **Location**: `../Data/causal_critic_training/`
- **Format**: Premise-hypothesis pairs for NLI-based evaluation
- **Structure**: `train/` and `val/` directories with processed examples

**Data Source**: CRD3 Dataset Processing
- **Raw Data**: Critical Role D&D session transcripts
- **Extraction**: Player action → DM response pairs
- **Processing**: Premise-hypothesis pair generation via `data_prep.py`


## Files Overview

### Core Files

#### `causal_critic.py`

Main causal consistency evaluator using pre-trained NLI models.

**Main Class: `CausalCritic`**

- Zero-shot NLI-based evaluation (no training required)
- Uses DeBERTa-v3-base-mnli-fever-anli for robust entailment detection
- Converts NLI probabilities into causal consistency scores
- Optimized for both single evaluation and batch processing

**Key Methods:**

- `__init__()`: Loads pre-trained NLI model (DeBERTa-v3-base)
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
entailment_prob = F.softmax(logits, dim=-1)[2]  # Index 2 = entailment
score = entailment_prob.item()  # Range: 0.0 to 1.0
```

#### `data_prep.py`

Data extraction and preprocessing for causal pair generation from CRD3 dataset.

**Key Functions:**

- `extract_causal_pairs_from_crd3()`: Extracts player→DM interaction sequences
- `create_negative_pairs()`: Generates negative examples for training
- `save_to_datasets_format()`: Saves processed data in HuggingFace format
- `validate_pairs()`: Quality control for extracted pairs

**Data Processing Pipeline:**

1. **File Processing**: Iterates through CRD3 JSON files
2. **Turn Sequence Analysis**: Identifies speaker transitions
3. **Pair Extraction**: Captures Player action → DM response sequences
4. **Quality Filtering**: Removes low-quality or short interactions
5. **Negative Sampling**: Creates mismatched pairs for training
6. **Format Conversion**: Saves in standard dataset format

**Speaker Identification:**

- **DM Detection**: Identifies "MATT"/"MATTHEW" as Dungeon Master
- **Player Detection**: All other speakers classified as players
- **Sequential Logic**: Extracts Player→DM response patterns

**Quality Filters:**

- Minimum text length (20 characters)
- Maximum text length (500 characters)
- Content quality heuristics
- Speaker consistency checks

#### `train.py`

Training script for fine-tuning causal critic on domain-specific data.

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

### Zero-Shot NLI Approach

- **Pre-trained Model**: DeBERTa-v3-base-mnli-fever-anli
- **No Training Required**: Uses existing NLI capabilities
- **Domain Transfer**: Leverages general logical reasoning for D&D contexts
- **Robust Performance**: Trained on diverse NLI datasets (MNLI, FEVER, ANLI)

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
pip install numpy pandas
```

### Zero-Shot Evaluation

```python
from causal_critic import CausalCritic

# Initialize critic (no training required)
critic = CausalCritic()

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
```

### Data Preparation

```bash
# Extract causal pairs from CRD3 dataset
python data_prep.py

# This creates training data for fine-tuning (optional)
```

### Fine-Tuning (Optional)

```bash
# Fine-tune on domain-specific data
python train.py

# Produces domain-adapted model with improved D&D understanding
```

## Integration with PPO Training

### Reward Computation

```python
# Used in PPO/critics.py for RL training
class CausalCritic:
    def get_reward(self, texts: List[str]) -> torch.Tensor:
        # Convert player context + DM response to NLI format
        # Return causal consistency scores for PPO optimization
```

### Dynamic Weighting

```yaml
# PPO configuration for different player intents
ACTION:
  narrative: 0.3
  causal: 0.7 # High weight for action sequences

EXPLORE:
  narrative: 0.8
  causal: 0.2 # Lower weight for exploration
```

## Output Examples

### High Consistency Examples

```
Premise: "I cast Fireball at the goblin horde"
Hypothesis: "The goblins scatter as flames engulf them"
Score: 0.89 (Strong causal relationship)

Premise: "I search the room for secret doors"
Hypothesis: "Rolling perception, you notice a crack in the wall"
Score: 0.78 (Good logical follow-through)
```

### Low Consistency Examples

```
Premise: "I ask the bartender about rumors"
Hypothesis: "A dragon suddenly appears and attacks"
Score: 0.12 (Poor causal relationship)

Premise: "I heal my wounded companion"
Hypothesis: "You discover a hidden treasure chest"
Score: 0.15 (No logical connection)
```

## Research Applications

### Academic Use Cases

- **Narrative Consistency**: Measuring logical coherence in generated stories
- **Dialogue Systems**: Evaluating response appropriateness
- **Content Generation**: Quality control for automated content
- **Ablation Studies**: Component analysis in multi-critic systems

### Evaluation Framework

- **Baseline Comparison**: Compares against simple heuristics
- **Human Correlation**: Aligns well with human judgments
- **Cross-Domain**: Applicable beyond D&D scenarios
- **Reproducible**: Consistent results across runs


### Customization Options

- **Threshold Tuning**: Adjustable consistency thresholds
- **Context Length**: Variable input sequence lengths
- **Batch Sizes**: Configurable for different hardware
- **Output Format**: Multiple score formats and explanations
