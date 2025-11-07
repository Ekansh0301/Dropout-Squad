# Narrative Critic: Narrative Quality Assessment

This module implements a DeBERTa-based regression model for assessing narrative quality in D&D responses, providing continuous quality scores for multi-critic reinforcement learning.

## Purpose

Evaluates the narrative quality of generated DM responses using a fine-tuned DeBERTa model. The critic assesses aspects like descriptiveness, engagement, creativity, and overall writing quality to provide reward signals for reinforcement learning training.

## Dataset

**Primary Dataset**: Critic Training Data

- **Location**: `../Data/critic_training/`
- **Total Examples**: 40,906 (36,815 train + 4,091 validation)
- **Format**: Arrow files with structured narrative examples and quality labels

**Data Composition**:

- **ROCStories Source**: 30,000 examples from TinyStories dataset
  - Purpose: Baseline narrative coherence understanding
  - Content: Short, coherent stories for quality assessment training
- **D&D Pairs Source**: 10,906 examples from DM response pairs
  - Purpose: Domain-specific narrative quality training
  - Content: Actual DM responses with quality annotations

**Quality Categories** (10,571 examples each):

- **Coherent**: High-quality, well-structured narratives
- **Shuffled**: Scrambled text for negative examples
- **Repetitive**: Repetitive text detection training
- **Truncated**: 9,193 incomplete narrative examples

**Data Preprocessing**:

- Quality score normalization for regression training
- Text cleaning and tokenization for DeBERTa input
- Balanced sampling across quality categories
- Domain-specific filtering for D&D content relevance

## Files Overview

### Core Files

#### `narrative_critic.py`

Main training script for the narrative quality assessment model.

**Key Components:**

**`DetailedLoggingCallback`**

- Custom callback for comprehensive training metrics logging
- Records training history with step and epoch information
- Saves detailed logs for analysis and debugging
- Enables training visualization and performance tracking

**Key Functions:**

- `compute_metrics()`: Calculates evaluation metrics (MSE, MAE, R²)
- `load_and_preprocess_data()`: Loads training data with quality filtering
- `setup_model_and_tokenizer()`: Configures DeBERTa for regression
- `setup_training_args()`: Defines training parameters and optimization
- `create_trainer()`: Sets up HuggingFace Trainer with custom callbacks
- `train_model()`: Orchestrates complete training pipeline
- `save_final_model()`: Saves trained model with evaluation results

**Training Pipeline:**

1. **Data Loading**: Loads narrative quality dataset with score labels
2. **Model Setup**: Configures DeBERTa-v3-base for regression (num_labels=1)
3. **Training Configuration**: Sets up multi-GPU training with optimization
4. **Training Loop**: Fine-tunes model with comprehensive logging
5. **Evaluation**: Computes quality metrics on validation set
6. **Model Saving**: Saves final model with performance statistics

**Regression Configuration:**

```python
# Model setup for continuous quality scores
model = AutoModelForSequenceClassification.from_pretrained(
    "microsoft/deberta-v3-base",
    num_labels=1,                    # Single output for regression
    problem_type="regression"        # Regression objective
)
```

**Quality Metrics:**

- **Mean Squared Error (MSE)**: Primary training objective
- **Mean Absolute Error (MAE)**: Average prediction error
- **R² Score**: Coefficient of determination for model fit
- **Training Loss**: Progressive optimization tracking

#### `critic_config.yaml`

Comprehensive configuration for narrative critic training.

**Key Sections:**

**Data Configuration:**

```yaml
data:
  max_seq_length: 128 # Input sequence length
  train_path: critic_training/train
  val_path: critic_training/val
```

**Model Configuration:**

```yaml
model:
  name: microsoft/deberta-v3-base
  num_labels: 1 # Regression output
  problem_type: regression
```

**Training Configuration:**

```yaml
training:
  num_train_epochs: 3
  per_device_train_batch_size: 32
  learning_rate: 3e-5
  lr_scheduler_type: cosine
  warmup_ratio: 0.1
  weight_decay: 0.01
  fp16: true # Memory optimization
```

**Multi-GPU Optimization:**

- Data parallel training across multiple GPUs
- Optimized batch sizes for memory efficiency
- Mixed precision training (FP16)
- Efficient data loading with multiple workers

#### `model/`

Directory containing trained model artifacts and configuration files.

**Model Artifacts:**

- `pytorch_model.bin`: Trained DeBERTa weights
- `config.json`: Model configuration and architecture
- `tokenizer.json`: Tokenizer configuration
- `training_args.bin`: Training arguments for reproducibility

## Technical Implementation

### Model Architecture

- **Base Model**: DeBERTa-v3-base (139M parameters)
- **Task Adaptation**: Fine-tuned for regression on narrative quality
- **Output Layer**: Single neuron with continuous activation
- **Optimization**: AdamW with cosine learning rate scheduling

### Training Strategy

- **Transfer Learning**: Starts from pre-trained DeBERTa checkpoint
- **Fine-Tuning**: Domain adaptation on D&D narrative data
- **Regression Objective**: MSE loss for continuous quality scores
- **Regularization**: Dropout, weight decay, and early stopping

### Data Processing

- **Tokenization**: DeBERTa tokenizer with 128 token sequences
- **Quality Labels**: Continuous scores (0.0 to 1.0) for narrative quality
- **Data Augmentation**: Optional text perturbations for robustness
- **Validation Split**: Held-out data for unbiased evaluation

### Multi-GPU Training

- **Data Parallel**: Distributes batches across available GPUs
- **Gradient Synchronization**: Efficient allreduce operations
- **Memory Optimization**: FP16 precision and optimized batch sizes
- **Scalability**: Linear speedup with additional GPUs

## Usage

### Prerequisites

```bash
pip install torch transformers datasets
pip install scikit-learn matplotlib
pip install accelerate wandb
```

### Configuration

1. Edit `critic_config.yaml` for your setup:

   - Adjust data paths for your dataset
   - Configure training parameters for your hardware
   - Set model output directory

2. Prepare training data:
   - Ensure narrative quality dataset is available
   - Format: text samples with continuous quality scores
   - Split into train/validation sets

### Training

```bash
# Run narrative critic training
python narrative_critic.py

# Monitor training progress
# - Check console output for metrics
# - View training_history.json for detailed logs
# - Monitor GPU utilization and memory usage
```

### Model Loading

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Load trained narrative critic
model = AutoModelForSequenceClassification.from_pretrained(
    "models/narrative_critic"
)
tokenizer = AutoTokenizer.from_pretrained("models/narrative_critic")

# Evaluate narrative quality
def get_quality_score(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
        score = torch.sigmoid(outputs.logits).item()
    return score
```

## Integration with PPO Training

### Reward Computation

```python
# Used in PPO/critics.py for RL training
class NarrativeCritic:
    def get_reward(self, texts: List[str]) -> torch.Tensor:
        # Tokenize batch of DM responses
        # Forward pass through trained DeBERTa model
        # Apply sigmoid activation for bounded rewards
        # Return quality scores for PPO optimization
```

### Dynamic Weighting

```yaml
# PPO configuration for different player intents
EXPLORE:
  narrative: 0.8 # High weight for exploration
  causal: 0.2

DIALOGUE:
  narrative: 0.6 # Balanced weight for dialogue
  causal: 0.4
```

## Training Outputs

### Model Performance

- **Training Metrics**: MSE, MAE, and R² tracked throughout training
- **Validation Performance**: Unbiased evaluation on held-out data
- **Learning Curves**: Training and validation loss progression
- **Quality Distribution**: Score distribution analysis

### Training Artifacts

```
models/narrative_critic/
├── pytorch_model.bin         # Trained model weights
├── config.json              # Model configuration
├── tokenizer.json           # Tokenizer configuration
├── training_args.bin        # Training arguments
├── training_history.json    # Detailed training logs
└── evaluation_results.json  # Final performance metrics
```

### Evaluation Results

- **Quantitative Metrics**: MSE, MAE, R² on test set
- **Qualitative Analysis**: Example predictions with explanations
- **Error Analysis**: Common failure modes and improvements
- **Correlation Analysis**: Agreement with human quality judgments

## Quality Examples

### High Quality Examples

```
Text: "The ancient library stretched endlessly before you, its towering shelves groaning under the weight of countless leather-bound tomes. Dust motes danced in shafts of golden sunlight that filtered through stained glass windows, casting rainbow patterns across worn stone floors."
Score: 0.72 (Excellent descriptive quality)

Text: "Your blade finds its mark with a satisfying thud, the orc's eyes widening in surprise before it crumples to the ground. Behind you, the sound of steel on steel echoes as your companions engage the remaining bandits."
Score: 0.64 (Good action description)
```

### Low Quality Examples

```
Text: "You see a room. There is a door. There is a table."
Score: 0.35 (Poor descriptive quality)

Text: "The thing happens and then the other thing happens too."
Score: 0.28 (Very poor narrative quality)
```

## Advanced Features

### Model Variants

### Training Enhancements

- **Data Augmentation**: Text perturbations for robustness
- **Ensemble Methods**: Multiple model combination
- **Active Learning**: Iterative data collection and training
- **Transfer Learning**: Cross-domain adaptation

### Research Applications

- **Narrative Analysis**: Systematic study of narrative quality factors
- **Content Generation**: Quality control for automated writing
- **Educational Tools**: Feedback systems for creative writing
- **Cross-Domain**: Adaptation to other narrative evaluation tas


### Find Trained weights at 

https://iiithydresearch-my.sharepoint.com/:f:/g/personal/jayant_g_research_iiit_ac_in/EkRB3fIyNoNDlkCbtGfbzpwBLUXR7PlOIkGitap7J9o5pQ?e=BqhPDP


https://iiithydresearch-my.sharepoint.com/:f:/g/personal/jayant_g_research_iiit_ac_in/EncAETRa38RKnKcz9_om_SQBUegp4A_FzJXbN8Neyb8XAg?e=t24fST

