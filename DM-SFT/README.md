# DM-SFT: Supervised Fine-Tuning Baseline

This module implements supervised fine-tuning of Phi-2 (2.7B) for Dungeon Master response generation using QLoRA optimization for memory-efficient training on consumer GPUs.

## Production Model: `sft_phi2_improved`

**Current Status**: ⭐ Production-ready baseline for PPO training

**Model Architecture**:

- **Base Model**: microsoft/phi-2 (2.7B parameters)
- **Fine-tuning Method**: QLoRA (4-bit quantization with LoRA adapters)
- **LoRA Configuration**: r=32, α=64, dropout=0.1
- **Target Modules**: q_proj, k_proj, v_proj, dense, fc1, fc2

**Training Results**:

| Metric             | Value      |
| ------------------ | ---------- |
| Training Loss      | 1.987      |
| Evaluation Loss    | 1.660      |
| Training Runtime   | 3.87 hours |
| Samples/Second     | 5.37       |
| Training Samples   | 75,000     |
| Validation Samples | 3,000      |
| Epochs             | 1          |

**Dataset Statistics**:

- **Training Set**: 75,000 examples from combined CRD3 + LIGHT
- **Validation Set**: 3,000 examples
- **Test Set**: Available in data splits
- **Sequence Length**: 384 tokens (optimized for D&D responses)
- **Format**: Instruction-tuned with DM system prompts

**Quality Metrics** (from evaluation on 45 diverse scenarios):

- **Average Response Length**: 122 words (σ=5.4)
- **Complete Responses**: 13.3% (properly ended with punctuation)
- **Response Cutoff Rate**: 86.7% (responses reach max length)
- **D&D Terms per Response**: 1.38 (σ=2.1)
- **Repetition Score**: 0.462 (lower is better)

**Performance by Category**:

- **Combat**: 119 words avg, 2.83 D&D terms (highest technical density)
- **Exploration**: 124 words avg, immersive scene descriptions
- **Dialogue**: 124 words avg, character interaction focus
- **Problem Solving**: 121 words avg, logical puzzle integration
- **Magic**: 118 words avg, 2.67 D&D terms (spell mechanics)

**Model Location**: `models/sft_phi2_improved/`

## Purpose

Creates a baseline Dungeon Master model through supervised learning on combined D&D and fantasy dialogue data. This baseline serves as the foundation for subsequent multi-critic reinforcement learning training in the PPO module.

## Dataset

**Primary Dataset**: Data Splits (Combined LIGHT + CRD3)

- **Location**: `../Data/splits/`
- **Format**: Instruction-tuned format with DM system prompts
- **Content**: Combined LIGHT fantasy dialogue and CRD3 D&D transcripts
- **Structure**: Train/validation/test splits with consistent DM role formatting

**Data Sources**:

- **CRD3**: Critical Role D&D session transcripts with DM responses
- **LIGHT**: Fantasy dialogue scenarios and character interactions
- **Integration**: Both sources converted from Llama-2 format to Phi-2 format

**Prompt Format Conversion**:

The training script (`train_sft_phi2_optimized.py`) converts Llama-2 instruction format to Phi-2 compatible format:

**Original Llama-2 Format** (in data files):

```
<s>[INST] <<SYS>>
You are an expert Dungeon Master...
<</SYS>>

As the Dungeon Master, describe a scene or respond to the player: {player_input}
[/INST] {dm_response} </s>
```

**Converted Phi-2 Format** (used for training):

```
You are a Dungeon Master in a fantasy RPG game.

Player: {player_input}
Dungeon Master: {dm_response}
```

For DM dialogue without player context (common in CRD3):

```
You are a Dungeon Master in a fantasy RPG game.

Dungeon Master: {dm_response}
```

**Data Preprocessing**:

- Automatic format conversion from Llama-2 to Phi-2 style
- Role definition included for consistent DM behavior
- Quality filtering removes transcription artifacts and low-quality content
- Context preservation from both dialogue sources
- Text cleaning (removes excessive tokens, whitespace, inaudible markers)

## Files Overview

### Core Files

#### `train_sft.py` ⭐ **[PRODUCTION TRAINING SCRIPT]**

Main training script implementing QLoRA-optimized supervised fine-tuning for the production model.

**Key Functions:**

- `load_config()`: Loads training configuration from YAML
- `load_dataset_splits()`: Loads and preprocesses D&D dialogue data from LIGHT + CRD3
- `setup_model_and_tokenizer()`: Configures Phi-2 with QLoRA adapters and 4-bit quantization
- `setup_lora_config()`: Creates LoRA configuration for parameter-efficient fine-tuning
- `setup_training_args()`: Configures training parameters for GPU optimization
- `compute_metrics()`: Evaluation metrics during training
- `main()`: Orchestrates complete training pipeline

**Training Configuration**:

- **Batch Size**: Effective batch size of 8 (1 per device × 8 gradient accumulation)
- **Learning Rate**: 1.5e-4 with cosine scheduler
- **Optimizer**: Paged AdamW 8-bit for memory efficiency
- **Warmup**: 200 steps (5% warmup ratio)
- **Evaluation**: Every 1000 steps
- **Checkpointing**: Saves best model by eval_loss

**Memory Optimizations**:

- 4-bit quantization with NormalFloat4 (NF4)
- LoRA adapters (r=32, α=64) for parameter-efficient fine-tuning
- BF16 mixed precision training
- Gradient checkpointing for reduced memory footprint
- Paged optimizers for efficient VRAM usage

**Output**: Trained model saved to `models/sft_phi2_improved/`

#### `sft_config.yaml`

Legacy configuration file for earlier training experiments.

**Note**: The production model uses `sft_config_improved.yaml` with optimized hyperparameters. This file remains for reference and backward compatibility.

#### `sft_config_improved.yaml` ⭐ **[PRODUCTION CONFIG]**

Comprehensive training configuration for the current production model.

**Key Sections:**

- `project`: Experiment tracking and W&B integration
- `data`: Dataset paths, sequence length (384), sample limits (75K train, 3K eval)
- `model`: Phi-2 base model with 4-bit quantization settings
- `lora`: LoRA adapter configuration (r=32, α=64, dropout=0.1)
- `training`: Training hyperparameters and optimization settings
- `generation`: Inference parameters for validation samples

**Critical Parameters**:

```yaml
# Memory optimization for RTX 4080 Super
model:
  load_in_4bit: true
  bnb_4bit_quant_type: "nf4"
  bnb_4bit_use_double_quant: true
  bnb_4bit_compute_dtype: float16

# LoRA configuration (production settings)
lora:
  r: 32
  lora_alpha: 64
  lora_dropout: 0.1
  target_modules: ["q_proj", "k_proj", "v_proj", "dense", "fc1", "fc2"]

# Training schedule
training:
  num_train_epochs: 1
  per_device_train_batch_size: 1
  gradient_accumulation_steps: 8
  learning_rate: 0.00015
  lr_scheduler_type: cosine
  warmup_steps: 200
```

### Model Artifacts Directory

#### `models/sft_phi2_improved/` ⭐ **[PRODUCTION MODEL]**

Contains the trained production model and all associated artifacts.

**Model Files**:

- `adapter_model.safetensors`: Trained LoRA adapter weights
- `adapter_config.json`: LoRA configuration and metadata
- `training_config.yaml`: Complete training configuration used
- `training_metrics.json`: Final training metrics (loss, runtime, throughput)
- `eval_metrics.json`: Evaluation metrics on validation set

**Tokenizer Files**:

- `tokenizer.json`: Fast tokenizer configuration
- `tokenizer_config.json`: Tokenizer settings and special tokens
- `vocab.json`: Vocabulary mapping
- `added_tokens.json`: Custom tokens for D&D domain
- `special_tokens_map.json`: Special token definitions

**Evaluation Artifacts**:

- `generation_history.json`: Sample generations during training
- Located at: `../evaluation_results/`
  - `metrics/summary_statistics.json`: Comprehensive quality analysis
  - `samples/generated_samples.json`: 45 diverse scenario responses

**Key Metrics Summary**:

```json
{
  "train_loss": 1.987,
  "eval_loss": 1.660,
  "train_runtime": 13933.13s (3.87 hours),
  "train_samples_per_second": 5.37,
  "epoch": 1.0
}
```

#### `sft_model/` (Legacy)

**Note**: This directory is deprecated. The production model is in `models/sft_phi2_improved/`.

#### `models/sft_baseline_interim/` (Legacy)

**Note**: Earlier training checkpoint. Use `models/sft_phi2_improved/` for current production model.

### Evaluation Results

#### `evaluation_results/metrics/summary_statistics.json`

Comprehensive quality analysis across 45 test scenarios covering 9 D&D categories.

**Overall Quality Metrics**:

- **Response Length**: 122 ± 5.4 words (consistent output)
- **Sentence Length**: 10.4 ± 3.3 words per sentence
- **D&D Terminology**: 1.38 ± 2.1 terms per response
- **Repetition Score**: 0.462 (moderate repetition)
- **Complete Responses**: 6/45 (13.3%) properly ended
- **Cutoff Rate**: 86.7% (most responses hit max length)

**Category Performance**:

- **Best D&D Term Usage**: Combat (2.83), Magic (2.67), Investigation (3.0)
- **Lowest Repetition**: Problem Solving (0.412), Dialogue (0.413)
- **Highest Repetition**: Stealth (0.599), Investigation (0.509)

#### `evaluation_results/samples/generated_samples.json`

45 example responses across diverse D&D scenarios including:

- Combat encounters
- Exploration and discovery
- Social interactions and dialogue
- Problem-solving challenges
- Magic and spellcasting
- Stealth and investigation
- Survival situations

## Technical Implementation

### Model Architecture

- **Base Model**: microsoft/phi-2 (2.7B parameters)
- **Quantization**: 4-bit NormalFloat4 with double quantization
- **Adapters**: LoRA on attention and feed-forward layers
  - Query, Key, Value projections (q_proj, k_proj, v_proj)
  - Dense layers and feed-forward (dense, fc1, fc2)
- **Memory Usage**: ~11GB VRAM (fits RTX 4080 Super 16GB with headroom)
- **Inference Mode**: Adapters can be merged with base model for faster inference

### Training Strategy

- **Dataset**: 75,000 examples from combined CRD3 + LIGHT
- **Validation Set**: 3,000 examples for continuous evaluation
- **Sequence Length**: 384 tokens (optimized for D&D response length)
- **Epochs**: 1 epoch (~3.87 hours on RTX 4080 Super)
- **Batch Size**: Effective batch size of 8 via gradient accumulation
- **Learning Rate**: 1.5e-4 with cosine scheduler and 200-step warmup
- **Training Throughput**: 5.37 samples/second

### Hardware Optimization

- **Single GPU**: Optimized for RTX 4080 Super 16GB
- **Memory**: Gradient checkpointing, BF16 precision, 4-bit quantization
- **Optimizer**: Paged AdamW 8-bit for efficient memory usage
- **Data Loading**: 4 workers with memory pinning
- **Checkpoint Strategy**: Saves top 3 checkpoints by eval_loss

### Quality Improvements

The production model (`sft_phi2_improved`) includes several quality enhancements:

1. **Increased LoRA Rank**: r=32 (vs r=16 earlier) for higher capacity
2. **Higher LoRA Alpha**: α=64 for stronger adapter influence
3. **Extended Target Modules**: Includes fc1, fc2 for better feed-forward learning
4. **Larger Training Set**: 75K examples (vs 26K earlier subset)
5. **Optimized Sequence Length**: 384 tokens (vs 256) for fuller responses
6. **Improved Generation Parameters**:
   - Temperature: 0.8 (balanced creativity)
   - Top-p: 0.9 (nucleus sampling)
   - Repetition penalty: 1.2 (reduced repetition)
   - No-repeat n-gram: 3 (prevents immediate repetition)

## Usage

### Prerequisites

```bash
pip install torch transformers peft trl datasets
pip install accelerate bitsandbytes wandb
```

### Configuration

1. **Production Training** (recommended):

   ```bash
   # Uses sft_config_improved.yaml
   python train_sft.py
   ```

2. **Custom Configuration**:

   ```bash
   # Edit sft_config_improved.yaml for your hardware setup
   # Adjust data paths, batch size, and model parameters as needed
   python train_sft.py --config sft_config_improved.yaml
   ```

3. **Key Configuration Options**:
   - `data.max_train_samples`: Number of training examples (default: 75,000)
   - `data.max_seq_length`: Maximum sequence length (default: 384)
   - `training.per_device_train_batch_size`: Batch size per GPU (default: 1)
   - `training.gradient_accumulation_steps`: Effective batch multiplier (default: 8)
   - `lora.r`: LoRA rank, higher = more capacity (default: 32)
   - `training.learning_rate`: Learning rate (default: 1.5e-4)

### Training

```bash
# Run production training with optimized settings
python train_sft.py

# Monitor training progress
# - W&B dashboard for real-time metrics
# - Console logs for step-by-step progress
# - Validation loss tracked every 1000 steps
```

**Expected Training Time**:

- **RTX 4080 Super 16GB**: ~3.9 hours for 75K samples (1 epoch)
- **Throughput**: ~5.4 samples/second
- **Checkpoint Saves**: Every 1000 steps + best model

### Monitoring

- **W&B Dashboard**: Real-time training metrics and loss curves
- **Local Logs**: Console output with progress tracking and ETA
- **Checkpoints**: Saved in `models/sft_phi2_improved/`
- **Validation Samples**: Generated every 250 steps for quality inspection

## Output

### Model Files

The training produces a complete model package in `models/sft_phi2_improved/`:

**Core Model Files**:

- `adapter_config.json`: LoRA configuration and hyperparameters
- `adapter_model.safetensors`: Trained adapter weights (~100MB)
- `tokenizer.json`: Fast tokenizer for inference
- `vocab.json`, `merges.txt`: Vocabulary and BPE merges
- `special_tokens_map.json`: Special token definitions

**Training Artifacts**:

- `training_config.yaml`: Complete configuration used for training
- `training_metrics.json`: Final training statistics
- `eval_metrics.json`: Validation set performance
- `generation_history.json`: Sample outputs during training

### Loading Trained Model

**For PPO Training** (LoRA adapters only):

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

# Load base model with quantization
base_model = AutoModelForCausalLM.from_pretrained(
    "microsoft/phi-2",
    load_in_4bit=True,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True
)

# Load LoRA adapters (for further training)
model = PeftModel.from_pretrained(
    base_model,
    "DM-SFT/models/sft_phi2_improved"
)
tokenizer = AutoTokenizer.from_pretrained("DM-SFT/models/sft_phi2_improved")
```

**For Inference** (merged model):

**For Inference** (merged model):

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

# Load base model
base_model = AutoModelForCausalLM.from_pretrained(
    "microsoft/phi-2",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map="auto"
)

# Load and merge LoRA adapters
model = PeftModel.from_pretrained(base_model, "DM-SFT/models/sft_phi2_improved")
model = model.merge_and_unload()  # Merge for faster inference
tokenizer = AutoTokenizer.from_pretrained("DM-SFT/models/sft_phi2_improved")

# Generate DM response using Phi-2 format
player_action = "I cautiously open the ancient door."
prompt = f"""You are a Dungeon Master in a fantasy RPG game.

Player: {player_action}
Dungeon Master:"""

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(
    **inputs,
    max_new_tokens=200,
    temperature=0.8,
    top_p=0.9,
    repetition_penalty=1.2,
    no_repeat_ngram_size=3,
    do_sample=True
)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)

# Extract just the DM's response
dm_response = response.split("Dungeon Master:")[-1].strip()
print(f"DM: {dm_response}")
```

### Performance Metrics

**Training Efficiency**:

- Loss Convergence: Train 1.987 → Eval 1.660 (no overfitting)
- Training Speed: 5.37 samples/second on RTX 4080 Super
- Memory Footprint: ~11GB VRAM (including optimizer states)

**Response Quality**:

- Consistent Length: 122 ± 5.4 words (tight distribution)
- D&D Terminology: 1.38 terms/response (domain-appropriate)
- Repetition: 0.462 score (moderate, can be improved in PPO)
- Completion: 13.3% proper endings (most hit max_length cutoff)

## Integration

This SFT baseline serves as the **policy model** for subsequent PPO training in the `../PPO/` module.

**Integration Flow**:

1. **SFT Training**: Fine-tunes Phi-2 on D&D dialogue (this module)
2. **Model Export**: Saves LoRA adapters to `models/sft_phi2_improved/`
3. **PPO Configuration**: Path configured in `../PPO/ppo_config.yaml`:
   ```yaml
   model:
     policy_path: "DM-SFT/models/sft_phi2_improved"
   ```
4. **PPO Training**: Loads SFT model as starting point for multi-critic RL

**Why SFT First?**:

- Provides domain knowledge (D&D rules, terminology, narrative style)
- Reduces PPO exploration time by starting from competent baseline
- Ensures coherent language generation before RL optimization
- Establishes response length and formatting patterns

**PPO Improvements Over SFT**:

- Causal consistency (via Causal Critic)
- Narrative quality (via Narrative Critic)
- Strategic gameplay decisions
- Reduced repetition through RL objective
- Better handling of edge cases and creative scenarios

## Troubleshooting

### Common Issues

**Out of Memory (OOM)**:

- Reduce `per_device_train_batch_size` (currently 1, cannot go lower)
- Reduce `gradient_accumulation_steps` (reduces effective batch size)
- Reduce `max_seq_length` from 384 to 256 or 192
- Enable additional memory optimizations in config
- Close other GPU-intensive applications

**Slow Training**:

- Increase `dataloader_num_workers` (default: 4)
- Check GPU utilization with `nvidia-smi`
- Ensure data is cached/preprocessed
- Verify not swapping to CPU

**Poor Quality Responses**:

- Increase training epochs (currently 1)
- Increase `max_train_samples` for more data
- Adjust generation parameters (temperature, top_p)
- Check data quality in training splits

**High Loss (Not Converging)**:

- Reduce learning rate (currently 1.5e-4)
- Increase warmup steps (currently 200)
- Check for data preprocessing issues
- Verify tokenizer is working correctly

### Memory Optimization Tips

**If still hitting OOM**:

```yaml
# Further reduce memory usage
training:
  gradient_accumulation_steps: 16 # Increase this
  per_device_train_batch_size: 1 # Already minimal

data:
  max_seq_length: 256 # Reduce from 384

# Enable additional flags
gradient_checkpointing: true
optim: "paged_adamw_8bit" # Already enabled
```

**For faster training (if memory allows)**:

```yaml
training:
  per_device_train_batch_size: 2 # Double batch size
  gradient_accumulation_steps: 4 # Halve accumulation
  # Keeps same effective batch size but trains faster
```

### Validation and Testing

**Check Model Quality**:

```bash
# Run evaluation on test scenarios
python evaluate_model.py --model models/sft_phi2_improved

# Generate sample responses
python generate_samples.py --model models/sft_phi2_improved --num_samples 10
```

**Evaluation Results Location**:

- Summary statistics: `evaluation_results/metrics/summary_statistics.json`
- Generated samples: `evaluation_results/samples/generated_samples.json`

### Known Limitations

1. **Response Cutoff**: 86.7% of responses hit max_length (384 tokens)

   - **Solution**: Increase `max_new_tokens` in generation config
   - **PPO Improvement**: RL training can learn to end responses naturally

2. **Moderate Repetition**: Repetition score of 0.462

   - **Solution**: Already using repetition_penalty=1.2
   - **PPO Improvement**: RL objective can further reduce repetition

3. **Variable D&D Term Density**: Ranges from 0 to 9 terms per response

   - **Context-dependent**: Combat/Magic naturally have more terms
   - **PPO Improvement**: Critics can encourage appropriate term usage

4. **Training Time**: ~4 hours for full training run
   - **Hardware-dependent**: Faster with RTX 4090 or A100
   - **Trade-off**: Can reduce training samples for faster iteration
