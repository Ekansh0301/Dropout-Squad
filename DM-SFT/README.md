# DM-SFT: Supervised Fine-Tuning Baseline

This module implements supervised fine-tuning of Llama-2-7B for Dungeon Master response generation using QLoRA optimization for memory-efficient training on consumer GPUs.

## Purpose

Creates a baseline Dungeon Master model through supervised learning on D&D dialogue data from the CRD3 dataset. This baseline serves as the foundation for subsequent multi-critic reinforcement learning training.

## Files Overview

### Core Files

#### `train_sft.py`

Main training script implementing QLoRA-optimized supervised fine-tuning.

**Key Functions:**

- `load_config()`: Loads training configuration from YAML
- `load_dataset_splits()`: Loads and preprocesses D&D dialogue data
- `setup_model_and_tokenizer()`: Configures Llama-2-7B with QLoRA adapters
- `setup_lora_config()`: Creates LoRA configuration for efficient fine-tuning
- `setup_training_args()`: Configures training parameters for multi-GPU setup
- `main()`: Orchestrates complete training pipeline

**Optimizations:**

- 4-bit quantization with NormalFloat4 for memory efficiency
- LoRA adapters (r=16, α=32) for parameter-efficient fine-tuning
- Mixed precision training with FP16
- Gradient checkpointing for memory optimization
- Multi-GPU data parallel training

#### `sft_config.yaml`

Comprehensive training configuration file.

**Key Sections:**

- `project`: Experiment tracking and naming
- `data`: Dataset paths and preprocessing parameters
- `model`: Base model and quantization settings
- `lora`: LoRA adapter configuration
- `training`: Training hyperparameters and optimization settings
- `hardware`: Multi-GPU and memory optimization settings

**Critical Parameters:**

```yaml
# Memory optimization for 4x GTX 1080 Ti
model:
  load_in_4bit: true
  bnb_4bit_quant_type: "nf4"

# LoRA configuration
lora:
  r: 16
  lora_alpha: 32
  target_modules: ["q_proj", "k_proj", "v_proj", "dense"]

# Training schedule
training:
  num_train_epochs: 1
  per_device_train_batch_size: 1
  gradient_accumulation_steps: 8
```

### Model Artifacts Directory

#### `sft_model/`

Contains model configurations and training artifacts.

#### `sft_model/training_config.yaml`

Detailed training configuration with hardware-specific optimizations.

**Key Features:**

- 4-bit quantization settings
- LoRA target module specifications
- Multi-GPU training parameters
- Memory optimization flags

## Technical Implementation

### Model Architecture

- **Base Model**: microsoft/phi-2 (alternative to Llama-2-7B for compatibility)
- **Quantization**: 4-bit NormalFloat4 with double quantization
- **Adapters**: LoRA on attention layers (q_proj, k_proj, v_proj, dense)
- **Memory Usage**: ~11GB per GPU (fits GTX 1080 Ti)

### Training Strategy

- **Data Subset**: 10% of CRD3 for faster iteration (26,329 examples)
- **Sequence Length**: 256 tokens (reduced from 512 for speed)
- **Epochs**: 1 epoch for interim training (~3 hours)
- **Batch Size**: Effective batch size of 32 via gradient accumulation

### Hardware Optimization

- **Multi-GPU**: Data parallel across 4x GTX 1080 Ti
- **Memory**: Gradient checkpointing, FP16, 4-bit quantization
- **Data Loading**: 4 workers with memory pinning
- **Synchronization**: Optimized DDP settings

## Usage

### Prerequisites

```bash
pip install torch transformers peft trl datasets
pip install accelerate bitsandbytes wandb
```

### Configuration

1. Edit `sft_config.yaml` for your hardware setup
2. Adjust data paths and model parameters as needed
3. Configure W&B logging (optional)

### Training

```bash
# Run training with default config
python train_sft.py

# Or specify custom config
python train_sft.py --config custom_config.yaml
```

### Monitoring

- **W&B Dashboard**: Real-time training metrics
- **Local Logs**: Console output with progress tracking
- **Checkpoints**: Saved in `models/sft_baseline_interim/`

## Output

### Model Files

The training produces a complete model package:

- `adapter_config.json`: LoRA configuration
- `adapter_model.safetensors`: Trained adapter weights
- `tokenizer.json`: Tokenizer configuration
- `training_args.bin`: Training arguments

### Loading Trained Model

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Load base model
base_model = AutoModelForCausalLM.from_pretrained(
    "microsoft/phi-2",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True
)

# Load LoRA adapters
model = PeftModel.from_pretrained(base_model, "models/sft_baseline_interim")
tokenizer = AutoTokenizer.from_pretrained("models/sft_baseline_interim")

# For inference, merge adapters
model = model.merge_and_unload()
```

## Integration

This SFT baseline serves as the policy model for subsequent PPO training in the `../PPO/` module. The trained model path is configured in `../PPO/ppo_config.yaml` as the starting point for multi-critic reinforcement learning.

## Troubleshooting

### Common Issues

- **OOM Errors**: Reduce batch size or enable additional memory optimizations
- **Slow Training**: Increase data loader workers or check GPU utilization
- **Poor Quality**: Increase training epochs or adjust learning rate

### Memory Optimization

- Enable gradient checkpointing
- Use FP16 mixed precision
- Reduce sequence length if needed
- Adjust gradient accumulation steps
