# Narrative Critic Training - Setup and Execution Guide

This guide will help you set up and train the Narrative Critic model for your ANLP project.

## Project Overview

The Narrative Critic is a DeBERTa-based model that evaluates narrative quality. It's trained on:
- **ROCStories** (training data): 5-sentence stories for positive examples
- **Story Cloze** (validation/test): Stories with correct/incorrect endings for evaluation

## Files in This Project

- `model_critic.py` - Main training script (adapted from run_glue.py)
- `train_narrative_critic.py` - Easy training launcher
- `test_narrative_critic.py` - Inference and testing script
- `training_config.json` - Configuration file with parameters
- `requirements.txt` - Python dependencies
- Data files: ROCStories and Story Cloze CSV files

## Step-by-Step Execution

### 1. Install Dependencies

First, install the required Python packages:

```powershell
C:/Users/sriva/Desktop/myenv/anlpproject/.venv/Scripts/python.exe -m pip install -r requirements.txt
```

This will install:
- PyTorch
- Transformers
- Datasets
- Accelerate
- Evaluate
- Other necessary packages

### 2. Quick Test Run (Recommended First)

Before running the full training, do a quick test to ensure everything works:

```powershell
C:/Users/sriva/Desktop/myenv/anlpproject/.venv/Scripts/python.exe train_narrative_critic.py --quick_test
```

This will:
- Train on 100 samples for 1 epoch
- Use smaller batch sizes
- Complete in ~5-10 minutes
- Verify your setup works correctly

### 3. Full Training

Once the quick test passes, run the full training:

```powershell
C:/Users/sriva/Desktop/myenv/anlpproject/.venv/Scripts/python.exe train_narrative_critic.py
```

Training parameters:
- Model: microsoft/deberta-v3-base
- Epochs: 3
- Batch size: 8 (training), 16 (evaluation)
- Learning rate: 2e-5
- Max sequence length: 512
- Output: `./narrative_critic_model/`

Expected training time: 2-4 hours (depending on hardware)

### 4. Alternative: JSON Configuration

You can also train using the JSON config file:

```powershell
C:/Users/sriva/Desktop/myenv/anlpproject/.venv/Scripts/python.exe model_critic.py training_config.json
```

### 5. Monitor Training

Training logs will be saved in `./logs/`. You can monitor progress with TensorBoard:

```powershell
C:/Users/sriva/Desktop/myenv/anlpproject/.venv/Scripts/python.exe -m tensorboard.main --logdir ./logs
```

### 6. Test the Trained Model

After training completes, test your model:

```powershell
# Interactive testing
C:/Users/sriva/Desktop/myenv/anlpproject/.venv/Scripts/python.exe test_narrative_critic.py --interactive

# Single text evaluation
C:/Users/sriva/Desktop/myenv/anlpproject/.venv/Scripts/python.exe test_narrative_critic.py --text "Your story text here"

# Compare story endings
C:/Users/sriva/Desktop/myenv/anlpproject/.venv/Scripts/python.exe test_narrative_critic.py --context "Story context" --ending1 "First ending" --ending2 "Second ending"

# Run demo examples
C:/Users/sriva/Desktop/myenv/anlpproject/.venv/Scripts/python.exe test_narrative_critic.py
```

## Expected Results

The model should achieve:
- **Accuracy**: ~75-85% on Story Cloze task
- **F1 Score**: ~0.75-0.85
- **Training Loss**: Should decrease to ~0.3-0.5
- **Validation Loss**: Should track training loss

## Model Output Structure

The trained model will be saved in `./narrative_critic_model/` with:
- `config.json` - Model configuration
- `pytorch_model.bin` - Model weights
- `tokenizer.json` - Tokenizer configuration  
- `training_args.bin` - Training arguments
- Evaluation results and metrics

## Usage in RL Pipeline

The trained model includes a `NarrativeCriticForRL` class for integration with your RL pipeline:

```python
from model_critic import NarrativeCriticForRL

# Initialize critic
critic = NarrativeCriticForRL("./narrative_critic_model/")

# Score narrative quality (0-1 scale)
score = critic.score_narrative(context="Story so far", continuation="New sentence")

# Compute RL reward (-1 to 1 scale)
reward = critic.compute_reward(context="Story so far", response="Generated text")
```

## Troubleshooting

### Common Issues:

1. **CUDA Out of Memory**: Reduce batch size in training_config.json
2. **Import Errors**: Make sure you're using the virtual environment
3. **File Not Found**: Ensure CSV files are in the project directory
4. **Slow Training**: Enable fp16 and gradient_checkpointing in config

### Hardware Requirements:

- **Minimum**: 8GB RAM, CPU training (~4 hours)
- **Recommended**: 16GB RAM, GPU with 8GB+ VRAM (~1 hour)
- **Storage**: ~2GB for model and cache files

## Next Steps

After training the Narrative Critic:

1. **Evaluate Performance**: Use the test script to validate quality
2. **Integration**: Import the critic into your RL training pipeline  
3. **Fine-tuning**: Adjust hyperparameters if needed
4. **Scaling**: Train on more data or larger models for better performance

## Configuration Options

Key parameters you can adjust in `training_config.json`:

- `num_train_epochs`: Number of training epochs (3-5 recommended)
- `learning_rate`: Learning rate (1e-5 to 5e-5)
- `per_device_train_batch_size`: Batch size (4-16 depending on GPU)
- `max_seq_length`: Maximum sequence length (256-512)
- `negative_sampling_ratio`: Ratio of negative examples (0.5-2.0)

Good luck with your training!