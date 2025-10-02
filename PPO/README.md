# PPO: Multi-Critic Reinforcement Learning

This module implements the core Multi-Critic Reinforcement Learning training pipeline using Proximal Policy Optimization (PPO) with dynamic reward weighting based on player intent classification.

## Purpose

Trains the Director LLM using multi-objective reinforcement learning that dynamically adjusts reward weights based on player action type. Combines narrative quality and causal consistency evaluation to produce contextually appropriate and engaging D&D responses.

## Files Overview

### Core Files

#### `ppo.py`

Main PPO training orchestrator implementing the complete multi-critic RL pipeline.

**Main Class: `PPOTrainingOrchestrator`**

- Orchestrates complete multi-critic RL training workflow
- Manages dynamic reward weighting based on player intent
- Provides comprehensive logging, checkpointing, and error handling
- Integrates all system components (policy model, critics, hybrid player)

**Key Methods:**

- `__init__()`: Initializes training with configuration and component loading
- `load_all_components()`: Loads policy model, critics, and hybrid player
- `setup_ppo_trainer()`: Configures PPO trainer with hyperparameters
- `dynamic_reward_weighting()`: Calculates intent-aware reward weights
- `compute_multi_critic_rewards()`: Combines narrative and causal rewards
- `train_step()`: Executes single PPO training iteration
- `train()`: Main training loop with evaluation and checkpointing
- `evaluate_model()`: Periodic evaluation during training
- `save_checkpoint()`: Saves training state and model weights
- `load_checkpoint()`: Resumes training from saved checkpoint

**Training Pipeline:**

1. Load all trained components (SFT baseline, critics, hybrid player)
2. Setup PPO trainer with multi-critic configuration
3. Generate prompts using hybrid player simulation
4. Generate responses using current policy model
5. Classify player intent for dynamic weighting
6. Compute multi-critic rewards (narrative + causal)
7. Apply dynamic reward weighting based on intent
8. Update policy using PPO algorithm
9. Log metrics and save checkpoints
10. Periodic evaluation and early stopping

**Dynamic Reward Weighting:**

```python
# Intent-based weight adjustment
EXPLORE: narrative=0.8, causal=0.2    # Prioritize descriptive quality
ACTION:  narrative=0.3, causal=0.7    # Prioritize logical consistency
DIALOGUE: narrative=0.6, causal=0.4   # Balance both aspects
```

#### `critics.py`

Unified reward engine providing efficient interfaces for critic models during PPO training.

**Classes:**

**`NarrativeCritic`**

- Trained DeBERTa-based regression model for narrative quality assessment
- Provides continuous reward scores (0.0 to 1.0)
- Uses sigmoid activation for bounded rewards
- Optimized for batch processing during RL training

**`CausalCritic`**

- NLI-based causal consistency evaluator using DeBERTa
- Evaluates logical coherence between player actions and DM responses
- Three-way classification: contradiction, neutral, entailment
- Converts NLI probabilities to reward scores

**Key Methods (Both Critics):**

- `get_reward(texts: List[str]) -> torch.Tensor`: Batch reward computation
- Efficient tokenization with padding and truncation
- No-gradient contexts for inference-only evaluation
- Device-aware processing for GPU acceleration

**Design Principles:**

- **Encapsulation**: Self-contained classes with internal model management
- **Efficiency**: Models loaded once, kept in evaluation mode
- **Batching**: All methods handle lists for efficient processing
- **Clear Rewards**: Tailored extraction for each critic's training approach

#### `ppo_config.yaml`

Comprehensive configuration for PPO training and dynamic reward weighting.

**Key Sections:**

- `project_name`: Experiment identification and W&B tracking
- `model_paths`: Paths to all trained components
- `ppo_hyperparameters`: PPO algorithm configuration
- `training_settings`: Training schedule and optimization
- `reward_weights`: Static reward weights (fallback)
- `dynamic_reward_weights`: Intent-based weight mapping
- `generation_settings`: Text generation parameters

**Critical Parameters:**

```yaml
ppo_hyperparameters:
  learning_rate: 1.41e-6 # Conservative RL learning rate
  batch_size: 16 # RL interaction batch size
  mini_batch_size: 4 # PPO update batch size
  ppo_epochs: 4 # PPO optimization epochs

training_settings:
  use_dynamic_weighting: true # Enable intent-aware weighting
  total_ppo_steps: 1000 # Total training steps
  max_reward_clip: 10.0 # Prevent extreme rewards

dynamic_reward_weights:
  EXPLORE: { narrative: 0.8, causal: 0.2 }
  ACTION: { narrative: 0.3, causal: 0.7 }
  DIALOGUE: { narrative: 0.6, causal: 0.4 }
```

## Technical Implementation

### Multi-Critic Architecture

- **Policy Model**: SFT baseline with LoRA adapters from `../DM-SFT/`
- **Narrative Critic**: DeBERTa regression model from `../narrative critic/`
- **Causal Critic**: DeBERTa NLI model from `../causal critic/`
- **Hybrid Player**: Intent classifier + generator from `../hybrid_player/`

### Dynamic Reward System

- **Intent Classification**: Classifies player prompts into EXPLORE/ACTION/DIALOGUE
- **Weight Adjustment**: Applies intent-specific critic weights
- **Reward Computation**: Combines weighted narrative and causal scores
- **Clipping**: Prevents extreme rewards from destabilizing training

### Training Optimization

- **Memory Management**: Efficient GPU memory usage across multiple models
- **Gradient Control**: PPO-specific gradient clipping and normalization
- **Checkpoint Recovery**: Robust training continuation from interruptions
- **Early Stopping**: Monitors reward trends for convergence detection

## Usage

### Prerequisites

```bash
pip install torch transformers peft trl
pip install wandb numpy matplotlib seaborn
pip install accelerate datasets tqdm
```

### Configuration

1. Edit `ppo_config.yaml` with paths to trained components:

   - Policy model from DM-SFT training
   - Narrative critic from narrative critic training
   - Causal critic from causal critic training
   - Hybrid player from hybrid player training

2. Adjust PPO hyperparameters for your hardware setup
3. Configure dynamic reward weights for your use case

### Training

```bash
# Run PPO training with default config
python ppo.py

# Or with custom configuration
python ppo.py --config custom_ppo_config.yaml
```

### Monitoring

- **W&B Dashboard**: Real-time training metrics and reward trends
- **Local Logs**: Detailed console output with step-by-step progress
- **Checkpoints**: Automatic saving every N steps for recovery

## Training Outputs

### Model Artifacts

```
models/director_ppo_final/
├── pytorch_model.bin        # Trained policy model weights
├── config.json             # Model configuration
├── tokenizer.json          # Tokenizer files
└── training_state.json     # Training state for resumption
```

### Training Logs

- **Reward Tracking**: Narrative, causal, and combined reward trends
- **Intent Distribution**: Frequency of different player intent types
- **Policy Metrics**: KL divergence, policy loss, value loss
- **Generation Quality**: Response length, diversity metrics

### Checkpoints

- **Model Weights**: Policy model state at checkpoint intervals
- **Training State**: Optimizer state, step counter, random seeds
- **Configuration**: Complete config snapshot for reproducibility

## Integration

### Input Dependencies

- **Policy Model**: Trained SFT baseline from `../DM-SFT/models/sft_baseline_interim/`
- **Narrative Critic**: Trained model from `../narrative critic/models/narrative_critic/`
- **Causal Critic**: Trained model from `../causal critic/models/causal_critic_finetuned/`
- **Hybrid Player**: Trained components from `../hybrid_player/models/`

### Output Usage

- **Final Model**: Used in `../Evaluation/` for comprehensive assessment
- **Checkpoints**: Enable training resumption and ablation studies
- **Logs**: Provide training analysis and hyperparameter optimization insights

## Performance Characteristics

### Training Metrics

- **Memory Usage**: ~10GB per GPU (policy + critics + hybrid player)
- **Training Duration**: Depends on configuration and convergence criteria
- **Stability**: Requires careful hyperparameter tuning for stable training

## Advanced Features

### Error Handling

- **Graceful Degradation**: Continues training despite minor errors
- **Automatic Recovery**: Resumes from last checkpoint on crash
- **Resource Monitoring**: Tracks GPU memory and prevents OOM
- **Signal Handling**: Clean shutdown on interruption

### Experimental Features

- **Reward Scheduling**: Dynamic adjustment of reward weights over training
- **Multi-Objective Tracking**: Separate monitoring of each critic
- **Ablation Support**: Easy disabling of components for analysis
- **Custom Reward Functions**: Extensible framework for new critics

## Troubleshooting

### Common Issues

- **OOM Errors**: Reduce batch sizes or enable gradient checkpointing
- **Reward Instability**: Adjust reward clipping or learning rate
- **Poor Convergence**: Check critic quality or adjust weights
- **Component Loading**: Verify all model paths in configuration

### Debugging Tools

- **Verbose Logging**: Detailed step-by-step execution logs
- **Reward Analysis**: Individual critic score tracking
- **Intent Distribution**: Monitor player intent classification balance
- **Generation Samples**: Periodic example outputs for quality assessment
