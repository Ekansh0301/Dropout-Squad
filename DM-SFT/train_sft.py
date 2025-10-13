"""
DM-SFT Training Script for Llama-2-7B with QLoRA optimization.
Trains Director LLM baseline on D&D dialogue data.

"""

import os
import sys
import torch
import yaml
import json
from pathlib import Path
from datetime import datetime
from datasets import load_from_disk
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    set_seed
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training
)

# Optional W&B integration for experiment tracking
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("⚠️  W&B not installed. Install with: pip install wandb")

def load_config(config_path="sft_config_fast.yaml"):
    """Load training configuration from YAML file."""
    print(f"\nLoading config from: {config_path}")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def print_config_summary(config):
    """Display key training configuration parameters."""
    print("\n" + "="*70)
    print("CONFIGURATION SUMMARY")
    print("="*70)
    print(f"\nProject: {config['project']['name']}")
    print(f"Description: {config['project']['description']}")
    print(f"\nModel: {config['model']['name']}")
    print(f"Data: {config['data']['max_train_samples']:,} training examples")
    print(f"Epochs: {config['training']['num_train_epochs']}")
    print(f"Sequence length: {config['data']['max_seq_length']} tokens")
    print(f"LoRA rank: {config['lora']['r']}")
    print(f"Effective batch size: {config['training']['per_device_train_batch_size'] * config['training']['gradient_accumulation_steps'] * torch.cuda.device_count()}")
    print(f"\nOutput: {config['training']['output_dir']}")
    print("="*70)

def check_gpu_availability():
    """Verify GPU availability and display hardware information."""
    print("\n" + "="*70)
    print("GPU INFORMATION")
    print("="*70)
    
    if not torch.cuda.is_available():
        print("\n❌ ERROR: No CUDA-capable GPU detected!")
        print("This script requires GPU. Exiting.")
        sys.exit(1)
    
    gpu_count = torch.cuda.device_count()
    print(f"\n✓ Found {gpu_count} GPU(s)")
    
    for i in range(gpu_count):
        props = torch.cuda.get_device_properties(i)
        memory_gb = props.total_memory / (1024**3)
        print(f"\nGPU {i}: {props.name}")
        print(f"  Memory: {memory_gb:.1f} GB")
        print(f"  Compute Capability: {props.major}.{props.minor}")
    
    print("="*70)
    
    return gpu_count

def setup_wandb(config):
    """Initialize Weights & Biases experiment tracking."""
    if not WANDB_AVAILABLE:
        print("\n⚠️  Skipping W&B (not installed)")
        return False
    
    if config['training']['report_to'] == "wandb":
        try:
            wandb.init(
                project=config['project']['name'],
                name=config['training']['run_name'],
                config={
                    'model': config['model']['name'],
                    'train_samples': config['data']['max_train_samples'],
                    'epochs': config['training']['num_train_epochs'],
                    'seq_length': config['data']['max_seq_length'],
                    'lora_r': config['lora']['r'],
                    'learning_rate': config['training']['learning_rate'],
                }
            )
            print("\n✓ W&B tracking initialized")
            return True
        except Exception as e:
            print(f"\n⚠️  W&B initialization failed: {e}")
            return False
    return False

def load_model_and_tokenizer(config):
    """Load Llama model with 4-bit quantization and tokenizer."""
    print("\n" + "="*70)
    print("LOADING MODEL & TOKENIZER")
    print("="*70)
    
    model_name = config['model']['name']
    print(f"\nModel: {model_name}")
    print("Quantization: 4-bit NF4")
    
    # Configure 4-bit quantization for memory efficiency
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=config['model']['load_in_4bit'],
        bnb_4bit_quant_type=config['model']['bnb_4bit_quant_type'],
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=config['model']['bnb_4bit_use_double_quant'],
    )
    
    # Device mapping for multi-GPU setup
    device_map = {"": int(os.environ.get("LOCAL_RANK", 0))}
    
    # Load base model with quantization
    print("\nLoading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map=device_map,
        trust_remote_code=True,
        torch_dtype=torch.float16,
    )
    
    # Load tokenizer with appropriate settings
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        padding_side="right"
    )
    
    # Configure padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    if hasattr(model.config, 'pad_token_id') and model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.pad_token_id
    
    # Prepare model for quantized training
    model = prepare_model_for_kbit_training(model)
    
    print("✓ Model and tokenizer loaded successfully")
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {total_params:,}")
    
    return model, tokenizer

def setup_lora(model, config):
    """Configure and apply LoRA adapters for parameter-efficient training."""
    print("\n" + "="*70)
    print("CONFIGURING LORA")
    print("="*70)
    
    lora_config = LoraConfig(
        r=config['lora']['r'],
        lora_alpha=config['lora']['lora_alpha'],
        lora_dropout=config['lora']['lora_dropout'],
        target_modules=config['lora']['target_modules'],
        bias=config['lora']['bias'],
        task_type=config['lora']['task_type'],
    )
    
    print(f"\nLoRA Configuration:")
    print(f"  Rank (r): {lora_config.r}")
    print(f"  Alpha: {lora_config.lora_alpha}")
    print(f"  Dropout: {lora_config.lora_dropout}")
    print(f"  Target modules: {lora_config.target_modules}")
    
    # Apply LoRA to model
    model = get_peft_model(model, lora_config)
    
    # Display parameter efficiency statistics
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_pct = 100 * trainable_params / total_params
    
    print(f"\nParameter efficiency:")
    print(f"  Trainable: {trainable_params:,} ({trainable_pct:.2f}%)")
    print(f"  Total: {total_params:,}")
    print(f"  Frozen: {total_params - trainable_params:,}")
    print("\n✓ LoRA applied successfully")
    
    return model

def load_and_prepare_datasets(config, tokenizer):
    """Load, subsample, and tokenize datasets with progress tracking."""
    print("\n" + "="*70)
    print("LOADING & PREPARING DATASETS")
    print("="*70)
    
    from tqdm import tqdm
    
    # Load datasets from disk
    print(f"\nLoading from disk:")
    print(f"  Train: {config['data']['train_path']}")
    print(f"  Val: {config['data']['val_path']}")
    
    train_dataset = load_from_disk(config['data']['train_path'])
    eval_dataset = load_from_disk(config['data']['val_path'])
    
    print(f"\nOriginal dataset sizes:")
    print(f"  Train: {len(train_dataset):,} examples")
    print(f"  Val: {len(eval_dataset):,} examples")
    
    # Validate dataset format
    print(f"\nDataset columns: {train_dataset.column_names}")
    if 'text' not in train_dataset.column_names:
        raise ValueError(f"Dataset must have 'text' column. Found: {train_dataset.column_names}")
    
    # Subsample for faster training iteration
    if 'max_train_samples' in config['data']:
        print(f"\n⚡ OPTIMIZATION: Using subset for interim submission")
        requested_samples = config['data']['max_train_samples']
        actual_samples = min(requested_samples, len(train_dataset))
        
        print(f"  Shuffling and selecting {actual_samples:,} examples...")
        train_dataset = train_dataset.shuffle(seed=config['seed']).select(
            range(actual_samples)
        )
        print(f"  ✓ Train subset: {len(train_dataset):,} examples ({len(train_dataset)/263287*100:.1f}% of full)")
    
    if 'max_eval_samples' in config['data']:
        requested_samples = config['data']['max_eval_samples']
        actual_samples = min(requested_samples, len(eval_dataset))
        eval_dataset = eval_dataset.shuffle(seed=config['seed']).select(
            range(actual_samples)
        )
        print(f"  ✓ Val subset: {len(eval_dataset):,} examples")
    
    # Display sample data before processing
    print(f"\nSample training example (before tokenization):")
    print(f"  Length: {len(train_dataset[0]['text'])} characters")
    print(f"  Preview: {train_dataset[0]['text'][:150]}...")
    
    # Tokenization function for language modeling
    def tokenize_function(examples):
        outputs = tokenizer(
            examples['text'],
            truncation=True,
            max_length=config['data']['max_seq_length'],
            padding=False,
            return_tensors=None,
        )
        outputs["labels"] = outputs["input_ids"].copy()
        return outputs
    
    # Apply tokenization with parallel processing
    print(f"\nTokenizing datasets (max length: {config['data']['max_seq_length']} tokens)...")
    
    print("\n[1/2] Tokenizing training set...")
    train_dataset = train_dataset.map(
        tokenize_function,
        batched=True,
        batch_size=1000,
        remove_columns=train_dataset.column_names,
        desc="Tokenizing train",
        num_proc=4,
        load_from_cache_file=True
    )
    print(f"  ✓ Training set tokenized: {len(train_dataset):,} examples")
    
    print("\n[2/2] Tokenizing validation set...")
    eval_dataset = eval_dataset.map(
        tokenize_function,
        batched=True,
        batch_size=1000,
        remove_columns=eval_dataset.column_names,
        desc="Tokenizing validation",
        num_proc=4,
        load_from_cache_file=True
    )
    print(f"  ✓ Validation set tokenized: {len(eval_dataset):,} examples")
    
    # Verify tokenization results
    print(f"\nTokenization verification:")
    sample_tokens = train_dataset[0]['input_ids']
    print(f"  Sample length: {len(sample_tokens)} tokens")
    print(f"  Sample tokens (first 10): {sample_tokens[:10]}")
    print(f"  Decoded sample: {tokenizer.decode(sample_tokens[:50])}...")
    
    # Calculate token statistics
    print(f"\nDataset statistics:")
    token_lengths = [len(example['input_ids']) for example in train_dataset.select(range(min(1000, len(train_dataset))))]
    avg_length = sum(token_lengths) / len(token_lengths)
    print(f"  Average token length (first 1000 examples): {avg_length:.1f}")
    print(f"  Min length: {min(token_lengths)}")
    print(f"  Max length: {max(token_lengths)}")
    
    print("\n✓ Datasets prepared successfully")
    
    return train_dataset, eval_dataset

def calculate_training_time(train_dataset, config, gpu_count):
    """Estimate total training time based on dataset size and hardware."""
    num_examples = len(train_dataset)
    epochs = config['training']['num_train_epochs']
    batch_size = config['training']['per_device_train_batch_size']
    grad_accum = config['training']['gradient_accumulation_steps']
    
    effective_batch = batch_size * grad_accum * gpu_count
    steps_per_epoch = num_examples // effective_batch
    total_steps = steps_per_epoch * epochs
    
    # Estimated processing time per step on 4x GTX 1080 Ti
    seconds_per_step = 1.5
    total_seconds = total_steps * seconds_per_step
    hours = total_seconds / 3600
    
    return {
        'total_steps': total_steps,
        'steps_per_epoch': steps_per_epoch,
        'estimated_hours': hours
    }

def train_model(model, tokenizer, train_dataset, eval_dataset, config):
    """Configure training parameters and execute model training."""
    print("\n" + "="*70)
    print("TRAINING CONFIGURATION")
    print("="*70)
    
    # Calculate training metrics
    gpu_count = torch.cuda.device_count()
    training_info = calculate_training_time(train_dataset, config, gpu_count)
    
    print(f"\nTraining details:")
    print(f"  Total steps: {training_info['total_steps']:,}")
    print(f"  Steps per epoch: {training_info['steps_per_epoch']:,}")
    print(f"  Estimated time: {training_info['estimated_hours']:.1f} hours")
    print(f"  Effective batch size: {config['training']['per_device_train_batch_size'] * config['training']['gradient_accumulation_steps'] * gpu_count}")
    
    # Prepare output directory
    output_dir = Path(config['training']['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save configuration for reproducibility
    config_save_path = output_dir / "training_config.yaml"
    with open(config_save_path, 'w') as f:
        yaml.dump(config, f)
    print(f"\n✓ Config saved to: {config_save_path}")
    
    # Configure training arguments
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=config['training']['num_train_epochs'],
        per_device_train_batch_size=config['training']['per_device_train_batch_size'],
        per_device_eval_batch_size=config['training']['per_device_eval_batch_size'],
        gradient_accumulation_steps=config['training']['gradient_accumulation_steps'],
        learning_rate=config['training']['learning_rate'],
        lr_scheduler_type=config['training']['lr_scheduler_type'],
        warmup_ratio=config['training']['warmup_ratio'],
        warmup_steps=config['training']['warmup_steps'],
        weight_decay=config['training']['weight_decay'],
        max_grad_norm=config['training']['max_grad_norm'],
        logging_steps=config['training']['logging_steps'],
        logging_first_step=config['training']['logging_first_step'],
        save_strategy=config['training']['save_strategy'],
        save_steps=config['training']['save_steps'],
        save_total_limit=config['training']['save_total_limit'],
        eval_strategy=config['training']['eval_strategy'],
        eval_steps=config['training']['eval_steps'],
        load_best_model_at_end=config['training']['load_best_model_at_end'],
        metric_for_best_model=config['training']['metric_for_best_model'],
        fp16=config['training']['fp16'],
        bf16=config['training']['bf16'],
        optim=config['training']['optim'],
        gradient_checkpointing=config['training']['gradient_checkpointing'],
        dataloader_num_workers=config['training']['dataloader_num_workers'],
        dataloader_pin_memory=config['training']['dataloader_pin_memory'],
        ddp_find_unused_parameters=config['training']['ddp_find_unused_parameters'],
        report_to=config['training']['report_to'],
        run_name=config['training']['run_name'],
        seed=config['seed'],
        remove_unused_columns=False,
    )
    
    # Set up data collator for language modeling
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )
    
    # Begin training process
    print("\n" + "="*70)
    print("🚀 STARTING TRAINING")
    print("="*70)
    print(f"\nStart time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\nMonitor progress:")
    print("  - Watch terminal output for loss/metrics")
    print("  - Check W&B dashboard if enabled")
    print("  - Run 'nvidia-smi' in another terminal to monitor GPU")
    print("\n" + "="*70 + "\n")
    
    # Execute training
    train_result = trainer.train()
    
    # Training completion summary
    print("\n" + "="*70)
    print("✓ TRAINING COMPLETE")
    print("="*70)
    print(f"\nEnd time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total training time: {train_result.metrics['train_runtime'] / 3600:.2f} hours")
    print(f"Final loss: {train_result.metrics['train_loss']:.4f}")
    
    # Save trained model and tokenizer
    print("\nSaving final model...")
    trainer.save_model()
    tokenizer.save_pretrained(config['training']['output_dir'])
    
    # Save training metrics
    metrics_path = output_dir / "training_metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(train_result.metrics, f, indent=2)
    print(f"✓ Metrics saved to: {metrics_path}")
    
    # Perform final evaluation
    print("\nRunning final evaluation...")
    eval_results = trainer.evaluate()
    
    print(f"\nFinal evaluation metrics:")
    print(f"  Eval loss: {eval_results['eval_loss']:.4f}")
    print(f"  Perplexity: {torch.exp(torch.tensor(eval_results['eval_loss'])):.2f}")
    
    # Save evaluation metrics
    eval_metrics_path = output_dir / "eval_metrics.json"
    with open(eval_metrics_path, 'w') as f:
        json.dump(eval_results, f, indent=2)
    
    print(f"\n✓ Model saved to: {config['training']['output_dir']}")
    
    return trainer, train_result, eval_results

def main():
    """Execute complete training pipeline."""
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
    
    print("\n" + "="*70)
    print("DM-SFT TRAINING - INTERIM SUBMISSION MODE")
    print("="*70)
    print("\nOPTIMIZATIONS APPLIED FOR 3-DAY WINDOW:")
    print("  - Using 10% of training data (26K examples)")
    print("  - Training for 1 epoch instead of 3")
    print("  - Shorter sequences (256 tokens)")
    print("  - Less frequent checkpointing")
    print("\nFULL VERSION (documented for report):")
    print("  - 100% training data (263K examples)")
    print("  - 3 epochs")
    print("  - 512 token sequences")
    print("  - Would require ~10-12 hours on 4x GTX 1080 Ti")
    print("="*70)
    
    # Load and validate configuration
    config = load_config()
    print_config_summary(config)
    
    # Verify hardware requirements
    gpu_count = check_gpu_availability()
    if gpu_count != 4:
        print(f"\n⚠️  WARNING: Expected 4 GPUs, found {gpu_count}")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            print("Exiting.")
            return
    
    # Set reproducible random seed
    set_seed(config['seed'])
    print(f"\n✓ Random seed set to: {config['seed']}")
    
    # Initialize experiment tracking
    setup_wandb(config)
    
    # Load model with quantization and tokenizer
    model, tokenizer = load_model_and_tokenizer(config)
    
    # Apply LoRA for efficient training
    model = setup_lora(model, config)
    
    # Prepare training and validation datasets
    train_dataset, eval_dataset = load_and_prepare_datasets(config, tokenizer)
    
    # Execute training process
    trainer, train_result, eval_results = train_model(
        model, tokenizer, train_dataset, eval_dataset, config
    )
    
    # Display completion summary
    print("\n✓ Training pipeline complete!")
    print("="*70 + "\n")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user")
        print("Model checkpoints are saved in the output directory.")
    except Exception as e:
        print(f"\n\n❌ ERROR: Training failed")
        print(f"Error details: {e}")
        import traceback
        traceback.print_exc()