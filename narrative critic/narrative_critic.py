"""
Narrative critic training with comprehensive metrics and multi-GPU optimization.
Trains DeBERTa model for narrative quality assessment with detailed instrumentation.
"""
import torch
import yaml
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    set_seed,
    TrainerCallback
)
from datasets import load_from_disk
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Custom training callback for detailed metrics logging
class DetailedLoggingCallback(TrainerCallback):
    """Callback to capture and save comprehensive training history."""
    
    def __init__(self, output_dir):
        self.output_dir = Path(output_dir)
        self.training_history = []
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        """Record each training log entry with step and epoch information."""
        if logs:
            log_entry = {
                'step': state.global_step,
                'epoch': state.epoch,
                **logs
            }
            self.training_history.append(log_entry)
    
    def on_train_end(self, args, state, control, **kwargs):
        """Save complete training history to disk."""
        history_file = self.output_dir / "training_history.json"
        with open(history_file, 'w') as f:
            json.dump(self.training_history, f, indent=2)
        print(f"\n✓ Training history saved: {history_file}")

def compute_metrics(eval_pred):
    """Calculate comprehensive evaluation metrics for narrative quality."""
    predictions, labels = eval_pred
    predictions = predictions.squeeze()
    
    # Apply sigmoid activation for 0-1 probability scores
    predictions_sigmoid = 1 / (1 + np.exp(-predictions))
    
    # Regression metrics
    mse = mean_squared_error(labels, predictions_sigmoid)
    mae = mean_absolute_error(labels, predictions_sigmoid)
    rmse = np.sqrt(mse)
    r2 = r2_score(labels, predictions_sigmoid)
    
    # Correlation analysis
    correlation = np.corrcoef(labels, predictions_sigmoid)[0, 1]
    
    # Classification-style accuracy (within tolerance)
    within_threshold = np.mean(np.abs(predictions_sigmoid - labels) < 0.2)
    
    return {
        'mse': mse,
        'mae': mae,
        'rmse': rmse,
        'r2_score': r2,
        'correlation': correlation,
        'accuracy_0.2': within_threshold
    }

def load_config(config_path="critic_config_fast.yaml"):
    """Load training configuration from YAML file."""
    print(f"\nLoading config: {config_path}")
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    """Execute complete narrative critic training pipeline."""
    print("\n" + "="*70)
    print("NARRATIVE CRITIC TRAINING - FULL INSTRUMENTATION")
    print("="*70)
    print("\nOptimizations applied:")
    print("  - 4 GPUs (parallelized)")
    print("  - Batch size 32 per GPU (128 effective)")
    print("  - Sequence length 128 (2x faster than 256)")
    print("  - 2 epochs (reduced from 3)")
    print("\nExpected time: 1-1.5 hours")
    print("="*70)
    
    # Load configuration and set reproducible seed
    config = load_config()
    set_seed(config['seed'])
    
    # Display GPU information
    gpu_count = torch.cuda.device_count()
    print(f"\n✓ GPUs detected: {gpu_count}")
    for i in range(gpu_count):
        print(f"  GPU {i}: {torch.cuda.get_device_properties(i).name}")
    
    # Setup output directory
    output_dir = Path(config['training']['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save configuration for reproducibility
    config_save = output_dir / "critic_config.yaml"
    with open(config_save, 'w') as f:
        yaml.dump(config, f)
    print(f"\n✓ Config saved: {config_save}")
    
    # Load pre-trained model and tokenizer
    print("\n" + "="*70)
    print("[1/4] LOADING MODEL")
    print("="*70)
    
    model = AutoModelForSequenceClassification.from_pretrained(
        config['model']['name'],
        num_labels=config['model']['num_labels'],
        problem_type=config['model']['problem_type']
    )
    
    tokenizer = AutoTokenizer.from_pretrained(config['model']['name'])
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n✓ Model: {config['model']['name']}")
    print(f"  Parameters: {total_params:,}")
    
    # Load training and validation datasets
    print("\n" + "="*70)
    print("[2/4] LOADING DATA")
    print("="*70)
    
    print(f"\nPaths:")
    print(f"  Train: {config['data']['train_path']}")
    print(f"  Val: {config['data']['val_path']}")
    
    train_dataset = load_from_disk(config['data']['train_path'])
    eval_dataset = load_from_disk(config['data']['val_path'])
    
    print(f"\n✓ Loaded:")
    print(f"  Train: {len(train_dataset):,} examples")
    print(f"  Val: {len(eval_dataset):,} examples")
    
    # Display sample data
    print(f"\nSample example:")
    print(f"  Text: {train_dataset[0]['text'][:100]}...")
    print(f"  Label: {train_dataset[0]['label_float']}")
    
    # Tokenize datasets
    print("\n" + "="*70)
    print("[3/4] TOKENIZING")
    print("="*70)
    
    def tokenize_function(examples):
        """Tokenize text examples with truncation and padding."""
        return tokenizer(
            examples['text'],
            truncation=True,
            max_length=config['data']['max_seq_length'],
            padding=False
        )
    
    print(f"\nMax sequence length: {config['data']['max_seq_length']}")
    
    train_dataset = train_dataset.map(
        tokenize_function,
        batched=True,
        batch_size=1000,
        num_proc=4,
        remove_columns=['text', 'label', 'source', 'type'],
        desc="Train"
    )
    
    eval_dataset = eval_dataset.map(
        tokenize_function,
        batched=True,
        batch_size=1000,
        num_proc=4,
        remove_columns=['text', 'label', 'source', 'type'],
        desc="Val"
    )
    
    train_dataset = train_dataset.rename_column('label_float', 'labels')
    eval_dataset = eval_dataset.rename_column('label_float', 'labels')
    
    print("\n✓ Tokenization complete")
    
    # Configure and execute training
    print("\n" + "="*70)
    print("[4/4] TRAINING")
    print("="*70)
    
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=config['training']['num_train_epochs'],
        per_device_train_batch_size=config['training']['per_device_train_batch_size'],
        per_device_eval_batch_size=config['training']['per_device_eval_batch_size'],
        gradient_accumulation_steps=config['training']['gradient_accumulation_steps'],
        learning_rate=float(config['training']['learning_rate']),
        lr_scheduler_type=config['training']['lr_scheduler_type'],
        warmup_ratio=config['training']['warmup_ratio'],
        weight_decay=config['training']['weight_decay'],
        max_grad_norm=config['training']['max_grad_norm'],
        logging_steps=config['training']['logging_steps'],
        eval_strategy=config['training']['eval_strategy'],
        eval_steps=config['training']['eval_steps'],
        save_strategy=config['training']['save_strategy'],
        save_steps=config['training']['save_steps'],
        save_total_limit=config['training']['save_total_limit'],
        load_best_model_at_end=config['training']['load_best_model_at_end'],
        metric_for_best_model=config['training']['metric_for_best_model'],
        fp16=config['training']['fp16'],
        dataloader_num_workers=config['training']['dataloader_num_workers'],
        dataloader_pin_memory=config['training']['dataloader_pin_memory'],
        report_to=config['training']['report_to'],
        seed=config['seed'],
    )
    
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    # Add detailed logging callback
    logging_callback = DetailedLoggingCallback(output_dir)
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[logging_callback]
    )
    
    # Calculate training info
    effective_batch = config['training']['per_device_train_batch_size'] * gpu_count
    steps_per_epoch = len(train_dataset) // effective_batch
    total_steps = steps_per_epoch * config['training']['num_train_epochs']
    
    print(f"\nTraining configuration:")
    print(f"  Epochs: {config['training']['num_train_epochs']}")
    print(f"  Batch size per GPU: {config['training']['per_device_train_batch_size']}")
    print(f"  Effective batch: {effective_batch}")
    print(f"  Steps per epoch: {steps_per_epoch}")
    print(f"  Total steps: {total_steps}")
    print(f"\nStart time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\n" + "="*70 + "\n")
    
    # Train
    train_result = trainer.train()
    
    # Training complete
    print("\n" + "="*70)
    print("✓ TRAINING COMPLETE")
    print("="*70)
    print(f"\nEnd time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Training time: {train_result.metrics['train_runtime'] / 3600:.2f} hours")
    print(f"Final loss: {train_result.metrics.get('train_loss', 'N/A')}")
    
    # Save model
    print("\nSaving model...")
    trainer.save_model()
    tokenizer.save_pretrained(str(output_dir))
    
    # Save training metrics
    metrics_file = output_dir / "training_metrics.json"
    with open(metrics_file, 'w') as f:
        json.dump(train_result.metrics, f, indent=2)
    print(f"✓ Training metrics: {metrics_file}")
    
    # Final evaluation
    print("\n" + "="*70)
    print("FINAL EVALUATION")
    print("="*70)
    
    eval_results = trainer.evaluate()
    
    print(f"\nValidation metrics:")
    for key, value in eval_results.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
    
    # Save eval metrics
    eval_file = output_dir / "eval_metrics.json"
    with open(eval_file, 'w') as f:
        json.dump(eval_results, f, indent=2)
    print(f"\n✓ Eval metrics: {eval_file}")
    
    # Create training plots
    print("\n" + "="*70)
    print("GENERATING PLOTS")
    print("="*70)
    
    create_training_plots(logging_callback.training_history, output_dir)
    
    # Test on examples
    print("\n" + "="*70)
    print("TESTING CRITIC")
    print("="*70)
    
    test_examples = [
        ("You enter a dimly lit tavern, thick with pipe smoke. A grizzled dwarf eyes you suspiciously.", "High quality"),
        ("You see a room. There are things. You can do stuff.", "Low quality"),
        ("The dragon roars and breathes fire. The dragon roars and breathes fire. The dragon roars.", "Repetitive"),
    ]
    
    print("\nSample predictions:")
    for text, description in test_examples:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128).to(model.device)
        with torch.no_grad():
            outputs = model(**inputs)
            score = torch.sigmoid(outputs.logits).item()
        
        print(f"\n  {description}:")
        print(f"  Text: {text[:60]}...")
        print(f"  Score: {score:.3f}")
    
    # Summary
    print("\n" + "="*70)
    print("CRITIC TRAINING SUMMARY")
    print("="*70)
    print(f"\nModel saved to: {output_dir}")
    print(f"\nFiles created:")
    print(f"  - pytorch_model.bin (trained weights)")
    print(f"  - critic_config.yaml (configuration)")
    print(f"  - training_metrics.json (final metrics)")
    print(f"  - eval_metrics.json (validation results)")
    print(f"  - training_history.json (step-by-step logs)")
    print(f"  - training_loss.png (loss curve)")
    print(f"  - eval_metrics_plot.png (validation metrics)")
    print("\n✓ All artifacts saved for report")
    print("="*70)

def create_training_plots(history, output_dir):
    """Generate training visualization plots"""
    
    if not history:
        print("  ⚠️  No training history to plot")
        return
    
    output_dir = Path(output_dir)
    
    # Extract data
    steps = [h['step'] for h in history if 'loss' in h]
    train_loss = [h['loss'] for h in history if 'loss' in h]
    
    eval_steps = [h['step'] for h in history if 'eval_loss' in h]
    eval_loss = [h['eval_loss'] for h in history if 'eval_loss' in h]
    eval_mae = [h.get('eval_mae', None) for h in history if 'eval_mae' in h]
    
    # Plot 1: Training loss
    if train_loss:
        plt.figure(figsize=(10, 6))
        plt.plot(steps, train_loss, label='Training Loss', linewidth=2)
        if eval_loss:
            plt.plot(eval_steps, eval_loss, label='Validation Loss', 
                    marker='o', linewidth=2)
        plt.xlabel('Training Steps')
        plt.ylabel('Loss')
        plt.title('Narrative Critic Training Progress')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / "training_loss.png", dpi=300)
        print("  ✓ training_loss.png")
    
    # Plot 2: Validation metrics over time
    if eval_mae and any(m is not None for m in eval_mae):
        plt.figure(figsize=(10, 6))
        
        valid_eval_mae = [m for m in eval_mae if m is not None]
        valid_eval_steps = eval_steps[:len(valid_eval_mae)]
        
        plt.plot(valid_eval_steps, valid_eval_mae, 
                marker='s', linewidth=2, label='MAE')
        plt.xlabel('Training Steps')
        plt.ylabel('Mean Absolute Error')
        plt.title('Critic Validation Performance')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / "eval_metrics_plot.png", dpi=300)
        print("  ✓ eval_metrics_plot.png")
    
    print("\n✓ Plots generated")

if __name__ == "__main__":
    main()