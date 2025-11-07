"""
Train 3-class causal critic: entailment, contradiction, neutral.
"""
import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)
from datasets import load_from_disk
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
import numpy as np
import json
from pathlib import Path

def compute_metrics(eval_pred):
    """Calculate comprehensive metrics for 3-class evaluation."""
    predictions, labels = eval_pred
    preds = np.argmax(predictions, axis=-1)
    
    # Overall accuracy
    accuracy = accuracy_score(labels, preds)
    
    # Per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        labels, preds, average=None, labels=[0, 1, 2], zero_division=0
    )
    
    # Macro-averaged metrics
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        labels, preds, average='macro', zero_division=0
    )
    
    return {
        'accuracy': accuracy,
        'macro_f1': macro_f1,
        'macro_precision': macro_precision,
        'macro_recall': macro_recall,
        'contradiction_f1': f1[0],
        'neutral_f1': f1[1],
        'entailment_f1': f1[2],
        'contradiction_precision': precision[0],
        'neutral_precision': precision[1],
        'entailment_precision': precision[2],
        'contradiction_recall': recall[0],
        'neutral_recall': recall[1],
        'entailment_recall': recall[2],
    }

def train_causal_critic_3class():
    """Train 3-class causal critic model on prepared dataset."""
    print("\n" + "="*60)
    print("TRAINING 3-CLASS CAUSAL CRITIC")
    print("="*60)
    
    # Check if data exists
    data_dir = Path("../data/causal_critic_training_3class")
    if not data_dir.exists():
        print(f"\n❌ Data directory not found: {data_dir}")
        print("Please run data_prep_3class.py first to prepare the data.")
        return
    
    print(f"\nLoading data from: {data_dir}")
    
    # Load pre-trained NLI model
    # Using RoBERTa instead of DeBERTa to avoid tokenizer issues
    model_name = "FacebookAI/roberta-base"  
    print(f"\nLoading model: {model_name}")
    
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=3,
        problem_type="single_label_classification"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Load prepared datasets
    print("\nLoading datasets...")
    train_dataset = load_from_disk(data_dir / "train")
    eval_dataset = load_from_disk(data_dir / "val")
    test_dataset = load_from_disk(data_dir / "test")
    
    print(f"  Train: {len(train_dataset):,} examples")
    print(f"  Val: {len(eval_dataset):,} examples")
    print(f"  Test: {len(test_dataset):,} examples")
    
    def tokenize(examples):
        """Tokenize premise-hypothesis pairs for NLI."""
        return tokenizer(
            examples['premise'], 
            examples['hypothesis'], 
            truncation=True, 
            max_length=256, 
            padding=False
        )
    
    # Tokenize datasets
    print("\nTokenizing datasets...")
    train_dataset = train_dataset.map(tokenize, batched=True, num_proc=4)
    eval_dataset = eval_dataset.map(tokenize, batched=True, num_proc=4)
    test_dataset = test_dataset.map(tokenize, batched=True, num_proc=4)
    
    # Rename label column for trainer compatibility
    train_dataset = train_dataset.rename_column('label', 'labels')
    eval_dataset = eval_dataset.rename_column('label', 'labels')
    test_dataset = test_dataset.rename_column('label', 'labels')
    
    # Configure training parameters
    output_dir = "models/causal_critic_3class"
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        learning_rate=2e-5,
        weight_decay=0.01,
        eval_strategy="steps",
        eval_steps=500,
        save_strategy="steps",
        save_steps=500,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        fp16=torch.cuda.is_available(),
        logging_steps=100,
        report_to="none",
        seed=42
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics
    )
    
    # Train model
    print("\n" + "="*60)
    print("STARTING TRAINING")
    print("="*60)
    trainer.train()
    
    # Evaluate on test set
    print("\n" + "="*60)
    print("EVALUATING ON TEST SET")
    print("="*60)
    test_results = trainer.predict(test_dataset)
    
    test_preds = np.argmax(test_results.predictions, axis=-1)
    test_labels = test_results.label_ids
    
    # Generate detailed classification report
    label_names = ['contradiction', 'neutral', 'entailment']
    report = classification_report(test_labels, test_preds, target_names=label_names, digits=4)
    
    print("\n" + "="*60)
    print("TEST SET RESULTS")
    print("="*60)
    print(report)
    
    # Save model and tokenizer
    print("\n" + "="*60)
    print("SAVING MODEL")
    print("="*60)
    model_save_path = Path("../model_causalcritic_3class")
    model_save_path.mkdir(parents=True, exist_ok=True)
    
    trainer.save_model(model_save_path)
    tokenizer.save_pretrained(model_save_path)
    
    # Save test results and summary
    test_metrics = test_results.metrics
    summary = {
        "model_name": model_name,
        "task": "3-Class Causal Consistency (NLI)",
        "dataset": {
            "train_size": len(train_dataset),
            "val_size": len(eval_dataset),
            "test_size": len(test_dataset)
        },
        "training": {
            "epochs": training_args.num_train_epochs,
            "batch_size": training_args.per_device_train_batch_size,
            "learning_rate": training_args.learning_rate
        },
        "test_metrics": {
            "accuracy": float(test_metrics.get('test_accuracy', 0)),
            "macro_f1": float(test_metrics.get('test_macro_f1', 0)),
            "macro_precision": float(test_metrics.get('test_macro_precision', 0)),
            "macro_recall": float(test_metrics.get('test_macro_recall', 0))
        },
        "per_class_metrics": {
            "contradiction": {
                "precision": float(test_metrics.get('test_contradiction_precision', 0)),
                "recall": float(test_metrics.get('test_contradiction_recall', 0)),
                "f1": float(test_metrics.get('test_contradiction_f1', 0))
            },
            "neutral": {
                "precision": float(test_metrics.get('test_neutral_precision', 0)),
                "recall": float(test_metrics.get('test_neutral_recall', 0)),
                "f1": float(test_metrics.get('test_neutral_f1', 0))
            },
            "entailment": {
                "precision": float(test_metrics.get('test_entailment_precision', 0)),
                "recall": float(test_metrics.get('test_entailment_recall', 0)),
                "f1": float(test_metrics.get('test_entailment_f1', 0))
            }
        },
        "label_mapping": {
            "0": "contradiction",
            "1": "neutral",
            "2": "entailment"
        }
    }
    
    with open(model_save_path / "training_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n✓ Model saved to: {model_save_path}")
    print(f"✓ Training summary saved to: {model_save_path / 'training_summary.json'}")
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    print(f"\nTest Accuracy: {test_metrics.get('test_accuracy', 0):.2%}")
    print(f"Macro F1: {test_metrics.get('test_macro_f1', 0):.4f}")

if __name__ == "__main__":
    train_causal_critic_3class()
