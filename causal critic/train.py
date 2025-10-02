"""Fine-tune causal critic on D&D causal consistency pairs."""
import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)
from datasets import load_from_disk
import numpy as np

def compute_metrics(eval_pred):
    """Calculate accuracy for evaluation."""
    predictions, labels = eval_pred
    preds = np.argmax(predictions, axis=-1)
    accuracy = (preds == labels).mean()
    return {'accuracy': accuracy}

def train_causal_critic():
    """Train causal critic model on prepared dataset."""
    print("Training Causal Critic...")
    
    # Load pre-trained NLI model
    model = AutoModelForSequenceClassification.from_pretrained(
        "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli",
        num_labels=3
    )
    tokenizer = AutoTokenizer.from_pretrained("MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli")
    
    # Load prepared datasets
    train_dataset = load_from_disk("data/causal_critic_training/train")
    eval_dataset = load_from_disk("data/causal_critic_training/val")
    
    def tokenize(examples):
        """Tokenize premise-hypothesis pairs for NLI."""
        return tokenizer(examples['premise'], examples['hypothesis'], 
                        truncation=True, max_length=256, padding=False)
    
    # Tokenize datasets
    train_dataset = train_dataset.map(tokenize, batched=True, num_proc=4)
    eval_dataset = eval_dataset.map(tokenize, batched=True, num_proc=4)
    
    # Rename label column for trainer compatibility
    train_dataset = train_dataset.rename_column('label', 'labels')
    eval_dataset = eval_dataset.rename_column('label', 'labels')
    
    # Configure training parameters
    training_args = TrainingArguments(
        output_dir="models/causal_critic_finetuned",
        num_train_epochs=2,
        per_device_train_batch_size=32,
        learning_rate=2e-5,
        eval_strategy="steps",
        eval_steps=500,
        save_steps=500,
        fp16=True,
        logging_steps=100,
        report_to="none"
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
    
    # Train and save model
    trainer.train()
    trainer.save_model()
    tokenizer.save_pretrained("models/causal_critic_finetuned")
    
    print("✓ Causal Critic trained")

if __name__ == "__main__":
    train_causal_critic()