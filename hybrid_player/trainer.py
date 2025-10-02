import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    Trainer, TrainingArguments, DataCollatorForLanguageModeling,
    DataCollatorWithPadding, EarlyStoppingCallback
)
import pandas as pd
from typing import Dict, Any
import logging
from .models import PlayerLanguageModel, IntentClassifier, HybridPlayerModel
from .data_loader import HybridPlayerDataProcessor

logger = logging.getLogger(__name__)

class PlayerTextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=128):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': encoding['input_ids'].flatten()
        }

class IntentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

class HybridPlayerTrainer:
    def __init__(self, config):
        self.config = config
        self.data_processor = HybridPlayerDataProcessor(config.data)
        self.model = HybridPlayerModel(config.model)
        
    def train_language_model(self, train_df: pd.DataFrame, val_df: pd.DataFrame):
        """Train the language model on player utterances"""
        logger.info("Training language model...")
        
        model, tokenizer = self.model.language_model.get_model_and_tokenizer()
        
        # Prepare datasets
        train_dataset = PlayerTextDataset(
            train_df['text'].tolist(), tokenizer, self.config.model.lm_max_length
        )
        val_dataset = PlayerTextDataset(
            val_df['text'].tolist(), tokenizer, self.config.model.lm_max_length
        )
        
        # Training arguments - compatible with newer transformers versions
        training_args = TrainingArguments(
            output_dir="./models/language_model",
            overwrite_output_dir=True,
            num_train_epochs=self.config.model.lm_epochs,
            per_device_train_batch_size=self.config.model.lm_batch_size,
            per_device_eval_batch_size=self.config.model.lm_batch_size,
            warmup_steps=500,
            logging_steps=100,
            eval_strategy="epoch",  
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            seed=self.config.model.seed,
        )
        
        # Initialize trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=DataCollatorForLanguageModeling(
                tokenizer=tokenizer,
                mlm=False,
            ),
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
        )
        
        # Train model
        trainer.train()
        
        # Save model
        trainer.save_model("./models/language_model/final")
        tokenizer.save_pretrained("./models/language_model/final")
        logger.info("Language model training completed and saved")
        
        return trainer
    
    def train_intent_classifier(self, train_df: pd.DataFrame, val_df: pd.DataFrame):
        """Train the intent classifier"""
        logger.info("Training intent classifier...")
        
        model, tokenizer = self.model.intent_classifier.get_model_and_tokenizer()
        
        # Prepare datasets
        train_dataset = IntentDataset(
            train_df['text'].tolist(),
            train_df['intent_id'].tolist(),
            tokenizer,
            self.config.model.lm_max_length
        )
        val_dataset = IntentDataset(
            val_df['text'].tolist(),
            val_df['intent_id'].tolist(),
            tokenizer,
            self.config.model.lm_max_length
        )
        
        # Training arguments - compatible with newer transformers versions
        training_args = TrainingArguments(
            output_dir="./models/intent_classifier",
            overwrite_output_dir=True,
            num_train_epochs=self.config.model.classifier_epochs,
            per_device_train_batch_size=self.config.model.classifier_batch_size,
            per_device_eval_batch_size=self.config.model.classifier_batch_size,
            warmup_steps=100,
            logging_steps=50,
            eval_strategy="epoch",  
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            seed=self.config.model.seed,
        )
        
        # Initialize trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
            callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
        )
        
        # Train model
        trainer.train()
        
        # Save model
        trainer.save_model("./models/intent_classifier/final")
        tokenizer.save_pretrained("./models/intent_classifier/final")
        logger.info("Intent classifier training completed and saved")
        
        return trainer
    
    def train_all(self):
        """Complete training pipeline"""
        logger.info("Starting complete hybrid player training...")
        
        # Process data
        df = self.data_processor.process_all_data()
        train_df, val_df, test_df = self.data_processor.train_val_test_split(df)
        
        # Train language model
        self.train_language_model(train_df, val_df)
        
        # Train intent classifier
        self.train_intent_classifier(train_df, val_df)
        
        logger.info("Hybrid player training completed!")
        
        return train_df, val_df, test_df