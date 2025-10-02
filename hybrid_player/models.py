"""Model classes for hybrid player language generation and intent classification."""

import torch
import torch.nn as nn
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, AutoModelForSequenceClassification,
    GPT2LMHeadModel, DistilBertForSequenceClassification, Trainer, TrainingArguments
)
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

class PlayerLanguageModel:
    """DistilGPT-2 model for generating player utterances."""
    
    def __init__(self, model_name: str = "distilgpt2"):
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        
    def load_model(self):
        """Load pre-trained model and configure tokenizer."""
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
        logger.info(f"Loaded language model: {self.model_name}")
    
    def get_model_and_tokenizer(self):
        """Return model and tokenizer, loading if necessary."""
        if self.model is None or self.tokenizer is None:
            self.load_model()
        return self.model, self.tokenizer

class IntentClassifier:
    """DistilBERT classifier for predicting player intent categories."""
    
    def __init__(self, model_name: str = "distilbert-base-uncased", num_labels: int = 3):
        self.model_name = model_name
        self.num_labels = num_labels
        self.model = None
        self.tokenizer = None
        self.id2label = {0: "EXPLORE", 1: "ACTION", 2: "DIALOGUE"}
        self.label2id = {"EXPLORE": 0, "ACTION": 1, "DIALOGUE": 2}
        
    def load_model(self):
        """Load pre-trained classifier with custom label mapping."""
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name, num_labels=self.num_labels, id2label=self.id2label, label2id=self.label2id
        )
        logger.info(f"Loaded classifier model: {self.model_name}")
    
    def get_model_and_tokenizer(self):
        """Return model and tokenizer, loading if necessary."""
        if self.model is None or self.tokenizer is None:
            self.load_model()
        return self.model, self.tokenizer

class HybridPlayerModel:
    """Combined model for player simulation with generation and intent classification."""
    
    def __init__(self, config):
        self.config = config
        self.language_model = PlayerLanguageModel(config.lm_model_name)
        self.intent_classifier = IntentClassifier(
            config.classifier_model_name, 
            config.classifier_num_labels
        )
        
    def load_models(self):
        """Initialize both language model and intent classifier."""
        self.language_model.load_model()
        self.intent_classifier.load_model()
        logger.info("Loaded both language model and intent classifier")