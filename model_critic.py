#!/usr/bin/env python
"""
Fine-tuning DeBERTa for Narrative Quality Assessment using ROCStories Dataset
Adapted from HuggingFace's run_glue.py for the Director LLM project
"""

import logging
import os
import random
import sys
from dataclasses import dataclass, field
from typing import Optional, Dict, List
import pandas as pd
import numpy as np
from collections import Counter

import datasets
import evaluate
import torch
from datasets import Dataset, DatasetDict

import transformers
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version

# Will error if the minimal version of Transformers is not installed
check_min_version("4.30.0")

logger = logging.getLogger(__name__)


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    
    train_file: str = field(
        default="ROCStories__spring2016 - ROCStories_spring2016.csv",
        metadata={"help": "The ROCStories training data CSV file."}
    )
    validation_file: str = field(
        default="cloze_test_val__spring2016 - cloze_test_ALL_val.csv",
        metadata={"help": "The Story Cloze validation CSV file."}
    )
    test_file: str = field(
        default="cloze_test_test__spring2016 - cloze_test_ALL_test.csv",
        metadata={"help": "The Story Cloze test CSV file."}
    )
    max_seq_length: int = field(
        default=512,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached preprocessed datasets or not."}
    )
    pad_to_max_length: bool = field(
        default=True,
        metadata={
            "help": (
                "Whether to pad all samples to `max_seq_length`. "
                "If False, will pad the samples dynamically when batching to the maximum length in the batch."
            )
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of prediction examples to this "
                "value if set."
            )
        },
    )
    negative_sampling_ratio: float = field(
        default=1.0,
        metadata={"help": "Ratio of negative examples to create from training data"}
    )


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """
    
    model_name_or_path: str = field(
        default="microsoft/deberta-v3-base",
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    token: str = field(
        default=None,
        metadata={"help": "The token to use as HTTP bearer authorization for remote files."}
    )
    trust_remote_code: bool = field(
        default=False,
        metadata={"help": "Whether to trust the execution of code from datasets/models defined on the Hub."}
    )
    ignore_mismatched_sizes: bool = field(
        default=False,
        metadata={"help": "Will enable to load a pretrained model whose head dimensions are different."},
    )


class ROCStoriesDataProcessor:
    """Process ROCStories and Story Cloze datasets for narrative quality assessment"""
    
    def __init__(self, tokenizer, max_seq_length: int = 512):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
    
    def load_rocstories_train(self, filepath: str, max_samples: Optional[int] = None) -> List[Dict]:
        """Load ROCStories training data and create positive examples"""
        logger.info(f"Loading ROCStories training data from {filepath}")
        
        # Read CSV file
        df = pd.read_csv(filepath)
        
        # ROCStories columns: storyid, storytitle, sentence1, sentence2, sentence3, sentence4, sentence5
        required_cols = ['sentence1', 'sentence2', 'sentence3', 'sentence4', 'sentence5']
        
        # Check if all required columns exist
        for col in required_cols:
            if col not in df.columns:
                # Try alternative naming (InputSentence1, etc.)
                alt_col = f'InputSentence{col[-1]}'
                if alt_col in df.columns:
                    df[col] = df[alt_col]
                else:
                    raise ValueError(f"Column {col} not found in {filepath}")
        
        if max_samples:
            df = df.sample(n=min(max_samples, len(df)), random_state=42)
        
        examples = []
        for _, row in df.iterrows():
            # Create positive example with full story
            story_text = f"{row['sentence1']} {row['sentence2']} {row['sentence3']} {row['sentence4']} {row['sentence5']}"
            examples.append({
                'text': story_text,
                'label': 1,  # Good narrative
                'sentences': [row[f'sentence{i}'] for i in range(1, 6)]
            })
        
        return examples
    
    def load_story_cloze(self, filepath: str, max_samples: Optional[int] = None) -> List[Dict]:
        """Load Story Cloze validation/test data"""
        logger.info(f"Loading Story Cloze data from {filepath}")
        
        # Read CSV file
        df = pd.read_csv(filepath)
        
        # Story Cloze columns vary, but typically include:
        # InputSentence1-4, RandomFifthSentenceQuiz1, RandomFifthSentenceQuiz2, AnswerRightEnding
        examples = []
        
        for _, row in df.iterrows():
            # Get context sentences
            context_sentences = []
            for i in range(1, 5):
                col_name = f'InputSentence{i}'
                if col_name in df.columns:
                    context_sentences.append(row[col_name])
                elif f'sentence{i}' in df.columns:
                    context_sentences.append(row[f'sentence{i}'])
            
            context = ' '.join(context_sentences)
            
            # Get the two possible endings
            ending1 = row.get('RandomFifthSentenceQuiz1', row.get('sentence5a', ''))
            ending2 = row.get('RandomFifthSentenceQuiz2', row.get('sentence5b', ''))
            correct_ending = row.get('AnswerRightEnding', 1)
            
            # Create example with correct ending
            if correct_ending == 1:
                correct_text = f"{context} {ending1}"
                incorrect_text = f"{context} {ending2}"
            else:
                correct_text = f"{context} {ending2}"
                incorrect_text = f"{context} {ending1}"
            
            examples.append({
                'text': correct_text,
                'label': 1,  # Correct continuation
            })
            examples.append({
                'text': incorrect_text,
                'label': 0,  # Incorrect continuation
            })
        
        if max_samples:
            examples = examples[:max_samples]
        
        return examples
    
    def create_negative_examples(self, positive_examples: List[Dict], ratio: float = 1.0) -> List[Dict]:
        """Create negative examples by mismatching story endings"""
        negative_examples = []
        num_negatives = int(len(positive_examples) * ratio)
        
        for i in range(num_negatives):
            # Take context from one story and ending from another
            story1_idx = i % len(positive_examples)
            story2_idx = (i + len(positive_examples) // 2) % len(positive_examples)
            
            if 'sentences' in positive_examples[story1_idx]:
                # From training data
                context = ' '.join(positive_examples[story1_idx]['sentences'][:4])
                wrong_ending = positive_examples[story2_idx]['sentences'][4] if 'sentences' in positive_examples[story2_idx] else ""
                
                negative_examples.append({
                    'text': f"{context} {wrong_ending}",
                    'label': 0  # Poor narrative
                })
        
        return negative_examples
    
    def prepare_dataset(self, examples: List[Dict], padding: str = "max_length") -> Dataset:
        """Convert examples to HuggingFace Dataset format"""
        
        def tokenize_function(examples):
            result = self.tokenizer(
                examples['text'],
                padding=padding,
                max_length=self.max_seq_length,
                truncation=True
            )
            result['labels'] = examples['label']
            return result
        
        # Convert to Dataset
        dataset = Dataset.from_dict({
            'text': [ex['text'] for ex in examples],
            'label': [ex['label'] for ex in examples]
        })
        
        # Tokenize
        dataset = dataset.map(
            tokenize_function,
            batched=True,
            desc="Running tokenizer on dataset"
        )
        
        return dataset


def main():
    # Parse arguments
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    
    if training_args.should_log:
        transformers.utils.logging.set_verbosity_info()
    
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()
    
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}, "
        + f"distributed training: {training_args.parallel_mode.value == 'distributed'}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")
    
    # Detecting last checkpoint
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}."
            )
    
    # Set seed
    set_seed(training_args.seed)
    
    # Load pretrained model and tokenizer
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=2,  # Binary classification: good/poor narrative
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        token=model_args.token,
        trust_remote_code=model_args.trust_remote_code,
    )
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        token=model_args.token,
        trust_remote_code=model_args.trust_remote_code,
    )
    
    model = AutoModelForSequenceClassification.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        token=model_args.token,
        trust_remote_code=model_args.trust_remote_code,
        ignore_mismatched_sizes=model_args.ignore_mismatched_sizes,
    )
    
    # Initialize data processor
    processor = ROCStoriesDataProcessor(tokenizer, data_args.max_seq_length)
    
    # Load and prepare datasets
    if training_args.do_train:
        logger.info("Loading training data...")
        train_examples = processor.load_rocstories_train(
            data_args.train_file,
            max_samples=data_args.max_train_samples
        )
        
        # Create negative examples
        negative_examples = processor.create_negative_examples(
            train_examples,
            ratio=data_args.negative_sampling_ratio
        )
        
        # Combine positive and negative examples
        all_train_examples = train_examples + negative_examples
        random.shuffle(all_train_examples)
        
        # Create dataset
        train_dataset = processor.prepare_dataset(
            all_train_examples,
            padding="max_length" if data_args.pad_to_max_length else False
        )
        
        # Log class distribution
        label_counts = Counter([ex['label'] for ex in all_train_examples])
        logger.info(f"Training set class distribution:")
        logger.info(f"  Positive (good narrative): {label_counts[1]} ({label_counts[1]/len(all_train_examples):.2%})")
        logger.info(f"  Negative (poor narrative): {label_counts[0]} ({label_counts[0]/len(all_train_examples):.2%})")
    
    if training_args.do_eval:
        logger.info("Loading validation data...")
        eval_examples = processor.load_story_cloze(
            data_args.validation_file,
            max_samples=data_args.max_eval_samples
        )
        
        eval_dataset = processor.prepare_dataset(
            eval_examples,
            padding="max_length" if data_args.pad_to_max_length else False
        )
        
        # Log class distribution
        label_counts = Counter([ex['label'] for ex in eval_examples])
        logger.info(f"Validation set class distribution:")
        logger.info(f"  Positive (correct ending): {label_counts[1]} ({label_counts[1]/len(eval_examples):.2%})")
        logger.info(f"  Negative (wrong ending): {label_counts[0]} ({label_counts[0]/len(eval_examples):.2%})")
    
    if training_args.do_predict:
        logger.info("Loading test data...")
        test_examples = processor.load_story_cloze(
            data_args.test_file,
            max_samples=data_args.max_predict_samples
        )
        
        predict_dataset = processor.prepare_dataset(
            test_examples,
            padding="max_length" if data_args.pad_to_max_length else False
        )
    
    # Load metrics
    metric = evaluate.load("accuracy", cache_dir=model_args.cache_dir)
    precision_metric = evaluate.load("precision", cache_dir=model_args.cache_dir)
    recall_metric = evaluate.load("recall", cache_dir=model_args.cache_dir)
    f1_metric = evaluate.load("f1", cache_dir=model_args.cache_dir)
    
    def compute_metrics(p: EvalPrediction):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        preds = np.argmax(preds, axis=1)
        labels = p.label_ids
        
        accuracy = metric.compute(predictions=preds, references=labels)
        precision = precision_metric.compute(predictions=preds, references=labels, average='binary')
        recall = recall_metric.compute(predictions=preds, references=labels, average='binary')
        f1 = f1_metric.compute(predictions=preds, references=labels, average='binary')
        
        return {
            'accuracy': accuracy['accuracy'],
            'precision': precision['precision'],
            'recall': recall['recall'],
            'f1': f1['f1'],
        }
    
    # Data collator
    if data_args.pad_to_max_length:
        data_collator = default_data_collator
    elif training_args.fp16:
        data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
    else:
        data_collator = None
    
    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        compute_metrics=compute_metrics,
        processing_class=tokenizer,
        data_collator=data_collator,
    )
    
    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        metrics = train_result.metrics
        
        trainer.save_model()
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()
        
        logger.info("Training completed!")
    
    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate(eval_dataset=eval_dataset)
        
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)
        
        logger.info(f"Evaluation metrics: {metrics}")
    
    # Prediction
    if training_args.do_predict:
        logger.info("*** Predict ***")
        
        predictions = trainer.predict(predict_dataset, metric_key_prefix="predict")
        preds = np.argmax(predictions.predictions, axis=1)
        
        output_predict_file = os.path.join(training_args.output_dir, "predictions.txt")
        if trainer.is_world_process_zero():
            with open(output_predict_file, "w") as writer:
                logger.info("***** Predict results *****")
                writer.write("index\tprediction\tlabel\n")
                for index, (pred, label) in enumerate(zip(preds, predictions.label_ids)):
                    writer.write(f"{index}\t{pred}\t{label}\n")
        
        # Calculate and log test metrics
        test_metrics = compute_metrics(predictions)
        logger.info(f"Test metrics: {test_metrics}")
    
    # Create model card
    kwargs = {
        "finetuned_from": model_args.model_name_or_path,
        "tasks": "narrative-quality-assessment",
        "dataset": "ROCStories/Story Cloze",
        "tags": ["narrative-quality", "story-cloze", "deberta"],
    }
    
    if training_args.push_to_hub:
        trainer.push_to_hub(**kwargs)
    else:
        trainer.create_model_card(**kwargs)


# Utility function for integration with RL pipeline
class NarrativeCriticForRL:
    """Wrapper class for using the trained model in RL pipeline"""
    
    def __init__(self, model_path: str, device: str = None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()
    
    def score_narrative(self, context: str, continuation: str) -> float:
        """Score a narrative continuation (0-1, higher is better)"""
        text = f"{context} {continuation}"
        
        inputs = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=512,
            return_tensors='pt'
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)
            score = float(probs[0, 1])  # Probability of good narrative
        
        return score
    
    def compute_reward(self, context: str, response: str) -> float:
        """Compute reward for RL training (-1 to 1)"""
        score = self.score_narrative(context, response)
        reward = 2 * score - 1  # Map [0,1] to [-1,1]
        
        # Apply reward shaping
        if score > 0.8:
            reward *= 1.2
        elif score < 0.3:
            reward *= 1.5
        
        return np.clip(reward, -1, 1)


if __name__ == "__main__":
    main()