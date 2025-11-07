# train_with_metrics_images.py
import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import (
    DebertaV2Tokenizer,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)
from datasets import Dataset
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc,
    mean_squared_error, mean_absolute_error
)
from scipy.stats import pearsonr
import pandas as pd
from sklearn.calibration import calibration_curve
import warnings
warnings.filterwarnings('ignore')

class ComprehensiveNarrativeCriticTrainer:
    def __init__(self):
        self.config = self.load_config()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.images_dir = "images"
        self.setup_directories()
        self.setup_model()
    
    def setup_directories(self):
        """Create images directory if it doesn't exist"""
        os.makedirs(self.images_dir, exist_ok=True)
        print(f"Created images directory: {self.images_dir}")
    
    def load_config(self):
        return {
            'model_name': 'microsoft/deberta-v3-base',
            'train_path': 'data/balanced_rocstoriestrain.json',
            'val_path': 'data/balanced_rocstoriesval.json',
            'max_seq_length': 128,
            'output_dir': 'models/narrative_critic_comprehensive',
            'num_train_epochs': 3,
            'per_device_train_batch_size': 8,
            'per_device_eval_batch_size': 8,
            'gradient_accumulation_steps': 4,
            'learning_rate': 2e-5,
            'warmup_ratio': 0.1,
            'weight_decay': 0.01,
            'max_grad_norm': 1.0,
            'logging_steps': 50,
            'eval_steps': 100,
            'save_steps': 200,
            'save_total_limit': 2,
            'fp16': False,
            'gradient_checkpointing': False,
            'dataloader_num_workers': 2,
            'seed': 42
        }
    
    def setup_model(self):
        print("Loading model and tokenizer...")
        self.tokenizer = DebertaV2Tokenizer.from_pretrained(self.config['model_name'])
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        from narrative_critic import NarrativeCritic
        from transformers import DebertaV2Config
        
        model_config = DebertaV2Config.from_pretrained(self.config['model_name'])
        model_config.num_labels = 1
        self.model = NarrativeCritic.from_pretrained(
            self.config['model_name'],
            config=model_config,
            ignore_mismatched_sizes=True
        )
        
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def load_data(self):
        print("Loading training data...")
        
        with open(self.config['train_path'], 'r') as f:
            train_data = json.load(f)
        with open(self.config['val_path'], 'r') as f:
            val_data = json.load(f)
        
        # Convert to binary classification for some metrics
        for data in [train_data, val_data]:
            for example in data:
                example['binary_label'] = 1 if example['label'] >= 0.5 else 0
        
        train_dataset = Dataset.from_list(train_data)
        val_dataset = Dataset.from_list(val_data)
        
        print(f"Training samples: {len(train_dataset)}")
        print(f"Validation samples: {len(val_dataset)}")
        
        # Analyze data distribution
        train_labels = [ex['label'] for ex in train_data]
        val_labels = [ex['label'] for ex in val_data]
        
        print(f"Training label stats: Min={min(train_labels):.3f}, Max={max(train_labels):.3f}, Mean={np.mean(train_labels):.3f}")
        print(f"Validation label stats: Min={min(val_labels):.3f}, Max={max(val_labels):.3f}, Mean={np.mean(val_labels):.3f}")
        
        # Tokenize
        def tokenize_function(examples):
            tokenized = self.tokenizer(
                examples['text'],
                padding='max_length',
                truncation=True,
                max_length=self.config['max_seq_length'],
                return_tensors=None
            )
            tokenized['labels'] = examples['label']
            return tokenized
        
        tokenized_train = train_dataset.map(tokenize_function, batched=True, batch_size=1000)
        tokenized_val = val_dataset.map(tokenize_function, batched=True, batch_size=1000)
        
        return tokenized_train, tokenized_val, train_data, val_data
    
    def compute_metrics(self, eval_pred):
        """Compute comprehensive metrics for evaluation"""
        predictions, labels = eval_pred
        predictions = predictions.flatten()
        
        # Apply sigmoid for probability scores
        prob_predictions = 1 / (1 + np.exp(-predictions))
        
        # Regression metrics
        mse = mean_squared_error(labels, prob_predictions)
        mae = mean_absolute_error(labels, prob_predictions)
        rmse = np.sqrt(mse)
        pearson_corr, _ = pearsonr(labels, prob_predictions)
        
        # Binary classification metrics (threshold at 0.5)
        binary_preds = (prob_predictions >= 0.5).astype(int)
        binary_labels = (labels >= 0.5).astype(int)
        
        accuracy = accuracy_score(binary_labels, binary_preds)
        precision = precision_score(binary_labels, binary_preds, zero_division=0)
        recall = recall_score(binary_labels, binary_preds, zero_division=0)
        f1 = f1_score(binary_labels, binary_preds, zero_division=0)
        
        # Additional metrics
        within_0_1 = np.mean(np.abs(labels - prob_predictions) <= 0.1)
        within_0_2 = np.mean(np.abs(labels - prob_predictions) <= 0.2)
        within_0_3 = np.mean(np.abs(labels - prob_predictions) <= 0.3)
        
        return {
            # Regression metrics
            'mse': mse,
            'mae': mae,
            'rmse': rmse,
            'pearson_correlation': pearson_corr,
            
            # Classification metrics
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            
            # Threshold-based metrics
            'within_0.1': within_0_1,
            'within_0.2': within_0_2,
            'within_0.3': within_0_3,
        }
    
    def plot_confusion_matrix(self, binary_labels, binary_preds):
        """Plot and save confusion matrix"""
        plt.figure(figsize=(8, 6))
        cm = confusion_matrix(binary_labels, binary_preds)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Low Quality', 'High Quality'],
                   yticklabels=['Low Quality', 'High Quality'])
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(f'{self.images_dir}/confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: {self.images_dir}/confusion_matrix.png")
    
    def plot_roc_curve(self, binary_labels, prob_predictions):
        """Plot and save ROC curve"""
        plt.figure(figsize=(8, 6))
        fpr, tpr, _ = roc_curve(binary_labels, prob_predictions)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{self.images_dir}/roc_curve.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: {self.images_dir}/roc_curve.png")
        return roc_auc
    
    def plot_prediction_vs_true(self, true_labels, prob_predictions):
        """Plot and save predictions vs true labels"""
        plt.figure(figsize=(8, 6))
        plt.scatter(true_labels, prob_predictions, alpha=0.5, s=10)
        plt.plot([0, 1], [0, 1], 'r--', alpha=0.8)
        plt.xlabel('True Labels')
        plt.ylabel('Predicted Probabilities')
        plt.title('Predictions vs True Labels')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{self.images_dir}/predictions_vs_true.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: {self.images_dir}/predictions_vs_true.png")
    
    def plot_residuals(self, prob_predictions, true_labels):
        """Plot and save residual plot"""
        plt.figure(figsize=(8, 6))
        residuals = prob_predictions - true_labels
        plt.scatter(prob_predictions, residuals, alpha=0.5, s=10)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel('Predicted Probabilities')
        plt.ylabel('Residuals')
        plt.title('Residual Plot')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{self.images_dir}/residual_plot.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: {self.images_dir}/residual_plot.png")
    
    def plot_prediction_distribution(self, prob_predictions):
        """Plot and save distribution of predictions"""
        plt.figure(figsize=(8, 6))
        plt.hist(prob_predictions, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        plt.xlabel('Predicted Probability')
        plt.ylabel('Frequency')
        plt.title('Distribution of Predictions')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{self.images_dir}/prediction_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: {self.images_dir}/prediction_distribution.png")
    
    def plot_calibration_curve(self, true_labels, prob_predictions):
        """Plot and save calibration curve"""
        plt.figure(figsize=(8, 6))
        fraction_of_positives, mean_predicted_value = calibration_curve(
            true_labels, prob_predictions, n_bins=10, strategy='quantile'
        )
        plt.plot(mean_predicted_value, fraction_of_positives, "s-", label="Model")
        plt.plot([0, 1], [0, 1], "k--", label="Perfectly calibrated")
        plt.xlabel("Mean Predicted Value")
        plt.ylabel("Fraction of Positives")
        plt.title("Calibration Curve")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{self.images_dir}/calibration_curve.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: {self.images_dir}/calibration_curve.png")
        return np.mean(np.abs(fraction_of_positives - mean_predicted_value))
    
    def plot_error_distribution(self, prob_predictions, true_labels):
        """Plot and save error distribution"""
        plt.figure(figsize=(8, 6))
        errors = np.abs(prob_predictions - true_labels)
        plt.hist(errors, bins=50, alpha=0.7, color='lightcoral', edgecolor='black')
        plt.xlabel('Absolute Error')
        plt.ylabel('Frequency')
        plt.title('Error Distribution')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{self.images_dir}/error_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: {self.images_dir}/error_distribution.png")
    
    def plot_training_history(self, trainer):
        """Plot and save training history"""
        log_history = trainer.state.log_history
        
        # Extract metrics
        train_losses = [log['loss'] for log in log_history if 'loss' in log and 'eval_loss' not in log]
        eval_losses = [log['eval_loss'] for log in log_history if 'eval_loss' in log]
        eval_f1 = [log.get('eval_f1', 0) for log in log_history if 'eval_f1' in log]
        eval_pearson = [log.get('eval_pearson_correlation', 0) for log in log_history if 'eval_pearson_correlation' in log]
        eval_accuracy = [log.get('eval_accuracy', 0) for log in log_history if 'eval_accuracy' in log]
        
        # Plot training and validation loss
        plt.figure(figsize=(10, 8))
        
        plt.subplot(2, 2, 1)
        if train_losses:
            plt.plot(range(len(train_losses)), train_losses, 'b-', label='Training Loss', alpha=0.7)
        if eval_losses:
            plt.plot(range(len(eval_losses)), eval_losses, 'r-', label='Validation Loss', alpha=0.7)
        plt.xlabel('Steps')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot F1 score
        plt.subplot(2, 2, 2)
        if eval_f1:
            plt.plot(range(len(eval_f1)), eval_f1, 'g-', label='F1 Score')
            plt.xlabel('Evaluation Steps')
            plt.ylabel('F1 Score')
            plt.title('F1 Score Over Time')
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        # Plot Pearson correlation
        plt.subplot(2, 2, 3)
        if eval_pearson:
            plt.plot(range(len(eval_pearson)), eval_pearson, 'purple', label='Pearson Correlation')
            plt.xlabel('Evaluation Steps')
            plt.ylabel('Pearson Correlation')
            plt.title('Pearson Correlation Over Time')
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        # Plot accuracy
        plt.subplot(2, 2, 4)
        if eval_accuracy:
            plt.plot(range(len(eval_accuracy)), eval_accuracy, 'orange', label='Accuracy')
            plt.xlabel('Evaluation Steps')
            plt.ylabel('Accuracy')
            plt.title('Accuracy Over Time')
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.images_dir}/training_history.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: {self.images_dir}/training_history.png")
    
    def plot_metrics_summary(self, final_metrics):
        """Plot and save metrics summary table"""
        plt.figure(figsize=(10, 6))
        plt.axis('off')
        
        # Create table data
        metric_names = list(final_metrics.keys())
        metric_values = [f'{final_metrics[k]:.4f}' for k in metric_names]
        
        table_data = [[name, value] for name, value in zip(metric_names, metric_values)]
        table = plt.table(cellText=table_data,
                         colLabels=['Metric', 'Value'],
                         loc='center',
                         cellLoc='center',
                         colWidths=[0.4, 0.2])
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1, 2)
        plt.title('Final Metrics Summary', fontsize=16, pad=20)
        
        plt.tight_layout()
        plt.savefig(f'{self.images_dir}/metrics_summary.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: {self.images_dir}/metrics_summary.png")
    
    def plot_comprehensive_metrics(self, trainer, val_data):
        """Generate and save all comprehensive evaluation plots"""
        print("Generating comprehensive evaluation plots...")
        
        # Get predictions
        predictions_output = trainer.predict(trainer.eval_dataset)
        predictions = predictions_output.predictions.flatten()
        true_labels = predictions_output.label_ids
        
        # Apply sigmoid
        prob_predictions = 1 / (1 + np.exp(-predictions))
        binary_preds = (prob_predictions >= 0.5).astype(int)
        binary_labels = (true_labels >= 0.5).astype(int)
        
        # Generate all individual plots
        self.plot_confusion_matrix(binary_labels, binary_preds)
        roc_auc = self.plot_roc_curve(binary_labels, prob_predictions)
        self.plot_prediction_vs_true(true_labels, prob_predictions)
        self.plot_residuals(prob_predictions, true_labels)
        self.plot_prediction_distribution(prob_predictions)
        calibration_error = self.plot_calibration_curve(true_labels, prob_predictions)
        self.plot_error_distribution(prob_predictions, true_labels)
        self.plot_training_history(trainer)
        
        # Calculate final metrics
        final_metrics = {
            'MSE': mean_squared_error(true_labels, prob_predictions),
            'MAE': mean_absolute_error(true_labels, prob_predictions),
            'RMSE': np.sqrt(mean_squared_error(true_labels, prob_predictions)),
            'Pearson': pearsonr(true_labels, prob_predictions)[0],
            'Accuracy': accuracy_score(binary_labels, binary_preds),
            'Precision': precision_score(binary_labels, binary_preds, zero_division=0),
            'Recall': recall_score(binary_labels, binary_preds, zero_division=0),
            'F1': f1_score(binary_labels, binary_preds, zero_division=0),
            'AUC': roc_auc,
            'Calibration_Error': calibration_error,
        }
        
        # Save metrics summary
        self.plot_metrics_summary(final_metrics)
        
        # Print detailed classification report
        print("\n" + "="*60)
        print("DETAILED CLASSIFICATION REPORT")
        print("="*60)
        print(classification_report(binary_labels, binary_preds, 
                                  target_names=['Low Quality', 'High Quality']))
        
        # Print confusion matrix details
        cm = confusion_matrix(binary_labels, binary_preds)
        print("\nCONFUSION MATRIX DETAILS:")
        print(f"True Negatives (Low->Low): {cm[0,0]}")
        print(f"False Positives (Low->High): {cm[0,1]}")
        print(f"False Negatives (High->Low): {cm[1,0]}")
        print(f"True Positives (High->High): {cm[1,1]}")
        
        return final_metrics
    
    def train(self):
        """Comprehensive training with full evaluation"""
        tokenized_train, tokenized_val, train_data, val_data = self.load_data()
        
        os.makedirs(self.config['output_dir'], exist_ok=True)
        
        training_args = TrainingArguments(
            output_dir=self.config['output_dir'],
            num_train_epochs=self.config['num_train_epochs'],
            per_device_train_batch_size=self.config['per_device_train_batch_size'],
            per_device_eval_batch_size=self.config['per_device_eval_batch_size'],
            gradient_accumulation_steps=self.config['gradient_accumulation_steps'],
            learning_rate=self.config['learning_rate'],
            warmup_ratio=self.config['warmup_ratio'],
            weight_decay=self.config['weight_decay'],
            max_grad_norm=self.config['max_grad_norm'],
            logging_steps=self.config['logging_steps'],
            eval_steps=self.config['eval_steps'],
            save_steps=self.config['save_steps'],
            save_total_limit=self.config['save_total_limit'],
            fp16=self.config['fp16'],
            gradient_checkpointing=self.config['gradient_checkpointing'],
            dataloader_num_workers=self.config['dataloader_num_workers'],
            eval_strategy="steps",
            save_strategy="steps",
            load_best_model_at_end=True,
            metric_for_best_model="eval_pearson_correlation",
            greater_is_better=True,
            report_to="none",
            seed=self.config['seed'],
            remove_unused_columns=False,
        )
        
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_train,
            eval_dataset=tokenized_val,
            compute_metrics=self.compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
        )
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        print("Starting comprehensive training...")
        trainer.train()
        
        # Generate comprehensive evaluation
        final_metrics = self.plot_comprehensive_metrics(trainer, val_data)
        
        # Save final model
        trainer.save_model()
        self.tokenizer.save_pretrained(self.config['output_dir'])
        
        print(f"\nTraining completed. Model saved to {self.config['output_dir']}")
        
        return trainer, final_metrics

def test_model_comprehensively(model_path):
    """Comprehensive testing of the trained model"""
    from narrative_critic import NarrativeCritic
    from transformers import DebertaV2Tokenizer
    
    tokenizer = DebertaV2Tokenizer.from_pretrained(model_path)
    model = NarrativeCritic.from_pretrained(model_path)
    model.eval()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Diverse test examples
    test_examples = [
        {
            "text": "The ancient library stretched endlessly before you, its towering shelves groaning under the weight of countless leather-bound tomes. Dust motes danced in shafts of golden sunlight that pierced the stained-glass windows.",
            "expected": 0.85,
            "type": "High Quality - Descriptive"
        },
        {
            "text": "Your blade finds its mark with a satisfying thud, cutting deep across the orc's shoulder. It roars in pain and staggers backward, dark blood spraying across the stone floor.",
            "expected": 0.80,
            "type": "High Quality - Action"
        },
        {
            "text": "You see a room. There is a door. There is a table.",
            "expected": 0.15,
            "type": "Low Quality - Minimal"
        },
        {
            "text": "The thing happens and then the other thing happens too. Stuff occurs.",
            "expected": 0.10,
            "type": "Low Quality - Vague"
        },
        {
            "text": "The wizard carefully examines the ancient runes, his fingers tracing the intricate patterns carved into the stone.",
            "expected": 0.70,
            "type": "Medium Quality - Descriptive"
        },
        {
            "text": "You walk into the room and look around. There are some things there.",
            "expected": 0.25,
            "type": "Low Quality - Generic"
        }
    ]
    
    print("\n" + "="*70)
    print("COMPREHENSIVE MODEL TESTING")
    print("="*70)
    
    results = []
    for example in test_examples:
        with torch.no_grad():
            inputs = tokenizer(
                example["text"], 
                return_tensors="pt", 
                truncation=True, 
                padding=True, 
                max_length=128
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            outputs = model(**inputs)
            raw_score = outputs.logits.item()
            predicted_score = torch.sigmoid(outputs.logits).item()
        
        expected = example["expected"]
        diff = abs(predicted_score - expected)
        binary_pred = "HIGH" if predicted_score >= 0.5 else "LOW"
        binary_true = "HIGH" if expected >= 0.5 else "LOW"
        correct = binary_pred == binary_true
        
        results.append({
            'type': example["type"],
            'predicted': predicted_score,
            'expected': expected,
            'diff': diff,
            'raw': raw_score,
            'binary_pred': binary_pred,
            'binary_true': binary_true,
            'correct': correct
        })
        
        print(f"\n{example['type']}")
        print(f"Text: {example['text'][:80]}...")
        print(f"Predicted: {predicted_score:.3f} | Expected: {expected:.3f} | Diff: {diff:.3f}")
        print(f"Binary: {binary_pred} (True: {binary_true}) | Correct: {correct}")
    
    # Calculate test metrics
    binary_accuracy = sum([1 for r in results if r['correct']]) / len(results)
    avg_diff = np.mean([r['diff'] for r in results])
    
    print(f"\nTest Set Performance:")
    print(f"Binary Accuracy: {binary_accuracy:.3f}")
    print(f"Average Difference: {avg_diff:.3f}")
    print(f"Correct classifications: {sum([1 for r in results if r['correct']])}/{len(results)}")
    
    # Plot test results
    plt.figure(figsize=(12, 6))
    types = [r['type'] for r in results]
    predicted = [r['predicted'] for r in results]
    expected = [r['expected'] for r in results]
    
    x = np.arange(len(results))
    width = 0.35
    
    plt.bar(x - width/2, predicted, width, label='Predicted', alpha=0.7, color='blue')
    plt.bar(x + width/2, expected, width, label='Expected', alpha=0.7, color='orange')
    
    plt.xlabel('Example Type')
    plt.ylabel('Score')
    plt.title('Model Performance on Test Examples')
    plt.xticks(x, [t.split(' - ')[0] for t in types], rotation=45, ha='right')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add value labels
    for i, (p, e) in enumerate(zip(predicted, expected)):
        plt.text(i - width/2, p + 0.02, f'{p:.2f}', ha='center', va='bottom', fontsize=8)
        plt.text(i + width/2, e + 0.02, f'{e:.2f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig('images/test_examples_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: images/test_examples_comparison.png")
    
    print("\n" + "="*70)
    return results

if __name__ == "__main__":
    # Set plotting style
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    
    # Clear GPU memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    print("=== COMPREHENSIVE NARRATIVE CRITIC TRAINING ===")
    print("This will train with full metrics and evaluation")
    print("All plots will be saved in the 'images/' folder")
    print()
    
    # Train with comprehensive metrics
    trainer = ComprehensiveNarrativeCriticTrainer()
    trainer_instance, final_metrics = trainer.train()
    
    # Comprehensive testing
    test_results = test_model_comprehensively(trainer.config['output_dir'])
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE")
    print("="*70)
    print("All plots saved in 'images/' folder:")
    print("1. confusion_matrix.png - Classification performance")
    print("2. roc_curve.png - ROC curve with AUC")
    print("3. predictions_vs_true.png - Scatter plot of predictions")
    print("4. residual_plot.png - Error distribution")
    print("5. prediction_distribution.png - Histogram of predictions")
    print("6. calibration_curve.png - Model calibration")
    print("7. error_distribution.png - Absolute error distribution")
    print("8. training_history.png - Loss and metrics over time")
    print("9. metrics_summary.png - Final metrics table")
    print("10. test_examples_comparison.png - Performance on test examples")
    print("="*70)