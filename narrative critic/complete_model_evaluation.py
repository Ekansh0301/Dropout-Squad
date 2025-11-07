"""
Complete Model Evaluation Script for Narrative Critic
Evaluates the trained model on validation data with comprehensive metrics and diagnostics.
"""

import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    confusion_matrix, classification_report
)
from scipy import stats
import pandas as pd

# Configuration
MODEL_PATH = "./model_jayant"
VAL_DATA_PATH = "./dataset_json/rocstoriesval.json"
TRAIN_DATA_PATH = "./dataset_json/rocstoriestrain.json"
OUTPUT_DIR = "./evaluation_results"

# Create output directory
Path(OUTPUT_DIR).mkdir(exist_ok=True)

print("="*80)
print("NARRATIVE CRITIC - COMPLETE MODEL EVALUATION")
print("="*80)

# ============================================================================
# 1. LOAD MODEL AND DATA
# ============================================================================
print("\n[1/8] Loading model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
model.eval()
print(f"✓ Model loaded on {device}")

print("\n[2/8] Loading datasets...")
with open(VAL_DATA_PATH, 'r') as f:
    val_data = json.load(f)
with open(TRAIN_DATA_PATH, 'r') as f:
    train_data = json.load(f)
    
print(f"✓ Validation data: {len(val_data)} examples")
print(f"✓ Training data: {len(train_data)} examples")

# ============================================================================
# 2. RUN PREDICTIONS ON VALIDATION SET
# ============================================================================
print("\n[3/8] Running predictions on validation set...")
predictions = []
true_labels = []
story_texts = []
story_types = []

with torch.no_grad():
    for i, example in enumerate(val_data):
        if i % 500 == 0:
            print(f"  Processing {i}/{len(val_data)}...")
        
        # Tokenize
        inputs = tokenizer(
            example['text'],
            return_tensors='pt',
            truncation=True,
            max_length=256,
            padding='max_length'
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Predict
        logits = model(**inputs).logits.squeeze()
        score = torch.sigmoid(logits).cpu().item()
        
        predictions.append(score)
        true_labels.append(example['label_float'])
        story_texts.append(example['text'])
        story_types.append(example['type'])

predictions = np.array(predictions)
true_labels = np.array(true_labels)
print(f"✓ Predictions complete")

# ============================================================================
# 3. COLLAPSE DETECTION
# ============================================================================
print("\n[4/8] Checking for model collapse...")
pred_std = np.std(predictions)
pred_range = np.max(predictions) - np.min(predictions)
pred_mean = np.mean(predictions)
pred_median = np.median(predictions)

print(f"\nPrediction Statistics:")
print(f"  Mean:              {pred_mean:.4f}")
print(f"  Median:            {pred_median:.4f}")
print(f"  Std Dev:           {pred_std:.4f}")
print(f"  Min:               {np.min(predictions):.4f}")
print(f"  Max:               {np.max(predictions):.4f}")
print(f"  Range:             {pred_range:.4f}")

# Collapse verdict
if pred_std < 0.05:
    print("\n🛑 CRITICAL: MODEL HAS COLLAPSED!")
    print("   All predictions are nearly identical")
    print("   This model should NOT be used in production")
    collapse_status = "COLLAPSED"
elif pred_std < 0.10:
    print("\n⚠️ WARNING: Low prediction diversity")
    print("   Model may be partially collapsed")
    print("   Use with caution")
    collapse_status = "MARGINAL"
else:
    print("\n✅ PASS: Model shows good prediction diversity")
    collapse_status = "HEALTHY"

# ============================================================================
# 4. REGRESSION METRICS
# ============================================================================
print("\n[5/8] Calculating regression metrics...")
mse = mean_squared_error(true_labels, predictions)
mae = mean_absolute_error(true_labels, predictions)
rmse = np.sqrt(mse)
r2 = r2_score(true_labels, predictions)
correlation, p_value = stats.pearsonr(true_labels, predictions)
spearman_corr, spearman_p = stats.spearmanr(true_labels, predictions)

# Accuracy within thresholds
within_01 = np.mean(np.abs(predictions - true_labels) < 0.1)
within_02 = np.mean(np.abs(predictions - true_labels) < 0.2)
within_03 = np.mean(np.abs(predictions - true_labels) < 0.3)

print(f"\nRegression Metrics:")
print(f"  MSE:                {mse:.4f}")
print(f"  MAE:                {mae:.4f}")
print(f"  RMSE:               {rmse:.4f}")
print(f"  R² Score:           {r2:.4f}")
print(f"  Pearson Corr:       {correlation:.4f} (p={p_value:.2e})")
print(f"  Spearman Corr:      {spearman_corr:.4f} (p={spearman_p:.2e})")
print(f"\nAccuracy (within threshold):")
print(f"  ±0.1:               {within_01:.2%}")
print(f"  ±0.2:               {within_02:.2%}")
print(f"  ±0.3:               {within_03:.2%}")

# ============================================================================
# 5. PER-TYPE ANALYSIS
# ============================================================================
print("\n[6/8] Analyzing performance by quality type...")
unique_types = sorted(set(story_types))
type_metrics = {}

for qtype in unique_types:
    mask = np.array([t == qtype for t in story_types])
    type_preds = predictions[mask]
    type_labels = true_labels[mask]
    
    type_metrics[qtype] = {
        'count': mask.sum(),
        'mean_pred': np.mean(type_preds),
        'std_pred': np.std(type_preds),
        'mean_label': np.mean(type_labels),
        'mae': mean_absolute_error(type_labels, type_preds),
        'r2': r2_score(type_labels, type_preds),
        'correlation': np.corrcoef(type_labels, type_preds)[0, 1]
    }

print(f"\nPer-Type Performance:")
print(f"{'Type':<15} {'Count':<8} {'Pred Mean':<12} {'Pred Std':<12} {'Label Mean':<12} {'MAE':<10} {'R²':<10}")
print("-" * 85)
for qtype in unique_types:
    m = type_metrics[qtype]
    print(f"{qtype:<15} {m['count']:<8} {m['mean_pred']:<12.4f} {m['std_pred']:<12.4f} "
          f"{m['mean_label']:<12.4f} {m['mae']:<10.4f} {m['r2']:<10.4f}")

# ============================================================================
# 6. BINNED CLASSIFICATION ANALYSIS
# ============================================================================
print("\n[7/8] Analyzing as classification problem (binned scores)...")

# Bin predictions and labels
def bin_scores(scores):
    """Bin continuous scores into quality categories."""
    bins = np.array([0.0, 0.3, 0.5, 0.7, 1.0])
    labels = ['low', 'medium-low', 'medium-high', 'high']
    return pd.cut(scores, bins=bins, labels=labels, include_lowest=True)

pred_bins = bin_scores(predictions)
label_bins = bin_scores(true_labels)

# Classification report
print("\nClassification Report (binned scores):")
print(classification_report(label_bins, pred_bins, zero_division=0))

# Confusion matrix
cm = confusion_matrix(label_bins, pred_bins, labels=['low', 'medium-low', 'medium-high', 'high'])
print("\nConfusion Matrix:")
print(f"{'':>15} {'low':<15} {'medium-low':<15} {'medium-high':<15} {'high':<15}")
for i, true_label in enumerate(['low', 'medium-low', 'medium-high', 'high']):
    print(f"{true_label:>15} {cm[i][0]:<15} {cm[i][1]:<15} {cm[i][2]:<15} {cm[i][3]:<15}")

# ============================================================================
# 7. EXAMPLE PREDICTIONS
# ============================================================================
print("\n[8/8] Showing example predictions...")

# Get diverse examples
examples_per_type = 2
example_indices = []
for qtype in unique_types:
    type_indices = [i for i, t in enumerate(story_types) if t == qtype]
    example_indices.extend(np.random.choice(type_indices, min(examples_per_type, len(type_indices)), replace=False))

print(f"\nExample Predictions ({len(example_indices)} samples):")
print("="*80)
for idx in example_indices:
    print(f"\nType: {story_types[idx]}")
    print(f"Text: {story_texts[idx][:150]}...")
    print(f"True Label: {true_labels[idx]:.4f}")
    print(f"Prediction: {predictions[idx]:.4f}")
    print(f"Error:      {abs(predictions[idx] - true_labels[idx]):.4f}")
    print("-"*80)

# ============================================================================
# 8. VISUALIZATIONS
# ============================================================================
print("\n[Bonus] Creating visualizations...")

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Figure 1: Scatter plot with regression line
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# Scatter plot
ax = axes[0, 0]
ax.scatter(true_labels, predictions, alpha=0.3, s=10)
ax.plot([0, 1], [0, 1], 'r--', label='Perfect prediction')
z = np.polyfit(true_labels, predictions, 1)
p = np.poly1d(z)
ax.plot(true_labels, p(true_labels), "b-", alpha=0.8, label=f'Fit: y={z[0]:.2f}x+{z[1]:.2f}')
ax.set_xlabel('True Labels')
ax.set_ylabel('Predictions')
ax.set_title(f'Predictions vs True Labels\nR²={r2:.4f}, Corr={correlation:.4f}')
ax.legend()
ax.grid(True, alpha=0.3)

# Distribution comparison
ax = axes[0, 1]
ax.hist(true_labels, bins=30, alpha=0.5, label='True Labels', density=True)
ax.hist(predictions, bins=30, alpha=0.5, label='Predictions', density=True)
ax.set_xlabel('Score')
ax.set_ylabel('Density')
ax.set_title('Score Distributions')
ax.legend()
ax.grid(True, alpha=0.3)

# Error distribution
ax = axes[1, 0]
errors = predictions - true_labels
ax.hist(errors, bins=50, edgecolor='black', alpha=0.7)
ax.axvline(0, color='red', linestyle='--', linewidth=2, label='Zero error')
ax.axvline(np.mean(errors), color='green', linestyle='--', linewidth=2, label=f'Mean: {np.mean(errors):.4f}')
ax.set_xlabel('Prediction Error')
ax.set_ylabel('Frequency')
ax.set_title(f'Error Distribution\nMAE={mae:.4f}, Std={np.std(errors):.4f}')
ax.legend()
ax.grid(True, alpha=0.3)

# Per-type box plot
ax = axes[1, 1]
type_data = []
type_labels_plot = []
for qtype in unique_types:
    mask = np.array([t == qtype for t in story_types])
    type_data.append(predictions[mask])
    type_labels_plot.append(qtype)

bp = ax.boxplot(type_data, labels=type_labels_plot, patch_artist=True)
for patch in bp['boxes']:
    patch.set_facecolor('lightblue')
ax.set_xlabel('Quality Type')
ax.set_ylabel('Predicted Score')
ax.set_title('Predictions by Quality Type')
ax.grid(True, alpha=0.3, axis='y')
plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/evaluation_plots.png", dpi=300, bbox_inches='tight')
print(f"✓ Saved visualization: {OUTPUT_DIR}/evaluation_plots.png")

# Figure 2: Confusion matrix heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['low', 'medium-low', 'medium-high', 'high'],
            yticklabels=['low', 'medium-low', 'medium-high', 'high'])
plt.title('Confusion Matrix (Binned Scores)')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/confusion_matrix.png", dpi=300, bbox_inches='tight')
print(f"✓ Saved confusion matrix: {OUTPUT_DIR}/confusion_matrix.png")

# ============================================================================
# 9. SAVE RESULTS
# ============================================================================
print("\nSaving evaluation results...")

results = {
    'collapse_status': collapse_status,
    'prediction_stats': {
        'mean': float(pred_mean),
        'median': float(pred_median),
        'std': float(pred_std),
        'min': float(np.min(predictions)),
        'max': float(np.max(predictions)),
        'range': float(pred_range)
    },
    'regression_metrics': {
        'mse': float(mse),
        'mae': float(mae),
        'rmse': float(rmse),
        'r2_score': float(r2),
        'pearson_correlation': float(correlation),
        'pearson_p_value': float(p_value),
        'spearman_correlation': float(spearman_corr),
        'spearman_p_value': float(spearman_p)
    },
    'accuracy_thresholds': {
        'within_0.1': float(within_01),
        'within_0.2': float(within_02),
        'within_0.3': float(within_03)
    },
    'per_type_metrics': {
        qtype: {k: float(v) if isinstance(v, (np.floating, np.integer)) else v 
                for k, v in metrics.items()}
        for qtype, metrics in type_metrics.items()
    }
}

with open(f"{OUTPUT_DIR}/evaluation_results.json", 'w') as f:
    json.dump(results, f, indent=2)
print(f"✓ Saved results: {OUTPUT_DIR}/evaluation_results.json")

# ============================================================================
# 10. FINAL REPORT
# ============================================================================
print("\n" + "="*80)
print("EVALUATION SUMMARY")
print("="*80)

print(f"\n📊 Model Status: {collapse_status}")
if collapse_status == "COLLAPSED":
    print("   🛑 MODEL UNUSABLE - Do not deploy!")
elif collapse_status == "MARGINAL":
    print("   ⚠️ USE WITH CAUTION - Limited discriminative power")
else:
    print("   ✅ Model appears functional")

print(f"\n📈 Key Metrics:")
print(f"   MAE:         {mae:.4f} {'✅' if mae < 0.20 else '⚠️' if mae < 0.25 else '❌'}")
print(f"   R² Score:    {r2:.4f} {'✅' if r2 > 0.3 else '⚠️' if r2 > 0 else '❌'}")
print(f"   Correlation: {correlation:.4f} {'✅' if correlation > 0.7 else '⚠️'}")
print(f"   Pred Std:    {pred_std:.4f} {'✅' if pred_std > 0.12 else '⚠️' if pred_std > 0.08 else '❌'}")

print(f"\n🎯 Recommendations:")
if collapse_status == "COLLAPSED":
    print("   1. RETRAIN with learning rate 3e-6 or lower")
    print("   2. Use ultra-conservative configuration")
    print("   3. Monitor pred_std during training")
elif collapse_status == "MARGINAL":
    print("   1. Consider retraining with lower learning rate")
    print("   2. Test thoroughly before production use")
    print("   3. Apply calibration to improve score distribution")
elif mae > 0.20 or r2 < 0.3:
    print("   1. Apply calibration to improve accuracy")
    print("   2. Consider fine-tuning with more data")
    print("   3. May need architectural changes for better performance")
else:
    print("   1. Apply calibration for production use")
    print("   2. Test on diverse examples")
    print("   3. Monitor performance in production")

print(f"\n📁 Output files saved to: {OUTPUT_DIR}/")
print("   - evaluation_results.json")
print("   - evaluation_plots.png")
print("   - confusion_matrix.png")

print("\n" + "="*80)
print("EVALUATION COMPLETE")
print("="*80)
