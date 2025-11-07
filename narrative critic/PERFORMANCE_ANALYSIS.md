# Narrative Critic - Performance Analysis and Diagnostics

## 📊 Your Training Results Analysis

Based on your validation metrics, here's what's happening:

### Metrics Received:
```
eval_correlation: 0.8973  ✅ EXCELLENT
eval_mae: 0.2556          ⚠️  HIGH (expected ~0.10)
eval_r2_score: -0.1075    ❌ NEGATIVE (expected 0.75-0.85)
eval_accuracy_0.2: 0.3127 ⚠️  LOW (expected ~0.85-0.90)
```

### 🔍 Diagnosis: Score Calibration Issue

The **high correlation (0.90)** with **negative R²** means:
- ✅ Model correctly ranks quality (high vs low)
- ❌ Absolute scores are systematically off

This is a **calibration problem**, not a learning problem!

---

## 🛠️ Likely Causes & Solutions

### Issue 1: Sigmoid Saturation
**Problem**: DeBERTa outputs might be too large/small for sigmoid to work well

**Check**: 
```python
# After getting predictions
print("Raw logits range:", predictions.predictions.min(), predictions.predictions.max())
print("After sigmoid range:", pred_scores.min(), pred_scores.max())
```

**Fix**: Temperature scaling
```python
# Instead of sigmoid(logits)
pred_scores = torch.sigmoid(predictions.predictions / temperature).squeeze()
# Try temperature = 2.0 or 3.0
```

### Issue 2: Training Data Distribution Mismatch
**Problem**: Model trained on continuous 0-1 labels but outputs are compressed

**Check**:
```python
# Plot predicted vs true distributions
plt.hist(true_scores, alpha=0.5, bins=50, label='True')
plt.hist(pred_scores, alpha=0.5, bins=50, label='Predicted')
plt.legend()
```

**Fix**: Check if `prepare_dataset.py` generated correct score distributions

### Issue 3: Regression Head Initialization
**Problem**: Linear layer might have poor initialization

**Fix**: Retrain with better initialization or add dropout

---

## 🎯 Quick Diagnostic Tests

### Test 1: Check Score Distributions
```python
# Get predictions on validation set
predictions = trainer.predict(val_dataset)
pred_scores = 1 / (1 + np.exp(-predictions.predictions.squeeze()))
true_scores = predictions.label_ids

print("True scores:")
print(f"  Min: {true_scores.min():.3f}, Max: {true_scores.max():.3f}")
print(f"  Mean: {true_scores.mean():.3f}, Std: {true_scores.std():.3f}")

print("\nPredicted scores:")
print(f"  Min: {pred_scores.min():.3f}, Max: {pred_scores.max():.3f}")
print(f"  Mean: {pred_scores.mean():.3f}, Std: {pred_scores.std():.3f}")
```

### Test 2: Visualize Prediction Scatter
```python
plt.figure(figsize=(10, 10))
plt.scatter(true_scores, pred_scores, alpha=0.3, s=10)
plt.plot([0, 1], [0, 1], 'r--', label='Perfect Prediction')
plt.xlabel('True Score')
plt.ylabel('Predicted Score')
plt.title('Predicted vs True Scores')
plt.legend()
plt.grid(True)
plt.show()
```

### Test 3: Check Per-Type Performance
```python
results_df = val_df.copy()
results_df['predicted_score'] = pred_scores
results_df['true_score'] = true_scores

for ntype in ['coherent', 'shuffled', 'repetitive', 'truncated']:
    subset = results_df[results_df['type'] == ntype]
    print(f"\n{ntype.upper()}:")
    print(f"  True mean: {subset['true_score'].mean():.3f}")
    print(f"  Pred mean: {subset['predicted_score'].mean():.3f}")
    print(f"  MAE: {np.abs(subset['true_score'] - subset['predicted_score']).mean():.3f}")
```

---

## 🔧 Recommended Fixes

### Option 1: Post-Training Calibration (Fastest)
```python
# Fit a simple linear calibration on validation set
from sklearn.linear_model import LinearRegression

calibrator = LinearRegression()
calibrator.fit(pred_scores.reshape(-1, 1), true_scores)

# Apply calibration
calibrated_scores = calibrator.predict(pred_scores.reshape(-1, 1))

# Check new metrics
from sklearn.metrics import r2_score, mean_absolute_error
print(f"Calibrated R²: {r2_score(true_scores, calibrated_scores):.3f}")
print(f"Calibrated MAE: {mean_absolute_error(true_scores, calibrated_scores):.3f}")
```

### Option 2: Retrain with Different Loss (Better)
```python
# Use Huber loss instead of MSE (more robust)
from torch import nn

class RegressionModel(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base = base_model
        self.loss_fn = nn.HuberLoss(delta=0.1)
    
    def forward(self, **inputs):
        outputs = self.base(**inputs)
        # Custom loss calculation
        return outputs
```

### Option 3: Adjust Quality Score Ranges (Easiest)
```python
# In prepare_dataset.py, modify score generation
def generate_quality_labels(quality_type: str) -> float:
    if quality_type == 'coherent':
        return round(random.uniform(0.6, 0.9), 3)  # Narrower range
    elif quality_type == 'shuffled':
        return round(random.uniform(0.1, 0.4), 3)  # Narrower range
    # ... etc
```

---

## 📈 What Good Results Look Like

After fixes, you should see:

```
✅ eval_r2_score: 0.75 - 0.85
✅ eval_mae: 0.08 - 0.12
✅ eval_correlation: 0.85 - 0.92 (you already have this!)
✅ eval_accuracy_0.2: 0.85 - 0.92
```

---

## 🎯 Action Plan

1. **Run diagnostic tests** above to understand the issue
2. **Try Option 1** (calibration) first - quick fix
3. **If needed**, regenerate data with narrower score ranges
4. **Retrain** with adjusted configuration
5. **Validate** on custom D&D examples

---

## 💡 Key Insight

Your model **learned the task** correctly (0.90 correlation!) but has a **calibration issue**. This is much easier to fix than a model that didn't learn at all. The ranking is perfect, we just need to adjust the scale!

---

## 📞 Next Steps

1. Add these diagnostic cells to your notebook
2. Run the tests to see exact issue
3. Apply calibration (Option 1)
4. If results improve, save calibrated model
5. If not, try Option 3 (regenerate data)

Good luck! 🚀
