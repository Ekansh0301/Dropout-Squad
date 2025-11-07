# Complete Model Evaluation Report
## Narrative Critic Model (model_jayant)

**Evaluation Date:** November 7, 2025  
**Model Path:** `./model_jayant`  
**Validation Dataset:** 3,000 examples from ROCStories  
**Training Duration:** 20 epochs, ~85 minutes

---

## 🚨 CRITICAL FINDINGS

### Model Status: **MARGINAL** ⚠️

**The model is PARTIALLY COLLAPSED and NOT suitable for production use without fixes.**

### Key Issues Identified:

1. **Very Low Prediction Diversity**
   - Standard deviation: **0.058** (needs > 0.12)
   - Prediction range: **0.172** (only 0.535 to 0.707)
   - All predictions cluster around **0.60**
   
2. **Negative R² Score**
   - R²: **-0.1248** (worse than predicting the mean!)
   - Model is not learning meaningful patterns
   
3. **High Mean Absolute Error**
   - MAE: **0.2579** (target < 0.20)
   - 69% of predictions within ±0.3, but only 8% within ±0.1

4. **Misleading Correlation**
   - Pearson correlation: **0.8907** (seems good but misleading!)
   - High correlation occurs because validation set has consistent score ranges per type
   - Model just predicts values near dataset mean, not actual discrimination

---

## 📊 Detailed Metrics

### Overall Performance

| Metric | Value | Status | Target |
|--------|-------|--------|--------|
| **Mean Absolute Error (MAE)** | 0.2579 | ❌ Bad | < 0.20 |
| **Mean Squared Error (MSE)** | 0.0806 | ❌ Bad | < 0.05 |
| **Root Mean Squared Error (RMSE)** | 0.2839 | ❌ Bad | < 0.22 |
| **R² Score** | -0.1248 | ❌ Very Bad | > 0.30 |
| **Pearson Correlation** | 0.8907 | ⚠️ Misleading | > 0.70 |
| **Spearman Correlation** | 0.8302 | ⚠️ Misleading | > 0.70 |

### Prediction Statistics

| Statistic | Value | Interpretation |
|-----------|-------|----------------|
| Mean | 0.6054 | Predicting near dataset mean (0.425) after sigmoid |
| Median | 0.6000 | Confirms clustering |
| Std Dev | 0.0580 | **CRITICAL: Too low!** (needs > 0.12) |
| Min | 0.5350 | Narrow range |
| Max | 0.7066 | Narrow range |
| Range | 0.1716 | Only 17% of possible range used |

### Accuracy Within Thresholds

| Threshold | Accuracy | Status |
|-----------|----------|--------|
| ±0.1 | 8.00% | ❌ Terrible |
| ±0.2 | 31.00% | ⚠️ Poor |
| ±0.3 | 69.23% | ⚠️ Marginal |

---

## 🎯 Per-Type Analysis

### Performance by Quality Type

| Type | Count | Pred Mean | Pred Std | Label Mean | MAE | R² |
|------|-------|-----------|----------|------------|-----|-----|
| **Coherent** | 731 | 0.6943 | 0.0268 | 0.8494 | 0.1552 | -3.28 |
| **Repetitive** | 769 | 0.5771 | 0.0004 | 0.2942 | 0.2829 | -24.34 |
| **Shuffled** | 719 | 0.5493 | 0.0352 | 0.1521 | 0.3972 | -21.18 |
| **Truncated** | 781 | 0.6017 | 0.0034 | 0.4003 | 0.2014 | -11.92 |

### Critical Observations:

1. **Repetitive Type:** Pred std = 0.0004 (**COMPLETELY COLLAPSED!**)
   - All repetitive stories get nearly identical scores (~0.577)
   - Model cannot distinguish between repetitive examples
   
2. **Truncated Type:** Pred std = 0.0034 (**COLLAPSED!**)
   - All truncated stories get ~0.602
   - Zero discriminative power
   
3. **Coherent Type:** Pred std = 0.0268 (low but better)
   - Best performance but still poor (R² = -3.28)
   - Predicts ~0.694 but should be ~0.849
   
4. **Shuffled Type:** Pred std = 0.0352 (low)
   - Predicts ~0.549 but should be ~0.152
   - Massive systematic error

**Conclusion:** Model has learned to output approximately constant values per type, but NOT learned to discriminate within types or accurately predict scores.

---

## 📉 Classification Analysis (Binned Scores)

When treating as 4-class classification problem:

### Confusion Matrix

|  | Predicted Low | Predicted Med-Low | Predicted Med-High | Predicted High |
|---|---|---|---|---|
| **True Low** | 0 | 0 | 1137 | 7 |
| **True Med-Low** | 0 | 0 | 1125 | 0 |
| **True Med-High** | 0 | 0 | 2 | 0 |
| **True High** | 0 | 0 | 277 | 452 |

### Findings:

- **Accuracy: 15%** (essentially random for 4 classes)
- Model predicts "medium-high" for 89% of examples
- Only "high" class has any recall (62%)
- Low and medium-low classes: **0% recall**

**Interpretation:** Model outputs are so narrow (0.535-0.707) that almost everything falls into medium-high bin. Cannot distinguish low-quality from high-quality narratives.

---

## 🔍 Why Correlation is Misleading

**How can correlation be 89% if the model is collapsed?**

### The Explanation:

1. **Validation set has consistent ranges per type:**
   - Coherent: mostly 0.7-1.0
   - Repetitive: mostly 0.2-0.4
   - Shuffled: mostly 0.0-0.3
   - Truncated: mostly 0.3-0.5

2. **Model learned type-level means:**
   - Coherent → 0.694 (high-ish)
   - Repetitive → 0.577 (mid-ish)
   - Shuffled → 0.549 (mid-low-ish)
   - Truncated → 0.602 (mid-ish)

3. **Correlation measures ranking, not accuracy:**
   - Since types have different target ranges
   - And model outputs different (narrow) ranges per type
   - Correlation can be high even with zero discrimination

**Analogy:** Like a broken thermometer that shows 70°F for all temperatures between 60-80°F, 50°F for 40-60°F, etc. It will correlate with actual temperature, but is useless for precise measurement.

---

## 📈 Example Predictions

### Good Example (Coherent):
```
Text: "The school had a ceremony. The principal stood up..."
True: 0.7110
Pred: 0.7008
Error: 0.0102 ✅
```

### Bad Example (Shuffled):
```
Text: "Jason was chosen to lead the team. Jason was the best athlete..."
True: 0.0600 (very bad quality)
Pred: 0.5408 (mediocre quality)
Error: 0.4808 ❌❌❌
```

### Bad Example (Repetitive):
```
Text: "I called my son today. I called my son today. He said that he was..."
True: 0.3370
Pred: 0.5768
Error: 0.2398 ❌
```

**Pattern:** Model struggles most with low-quality examples, predicting them as mediocre instead of bad.

---

## 🔧 Root Cause Analysis

### Why Did This Model Collapse?

Based on training history (20 epochs, 85 minutes):

1. **Learning rate too high:**
   - Model was trained with higher LR than ultra-conservative 3e-6
   - Gradients pushed predictions toward dataset mean
   
2. **Insufficient warmup:**
   - Early training may have caused gradient explosions
   - Model settled into predicting near-mean values
   
3. **Too many epochs without early stopping:**
   - 20 epochs is excessive if model collapsed early
   - Should have stopped when MAE stopped improving
   
4. **No collapse detection during training:**
   - pred_std and R² weren't monitored
   - Training continued despite collapse

### Evidence from Training Metrics:

```json
{
  "train_loss": 0.0112,  // Very low (appears to be learning)
  "eval_loss": 0.0150,   // Low (no overfitting)
  "eval_mae": 0.2579,    // HIGH (not actually learning well)
  "eval_r2": -0.1248,    // NEGATIVE (worse than baseline)
  "eval_correlation": 0.8907  // High but misleading
}
```

**Takeaway:** Loss can decrease while model collapses! Must monitor MAE, R², and pred_std.

---

## ✅ What Works / ❌ What Doesn't

### ✅ Positives:

1. Model runs and makes predictions (not completely broken)
2. Coherent stories get slightly higher scores on average
3. Training completed without crashes
4. Tokenizer and model architecture are correct

### ❌ Critical Issues:

1. **Collapsed predictions** (std = 0.058)
2. **Negative R²** (worse than predicting mean)
3. **Cannot distinguish quality levels** within types
4. **Systematic bias** toward predicting ~0.60 for everything
5. **Useless for RL reward signal** (would give similar rewards to good and bad responses)

---

## 🎯 Recommendations

### Immediate Actions:

1. **DO NOT USE THIS MODEL IN PRODUCTION**
   - It will give similar scores to high and low quality narratives
   - RL training would not learn meaningful improvements
   
2. **RETRAIN from scratch** with:
   - Learning rate: **3e-6** (ultra-conservative)
   - Batch size: **8**
   - Warmup: **30%**
   - Gradient clipping: **0.3**
   - Early stopping on **MAE** (not loss)
   - Monitor **pred_std** every eval (must be > 0.10)

3. **Add collapse detection** during training:
   - Stop if pred_std < 0.08 after step 1000
   - Stop if R² stays negative after epoch 2
   - Stop if MAE doesn't improve for 5 evals

### Validation Strategy:

After retraining, verify:
- [ ] pred_std > 0.15 (good diversity)
- [ ] R² > 0.30 (positive learning)
- [ ] MAE < 0.20 (acceptable accuracy)
- [ ] Predictions span 0.1 to 0.9 range
- [ ] Low quality stories get < 0.3 scores
- [ ] High quality stories get > 0.7 scores

### Alternative Approaches (if retraining fails):

1. **Freeze DeBERTa layers, train only head:**
   ```python
   model.deberta.requires_grad_(False)
   # Train for 1 epoch
   model.deberta.requires_grad_(True)
   # Continue training with 1e-6 LR
   ```

2. **Use smaller model:**
   - Try `microsoft/deberta-v3-small` (44M params)
   - Smaller models less prone to collapse
   
3. **Change architecture:**
   - Add intermediate layers with dropout
   - Use multiple regression heads with ensemble

4. **Synthetic data augmentation:**
   - Generate more extreme examples
   - Add noise to prevent overfitting to type means

---

## 📁 Output Files

Evaluation results saved to: `./evaluation_results/`

1. **evaluation_results.json** - Complete metrics in JSON format
2. **evaluation_plots.png** - 4-panel visualization:
   - Scatter plot with regression line
   - Score distribution comparison
   - Error distribution histogram
   - Per-type box plots
3. **confusion_matrix.png** - Heatmap of classification confusion

---

## 🎓 Key Learnings

### For Future Training:

1. **Correlation ≠ Accuracy** for regression tasks
   - Always check R², MAE, and pred_std alongside correlation
   
2. **Monitor prediction diversity** during training
   - If pred_std drops below 0.10, model is collapsing
   
3. **DeBERTa needs extremely low LR** for regression
   - 3e-6 or even 1e-6 (much lower than classification)
   
4. **Early stopping crucial** for regression
   - Use MAE or R² as metric, not loss
   
5. **Test on diverse examples** before deployment
   - High/low quality pairs should have very different scores

---

## 📞 Next Steps

1. **Review this report** carefully
2. **Run diagnostic_collapse.py** to confirm findings
3. **Update notebook** with ultra-conservative config (already done)
4. **Retrain on Kaggle** with new configuration
5. **Re-evaluate** after training completes
6. **Apply calibration** only if model is healthy

**Do not proceed to calibration or production deployment until the model shows:**
- ✅ pred_std > 0.15
- ✅ R² > 0.30
- ✅ MAE < 0.20

---

## Summary

Your current model (`model_jayant`) has **collapsed** and is **not usable**. It predicts values clustered around 0.60 for nearly all inputs, with insufficient diversity to discriminate between quality levels. The 89% correlation is misleading—it comes from the model predicting slightly different constant values for each quality type, not from true understanding.

**Action Required:** Retrain with ultra-conservative configuration (LR=3e-6, batch=8, warmup=30%).

---

*Report generated by: complete_model_evaluation.py*  
*Model evaluated: model_jayant (20 epochs)*  
*Evaluation date: November 7, 2025*
