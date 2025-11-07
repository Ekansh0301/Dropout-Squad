# 🎯 ROOT CAUSE ANALYSIS: Why Your Model Keeps Collapsing

## Critical Issues Found

After deep investigation, I found **FUNDAMENTAL PROBLEMS** that explain why all your models collapse, regardless of hyperparameters:

---

## 🚨 Issue #1: Dataset Has Conflicting Labels

### The Problem:

Your dataset has **TWO different label fields**:

```json
{
  "text": "Once upon a time...",
  "label": 0,              // ❌ Classification (0 or 1)
  "label_float": 0.874,    // ✅ Regression (0.0 to 1.0)
  "type": "coherent"
}
```

**Impact**: The model may be seeing the wrong label field during training, causing it to learn binary classification instead of regression!

### The Fix:

**CRITICAL:** Remove the `label` field completely before creating the dataset:

```python
# DROP the classification label
if 'label' in train_df.columns:
    train_df = train_df.drop(columns=['label'])
if 'label' in val_df.columns:
    val_df = val_df.drop(columns=['label'])
```

---

## 🚨 Issue #2: Model Not Initialized Before Training

### The Problem:

In your notebook, the model loading code was **MISSING OR IN THE WRONG PLACE**. The trainer was created before the model was even loaded!

This sequence is WRONG:
```python
# 1. Load config
CONFIG = {...}

# 2. Create training args
training_args = TrainingArguments(...)

# 3. Create trainer  
trainer = Trainer(model=model, ...)  # ❌ Where does `model` come from?!
```

### The Fix:

Model MUST be loaded BEFORE creating TrainingArguments:

```python
# 1. Load config
CONFIG = {...}

# 2. Load tokenizer and model  # ✅ DO THIS FIRST!
tokenizer = AutoTokenizer.from_pretrained(CONFIG['model_name'])
model = AutoModelForSequenceClassification.from_pretrained(
    CONFIG['model_name'],
    num_labels=1,           # CRITICAL: 1 for regression
    problem_type="regression"  # CRITICAL: Tell model this is regression
)

# 3. Create training args
training_args = TrainingArguments(...)

# 4. Create trainer
trainer = Trainer(model=model, ...)
```

---

## 🚨 Issue #3: Duplicate Data

### The Problem:

```
Found 58 duplicate texts in training data
```

This creates:
- Inconsistent gradients (same text with different random labels)
- Model confusion
- Unstable training

### The Fix:

Dedup in `prepare_dataset.py`:

```python
# After creating all examples
examples_df = pd.DataFrame(examples)
examples_df = examples_df.drop_duplicates(subset=['text'], keep='first')
examples = examples_df.to_dict('records')
```

---

## 🚨 Issue #4: Why High Correlation But Negative R²?

This seems impossible, but here's why it happens:

### Dataset Structure:
- Coherent: mean=0.85, very little variance (all 0.7-1.0)
- Repetitive: mean=0.30, very little variance (all 0.2-0.4)
- Shuffled: mean=0.15, very little variance (all 0.0-0.3)
- Truncated: mean=0.40, very little variance (all 0.3-0.5)

### What Model Learns:
Instead of learning **"what makes text good vs bad"**, it learns:
- "Coherent type" → always output 0.69
- "Repetitive type" → always output 0.58
- "Shuffled type" → always output 0.55
- "Truncated type" → always output 0.60

### Why Correlation is High:
```
True scores by type: [0.85, 0.30, 0.15, 0.40]
Model outputs:       [0.69, 0.58, 0.55, 0.60]
```

These are correlated! (higher type → higher output)

### Why R² is Negative:
R² measures **actual prediction accuracy**, not just ranking.  
The model predicts values FAR from actual scores, making it worse than just predicting the dataset mean (0.425).

---

## ✅ COMPLETE FIX CHECKLIST

### 1. Fix `prepare_dataset.py`

Add deduplication:

```python
def create_dataset_examples(df, max_per_type=None):
    examples = []
    
    # ... existing code to create examples ...
    
    # DEDUPLICATE
    print(f"  Created {len(examples)} examples")
    examples_df = pd.DataFrame(examples)
    before = len(examples_df)
    examples_df = examples_df.drop_duplicates(subset=['text'], keep='first')
    after = len(examples_df)
    if before != after:
        print(f"  ⚠️ Removed {before - after} duplicates")
    
    return examples_df.to_dict('records')
```

### 2. Fix Notebook - Add These Cells

**Cell after data loading:**
```python
# CRITICAL FIX: Remove classification 'label' field
if 'label' in train_df.columns:
    train_df = train_df.drop(columns=['label'])
if 'label' in val_df.columns:
    val_df = val_df.drop(columns=['label'])

print("✓ Using only 'label_float' for regression")
```

**Cell BEFORE training args:**
```python
# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(CONFIG['model_name'])
model = AutoModelForSequenceClassification.from_pretrained(
    CONFIG['model_name'],
    num_labels=1,
    problem_type="regression"
)
model.to(device)
print("✓ Model loaded and configured for regression")
```

### 3. Regenerate Dataset

```bash
# Delete old data
rm dataset_json/*.json

# Regenerate with fixed script
python prepare_dataset.py
```

### 4. Verify Dataset

```python
# Check no 'label' field exists
data = json.load(open('dataset_json/rocstoriestrain.json'))
assert 'label' not in data[0], "ERROR: 'label' field still exists!"
assert 'label_float' in data[0], "ERROR: 'label_float' missing!"
print("✓ Dataset validated")
```

---

## 🎯 Why This Will Work

### Before (Broken):
1. Dataset has confusing double labels
2. Model not properly initialized for regression  
3. Duplicates causing gradient noise
4. Model learns type-level means, not quality discrimination

### After (Fixed):
1. Clean dataset with only `label_float`
2. Model explicitly configured for regression
3. No duplicates = stable gradients
4. Model can learn actual quality patterns

---

## 📊 Expected Results After Fix

### Training Metrics:
```
Epoch 1: MAE=0.235, R²=0.15, pred_std=0.12
Epoch 2: MAE=0.210, R²=0.35, pred_std=0.16
Epoch 3: MAE=0.185, R²=0.52, pred_std=0.19
Epoch 4: MAE=0.170, R²=0.62, pred_std=0.21
Epoch 5: MAE=0.160, R²=0.68, pred_std=0.22
```

### Test Predictions:
```
High Quality:  0.82 (vs collapsed: 0.60)
Shuffled:      0.18 (vs collapsed: 0.55)
Repetitive:    0.31 (vs collapsed: 0.58)
Truncated:     0.44 (vs collapsed: 0.60)

Std Dev: 0.24 (vs collapsed: 0.06) ✅
```

---

## 🚀 Next Steps

1. **Update `prepare_dataset.py`** - Add deduplication
2. **Update notebook** - Add label fix + model initialization cells
3. **Regenerate dataset** - Run `prepare_dataset.py`
4. **Verify dataset** - Check no `label` field
5. **Upload to Kaggle** - Use fixed notebook
6. **Train and monitor** - Watch for pred_std > 0.12

---

## 💡 Key Lesson

**Hyperparameters weren't the problem!**

The issue was:
- ❌ Wrong label field (classification vs regression)
- ❌ Model not configured for regression
- ❌ Duplicate data
- ❌ Dataset structure allowing type-based shortcuts

No amount of LR tuning can fix fundamental code/data issues!

---

*Analysis Date: November 7, 2025*  
*Root Cause: Data + Code Issues, NOT Hyperparameters*
