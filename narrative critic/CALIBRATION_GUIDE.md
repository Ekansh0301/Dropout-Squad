# How to Permanently Apply Calibration to Your Narrative Critic Model

## 🎯 Three Approaches to Permanent Calibration

---

## ✅ **Option 1: Save Calibrator Separately (Recommended)**

This is the cleanest and most flexible approach.

### Step 1: Train and Save Calibrator

```python
# After training your model, fit the calibrator on validation set
from sklearn.linear_model import LinearRegression
import pickle

# Get predictions
predictions = trainer.predict(val_dataset)
pred_scores = 1 / (1 + np.exp(-predictions.predictions.squeeze()))
true_scores = predictions.label_ids

# Fit calibrator
calibrator = LinearRegression()
calibrator.fit(pred_scores.reshape(-1, 1), true_scores)

# Save calibrator alongside model
import pickle
with open(f"{CONFIG['output_dir']}/calibrator.pkl", 'wb') as f:
    pickle.dump(calibrator, f)

print(f"✓ Calibrator saved: {CONFIG['output_dir']}/calibrator.pkl")
print(f"  Slope: {calibrator.coef_[0]:.4f}")
print(f"  Intercept: {calibrator.intercept_:.4f}")
```

### Step 2: Load and Use in Production

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import pickle
import torch
import numpy as np

# Load model
model = AutoModelForSequenceClassification.from_pretrained("models/narrative_critic")
tokenizer = AutoTokenizer.from_pretrained("models/narrative_critic")

# Load calibrator
with open("models/narrative_critic/calibrator.pkl", 'rb') as f:
    calibrator = pickle.load(f)

def predict_quality(text):
    """Predict with calibration applied."""
    # Tokenize
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
    
    # Get raw prediction
    with torch.no_grad():
        outputs = model(**inputs)
        raw_score = torch.sigmoid(outputs.logits).item()
    
    # Apply calibration
    calibrated_score = calibrator.predict([[raw_score]])[0]
    calibrated_score = np.clip(calibrated_score, 0.0, 1.0)  # Ensure [0, 1] range
    
    return calibrated_score

# Use it
score = predict_quality("You enter a dimly lit tavern...")
print(f"Calibrated Quality Score: {score:.3f}")
```

### Step 3: Batch Prediction with Calibration

```python
def predict_batch(texts):
    """Batch prediction with calibration."""
    # Tokenize all texts
    inputs = tokenizer(texts, return_tensors="pt", padding=True, 
                      truncation=True, max_length=128)
    
    # Get raw predictions
    with torch.no_grad():
        outputs = model(**inputs)
        raw_scores = torch.sigmoid(outputs.logits).squeeze().numpy()
    
    # Handle single example case
    if raw_scores.ndim == 0:
        raw_scores = raw_scores.reshape(1)
    
    # Apply calibration
    calibrated_scores = calibrator.predict(raw_scores.reshape(-1, 1))
    calibrated_scores = np.clip(calibrated_scores, 0.0, 1.0)
    
    return calibrated_scores

# Use it
texts = ["Text 1", "Text 2", "Text 3"]
scores = predict_batch(texts)
```

---

## 🔧 **Option 2: Create a Wrapper Class**

Encapsulate model + calibrator in a single class for easy deployment.

```python
import torch
import pickle
import numpy as np
from transformers import AutoModelForSequenceClassification, AutoTokenizer

class CalibratedNarrativeCritic:
    """Narrative Critic with built-in calibration."""
    
    def __init__(self, model_path):
        """Load model and calibrator."""
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # Load calibrator
        calibrator_path = f"{model_path}/calibrator.pkl"
        try:
            with open(calibrator_path, 'rb') as f:
                self.calibrator = pickle.load(f)
            print(f"✓ Loaded calibrator from {calibrator_path}")
        except FileNotFoundError:
            print(f"⚠️  No calibrator found at {calibrator_path}")
            print("   Using raw predictions (no calibration)")
            self.calibrator = None
        
        self.model.eval()
        
        # Move to GPU if available
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
    
    def predict(self, text):
        """Predict quality score for a single text."""
        return self.predict_batch([text])[0]
    
    def predict_batch(self, texts):
        """Predict quality scores for batch of texts."""
        # Tokenize
        inputs = self.tokenizer(
            texts, 
            return_tensors="pt", 
            padding=True,
            truncation=True, 
            max_length=128
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Get raw predictions
        with torch.no_grad():
            outputs = self.model(**inputs)
            raw_scores = torch.sigmoid(outputs.logits).squeeze()
        
        # Move to CPU and convert to numpy
        raw_scores = raw_scores.cpu().numpy()
        
        # Handle single example
        if raw_scores.ndim == 0:
            raw_scores = raw_scores.reshape(1)
        
        # Apply calibration if available
        if self.calibrator is not None:
            scores = self.calibrator.predict(raw_scores.reshape(-1, 1))
            scores = np.clip(scores, 0.0, 1.0)
        else:
            scores = raw_scores
        
        return scores
    
    def __call__(self, text):
        """Make the class callable."""
        return self.predict(text)


# Usage
critic = CalibratedNarrativeCritic("models/narrative_critic")

# Single prediction
score = critic("You enter a dimly lit tavern...")
print(f"Quality: {score:.3f}")

# Batch prediction
scores = critic.predict_batch(["Text 1", "Text 2", "Text 3"])
print(f"Scores: {scores}")

# Callable interface
score = critic("Some narrative text...")
```

### Save the Wrapper Class

```python
# Save as narrative_critic_wrapper.py
# Then use:
from narrative_critic_wrapper import CalibratedNarrativeCritic
critic = CalibratedNarrativeCritic("models/narrative_critic")
```

---

## 🏗️ **Option 3: Modify Model Architecture (Advanced)**

Add calibration layer directly into the model architecture.

```python
import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification

class CalibratedRegressionModel(nn.Module):
    """DeBERTa with calibration layer."""
    
    def __init__(self, base_model_name, calibration_slope=1.0, calibration_intercept=0.0):
        super().__init__()
        
        # Load base model
        self.base_model = AutoModelForSequenceClassification.from_pretrained(
            base_model_name,
            num_labels=1,
            problem_type="regression"
        )
        
        # Calibration parameters (learnable or fixed)
        self.calibration_slope = nn.Parameter(
            torch.tensor([calibration_slope]), 
            requires_grad=False  # Fixed after fitting
        )
        self.calibration_intercept = nn.Parameter(
            torch.tensor([calibration_intercept]),
            requires_grad=False
        )
    
    def forward(self, **inputs):
        # Get base model output
        outputs = self.base_model(**inputs)
        
        # Apply sigmoid
        probs = torch.sigmoid(outputs.logits)
        
        # Apply calibration
        calibrated = self.calibration_slope * probs + self.calibration_intercept
        calibrated = torch.clamp(calibrated, 0.0, 1.0)
        
        # Return in same format
        outputs.logits = calibrated
        return outputs
    
    def set_calibration(self, slope, intercept):
        """Set calibration parameters."""
        self.calibration_slope.data = torch.tensor([slope])
        self.calibration_intercept.data = torch.tensor([intercept])
        print(f"✓ Calibration set: y = {slope:.4f} * x + {intercept:.4f}")


# Usage after training
# 1. Load your trained model
base_model = AutoModelForSequenceClassification.from_pretrained("models/narrative_critic")

# 2. Create calibrated wrapper
calibrated_model = CalibratedRegressionModel("models/narrative_critic")
calibrated_model.base_model = base_model  # Use trained weights

# 3. Fit calibration on validation set
from sklearn.linear_model import LinearRegression
calibrator = LinearRegression()
calibrator.fit(pred_scores.reshape(-1, 1), true_scores)

# 4. Set calibration parameters
calibrated_model.set_calibration(
    slope=calibrator.coef_[0],
    intercept=calibrator.intercept_
)

# 5. Save the calibrated model
calibrated_model.save_pretrained("models/narrative_critic_calibrated")
tokenizer.save_pretrained("models/narrative_critic_calibrated")
```

---

## 📦 **Complete Save/Load Example (Option 1 - Recommended)**

### In Your Kaggle Notebook (After Training)

```python
# ===== AFTER EVALUATION SECTION =====

print("\n" + "="*70)
print("FITTING AND SAVING CALIBRATOR")
print("="*70)

from sklearn.linear_model import LinearRegression
import pickle
import json

# Get predictions on validation set
predictions = trainer.predict(val_dataset)
pred_scores = 1 / (1 + np.exp(-predictions.predictions.squeeze()))
true_scores = predictions.label_ids

# Fit calibrator
calibrator = LinearRegression()
calibrator.fit(pred_scores.reshape(-1, 1), true_scores)

# Calculate calibrated metrics
calibrated_scores = calibrator.predict(pred_scores.reshape(-1, 1))
calibrated_scores = np.clip(calibrated_scores, 0.0, 1.0)

cal_metrics = {
    'calibration_slope': float(calibrator.coef_[0]),
    'calibration_intercept': float(calibrator.intercept_),
    'calibrated_mae': float(mean_absolute_error(true_scores, calibrated_scores)),
    'calibrated_r2': float(r2_score(true_scores, calibrated_scores)),
    'calibrated_correlation': float(np.corrcoef(true_scores, calibrated_scores)[0, 1])
}

print(f"\n✓ Calibration fitted:")
print(f"  Formula: y = {cal_metrics['calibration_slope']:.4f} * pred + {cal_metrics['calibration_intercept']:.4f}")
print(f"\n✓ Calibrated metrics:")
print(f"  MAE: {cal_metrics['calibrated_mae']:.4f}")
print(f"  R²: {cal_metrics['calibrated_r2']:.4f}")
print(f"  Correlation: {cal_metrics['calibrated_correlation']:.4f}")

# Save calibrator
output_dir = CONFIG['output_dir']
with open(f"{output_dir}/calibrator.pkl", 'wb') as f:
    pickle.dump(calibrator, f)
print(f"\n✓ Saved: {output_dir}/calibrator.pkl")

# Save calibration metadata
with open(f"{output_dir}/calibration_info.json", 'w') as f:
    json.dump(cal_metrics, f, indent=2)
print(f"✓ Saved: {output_dir}/calibration_info.json")

# Also save as numpy arrays for other languages
np.save(f"{output_dir}/calibration_params.npy", 
        np.array([calibrator.coef_[0], calibrator.intercept_]))
print(f"✓ Saved: {output_dir}/calibration_params.npy")

print("\n" + "="*70)
print("✓ CALIBRATOR SAVED SUCCESSFULLY")
print("="*70)
```

### In Your Production Code

```python
# production_inference.py

import torch
import pickle
import numpy as np
from transformers import AutoModelForSequenceClassification, AutoTokenizer

class NarrativeCritic:
    """Production-ready Narrative Critic with calibration."""
    
    def __init__(self, model_path):
        # Load model and tokenizer
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model.eval()
        
        # Load calibrator
        with open(f"{model_path}/calibrator.pkl", 'rb') as f:
            self.calibrator = pickle.load(f)
        
        # Move to GPU if available
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        print(f"✓ Narrative Critic loaded from {model_path}")
        print(f"  Device: {self.device}")
    
    def score(self, text):
        """Get calibrated quality score for text."""
        inputs = self.tokenizer(text, return_tensors="pt", 
                               truncation=True, max_length=128)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            raw_score = torch.sigmoid(outputs.logits).item()
        
        # Apply calibration
        calibrated = self.calibrator.predict([[raw_score]])[0]
        return float(np.clip(calibrated, 0.0, 1.0))
    
    def score_batch(self, texts):
        """Get calibrated scores for batch of texts."""
        inputs = self.tokenizer(texts, return_tensors="pt", padding=True,
                               truncation=True, max_length=128)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            raw_scores = torch.sigmoid(outputs.logits).squeeze().cpu().numpy()
        
        if raw_scores.ndim == 0:
            raw_scores = raw_scores.reshape(1)
        
        calibrated = self.calibrator.predict(raw_scores.reshape(-1, 1))
        return np.clip(calibrated, 0.0, 1.0).tolist()


# Use it
if __name__ == "__main__":
    critic = NarrativeCritic("models/narrative_critic")
    
    # Test examples
    examples = [
        "You enter a dimly lit tavern filled with rowdy adventurers...",
        "You see room. There is door.",
        "The dragon roars. The dragon roars. The dragon roars."
    ]
    
    for text in examples:
        score = critic.score(text)
        print(f"Score: {score:.3f} - {text[:50]}...")
```

---

## 🎯 **Recommendation**

**Use Option 1** (Save Calibrator Separately):

✅ **Pros:**
- Simple to implement
- Easy to update calibration independently
- No model architecture changes
- Works with any framework
- Can version calibrators separately

❌ **Cons:**
- Need to distribute two files (model + calibrator)
- Slightly more code in inference

**Use Option 2** (Wrapper Class) if you want a cleaner interface for your team.

**Use Option 3** (Modified Architecture) only if you need a single-file solution and are comfortable with custom model code.

---

## 📋 **Complete Checklist**

- [ ] Add calibration fitting code to your notebook
- [ ] Train model and fit calibrator on validation set
- [ ] Save calibrator.pkl alongside model
- [ ] Save calibration_info.json for documentation
- [ ] Test loading in separate script
- [ ] Verify calibrated predictions match expected range
- [ ] Update your inference code to use calibration
- [ ] Document calibration in your model card/README

---

## 🚀 **Next Steps**

1. Add the calibration code to your Kaggle notebook
2. Run it to generate and save the calibrator
3. Download both the model and calibrator.pkl
4. Use the production code above in your D&D system
5. Enjoy properly calibrated quality scores! ✨
