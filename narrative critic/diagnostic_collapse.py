"""
Diagnose Model Collapse
Run this to check if your model has collapsed to predicting constants
"""

import json
import numpy as np
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

print("="*70)
print("MODEL COLLAPSE DIAGNOSTIC")
print("="*70)

# Load model
model_path = "models/narrative_critic"  # Adjust path
print(f"\nLoading model from: {model_path}")

try:
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model.eval()
    print("✓ Model loaded")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    exit(1)

# Test on diverse examples
test_texts = [
    # High quality
    "The ancient library stretched endlessly before you, its towering shelves groaning under countless leather-bound tomes. Dust motes danced in golden sunlight filtering through stained glass windows, casting rainbow patterns across worn stone floors. You breathe in the scent of old parchment and aged wood, feeling the weight of centuries of knowledge surrounding you.",
    
    # Medium quality
    "You enter a tavern. It's busy and noisy. People are drinking and talking. The bartender looks at you.",
    
    # Low quality
    "Room. Door. Table. Chair. Window.",
    
    # Repetitive
    "The dragon roars. The dragon roars. The dragon roars. The dragon roars. The dragon roars.",
    
    # Truncated
    "You see the",
    
    # Another high quality
    "As you approach the massive oak doors, you notice intricate carvings depicting ancient battles. The brass handles are worn smooth by countless hands over the years.",
    
    # Random text
    "The purple elephant jumped over the singing pancake while the moon discussed philosophy with a potato.",
    
    # Very short
    "Hi.",
    
    # Numbers
    "1 2 3 4 5 6 7 8 9 10",
    
    # Empty-ish
    ".",
]

print("\n" + "="*70)
print("TESTING ON DIVERSE EXAMPLES")
print("="*70)

raw_scores = []
sigmoid_scores = []

for i, text in enumerate(test_texts, 1):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
    
    with torch.no_grad():
        outputs = model(**inputs)
        raw_logit = outputs.logits.item()
        sigmoid_score = torch.sigmoid(outputs.logits).item()
    
    raw_scores.append(raw_logit)
    sigmoid_scores.append(sigmoid_score)
    
    print(f"\n{i}. {text[:60]}...")
    print(f"   Raw logit: {raw_logit:.4f}")
    print(f"   Sigmoid score: {sigmoid_score:.4f}")

# Analysis
print("\n" + "="*70)
print("COLLAPSE ANALYSIS")
print("="*70)

raw_std = np.std(raw_scores)
sigmoid_std = np.std(sigmoid_scores)
raw_range = np.max(raw_scores) - np.min(raw_scores)
sigmoid_range = np.max(sigmoid_scores) - np.min(sigmoid_scores)

print(f"\nRaw logits:")
print(f"  Mean: {np.mean(raw_scores):.4f}")
print(f"  Std: {raw_std:.4f}")
print(f"  Range: {raw_range:.4f}")
print(f"  Min: {np.min(raw_scores):.4f}")
print(f"  Max: {np.max(raw_scores):.4f}")

print(f"\nSigmoid scores:")
print(f"  Mean: {np.mean(sigmoid_scores):.4f}")
print(f"  Std: {sigmoid_std:.4f}")
print(f"  Range: {sigmoid_range:.4f}")
print(f"  Min: {np.min(sigmoid_scores):.4f}")
print(f"  Max: {np.max(sigmoid_scores):.4f}")

print("\n" + "="*70)
print("DIAGNOSIS")
print("="*70)

collapsed = False

if sigmoid_std < 0.05:
    print("\n❌ SEVERE MODEL COLLAPSE DETECTED")
    print(f"   Sigmoid std ({sigmoid_std:.4f}) is very low")
    print("   Model is predicting nearly constant values!")
    collapsed = True
elif sigmoid_std < 0.10:
    print("\n⚠️  MODERATE MODEL COLLAPSE")
    print(f"   Sigmoid std ({sigmoid_std:.4f}) is low")
    print("   Model has limited discriminative power")
    collapsed = True
elif sigmoid_range < 0.2:
    print("\n⚠️  LIMITED OUTPUT RANGE")
    print(f"   Sigmoid range ({sigmoid_range:.4f}) is narrow")
    print("   Model not using full 0-1 range")
    collapsed = True
else:
    print("\n✅ NO COLLAPSE DETECTED")
    print(f"   Sigmoid std ({sigmoid_std:.4f}) is healthy")
    print(f"   Sigmoid range ({sigmoid_range:.4f}) is good")

if collapsed:
    print("\n" + "="*70)
    print("RECOMMENDED ACTIONS")
    print("="*70)
    print("\n1. ⚠️  DO NOT USE THIS MODEL - predictions are meaningless")
    print("\n2. 🔧 RETRAIN with fixed configuration:")
    print("   - Lower learning rate (1e-5 instead of 3e-5)")
    print("   - More warmup (0.2 instead of 0.1)")
    print("   - Stronger gradient clipping (0.5 instead of 1.0)")
    print("   - Use critic_config_fixed.yaml")
    
    print("\n3. 📊 CHECK DATA:")
    print("   - Verify dataset has good score distribution")
    print("   - Ensure labels are correctly assigned")
    print("   - Check for data loading errors")
    
    print("\n4. 🧪 TRAINING TIPS:")
    print("   - Monitor eval_mae during training")
    print("   - Stop if predictions become constant")
    print("   - Use early stopping")
    print("   - Check output variance every 100 steps")
    
    print("\n5. 🔄 ALTERNATIVE APPROACHES:")
    print("   - Try classification instead of regression")
    print("   - Use smaller model (distilbert)")
    print("   - Add output activation constraints")
    print("   - Use focal loss or Huber loss")

print("\n" + "="*70)
print("✓ DIAGNOSTIC COMPLETE")
print("="*70)

if collapsed:
    print("\n⚠️  Model has collapsed - retrain required!")
else:
    print("\n✅ Model looks healthy!")
