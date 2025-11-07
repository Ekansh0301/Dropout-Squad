# Narrative Critic - Quick Start Guide

## 🚀 Quick Setup (5 minutes)

### Option 1: Local Training
```bash
# 1. Prepare dataset
python prepare_dataset.py

# 2. Train model
python narrative_critic.py

# Output: models/narrative_critic/
```

### Option 2: Kaggle Training
```bash
# 1. Prepare dataset
python prepare_dataset.py

# 2. Upload dataset_json/ to Kaggle as "ROCStoriesData"

# 3. Run kaggle_narrative_critic_training.ipynb on Kaggle

# 4. Download trained model
```

## 📊 What Gets Created

### From `prepare_dataset.py`:
```
dataset_json/
├── rocstoriestrain.json  (~27K examples)
└── rocstoriesval.json    (~3K examples)
```

**Data Format**:
```json
{
  "text": "Dan's parents were overweight. Dan was...",
  "label": 1,
  "label_float": 0.85,
  "source": "rocstories",
  "type": "coherent",
  "story_id": "9a51198e-96f1-42c3-b09d-a3e1e067d803"
}
```

### From Training:
```
models/narrative_critic/
├── pytorch_model.bin        (Model weights)
├── config.json             (Model config)
├── tokenizer.json          (Tokenizer)
├── training_history.json   (Training logs)
├── eval_metrics.json       (Performance)
├── training_loss.png       (Loss curves)
└── eval_metrics_plot.png   (Metrics)
```

## 🎯 Dataset Breakdown

| Type | Count | Score Range | Purpose |
|------|-------|-------------|---------|
| Coherent | ~7,500 | 0.7-1.0 | High-quality examples |
| Shuffled | ~7,500 | 0.0-0.3 | Poor coherence |
| Repetitive | ~7,500 | 0.2-0.4 | Repetition detection |
| Truncated | ~7,500 | 0.3-0.5 | Incompleteness |

## 💡 Quick Usage

### Load and Use Model
```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

# Load
model = AutoModelForSequenceClassification.from_pretrained("models/narrative_critic")
tokenizer = AutoTokenizer.from_pretrained("models/narrative_critic")

# Predict
def score_text(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
        return torch.sigmoid(outputs.logits).item()

# Test
print(score_text("You enter a dimly lit tavern..."))  # ~0.75
print(score_text("You see room."))                     # ~0.25
```

### Batch Processing
```python
texts = ["Text 1", "Text 2", "Text 3", ...]
inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
outputs = model(**inputs)
scores = torch.sigmoid(outputs.logits).squeeze().tolist()
```

## 📈 Expected Performance

| Metric | Value | What it means |
|--------|-------|---------------|
| MAE | ~0.10 | Average error of 0.1 points |
| R² | ~0.80 | Explains 80% of variance |
| Accuracy (±0.2) | ~0.90 | 90% within ±0.2 of true score |

## 🎮 D&D Integration

### As RL Reward
```python
# In PPO training
narrative_score = critic.score(dm_response)
reward = 0.6 * narrative_score + 0.4 * other_rewards
ppo.step(reward)
```

### Quality Filtering
```python
# Generate multiple candidates
responses = [generate() for _ in range(5)]
scores = [score_text(r) for r in responses]
best = responses[np.argmax(scores)]  # Pick highest quality
```

### Threshold-based Selection
```python
while True:
    response = generate()
    if score_text(response) > 0.6:  # Quality threshold
        break
```

## ⚙️ Configuration

### Key Parameters (`critic_config.yaml`)
```yaml
# Model
model.name: "microsoft/deberta-v3-base"
model.num_labels: 1

# Data
data.max_seq_length: 128  # Shorter = faster

# Training
training.num_train_epochs: 3
training.per_device_train_batch_size: 32
training.learning_rate: 3e-5
```

### Customize for Your Needs

**Faster Training**:
```yaml
max_seq_length: 96        # Shorter sequences
num_train_epochs: 2       # Fewer epochs
per_device_train_batch_size: 64  # Larger batches (if GPU allows)
```

**Better Performance**:
```yaml
num_train_epochs: 5       # More training
learning_rate: 2e-5       # Lower LR
weight_decay: 0.02        # More regularization
```

## 🔍 Quality Score Interpretation

| Score | Quality | Use Case |
|-------|---------|----------|
| 0.8-1.0 | Excellent | Descriptive, engaging D&D narration |
| 0.6-0.8 | Good | Solid DM responses |
| 0.4-0.6 | Fair | Acceptable but could improve |
| 0.2-0.4 | Poor | Likely repetitive or incomplete |
| 0.0-0.2 | Very Poor | Incoherent or shuffled text |

## 📊 Example Predictions

```python
# High quality
text = "The ancient library stretched endlessly before you..."
score_text(text)  # → 0.82

# Low quality
text = "You see room. There is thing. You do stuff."
score_text(text)  # → 0.28

# Repetitive
text = "The dragon roars. The dragon roars. The dragon roars."
score_text(text)  # → 0.35

# Truncated
text = "You enter the tavern and see"
score_text(text)  # → 0.45
```

## 🛠️ Troubleshooting

### Issue: Model predicts same score for everything
**Solution**: Check training data balance, ensure varied quality scores

### Issue: Poor performance on D&D text
**Solution**: Fine-tune on D&D examples, adjust score ranges

### Issue: Out of memory
**Solution**: Reduce batch_size or max_seq_length

### Issue: Training too slow
**Solution**: Use GPU, increase batch_size, reduce sequence length

## 📁 File Structure

```
narrative critic/
├── prepare_dataset.py                    # Create training data
├── narrative_critic.py                   # Local training script
├── kaggle_narrative_critic_training.ipynb  # Kaggle notebook
├── critic_config.yaml                    # Configuration
├── PROJECT_DOCUMENTATION.md              # Full documentation
├── KAGGLE_SETUP.md                      # Kaggle guide
├── QUICK_START.md                       # This file
├── README.md                            # Project overview
│
├── ROCStories__spring2016 - ROCStories_spring2016.csv  # Source data
│
├── dataset_json/                        # Generated by prepare_dataset.py
│   ├── rocstoriestrain.json
│   └── rocstoriesval.json
│
└── models/                              # Generated by training
    └── narrative_critic/
        ├── pytorch_model.bin
        ├── config.json
        ├── tokenizer.json
        └── ...
```

## 🎓 Learning Path

### Beginner
1. ✅ Run `prepare_dataset.py` to understand data
2. ✅ Explore generated JSON files
3. ✅ Train on Kaggle (easiest)
4. ✅ Test with simple examples

### Intermediate
1. ✅ Train locally with `narrative_critic.py`
2. ✅ Analyze training plots
3. ✅ Experiment with hyperparameters
4. ✅ Test on custom D&D examples

### Advanced
1. ✅ Fine-tune on domain-specific data
2. ✅ Integrate with RL pipeline
3. ✅ Implement multi-aspect scoring
4. ✅ Add uncertainty estimation

## 📚 Resources

- **Documentation**: `PROJECT_DOCUMENTATION.md`
- **Kaggle Setup**: `KAGGLE_SETUP.md`
- **Model Card**: `README.md`
- **Config Reference**: `critic_config.yaml`

## 💬 Common Questions

**Q: How long does training take?**
A: 1-2 hours on GPU, 4-6 hours on CPU

**Q: Can I use a different base model?**
A: Yes! Change `model.name` in config (e.g., "roberta-base")

**Q: How do I add my own quality examples?**
A: Add to JSON files with your quality scores, retrain

**Q: What if my text is longer than 128 tokens?**
A: Increase `max_seq_length` or use sliding window approach

**Q: Can I deploy this as an API?**
A: Yes! Use FastAPI or Flask with the model loading code

## 🎯 Next Steps

1. **Train your model**: Follow Option 1 or 2 above
2. **Test predictions**: Try the quick usage examples
3. **Integrate**: Add to your D&D system
4. **Iterate**: Fine-tune on your specific use case

---

**Need help?** Check `PROJECT_DOCUMENTATION.md` for detailed explanations! 🚀
