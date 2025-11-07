# 📋 Narrative Critic - Complete Index

## 🎯 Start Here

Welcome to the **Narrative Critic** project! This is a complete system for training a DeBERTa-based model to assess narrative quality in D&D (Dungeons & Dragons) responses.

### 🚀 Quick Navigation

**New to the project?** → Start with [QUICK_START.md](QUICK_START.md)

**Want to understand the system?** → Read [ARCHITECTURE_VISUAL.md](ARCHITECTURE_VISUAL.md)

**Ready to train on Kaggle?** → Follow [KAGGLE_SETUP.md](KAGGLE_SETUP.md)

**Need technical details?** → Check [PROJECT_DOCUMENTATION.md](PROJECT_DOCUMENTATION.md)

**Looking for specific files?** → See [FILE_SUMMARY.md](FILE_SUMMARY.md)

---

## 📚 Documentation Library

### 📖 Core Documentation

| Document | Purpose | When to Read |
|----------|---------|--------------|
| **[QUICK_START.md](QUICK_START.md)** | Fast 5-minute guide | Getting started immediately |
| **[KAGGLE_SETUP.md](KAGGLE_SETUP.md)** | Complete Kaggle training guide | Training on Kaggle platform |
| **[PROJECT_DOCUMENTATION.md](PROJECT_DOCUMENTATION.md)** | Comprehensive technical docs | Understanding the entire system |
| **[ARCHITECTURE_VISUAL.md](ARCHITECTURE_VISUAL.md)** | Visual architecture guide | Visual learners, system overview |
| **[FILE_SUMMARY.md](FILE_SUMMARY.md)** | Complete file reference | Finding specific components |
| **[README.md](README.md)** | Original project overview | Project context and purpose |

---

## 💻 Code Files

### 🔧 Implementation Files

| File | Purpose | Output |
|------|---------|--------|
| **`prepare_dataset.py`** | Convert CSV to JSON training data | `dataset_json/*.json` |
| **`narrative_critic.py`** | Local training script (multi-GPU) | `models/narrative_critic/` |
| **`kaggle_narrative_critic_training.ipynb`** | Complete Kaggle notebook | Trained model + analysis |
| **`critic_config.yaml`** | Training configuration | Used by narrative_critic.py |

---

## 🎓 Learning Paths

### 👶 Beginner Path (2-3 hours)

```
1. Read: QUICK_START.md (10 min)
   ↓
2. Run: prepare_dataset.py (5 min)
   ↓
3. Explore: Generated JSON files (10 min)
   ↓
4. Read: KAGGLE_SETUP.md (15 min)
   ↓
5. Upload: Dataset to Kaggle (10 min)
   ↓
6. Train: Run notebook on Kaggle (1-2 hours)
   ↓
7. Analyze: Review outputs and metrics (30 min)
```

### 🎯 Intermediate Path (4-6 hours)

```
1. Read: ARCHITECTURE_VISUAL.md (20 min)
   ↓
2. Read: PROJECT_DOCUMENTATION.md (1 hour)
   ↓
3. Understand: Data transformation logic (30 min)
   ↓
4. Run: prepare_dataset.py + analyze output (30 min)
   ↓
5. Train: Local or Kaggle (1-2 hours)
   ↓
6. Experiment: Test on custom examples (1 hour)
   ↓
7. Integrate: Add to your D&D system (1 hour)
```

### 🚀 Advanced Path (8-12 hours)

```
1. Complete: Intermediate path
   ↓
2. Fine-tune: On domain-specific D&D data (2 hours)
   ↓
3. Optimize: Hyperparameter tuning (2 hours)
   ↓
4. Extend: Multi-aspect scoring (2 hours)
   ↓
5. Deploy: Production API setup (2 hours)
   ↓
6. Integrate: Full RL pipeline (2 hours)
```

---

## 🗺️ Project Workflow

```
┌─────────────────────────────────────────────────────────────┐
│                    COMPLETE WORKFLOW                         │
└─────────────────────────────────────────────────────────────┘

PHASE 1: DATA PREPARATION
├── Read: QUICK_START.md or KAGGLE_SETUP.md
├── Obtain: ROCStories CSV dataset
├── Run: prepare_dataset.py
└── Output: dataset_json/rocstoriestrain.json + rocstoriesval.json

PHASE 2: TRAINING SETUP
├── Choose: Local (narrative_critic.py) OR Kaggle (notebook)
├── If Kaggle:
│   ├── Upload: dataset_json/ as Kaggle dataset
│   ├── Upload: kaggle_narrative_critic_training.ipynb
│   └── Configure: Dataset paths in notebook
└── If Local:
    ├── Configure: critic_config.yaml
    └── Setup: GPU environment

PHASE 3: MODEL TRAINING
├── Execute: Training script or notebook
├── Monitor: Training progress (loss, metrics)
├── Wait: 1-2 hours (GPU) or 4-6 hours (CPU)
└── Output: Trained model in models/narrative_critic/

PHASE 4: EVALUATION
├── Review: Training metrics (MAE, R², etc.)
├── Analyze: Visualizations (loss curves, confusion matrix)
├── Test: Custom D&D examples
└── Validate: Performance meets expectations

PHASE 5: DEPLOYMENT
├── Load: Trained model in your application
├── Integrate: With D&D response generation
├── Use: As RL reward signal or quality filter
└── Monitor: Real-world performance
```

---

## 📊 Dataset Overview

### Source Data
- **File**: `ROCStories__spring2016 - ROCStories_spring2016.csv`
- **Stories**: ~45,000 five-sentence narratives
- **Format**: CSV with storyid, title, sentence1-5

### Generated Training Data
- **Total Examples**: ~30,000
- **Train Set**: ~27,000 (90%)
- **Validation Set**: ~3,000 (10%)

### Quality Types (7,500 each)
| Type | Score Range | Description |
|------|-------------|-------------|
| Coherent | 0.7-1.0 | Original, well-structured stories |
| Shuffled | 0.0-0.3 | Scrambled sentence order |
| Repetitive | 0.2-0.4 | Repeated sentences |
| Truncated | 0.3-0.5 | Incomplete stories |

---

## 🤖 Model Information

### Architecture
- **Base Model**: microsoft/deberta-v3-base
- **Parameters**: 139 million
- **Task**: Regression (quality scoring)
- **Input**: Text sequences (max 128 tokens)
- **Output**: Quality score (0.0-1.0)

### Training Configuration
- **Epochs**: 3
- **Batch Size**: 16-32 per device
- **Learning Rate**: 3e-5
- **Optimizer**: AdamW
- **Scheduler**: Cosine with warmup
- **Loss**: Mean Squared Error (MSE)

### Expected Performance
- **MAE**: 0.08-0.12
- **R²**: 0.75-0.85
- **Correlation**: 0.85-0.92
- **Accuracy (±0.2)**: 0.85-0.92

---

## 🎮 D&D Integration

### Use Cases
1. **RL Reward Signal**: Guide reinforcement learning training
2. **Quality Filtering**: Select best from multiple candidates
3. **Response Ranking**: Order responses by quality
4. **Threshold Gating**: Ensure minimum quality standards
5. **A/B Testing**: Compare model versions

### Integration Pattern
```python
# 1. Generate DM response
response = dm_generator(player_action)

# 2. Evaluate quality
quality = narrative_critic.score(response)

# 3. Use score
if quality > threshold:
    return response
else:
    regenerate()
```

---

## 📈 Performance Benchmarks

### Training Time
| Platform | GPU | Time |
|----------|-----|------|
| Kaggle | T4 x2 | 1-2 hours |
| Kaggle | P100 | 1.5-2.5 hours |
| Local | RTX 3090 | 45-90 min |
| Local | CPU | 4-6 hours |

### Inference Speed
| Batch Size | GPU | Examples/sec |
|------------|-----|--------------|
| 1 | T4 | ~100 |
| 8 | T4 | ~500 |
| 32 | T4 | ~1000 |

### Memory Requirements
| Component | Memory |
|-----------|--------|
| Model | ~550 MB |
| Training (batch 16) | ~4 GB |
| Training (batch 32) | ~8 GB |

---

## 🔧 Configuration Guide

### Quick Adjustments

**Faster Training**:
```yaml
max_seq_length: 96         # Shorter sequences
num_train_epochs: 2        # Fewer epochs
per_device_train_batch_size: 64  # Larger batches
```

**Better Performance**:
```yaml
num_train_epochs: 5        # More training
learning_rate: 2e-5        # Lower learning rate
weight_decay: 0.02         # More regularization
```

**Less Memory**:
```yaml
per_device_train_batch_size: 8   # Smaller batches
max_seq_length: 96         # Shorter sequences
gradient_accumulation_steps: 4   # Accumulate gradients
```

---

## 🛠️ Troubleshooting Index

| Issue | Solution Location |
|-------|-------------------|
| Dataset preparation errors | FILE_SUMMARY.md → prepare_dataset.py |
| Kaggle setup problems | KAGGLE_SETUP.md → Troubleshooting |
| Training crashes | PROJECT_DOCUMENTATION.md → Training Process |
| Poor performance | KAGGLE_SETUP.md → Optimization Tips |
| Integration questions | PROJECT_DOCUMENTATION.md → D&D Integration |
| Configuration changes | QUICK_START.md → Configuration |

---

## 📦 Output Files Reference

### After Dataset Preparation
```
dataset_json/
├── rocstoriestrain.json   # Training data
└── rocstoriesval.json     # Validation data
```

### After Training
```
models/narrative_critic/
├── pytorch_model.bin         # Model weights
├── config.json              # Model config
├── tokenizer.json           # Tokenizer
├── training_history.json    # Training logs
├── eval_metrics.json        # Performance
├── training_loss.png        # Loss curves
└── eval_metrics_plot.png    # Metrics
```

### After Notebook Execution
```
working/
├── narrative_critic_model/         # Trained model
├── training_progress.png           # Training plots
├── model_analysis.png              # Analysis
├── confusion_matrix.png            # Quality bins
├── validation_predictions.csv      # All predictions
└── model_summary.json              # Summary
```

---

## 💡 Quick Reference

### Load Model
```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

model = AutoModelForSequenceClassification.from_pretrained("models/narrative_critic")
tokenizer = AutoTokenizer.from_pretrained("models/narrative_critic")
```

### Score Text
```python
def score(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
        return torch.sigmoid(outputs.logits).item()
```

### Batch Scoring
```python
texts = ["Text 1", "Text 2", ...]
inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
outputs = model(**inputs)
scores = torch.sigmoid(outputs.logits).squeeze().tolist()
```

---

## 🎯 Success Criteria

### After Setup
- ✅ Dataset JSON files created (~30,000 examples)
- ✅ Balanced quality distribution
- ✅ Train/val split verified

### After Training
- ✅ Model trained without errors
- ✅ MAE < 0.15
- ✅ R² > 0.70
- ✅ Training curves show convergence

### After Integration
- ✅ Model loads successfully
- ✅ Predictions match expected ranges
- ✅ Inference speed acceptable
- ✅ Quality improvements observed

---

## 📞 Need Help?

1. **Quick answers**: Check [QUICK_START.md](QUICK_START.md) → Common Questions
2. **Setup issues**: See [KAGGLE_SETUP.md](KAGGLE_SETUP.md) → Troubleshooting
3. **Technical details**: Read [PROJECT_DOCUMENTATION.md](PROJECT_DOCUMENTATION.md)
4. **Understanding code**: Review [FILE_SUMMARY.md](FILE_SUMMARY.md)
5. **Visual overview**: Check [ARCHITECTURE_VISUAL.md](ARCHITECTURE_VISUAL.md)

---

## 🚀 Ready to Start?

### Recommended First Steps:

1. **Read** [QUICK_START.md](QUICK_START.md) (10 minutes)
2. **Run** `prepare_dataset.py` (5 minutes)
3. **Follow** [KAGGLE_SETUP.md](KAGGLE_SETUP.md) (2 hours total)

---

## 📝 Version Info

- **Project**: Narrative Critic
- **Model**: DeBERTa-v3-base
- **Task**: Narrative Quality Assessment
- **Application**: D&D Response Generation
- **Version**: 1.0
- **Date**: November 7, 2025

---

**You have everything you need to train and deploy the Narrative Critic!** 🎉

Start with [QUICK_START.md](QUICK_START.md) → Happy training! 🚀
