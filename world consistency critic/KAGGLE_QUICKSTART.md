# Kaggle Quick Start Guide - World Consistency Critic Training

## Prerequisites

### 1. Kaggle Dataset Setup
Upload `crd3_npc_dialogues.json` as a Kaggle dataset:

1. Go to kaggle.com/datasets
2. Click "New Dataset"
3. Upload `crd3_npc_dialogues.json`
4. Name: `crd3-npc-dialogues`
5. Make public or private
6. Note the path: `/kaggle/input/crd3-npc-dialogues/crd3_npc_dialogues.json`

### 2. Create New Notebook
1. Go to kaggle.com/code
2. Click "New Notebook"
3. Select "Notebook" (not script)
4. Enable GPU: Settings → Accelerator → GPU T4 x2 (or P100)
5. Add input data: Add Data → Your Datasets → crd3-npc-dialogues

---

## Option A: Upload Notebook (Recommended)

### Step 1: Upload `Train_World_Consistency_Critic.ipynb`
1. In Kaggle notebook, click File → Upload Notebook
2. Select `Train_World_Consistency_Critic.ipynb`
3. Wait for upload to complete

### Step 2: Verify Dataset Path
Check that dataset is available:
```python
!ls /kaggle/input/crd3-npc-dialogues/
```
Should show: `crd3_npc_dialogues.json`

### Step 3: Run All Cells
1. Click "Run All" or press Shift+Enter through each cell
2. Training will take **8-12 hours** on T4 GPU
3. Monitor progress in output

### Step 4: Download Trained Model
After training completes:
```bash
# Model saved at: /kaggle/working/world_consistency_critic_final/
```

1. Click "Output" tab
2. Download `world_consistency_critic_final` folder
3. Upload to new Kaggle dataset named `director-llm-critics`

---

## Option B: Copy-Paste Code

If upload doesn't work, manually create cells:

### Cell 1: Install Dependencies
```python
!pip install -q transformers datasets accelerate scikit-learn matplotlib seaborn
```

### Cell 2: Import Libraries
```python
import json
import random
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict, Counter
from tqdm.auto import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback
)

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
    roc_auc_score
)
from sklearn.preprocessing import label_binarize

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
```

### Cell 3: Load CRD3 Data
```python
CRD3_FILE = "/kaggle/input/crd3-npc-dialogues/crd3_npc_dialogues.json"

with open(CRD3_FILE, 'r', encoding='utf-8') as f:
    crd3_data = json.load(f)

print(f"Loaded {len(crd3_data):,} dialogue turns from CRD3")
```

### Cells 4-13: 
Copy code from `Train_World_Consistency_Critic.ipynb` cells sequentially.

---

## Expected Timeline

| Phase | Duration | Notes |
|-------|----------|-------|
| Setup & Data Loading | 5-10 min | Download dataset, install packages |
| Data Preparation | 15-30 min | Extract sequences, apply corruptions |
| Model Initialization | 2-5 min | Download DeBERTa-v3-large |
| Training Epoch 1 | 2.5-4 hours | ~1000 steps |
| Training Epoch 2 | 2.5-4 hours | ~1000 steps |
| Training Epoch 3 | 2.5-4 hours | ~1000 steps |
| Evaluation & Viz | 10-15 min | Test set metrics, plots |
| **Total** | **8-12 hours** | Full pipeline |

---

## Monitoring Training

### Check Progress
```python
# Training logs show:
# - Step number
# - Training loss
# - Learning rate
# - Samples/second

# Example output:
# [1000/3000] Loss: 0.234, LR: 0.00002, 15.2 samples/sec
```

### Early Stopping
Model saves best checkpoint based on validation F1:
```
✓ Best model: Epoch 2, Val F1 = 0.876
✓ Early stopping triggered (no improvement for 2 epochs)
```

---

## Common Issues

### Issue 1: Out of Memory
**Error:** `CUDA out of memory`

**Solution:**
```python
# Reduce batch size in TrainingArguments
per_device_train_batch_size=4,  # Instead of 8
gradient_accumulation_steps=8,   # Instead of 4
```

### Issue 2: Slow Training
**Issue:** Training taking > 15 hours

**Solution:**
- Check GPU is enabled (not CPU)
- Reduce dataset size:
  ```python
  num_examples=20000  # Instead of 40000
  ```

### Issue 3: Dataset Not Found
**Error:** `FileNotFoundError: /kaggle/input/crd3-npc-dialogues/...`

**Solution:**
1. Click "Add Data" → "Your Datasets"
2. Select `crd3-npc-dialogues`
3. Verify path with `!ls /kaggle/input/`

### Issue 4: Low Accuracy
**Issue:** Test accuracy < 80%

**Solutions:**
- Check class balance in training data
- Increase num_examples to 100K
- Train for more epochs (5 instead of 3)

---

## Saving & Using Model

### After Training
Model automatically saved to:
```
/kaggle/working/world_consistency_critic_final/
```

**Files:**
- `config.json` - Model configuration
- `pytorch_model.bin` - Trained weights (1.2 GB)
- `tokenizer.json` - Tokenizer
- `training_config.json` - Training metadata

### Create Kaggle Dataset

1. Click "Output" in notebook
2. Find `world_consistency_critic_final` folder
3. Click "⋮" → "Create Dataset"
4. Name: `world-consistency-critic-deberta`
5. Make public/private
6. Click "Create"

### Use in Main Notebook

```python
# Add dataset to notebook
# Input: /kaggle/input/world-consistency-critic-deberta/

# Load critic
from world_consistency_critic_deberta import WorldConsistencyCritic

critic = WorldConsistencyCritic(
    model_path="/kaggle/input/world-consistency-critic-deberta"
)

# Use it
score = critic.score("The door is locked.", history=["You unlock the door"])
print(score)  # 0.0 (contradiction)
```

---

## Verification Checklist

Before using trained model:

- [ ] Test accuracy > 85%
- [ ] All 4 classes have F1 > 80%
- [ ] Hand-crafted examples work correctly
- [ ] Model files saved to `/kaggle/working/`
- [ ] Inference code works on test examples
- [ ] Model uploaded as Kaggle dataset

---

## Performance Targets

### Minimum Acceptable
- Overall Accuracy: **80%**
- Macro F1: **78%**
- Per-class F1: **> 75%** for all

### Expected Performance
- Overall Accuracy: **85-90%**
- Macro F1: **83-88%**
- Per-class F1: **80-90%**

### Excellent Performance
- Overall Accuracy: **> 90%**
- Macro F1: **> 88%**
- Per-class F1: **> 85%** for all

---

## Next Steps

After successful training:

1. **Test on real scenarios**
   - Use in Director LLM notebook
   - Compare with other critics
   - Validate on new D&D dialogues

2. **Iterate if needed**
   - Collect failure cases
   - Add to training data
   - Retrain model

3. **Deploy**
   - Integrate into RL pipeline
   - Set weight `w_world` in reward function
   - Monitor performance during training

---

## Resources

- **DeBERTa Paper**: https://arxiv.org/abs/2006.03654
- **Transformers Docs**: https://huggingface.co/docs/transformers
- **CRD3 Dataset**: https://github.com/RevanthRameshkumar/CRD3

---

## Support

If training fails or results are poor:

1. Check this guide's troubleshooting section
2. Review `TRAINING_README.md` for detailed explanations
3. Verify dataset statistics match expected values
4. Post issue with:
   - Error message
   - Training logs
   - Dataset statistics
   - GPU type

---

**Happy Training! 🚀**
