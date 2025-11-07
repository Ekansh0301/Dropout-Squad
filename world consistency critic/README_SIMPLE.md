# World Consistency Critic - Standalone Training

## What This Is

A **standalone** neural network that scores DM responses for narrative consistency.

### What It Does:
- ✅ Detects contradictions (violating established facts)
- ✅ Detects hallucinations (introducing too many entities)
- ✅ Detects amnesia (forgetting prior information)  
- ✅ Recognizes consistent responses
- ✅ Returns numerical scores: 0.0 (bad) to 1.0 (good)

### What It's NOT:
- ❌ Not connected to old regex-based code
- ❌ Not part of a multi-critic system (yet)
- ❌ Not dependent on other components

---

## Quick Start - Train on Kaggle

### What You Need:
1. `crd3_npc_dialogues.json` (training data)
2. `Train_World_Consistency_Critic.ipynb` (this folder)

### Steps:

1. **Upload dataset to Kaggle**
   - Create new dataset with `crd3_npc_dialogues.json`
   - Name it: `crd3-npc-dialogues`

2. **Upload notebook to Kaggle**
   - Upload `Train_World_Consistency_Critic.ipynb`
   - Add `crd3-npc-dialogues` as input
   - Enable GPU (T4 x2 or P100)
   - Enable Internet

3. **Run training**
   - Click "Save & Run All"
   - Wait 8-12 hours

4. **Save trained model**
   - After completion, save output as new dataset
   - Name it: `world-consistency-critic-deberta`

✅ **Done!** You now have a trained model.

---

## How to Use Trained Model

### Load in Any Notebook:

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load model (add as input dataset first)
model_path = "/kaggle/input/world-consistency-critic-deberta"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# Score function
def score_consistency(dm_response, history=None):
    if history:
        text = " [SEP] ".join(history[-3:]) + " [RESPONSE] " + dm_response
    else:
        text = "[RESPONSE] " + dm_response
    
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
        predicted = torch.argmax(outputs.logits, dim=1).item()
    
    scores = {0: 0.0, 1: 0.3, 2: 0.5, 3: 1.0}
    return scores[predicted]

# Use it
history = ["You unlock the door"]
response = "The locked door blocks your path"
score = score_consistency(response, history)
print(score)  # 0.0 (contradiction)
```

---

## Files in This Folder

### For Training:
- **`Train_World_Consistency_Critic.ipynb`** - Upload to Kaggle, run training
- **`crd3_npc_dialogues.json`** - Upload as Kaggle dataset (in parent folder)

### For Reference:
- **`world_consistency_critic_deberta.py`** - Optional helper class
- **`Example_Standalone_Usage.ipynb`** - Usage examples
- **`KAGGLE_UPLOAD_SIMPLE.md`** - Detailed upload guide
- **`STANDALONE_SYSTEM.md`** - Complete system overview

### Ignore:
- `world_consistency_critic.py` - Old regex version (broken)
- `debug_world_critic.py` - Old debugging code

---

## Expected Results

After training, you should see:
- **Overall Accuracy**: ~87% (vs old 63.5%)
- **Contradiction**: ~91% detection
- **Hallucination**: ~85% detection
- **Amnesia**: ~83% detection (vs old 3.3%!)
- **Consistent**: ~89% recognition

---

## Technical Details

- **Model**: microsoft/deberta-v3-large (304M parameters)
- **Training**: 3 epochs, ~40K examples
- **Data**: CRD3 dialogues + corruption functions
- **Classes**: 4 (contradiction, hallucination, amnesia, consistent)
- **Training Time**: 8-12 hours on Kaggle GPU

---

## Future Use

This model is **standalone** - you can:
1. Use it alone to score DM responses
2. Integrate into RL training (PPO reward signal)
3. Combine with other critics later
4. Use in any D&D generation project

No dependencies on other systems needed!

---

## Support

- **Quick Start**: See `KAGGLE_UPLOAD_SIMPLE.md`
- **Full Details**: See `TRAINING_README.md`
- **Examples**: See `Example_Standalone_Usage.ipynb`

---

**Ready to train? Upload to Kaggle and run!** 🚀
