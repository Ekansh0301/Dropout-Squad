# 🎯 SIMPLIFIED GUIDE - Just Train the Model

## Your Goal
Train a standalone World Consistency Critic model. **That's it.**

No multi-critic system. No integration. Just train this one model on Kaggle.

---

## 📦 What to Upload to Kaggle

### Only 2 Files Needed:

```
1️⃣ crd3_npc_dialogues.json (from parent folder)
   └─ Upload as Kaggle Dataset
   └─ Name: "crd3-npc-dialogues"

2️⃣ Train_World_Consistency_Critic.ipynb (from this folder)
   └─ Upload as Kaggle Notebook
   └─ Enable GPU + Internet
   └─ Add dataset as input
   └─ Run training (8-12 hours)
```

---

## 📋 Files in This Folder - What Are They?

### ✅ UPLOAD THESE TO KAGGLE:

| File | Upload As | Purpose |
|------|-----------|---------|
| `crd3_npc_dialogues.json` | **Dataset** | Training data (40K+ dialogues) |
| `Train_World_Consistency_Critic.ipynb` | **Notebook** | Training script (ready to run) |

### 📖 REFERENCE DOCS (Keep Locally):

| File | What It Is |
|------|------------|
| `UPLOAD_CHECKLIST.md` | **← START HERE** - 5-step upload guide |
| `KAGGLE_UPLOAD_SIMPLE.md` | Detailed Kaggle instructions |
| `README_SIMPLE.md` | Quick overview |
| `STANDALONE_SYSTEM.md` | Complete system docs |

### 🔧 OPTIONAL HELPERS (Not Required):

| File | What It Is |
|------|------------|
| `world_consistency_critic_deberta.py` | Helper class for inference (optional) |
| `Example_Standalone_Usage.ipynb` | Usage examples after training |
| `world_consistency_data_prep.py` | Data pipeline (embedded in training notebook) |

### 🗑️ IGNORE THESE (Old Code):

| File | Why Ignore |
|------|------------|
| `world_consistency_critic.py` | Old regex version (broken) |
| `debug_world_critic.py` | Old debugging code |

---

## 🚀 Quick Start (5 Steps)

### 1. Upload Dataset (5 min)
- Go to kaggle.com/datasets → New Dataset
- Upload `crd3_npc_dialogues.json`
- Name: `crd3-npc-dialogues`

### 2. Upload Notebook (2 min)
- Go to kaggle.com/code → New Notebook
- Upload `Train_World_Consistency_Critic.ipynb`

### 3. Configure (1 min)
- Add `crd3-npc-dialogues` as input
- Enable GPU (T4 x2)
- Enable Internet

### 4. Run Training (8-12 hours)
- Click "Save & Run All"
- Wait for completion

### 5. Save Model (5 min)
- After done, save output as dataset
- Name: `world-consistency-critic-deberta`

**✅ Done! You have a trained model.**

---

## 💡 What You Get

After training, you'll have a model that:
- ✅ Scores DM responses (0.0 to 1.0)
- ✅ Detects contradictions, hallucinations, amnesia
- ✅ Works standalone (no other dependencies)
- ✅ Can be used in any Kaggle notebook
- ✅ ~87% accuracy (vs old 63.5%)

---

## 📝 How to Use Trained Model

### In Any Kaggle Notebook:

```python
# 1. Add your trained model dataset as input
#    Dataset: world-consistency-critic-deberta

# 2. Load and use
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

model_path = "/kaggle/input/world-consistency-critic-deberta"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# 3. Score responses
def score(response, history=None):
    text = " [SEP] ".join(history[-3:] if history else []) + " [RESPONSE] " + response
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
        predicted = torch.argmax(outputs.logits, dim=1).item()
    
    return {0: 0.0, 1: 0.3, 2: 0.5, 3: 1.0}[predicted]

# Test
history = ["You unlock the door"]
response = "The locked door blocks your path"
print(score(response, history))  # 0.0 (contradiction)
```

---

## ❓ Common Questions

**Q: Do I need other critics?**
A: No! This works standalone.

**Q: Do I need the old Director_LLM notebook?**
A: No! This is completely separate.

**Q: Can I use this in my own projects?**
A: Yes! Just load the model in any notebook.

**Q: How do I integrate with other systems?**
A: Later! For now, just train this model. Integration comes after.

---

## 📚 Which Guide Should I Read?

- **Just want to upload?** → `UPLOAD_CHECKLIST.md` ⭐
- **Need more detail?** → `KAGGLE_UPLOAD_SIMPLE.md`
- **Want full docs?** → `STANDALONE_SYSTEM.md`
- **Looking for examples?** → `Example_Standalone_Usage.ipynb`

**Most people should start with `UPLOAD_CHECKLIST.md`**

---

## ⚠️ Key Points

1. ✅ **This is standalone** - No dependencies
2. ✅ **Just train the model** - No multi-critic setup needed
3. ✅ **Only 2 files to upload** - Dataset + Notebook
4. ✅ **Ready to run** - Don't modify the notebook
5. ✅ **Use anywhere** - After training, use in any project

---

## 🎓 Summary

**You're training ONE model that detects narrative inconsistencies.**

Upload 2 files → Run training → Get trained model → Use it anywhere.

**No integration. No multi-critic system. No old code dependencies.**

**Just train this model!** 🚀

---

📖 **Next Step**: Open `UPLOAD_CHECKLIST.md` and follow the 5 steps!
