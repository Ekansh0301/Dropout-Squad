# ✅ UPLOAD CHECKLIST - World Consistency Critic Training

## What You're Doing
Training a **standalone** DeBERTa-v3-Large model to detect narrative inconsistencies in D&D responses.

**No multi-critic system. No integration. Just train this one model.**

---

## 📦 STEP 1: Upload Dataset to Kaggle

### File to Upload:
```
📁 From parent folder: crd3_npc_dialogues.json
```

### How to Upload:
1. Go to https://www.kaggle.com/datasets
2. Click "New Dataset"
3. Upload `crd3_npc_dialogues.json`
4. Settings:
   - **Title**: CRD3 NPC Dialogues
   - **Slug**: `crd3-npc-dialogues`
   - **Visibility**: Private or Public (your choice)
5. Click "Create"

### ✅ Result:
Dataset available at: `/kaggle/input/crd3-npc-dialogues/crd3_npc_dialogues.json`

---

## 📓 STEP 2: Upload Notebook to Kaggle

### File to Upload:
```
📁 From this folder: Train_World_Consistency_Critic.ipynb
```

### How to Upload:
1. Go to https://www.kaggle.com/code
2. Click "New Notebook"
3. Click "File" → "Upload Notebook"
4. Select `Train_World_Consistency_Critic.ipynb`
5. Click "Upload"

### ✅ Result:
Notebook ready to configure

---

## ⚙️ STEP 3: Configure Notebook Settings

### In Kaggle Notebook Interface:

1. **Add Input Dataset**
   - Click "Add Input" in right sidebar
   - Search: `crd3-npc-dialogues`
   - Click "Add"

2. **Enable GPU**
   - Click "Accelerator" dropdown
   - Select: **GPU T4 x2** (or P100)

3. **Enable Internet**
   - Toggle "Internet" to **ON**
   - Needed to download DeBERTa model

### ✅ Result:
Notebook configured and ready to run

---

## 🚀 STEP 4: Run Training

1. Click **"Save Version"** (top right)
2. Select **"Save & Run All (Commit)"**
3. Click **"Save"**

### What Happens:
- Loads CRD3 data
- Generates 40,000 training examples
- Trains DeBERTa-v3-Large for 3 epochs
- Saves model to `/kaggle/working/world_consistency_critic_final/`

### ⏱️ Duration:
**8-12 hours** (depending on GPU)

### ✅ Result:
Trained model in `/kaggle/working/`

---

## 💾 STEP 5: Save Trained Model as Dataset

### After Training Completes:

1. Go to notebook's **"Output"** tab
2. You'll see: `world_consistency_critic_final/` folder
3. Click **"Save Version"**
4. Settings:
   - **Title**: World Consistency Critic DeBERTa
   - **Slug**: `world-consistency-critic-deberta`
5. Click "Create"

### ✅ Result:
Model saved as reusable dataset at: `/kaggle/input/world-consistency-critic-deberta`

---

## 🎯 DONE! Now What?

### Your Trained Model Can:
- Score DM responses for consistency (0.0 to 1.0)
- Detect contradictions, hallucinations, amnesia
- Be used in ANY Kaggle notebook
- Be integrated into future projects

### To Use It:
```python
# In any notebook, add your model dataset as input
from transformers import AutoTokenizer, AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained(
    "/kaggle/input/world-consistency-critic-deberta"
)
```

See `Example_Standalone_Usage.ipynb` for complete examples.

---

## 📋 Files Summary

### Upload to Kaggle:
1. ✅ `crd3_npc_dialogues.json` → Dataset
2. ✅ `Train_World_Consistency_Critic.ipynb` → Notebook

### Keep Locally for Reference:
- `README_SIMPLE.md` - Quick overview
- `KAGGLE_UPLOAD_SIMPLE.md` - Detailed guide
- `world_consistency_critic_deberta.py` - Helper class
- `Example_Standalone_Usage.ipynb` - Usage examples

### You Can Ignore:
- `world_consistency_critic.py` - Old broken regex code
- `Director_LLM_Critics_Implementation.ipynb` - Old implementation (in parent folder)
- Anything referencing "Director LLM" integration

---

## ⚠️ Important Notes

1. **This is standalone** - No dependencies on other critics
2. **No multi-critic system yet** - Just training this one model
3. **Internet must be ON** - Needed to download DeBERTa from Hugging Face
4. **GPU required** - CPU training would take days
5. **Don't modify notebook** - It's ready to run as-is

---

## 🆘 Troubleshooting

**"Out of memory"?**
- Use smaller GPU (T4 instead of P100)
- Or reduce batch size in training config

**"Can't find dataset"?**
- Verify dataset slug is exactly `crd3-npc-dialogues`
- Check it's added as input

**Training seems stuck?**
- It's normal - each epoch takes 3-4 hours
- Check "Logs" tab to see progress

---

## That's All You Need!

Just follow these 5 steps:
1. ✅ Upload `crd3_npc_dialogues.json` as dataset
2. ✅ Upload `Train_World_Consistency_Critic.ipynb` as notebook
3. ✅ Configure GPU + Internet
4. ✅ Click "Save & Run All"
5. ✅ Save output as dataset

**No integration. No multi-critic setup. Just train the model!** 🚀
