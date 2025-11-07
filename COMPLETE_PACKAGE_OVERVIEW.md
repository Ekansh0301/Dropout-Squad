# 🎭 Character Voice Critic - Complete Package

## ✅ What You Have Now

### 1. Core Implementation
- **File**: `character voice critic/character_voice_critic.py`
- **Features**:
  - Complete standalone implementation
  - Can run from terminal: `python character_voice_critic.py --data ... --output ...`
  - Can be imported into notebooks
  - Includes all training, evaluation, and inference code
  - Character profile management
  - Model save/load functionality

### 2. Kaggle Notebook
- **File**: `Character_Voice_Critic_Implementation.ipynb` (root directory)
- **Features**:
  - Imports the .py file (no code duplication)
  - 11 comprehensive sections
  - Real-time training progress
  - 5 positive/negative example pairs
  - Training visualizations
  - Character embedding t-SNE plots
  - Evaluation metrics
  - Works on both Kaggle and local

### 3. Documentation
- **Files**:
  - `character voice critic/README.md` (existing technical doc)
  - `character voice critic/USAGE_GUIDE.md` (new detailed guide)
  - `CHARACTER_VOICE_CRITIC_SUMMARY.md` (new complete summary)
  - `QUICK_REFERENCE.md` (new quick start)

### 4. Testing
- **File**: `character voice critic/test_implementation.py`
- **Purpose**: Verify implementation works correctly
- **Run**: `python test_implementation.py`

### 5. Dependencies
- **File**: `character voice critic/requirements.txt`
- **Packages**: torch, transformers, numpy, scikit-learn, tqdm

## 🚀 How to Use

### Option 1: Local Training (Terminal)

```bash
cd "character voice critic"

# Install dependencies
pip install -r requirements.txt

# Test implementation
python test_implementation.py

# Train model
python character_voice_critic.py \
    --data ../crd3_npc_dialogues.json \
    --output ./character_voice_model \
    --epochs 3 \
    --batch_size 16 \
    --max_characters 50
```

### Option 2: Kaggle Training (Notebook)

1. **Prepare Datasets**:
   ```
   Upload to Kaggle Datasets:
   - Dataset 1: "director-llm-critics" 
     → Upload: character_voice_critic.py
   
   - Dataset 2: "crd3-npc-dialogues"
     → Upload: crd3_npc_dialogues.json
   ```

2. **Create Notebook**:
   - New Kaggle Notebook
   - Add both datasets as input
   - Copy code from `Character_Voice_Critic_Implementation.ipynb`
   - Set accelerator to GPU (T4 x2)

3. **Configure Paths** (already set in notebook):
   ```python
   USE_KAGGLE = True  # Already True in notebook
   
   CRITIC_MODULE_PATH = '/kaggle/input/director-llm-critics'
   DATASET_PATH = '/kaggle/input/crd3-npc-dialogues/crd3_npc_dialogues.json'
   ```

4. **Run**:
   - Execute all cells sequentially
   - Model saved to `/kaggle/working/character_voice_model`
   - Download after training

### Option 3: Local Notebook Execution

In the notebook, change:
```python
USE_KAGGLE = False  # Change to False
```

Then run locally in Jupyter/VS Code.

## 📊 What the Training Does

### Input Data
Your `crd3_npc_dialogues.json`:
```json
{
  "character": "Character Name",
  "text": "Their dialogue",
  "context": "Scene context",
  "episode": "C1E001",
  "turn_number": 123
}
```

### Training Process

1. **Group by character** (e.g., Scanlan, Pike, Keyleth, ...)
2. **Filter** characters with ≥10 dialogues
3. **Create positive pairs**:
   - (Scanlan, "Hello!", context) → Label 1.0 ✅
   - (Pike, "Holy light!", context) → Label 1.0 ✅

4. **Create negative pairs**:
   - (Scanlan, Pike's "Holy light!", context) → Label 0.0 ❌
   - (Pike, Scanlan's "Hello!", context) → Label 0.0 ❌

5. **Train DeBERTa** to distinguish matches from mismatches

### The 5 Example Pairs (Notebook Section 5)

For each of 5 characters, shows:

**✅ POSITIVE**: Character matched with their OWN dialogue
```
Character: Scanlan
Dialogue: "Well, well! What a fine establishment!"
Label: 1.0 (MATCH)
→ This IS Scanlan's dialogue, should score HIGH
```

**❌ NEGATIVE**: Character matched with ANOTHER's dialogue
```
Character: Scanlan (being evaluated)
Dialogue: "I will smite thee!" (actually Pike's)
Label: 0.0 (MISMATCH)
→ This is NOT Scanlan's dialogue, should score LOW
```

This demonstrates the **contradictory relationship**: Same character, different outcomes based on dialogue source.

## 🎯 Expected Results

### Training Metrics
- **Accuracy**: 85-90% (train), 80-85% (val)
- **Training Time**: 2-4 hours (GPU), 8-12 hours (CPU)
- **Model Size**: ~1 GB

### What the Model Learns
- Character-specific vocabulary
- Speech patterns (formal vs. casual)
- Personality markers (brave, witty, wise)
- Topic preferences (combat, magic, diplomacy)

### Example Scores
```python
# Scanlan's own dialogue (casual, playful)
score = critic.score("Scanlan", "Let's make this interesting!", context)
# Expected: 0.85 (high match)

# Scanlan evaluated on Pike's dialogue (holy, formal)
score = critic.score("Scanlan", "By the light of Sarenrae!", context)
# Expected: 0.20 (low match - not Scanlan's style)
```

## 🔗 Integration with MCRL Pipeline

After training, use in PPO:

```python
from character_voice_critic import CharacterVoiceCritic

# Initialize once
char_critic = CharacterVoiceCritic()
char_critic.load_model("./character_voice_model")

# During training episode
for step in episode:
    dm_response = dm_policy.generate(player_action)
    
    # Extract NPC dialogue (if present)
    npc_name, npc_dialogue = extract_npc_dialogue(dm_response)
    
    # Score character voice consistency
    if npc_dialogue:
        r_char = char_critic.score(
            character_name=npc_name,
            dialogue=npc_dialogue,
            context=player_action
        )
    else:
        r_char = 1.0  # No NPC, neutral score
    
    # Combine with other critics
    R = (w_narr * r_narr + 
         w_caus * r_caus + 
         w_world * r_world + 
         w_char * r_char)
    
    # PPO update
    update_policy(R)
```

## 📁 File Structure

```
Dropout-Squad/
├── character voice critic/
│   ├── character_voice_critic.py       ← Main implementation ✅
│   ├── test_implementation.py          ← Test script ✅
│   ├── USAGE_GUIDE.md                  ← Detailed guide ✅
│   ├── README.md                       ← Technical doc (existing)
│   └── requirements.txt                ← Dependencies (existing)
│
├── Character_Voice_Critic_Implementation.ipynb  ← Kaggle notebook ✅
├── CHARACTER_VOICE_CRITIC_SUMMARY.md            ← Summary ✅
├── QUICK_REFERENCE.md                           ← Quick start ✅
├── THIS_FILE.md                                 ← Overview ✅
└── crd3_npc_dialogues.json                      ← Dataset (existing)
```

## ✨ Key Features

### Python File (`character_voice_critic.py`)
✅ Standalone executable  
✅ Command-line interface  
✅ All training logic  
✅ Model save/load  
✅ Character embeddings  
✅ Importable for notebooks  

### Notebook
✅ Imports .py (no duplication)  
✅ 5 example pairs with explanations  
✅ Real-time progress bars  
✅ Training visualizations  
✅ t-SNE embeddings plot  
✅ Confusion matrix  
✅ Score distributions  
✅ Works on Kaggle + local  

## 🎓 Next Steps

### 1. Test Locally
```bash
cd "character voice critic"
python test_implementation.py
```

### 2. Quick Training Test (10 characters, 1 epoch)
```bash
python character_voice_critic.py \
    --data ../crd3_npc_dialogues.json \
    --output ./test_model \
    --epochs 1 \
    --max_characters 10
```

### 3. Full Local Training
```bash
python character_voice_critic.py \
    --data ../crd3_npc_dialogues.json \
    --output ./character_voice_model \
    --epochs 3 \
    --max_characters 50
```

### 4. Kaggle Training
- Upload datasets
- Copy notebook
- Run on GPU
- Download model

### 5. Use in Pipeline
```python
from character_voice_critic import CharacterVoiceCritic
critic = CharacterVoiceCritic()
critic.load_model("./character_voice_model")
score = critic.score("Character", "dialogue", "context")
```

## 🐛 Common Issues

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError` | Check sys.path or CRITIC_MODULE_PATH |
| CUDA OOM | Reduce batch_size to 8 or 4 |
| Too slow | Reduce max_characters to 20-30 |
| Character not found | Character not in training data (returns 0.5) |
| Import errors | Run `pip install -r requirements.txt` |

## 📚 Documentation Hierarchy

1. **QUICK_REFERENCE.md** ← Start here for commands
2. **USAGE_GUIDE.md** ← Detailed instructions
3. **CHARACTER_VOICE_CRITIC_SUMMARY.md** ← Complete overview
4. **README.md** ← Technical details
5. **THIS_FILE.md** ← Package overview

## ✅ Checklist

Before training:
- [ ] `character_voice_critic.py` exists
- [ ] `crd3_npc_dialogues.json` exists
- [ ] Dependencies installed (`pip install -r requirements.txt`)
- [ ] Test passed (`python test_implementation.py`)

For Kaggle:
- [ ] Uploaded `character_voice_critic.py` as dataset
- [ ] Uploaded `crd3_npc_dialogues.json` as dataset
- [ ] Copied notebook code
- [ ] Enabled GPU

## 🎉 You're All Set!

Everything is ready:
- ✅ Python implementation (terminal-ready)
- ✅ Jupyter notebook (Kaggle-ready)
- ✅ Test script
- ✅ Documentation
- ✅ Example usage

**Choose your path**:
- 🖥️ Terminal: `python character_voice_critic.py --data ...`
- 📓 Kaggle: Upload datasets → Run notebook
- 🔬 Local notebook: Change USE_KAGGLE to False

---

**Questions?** See:
- Quick commands: `QUICK_REFERENCE.md`
- Detailed guide: `USAGE_GUIDE.md`
- Complete info: `CHARACTER_VOICE_CRITIC_SUMMARY.md`
