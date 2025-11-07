# Character Voice Critic - Quick Reference

## 🚀 Quick Start

### Local Training (1 command)
```bash
cd "character voice critic"
python character_voice_critic.py --data ../crd3_npc_dialogues.json --output ./model --epochs 3 --max_characters 50
```

### Kaggle Paths (for notebook)
```python
CRITIC_MODULE_PATH = '/kaggle/input/director-llm-critics'
DATASET_PATH = '/kaggle/input/crd3-npc-dialogues/crd3_npc_dialogues.json'
```

## 📁 Files Created

| File | Purpose |
|------|---------|
| `character_voice_critic.py` | Main implementation (terminal + import) |
| `Character_Voice_Critic_Implementation.ipynb` | Kaggle notebook with visualizations |
| `USAGE_GUIDE.md` | Detailed usage instructions |
| `CHARACTER_VOICE_CRITIC_SUMMARY.md` | Complete summary document |

## 🎯 Key Features

### Python File (.py)
✅ Standalone executable from terminal  
✅ Importable into notebook  
✅ All training logic included  
✅ Save/load functionality  
✅ Character embedding management  

### Notebook (.ipynb)
✅ Imports .py file (no code duplication)  
✅ Real-time training progress  
✅ 5 positive/negative example pairs  
✅ Training visualizations  
✅ t-SNE character embeddings plot  
✅ Evaluation metrics & confusion matrix  

## 📊 Training Data Format

### Positive (Label 1.0)
Character matched with **their own** dialogue → Should score HIGH

### Negative (Label 0.0)
Character matched with **another character's** dialogue → Should score LOW

## 🔧 Command Line Arguments

```bash
--data           # Path to crd3_npc_dialogues.json (required)
--output         # Output directory (default: ./character_voice_model)
--epochs         # Training epochs (default: 3)
--batch_size     # Batch size (default: 16)
--learning_rate  # Learning rate (default: 2e-5)
--min_dialogues  # Min dialogues per character (default: 10)
--max_characters # Max characters to include (default: all)
```

## 💡 Common Use Cases

### Quick Test (Small Dataset)
```bash
python character_voice_critic.py --data ../crd3_npc_dialogues.json --output ./quick_test --epochs 1 --max_characters 10
```

### Full Training (All Characters)
```bash
python character_voice_critic.py --data ../crd3_npc_dialogues.json --output ./full_model --epochs 3
```

### Low Memory (Reduce Batch Size)
```bash
python character_voice_critic.py --data ../crd3_npc_dialogues.json --batch_size 4 --max_characters 20
```

## 🎓 Model Usage

### Load and Score
```python
from character_voice_critic import CharacterVoiceCritic

critic = CharacterVoiceCritic()
critic.load_model("./character_voice_model")

score = critic.score("Character Name", "dialogue text", "context")
print(f"Score: {score:.2f}")  # 0.0 (mismatch) to 1.0 (match)
```

### Detailed Evaluation
```python
result = critic.evaluate_with_explanation("Character", "dialogue", "context")
print(result['score'])           # 0.87
print(result['interpretation'])  # "Strong character voice match"
```

### Character Similarity
```python
similarity = critic.compare_characters("Character1", "Character2")
print(f"Similarity: {similarity:.2f}")  # -1.0 to 1.0
```

## 📈 Expected Performance

| Metric | Value |
|--------|-------|
| Train Accuracy | 85-90% |
| Val Accuracy | 80-85% |
| Training Time (GPU) | 2-4 hours |
| Training Time (CPU) | 8-12 hours |
| Model Size | ~1 GB |
| Inference Time (GPU) | ~50ms |

## 🔍 Score Interpretation

| Score Range | Meaning |
|-------------|---------|
| 0.8 - 1.0 | ✅ Strong voice match |
| 0.6 - 0.8 | 🟡 Moderate match |
| 0.4 - 0.6 | 🟠 Weak match |
| 0.0 - 0.4 | ❌ Voice mismatch |

## 🐛 Troubleshooting

| Problem | Solution |
|---------|----------|
| CUDA OOM | Reduce `--batch_size` to 8 or 4 |
| Too slow | Reduce `--max_characters` to 20-30 |
| Module not found | Check `CRITIC_MODULE_PATH` in notebook |
| Character not found | Character not in training set (returns 0.5) |

## 📦 Kaggle Setup (3 Steps)

1. **Upload Datasets**:
   - `director-llm-critics` → `character_voice_critic.py`
   - `crd3-npc-dialogues` → `crd3_npc_dialogues.json`

2. **Create Notebook**:
   - Add both datasets as input
   - Copy code from `Character_Voice_Critic_Implementation.ipynb`
   - Enable GPU (T4 x2)

3. **Run**:
   - Execute cells top to bottom
   - Download model from `/kaggle/working/`

## 🔗 Integration with MCRL

```python
# In PPO training
char_critic = CharacterVoiceCritic()
char_critic.load_model("./character_voice_model")

# Score NPC dialogue
r_char = char_critic.score(npc_name, npc_dialogue, context) if npc_dialogue else 1.0

# Combine rewards
R = w_narr * r_narr + w_caus * r_caus + w_world * r_world + w_char * r_char
```

## 📚 Files You Need

1. ✅ `character_voice_critic.py` - Already created
2. ✅ `Character_Voice_Critic_Implementation.ipynb` - Already created
3. ✅ `crd3_npc_dialogues.json` - Already in your directory

## 🎉 You're Ready!

- **Local**: `python character_voice_critic.py --data ../crd3_npc_dialogues.json --output ./model`
- **Kaggle**: Upload .py and .json, run notebook
- **Inference**: Load model and call `critic.score()`

---

**Need Help?** See `USAGE_GUIDE.md` or `CHARACTER_VOICE_CRITIC_SUMMARY.md`
