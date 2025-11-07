# Character Voice Critic - Implementation Summary

## What Has Been Created

### 1. Main Implementation File: `character_voice_critic.py`

**Location**: `character voice critic/character_voice_critic.py`

**Purpose**: Complete standalone Python implementation that can be:
- Run from terminal for local training
- Imported into the Kaggle notebook
- Used in the MCRL pipeline for inference

**Key Components**:
- `CharacterProfile`: Stores character dialogue history, traits, speech patterns
- `CharacterVoiceDataset`: PyTorch dataset for training data
- `CharacterVoiceModel`: DeBERTa-v3-base with character embedding layer
- `CharacterVoiceCritic`: Main class with training, evaluation, and scoring methods

**Standalone Usage**:
```bash
python character_voice_critic.py \
    --data ../crd3_npc_dialogues.json \
    --output ./character_voice_model \
    --epochs 3 \
    --batch_size 16 \
    --max_characters 50
```

### 2. Kaggle Notebook: `Character_Voice_Critic_Implementation.ipynb`

**Location**: `Character_Voice_Critic_Implementation.ipynb` (root directory)

**Purpose**: Interactive Kaggle notebook for training with visualizations

**Structure** (11 sections):

1. **Environment Setup** - Install packages, configure paths
2. **Load Module** - Import character_voice_critic.py from Kaggle input
3. **Explore Dataset** - Analyze CRD3 dialogues, show statistics
4. **Build Training Data** - Create positive/negative pairs
5. **Visualize Samples** - Show 5 example pairs with contradictions
6. **Initialize Model** - Set up DeBERTa with character embeddings
7. **Train Model** - Fine-tune with real-time progress bars
8. **Evaluate Performance** - Metrics, confusion matrix, score distribution
9. **Test Scoring** - Score sample dialogues
10. **Visualize Embeddings** - t-SNE plot of character space
11. **Save Model** - Export for later use

**Key Features**:
- ✅ Real-time training progress (tqdm)
- ✅ Live visualizations (matplotlib)
- ✅ 5 positive/negative pairs showing voice match vs mismatch
- ✅ Character embedding visualization
- ✅ Works on both Kaggle and local environments

**Configuration**:
```python
USE_KAGGLE = True  # Set to False for local execution

if USE_KAGGLE:
    CRITIC_MODULE_PATH = '/kaggle/input/director-llm-critics'
    DATASET_PATH = '/kaggle/input/crd3-npc-dialogues/crd3_npc_dialogues.json'
else:
    CRITIC_MODULE_PATH = './character voice critic'
    DATASET_PATH = './crd3_npc_dialogues.json'
```

### 3. Usage Guide: `USAGE_GUIDE.md`

**Location**: `character voice critic/USAGE_GUIDE.md`

**Content**:
- Local training instructions
- Kaggle setup steps
- Model loading examples
- Integration with MCRL pipeline
- Troubleshooting tips

## Dataset Format

The notebook expects `crd3_npc_dialogues.json` with this structure:

```json
[
  {
    "character": "Character Name",
    "text": "The dialogue text",
    "context": "Context of the scene",
    "episode": "C1E001",
    "turn_number": 123
  }
]
```

This matches the format in your existing `crd3_npc_dialogues.json`.

## Training Data Structure

The implementation creates balanced pairs:

### Positive Example (Label: 1.0)
```json
{
  "character": "Scanlan",
  "character_id": 5,
  "text": "Well, well! What a fine establishment!",
  "context": "in tavern",
  "label": 1.0
}
```
→ **This IS Scanlan's dialogue, voice should MATCH**

### Negative Example (Label: 0.0)
```json
{
  "character": "Scanlan",  // Being evaluated
  "character_id": 5,
  "text": "I will smite thee with holy fury!",  // Actually Pike's dialogue
  "context": "in battle",
  "label": 0.0
}
```
→ **This is NOT Scanlan's dialogue (it's Pike's), voice should NOT match**

## The 5 Example Pairs (Section 5 of Notebook)

The notebook demonstrates contradictory relationships:

**Pair 1-5**: Each shows the SAME character evaluated against:
1. ✅ Their own dialogue → High score expected
2. ❌ Another character's dialogue → Low score expected

This teaches the model to learn character-specific patterns.

## Model Architecture

```
Input: "[Character: {name}] [Context: {context}] [Dialogue: {text}]"
  ↓
DeBERTa-v3-base Encoder (184M params)
  ↓
[CLS] Token Embedding (768D)
  ↓
Character Embedding (128D) ←─ Learned per character
  ↓
Concatenate [768D + 128D = 896D]
  ↓
Classification Head:
  - Linear(896 → 512) + ReLU + Dropout
  - Linear(512 → 256) + ReLU + Dropout
  - Linear(256 → 1) + Sigmoid
  ↓
Output: Voice Match Score [0.0 - 1.0]
```

## How to Use

### Local Training (Terminal)

1. Navigate to the character voice critic directory:
```bash
cd "character voice critic"
```

2. Run training:
```bash
python character_voice_critic.py \
    --data ../crd3_npc_dialogues.json \
    --output ./character_voice_model \
    --epochs 3 \
    --max_characters 50
```

3. Model saved to `./character_voice_model/`

### Kaggle Training (Notebook)

1. **Upload Datasets**:
   - Create dataset `director-llm-critics` with `character_voice_critic.py`
   - Create dataset `crd3-npc-dialogues` with `crd3_npc_dialogues.json`

2. **Create Notebook**:
   - Copy code from `Character_Voice_Critic_Implementation.ipynb`
   - Add both datasets as input
   - Enable GPU (Settings → GPU T4 x2)

3. **Run Notebook**:
   - Execute cells sequentially
   - Model saved to `/kaggle/working/character_voice_model`

4. **Download Model**:
   - After training, download the output files
   - Use locally or in pipeline

### Using Trained Model

```python
from character_voice_critic import CharacterVoiceCritic

# Load model
critic = CharacterVoiceCritic()
critic.load_model("./character_voice_model")

# Score dialogue
score = critic.score(
    character_name="Scanlan",
    dialogue="Let's make this interesting with magic!",
    context="in battle"
)

print(f"Voice Match: {score:.2f}")  # e.g., 0.87 (strong match)
```

## Key Differences: .py vs .ipynb

### `character_voice_critic.py`
- ✅ Complete implementation
- ✅ Can run standalone from terminal
- ✅ No training code in notebook (imported)
- ✅ Reusable across projects
- ✅ Easy to version control

### Notebook
- ✅ Uses .py file (doesn't reimplement)
- ✅ Adds visualizations
- ✅ Adds example demonstrations
- ✅ Interactive exploration
- ✅ Progress tracking
- ✅ Educational (shows how it works)

## File Tree After Training

```
Dropout-Squad/
├── character voice critic/
│   ├── character_voice_critic.py    # Main implementation ✓
│   ├── USAGE_GUIDE.md               # Usage instructions ✓
│   ├── README.md                    # Technical documentation (existing)
│   ├── requirements.txt             # Dependencies (existing)
│   └── character_voice_model/       # Created after training
│       ├── model.pt
│       ├── character_info.json
│       ├── training_data.json
│       └── training_history.json
├── Character_Voice_Critic_Implementation.ipynb  # Kaggle notebook ✓
└── crd3_npc_dialogues.json          # Dataset (existing)
```

## Expected Results

After training, you should see:

- **Training Accuracy**: 85-90%
- **Validation Accuracy**: 80-85%
- **Training Time**: 2-4 hours (Kaggle GPU)
- **Model Size**: ~1 GB

**Score Interpretation**:
- 0.8-1.0: Strong voice match ✓
- 0.6-0.8: Moderate match
- 0.4-0.6: Weak match
- 0.0-0.4: Voice mismatch ✗

## Next Steps

1. **Test Locally**:
   ```bash
   cd "character voice critic"
   python character_voice_critic.py --data ../crd3_npc_dialogues.json --output ./test_model --epochs 1 --max_characters 10
   ```

2. **Test on Kaggle**:
   - Upload the .py file as dataset
   - Upload the .json file as dataset
   - Copy notebook code
   - Run on GPU

3. **Integrate with Pipeline**:
   - Load trained model in PPO code
   - Score NPC dialogues during generation
   - Combine with other critics

## Troubleshooting

**Q: Module not found in notebook?**
A: Check that `CRITIC_MODULE_PATH` points to the correct Kaggle input path

**Q: CUDA out of memory?**
A: Reduce `--batch_size` to 8 or 4

**Q: Training too slow?**
A: Reduce `--max_characters` to limit dataset size

**Q: Character not found during scoring?**
A: The model only knows characters from training. Unknown characters return 0.5

## Summary

✅ **Created**: Standalone Python file (`character_voice_critic.py`)
✅ **Created**: Kaggle notebook with visualizations
✅ **Created**: Usage guide
✅ **Ready**: For local terminal training
✅ **Ready**: For Kaggle GPU training
✅ **Ready**: For MCRL pipeline integration

All files use the existing `crd3_npc_dialogues.json` in your directory!
