# Character Voice Critic - Usage Guide

This directory contains the Character Voice Critic implementation for evaluating NPC dialogue consistency.

## Files

1. **character_voice_critic.py** - Main implementation (standalone executable)
2. **Character_Voice_Critic_Implementation.ipynb** - Kaggle notebook for training
3. **requirements.txt** - Python dependencies
4. **README.md** - This file

## Local Training (Terminal)

### Prerequisites

```bash
pip install -r requirements.txt
```

### Train the Critic

```bash
python character_voice_critic.py \
    --data ../crd3_npc_dialogues.json \
    --output ./character_voice_model \
    --epochs 3 \
    --batch_size 16 \
    --learning_rate 2e-5 \
    --min_dialogues 10 \
    --max_characters 50
```

### Arguments

- `--data`: Path to CRD3 NPC dialogues JSON file (required)
- `--output`: Output directory for trained model (default: ./character_voice_model)
- `--epochs`: Number of training epochs (default: 3)
- `--batch_size`: Batch size for training (default: 16)
- `--learning_rate`: Learning rate (default: 2e-5)
- `--min_dialogues`: Minimum dialogues per character (default: 10)
- `--max_characters`: Maximum number of characters to include (default: None, uses all)

### Example: Quick Training

```bash
# Train on top 30 characters for faster training
python character_voice_critic.py \
    --data ../crd3_npc_dialogues.json \
    --output ./quick_model \
    --epochs 2 \
    --max_characters 30
```

### Example: Full Training

```bash
# Train on all valid characters (takes longer)
python character_voice_critic.py \
    --data ../crd3_npc_dialogues.json \
    --output ./full_model \
    --epochs 3 \
    --batch_size 8
```

## Kaggle Training (Notebook)

### Setup

1. Upload **character_voice_critic.py** as a Kaggle Dataset:
   - Go to Kaggle → Datasets → New Dataset
   - Upload `character_voice_critic.py`
   - Name it: `director-llm-critics`

2. Upload **crd3_npc_dialogues.json** as a Kaggle Dataset:
   - Go to Kaggle → Datasets → New Dataset
   - Upload `crd3_npc_dialogues.json`
   - Name it: `crd3-npc-dialogues`

3. Create a new Kaggle Notebook:
   - Add both datasets as data sources
   - Copy the notebook code from `Character_Voice_Critic_Implementation.ipynb`
   - Set accelerator to GPU (Settings → Accelerator → GPU T4 x2)

4. Run the notebook:
   - All cells are ready to run sequentially
   - Training takes ~2-4 hours on Kaggle GPU
   - Outputs saved to `/kaggle/working/character_voice_model`

### Notebook Features

- ✅ Real-time training progress with tqdm
- ✅ Live loss/accuracy plots
- ✅ 5 positive/negative sample pairs visualization
- ✅ Character embedding visualization (t-SNE)
- ✅ Comprehensive evaluation metrics
- ✅ Model saving and loading examples

## Using the Trained Model

### Python Script

```python
from character_voice_critic import CharacterVoiceCritic

# Load trained model
critic = CharacterVoiceCritic()
critic.load_model("./character_voice_model")

# Score dialogue
score = critic.score(
    character_name="Character Name",
    dialogue="Some dialogue text",
    context="Context of the scene"
)

print(f"Voice Match Score: {score:.2f}")

# Detailed evaluation
result = critic.evaluate_with_explanation(
    character_name="Character Name",
    dialogue="Some dialogue text",
    context="Context"
)

print(f"Score: {result['score']:.2f}")
print(f"Interpretation: {result['interpretation']}")
```

### Integration with MCRL Pipeline

```python
# In PPO training loop
char_critic = CharacterVoiceCritic()
char_critic.load_model("./character_voice_model")

for episode in training_episodes:
    dm_response = policy.generate(player_action)
    
    # Extract NPC dialogue
    npc_name, npc_dialogue = extract_npc_dialogue(dm_response)
    
    if npc_dialogue:
        r_char = char_critic.score(npc_name, npc_dialogue, context=player_action)
    else:
        r_char = 1.0
    
    # Combine with other critics
    R = w_narr * r_narr + w_caus * r_caus + w_world * r_world + w_char * r_char
```

## Expected Performance

- **Training Accuracy**: ~85-90%
- **Validation Accuracy**: ~80-85%
- **Training Time**: 2-4 hours (GPU), 8-12 hours (CPU)
- **Model Size**: ~1 GB
- **Inference Time**: ~50ms per dialogue (GPU), ~200ms (CPU)

## Troubleshooting

### CUDA Out of Memory

```bash
# Reduce batch size
python character_voice_critic.py --data ... --batch_size 8
```

### Too Slow on CPU

```bash
# Reduce number of characters
python character_voice_critic.py --data ... --max_characters 20
```

### Character Not Found

The model only knows characters from the training data. If you score a character not in the training set, it returns a neutral score (0.5).

## File Structure After Training

```
character_voice_model/
├── model.pt                    # Model weights
├── character_info.json         # Character mappings and profiles
├── training_data.json          # Training examples
├── training_history.json       # Training metrics
├── tokenizer_config.json       # Tokenizer configuration
├── vocab.json                  # Vocabulary
└── ... (other tokenizer files)
```

## Credits

- **Dataset**: CRD3 (Critical Role Dungeons & Dragons Dataset)
- **Model**: DeBERTa-v3-base (Microsoft)
- **Framework**: PyTorch + Transformers (Hugging Face)

## License

See main project LICENSE file.
