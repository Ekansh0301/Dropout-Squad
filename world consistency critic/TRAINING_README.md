# World Consistency Critic - DeBERTa-v3-Large Training

## Overview

This directory contains the **new DeBERTa-based World Consistency Critic** that replaces the previous regex-based implementation with a trained neural classifier.

### Strategy
Following the **Character Voice Critic's negative sampling approach**:
1. Extract multi-turn sequences from CRD3 dataset
2. Apply corruption functions to create balanced training data
3. Train DeBERTa-v3-Large for 4-class classification

### Architecture
- **Model**: microsoft/deberta-v3-large (304M parameters)
- **Task**: Sequence classification with 4 output classes
- **Input**: `[HISTORY] turns [SEP] [RESPONSE] dm_response`
- **Output**: Classification into contradiction, hallucination, amnesia, or consistent

---

## Files

### 1. `world_consistency_data_prep.py`
Data preparation pipeline that:
- Extracts multi-turn sequences from CRD3
- Builds world state using regex patterns
- Applies corruption functions to create negative examples
- Generates balanced dataset (25% per class)

**Key Classes:**
- `WorldStateExtractor`: Regex-based entity/object/state extraction
- `CorruptionFunctions`: Inject contradictions, hallucinations, amnesia
- `CRD3SequenceExtractor`: Multi-turn sequence extraction

**Usage:**
```python
from world_consistency_data_prep import prepare_training_data

training_data = prepare_training_data(
    crd3_file="crd3_npc_dialogues.json",
    output_file="world_consistency_training_data.json",
    num_examples=100000,  # 25K per class
    window_size=5,
    seed=42
)
```

---

### 2. `Train_World_Consistency_Critic.ipynb`
Kaggle training notebook with comprehensive metrics and visualization.

**Sections:**
1. Install dependencies and import libraries
2. Load and explore CRD3 dataset
3. Implement world state extraction and corruption functions
4. Extract multi-turn sequences and generate training data
5. Create train/val/test splits (80/10/10)
6. Initialize DeBERTa-v3-Large model
7. Create custom dataset and data loaders
8. Define training configuration and metrics
9. Train the model with early stopping
10. Comprehensive evaluation on test set
11. Visualize training progress and results (confusion matrix, ROC curves)
12. Test model on hand-crafted examples
13. Save trained model for integration

**Expected Performance:**
- Overall Accuracy: **85-90%**
- Per-class F1: **80-90%** for all classes
- Training time: **8-12 hours** on Kaggle GPU (P100/T4)

---

### 3. `world_consistency_critic_deberta.py`
Production inference file for the trained model.

**Key Class:**
```python
class WorldConsistencyCritic:
    def __init__(self, model_path: str, device: str = None)
    def score(self, dm_response: str, history: List[str] = None) -> float
    def evaluate_with_explanation(self, dm_response: str, history: List[str] = None) -> Dict
    def batch_score(self, responses: List[str], histories: List[List[str]] = None) -> List[float]
```

**Usage in Director LLM:**
```python
from world_consistency_critic_deberta import WorldConsistencyCritic

# Initialize (load once)
world_critic = WorldConsistencyCritic(
    model_path="/kaggle/input/director-llm-critics/world_consistency_critic_final"
)

# Score DM response
history = [
    "You unlock the door with the rusty key",
    "The door swings open, revealing a dark corridor."
]
dm_response = "You step into the darkness. The door remains locked behind you."

score = world_critic.score(dm_response, history)
# Returns: 0.0 (contradiction - door was just unlocked)

# Get detailed explanation
result = world_critic.evaluate_with_explanation(dm_response, history)
print(result)
# {
#     'score': 0.0,
#     'predicted_class': 'contradiction',
#     'confidence': 0.92,
#     'probabilities': {...},
#     'explanation': 'Response contradicts established facts (confidence: 92.0%)'
# }
```

---

## Training Process

### Step 1: Prepare Dataset
Run on Kaggle with CRD3 dataset:

```python
# In Train_World_Consistency_Critic.ipynb
training_data = extract_sequences_and_generate_data(
    crd3_data,
    num_examples=40000,  # 10K per class
    window_size=5
)
```

**Output:**
- 10,000 Consistent examples (original CRD3)
- 10,000 Contradiction examples (flipped object states)
- 10,000 Hallucination examples (excessive entities)
- 10,000 Amnesia examples (removed information)

### Step 2: Train Model
```python
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
)

trainer.train()
```

**Hyperparameters:**
- Learning rate: 2e-5
- Batch size: 8 (with gradient accumulation = 4, effective 32)
- Epochs: 3
- Warmup: 500 steps
- Mixed precision (FP16): Enabled on GPU

### Step 3: Evaluate
Comprehensive metrics:
- Overall accuracy
- Per-class precision, recall, F1
- Confusion matrix
- ROC curves and AUC
- Hand-crafted example testing

### Step 4: Save Model
```python
model.save_pretrained("/kaggle/working/world_consistency_critic_final")
tokenizer.save_pretrained("/kaggle/working/world_consistency_critic_final")
```

**Saved Files:**
- `config.json`: Model configuration
- `pytorch_model.bin`: Trained weights
- `tokenizer.json`, `vocab.txt`: Tokenizer files
- `training_config.json`: Training metadata and label mappings

---

## Integration with Director LLM

### In Main Notebook

```python
# Load critic
from world_consistency_critic_deberta import WorldConsistencyCritic

world_critic = WorldConsistencyCritic(
    model_path="/kaggle/input/director-llm-critics/world_consistency_critic_final"
)

# Use in RL training loop
for episode in training_episodes:
    # Player action
    player_action = hybrid_player.generate()
    
    # DM response
    dm_response = policy.generate(player_action)
    
    # Score world consistency
    history = conversation_history[-3:]  # Last 3 turns
    r_world = world_critic.score(dm_response, history)
    
    # Combine with other critics
    R = w_narr * r_narr + w_caus * r_caus + w_world * r_world + w_char * r_char
```

---

## Corruption Functions Details

### 1. Contradiction (Label 0, Score 0.0)
**Mechanism:** Violates established object/entity states

**Examples:**
```python
Original: "The door swings open"
World State: {door: "open"}
Corrupted: "The door swings open. The door is locked."
```

**Detection Patterns:**
- locked ↔ open, unlocked
- open ↔ closed, locked
- lit ↔ unlit
- destroyed ↔ intact

### 2. Hallucination (Label 1, Score 0.3)
**Mechanism:** Introduces excessive new entities (8+)

**Examples:**
```python
Original: "You enter the tavern"
Corrupted: "You enter the tavern. Ten merchants, eight guards, five bards, and seven scholars fill the room."
```

**Templates:**
- "Five merchants approach... three guards and two scholars"
- "Ten goblins appear... along with a dragon and five wizards"
- "The room fills with seven bards, four innkeepers, six figures"

### 3. Amnesia (Label 2, Score 0.5)
**Mechanism:** Removes tracked information

**Examples:**
```python
Original: "Gregor the innkeeper greets you"
World State: {innkeeper: {name: "Gregor"}}
Corrupted: "The innkeeper greets you, though you can't recall his name"
```

**Types:**
- NPC name amnesia: Replace name with generic reference
- Password amnesia: "You can't recall the password"
- Object location amnesia: "You can't remember where the key is"
- Destroyed object amnesia: Reference consumed/destroyed object

### 4. Consistent (Label 3, Score 1.0)
**Mechanism:** Original CRD3 sequences, no corruption

---

## Comparison: Old vs New

| Aspect | Old (Regex-based) | New (DeBERTa) |
|--------|-------------------|---------------|
| **Architecture** | Hard-coded rules | Trained neural classifier |
| **Flan-T5 Usage** | Not used (despite references) | Not needed |
| **Model** | None | DeBERTa-v3-Large (304M) |
| **Contradiction** | 50.8% accuracy | **~90% expected** |
| **Hallucination** | 100% (threshold-based) | **~85-90% expected** |
| **Amnesia** | 3.3% (broken) | **~80-90% expected** |
| **Overall** | 63.5% | **~85-90% expected** |
| **Adaptability** | Fixed patterns | Learns from data |
| **False Positives** | High | Low (learned patterns) |
| **Maintenance** | Manual pattern updates | Retrain on new data |

---

## Dataset Statistics

### CRD3 Source
- **Total turns**: 105,485
- **Episodes**: 159
- **Characters**: Diverse (DM + NPCs)
- **Avg turn length**: ~150 characters

### Generated Training Data
- **Total examples**: 40,000
- **Per class**: 10,000 each
- **Train/Val/Test**: 80% / 10% / 10%
- **Sequence length**: 5 turns (window)

---

## Expected Metrics

### Overall Performance
- **Accuracy**: 85-90%
- **Macro F1**: 83-88%
- **Weighted F1**: 85-90%

### Per-Class Performance
| Class | Precision | Recall | F1 | AUC |
|-------|-----------|--------|----|----|
| Contradiction | 88-92% | 85-90% | 87-91% | 0.93-0.96 |
| Hallucination | 82-88% | 85-90% | 84-89% | 0.91-0.94 |
| Amnesia | 78-85% | 80-88% | 80-86% | 0.89-0.93 |
| Consistent | 90-95% | 88-92% | 89-93% | 0.95-0.98 |

---

## Troubleshooting

### Issue: Low accuracy on a specific class
**Solution:** Check class balance in training data. Regenerate with more examples.

### Issue: Model too slow
**Solution:** 
- Use batch_score() for multiple responses
- Reduce max_length from 512 to 256
- Use DeBERTa-v3-base instead of large

### Issue: Out of memory
**Solution:**
- Reduce batch size to 4
- Increase gradient accumulation to 8
- Use FP16 mixed precision

---

## Future Improvements

1. **Fine-tune on D&D-specific data**
   - Create synthetic D&D scenarios beyond CRD3
   - Add more domain-specific patterns

2. **Multi-task learning**
   - Joint training with character voice detection
   - Shared encoder for all critics

3. **Confidence calibration**
   - Temperature scaling for better probability estimates
   - Reject option for low-confidence predictions

4. **Active learning**
   - Collect hard examples from production
   - Iteratively retrain on difficult cases

5. **Ensemble methods**
   - Combine with rule-based fallback
   - Multiple model checkpoints voting

---

## Citation

```bibtex
@article{dropoutsquad2024worldcritic,
  title={World Consistency Critic: DeBERTa-based Narrative Consistency Detection},
  author={Dropout Squad},
  journal={IIIT Hyderabad},
  year={2024},
  note={Part of Director LLM Multi-Critic Framework}
}
```

---

## Contact

For questions or issues, please contact the Dropout Squad team at IIIT Hyderabad.
