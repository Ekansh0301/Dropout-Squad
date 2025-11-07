# Director LLM Critics - Implementation Summary

## Created Files

### 1. World Consistency Critic
**Location**: `world consistency critic/`

**Files**:
- `world_consistency_critic.py` - Main implementation
- `README.md` - Detailed documentation
- `requirements.txt` - Dependencies

**Key Features**:
- Hybrid symbolic tracker + neural extractor architecture
- Uses Flan-T5-Large for state extraction with few-shot prompting
- Detects three failure types:
  - Contradictions: 0.0 (violating established facts)
  - Hallucinations: 0.3 (introducing unmentioned entities)
  - Amnesia: 0.5 (forgetting prior facts)
  - Consistent: 1.0
- Maintains explicit world state (entities, objects, locations, facts)

**Classes**:
- `WorldStateTracker`: Symbolic state management
- `StateExtractor`: Flan-T5 few-shot extraction
- `WorldConsistencyCritic`: Main critic class

### 2. Character Voice Critic
**Location**: `character voice critic/`

**Files**:
- `character_voice_critic.py` - Main implementation
- `README.md` - Detailed documentation
- `requirements.txt` - Dependencies

**Key Features**:
- DeBERTa-v3-base (184M parameters) for character voice modeling
- Learns character-specific embeddings from CRD3 NPC dialogue
- Binary classification: voice match (1.0) vs. mismatch (0.0)
- Trains on positive (character + their dialogue) and negative (character + other's dialogue) pairs

**Classes**:
- `CharacterProfile`: Stores character information
- `NPCDialogueDataset`: Training data loader
- `CharacterVoiceModel`: DeBERTa + character embeddings
- `CharacterVoiceCritic`: Main critic class

### 3. Kaggle Implementation Notebook
**Location**: `Director_LLM_Critics_Implementation.ipynb`

**Contents**:
1. Environment setup and imports
2. World Consistency Critic initialization and testing
3. Multiple examples: consistent state, contradictions, hallucinations
4. Multi-turn conversation tracking
5. Character Voice Critic demonstration (training workflow)
6. Multi-critic integration with dynamic weighting
7. Intent-based reward comparison (EXPLORE/ACTION/DIALOGUE)
8. Performance benchmarking

## Methodology & Assumptions

### World Consistency Critic

**Assumptions**:
1. Partial state tracking (only explicit mentions tracked)
2. Flan-T5 extraction provides reasonable accuracy with few-shot prompting
3. Hallucination threshold: >2 new entities is excessive
4. Amnesia detection uses keyword-based heuristics
5. World state persists until explicit reset

**Scoring Logic**:
- Contradiction: Detected when new state conflicts with stored state → 0.0
- Hallucination: >2 unmentioned entities introduced → 0.3
- Amnesia: Recent important facts ignored → 0.5
- Consistent: No violations detected → 1.0

### Character Voice Critic

**Assumptions**:
1. Characters have stable personalities across episodes
2. CRD3 character labels are accurate
3. Minimum 10 dialogue examples needed per character
4. Voice consistency evaluable from single utterances
5. Random character swapping creates valid negative examples

**Training Process**:
1. Extract NPC dialogues from CRD3 with character attribution
2. Create positive pairs: (Character A, Context, A's dialogue) → 1.0
3. Create negative pairs: (Character A, Context, B's dialogue) → 0.0
4. Fine-tune DeBERTa with binary cross-entropy loss
5. Learn character-specific embeddings during training

## Integration with MCRL Pipeline

### Dynamic Weighting
```python
INTENT_WEIGHTS = {
    'EXPLORE': {'narrative': 0.8, 'causal': 0.2, 'world': 0.5, 'character': 0.3},
    'ACTION': {'narrative': 0.3, 'causal': 0.7, 'world': 0.6, 'character': 0.4},
    'DIALOGUE': {'narrative': 0.6, 'causal': 0.4, 'world': 0.3, 'character': 0.8}
}
```

### Reward Computation
```python
R = (w_narrative * r_narrative + 
     w_causal * r_causal + 
     w_world * r_world + 
     w_character * r_character) / sum(weights)
```

### MCRL Training Loop
1. Hybrid player generates action + classifies intent
2. Policy (Director Agent) generates DM response
3. All 4 critics evaluate response independently
4. Intent-based weights selected
5. Aggregated reward computed
6. PPO updates policy parameters

## Usage Instructions for Kaggle

### Step 1: Prepare Dataset
1. Create a Kaggle dataset named "director-llm-critics"
2. Upload these files:
   - `world_consistency_critic.py`
   - `character_voice_critic.py`
3. (Optional) Upload trained character voice model
4. (Optional) Upload CRD3 NPC dialogue data

### Step 2: Create Notebook
1. Create new Kaggle notebook
2. Add "director-llm-critics" dataset to inputs
3. Copy `Director_LLM_Critics_Implementation.ipynb` content
4. Update `INPUT_DIR` path to match your dataset

### Step 3: Run Implementation
Execute cells in order:
- Install dependencies
- Import critic modules
- Test world consistency critic
- (Optional) Train character voice critic if CRD3 data available
- Run multi-critic integration examples

### Step 4: Integrate with Your Training
Use critics in your MCRL training loop:
```python
world_critic = WorldConsistencyCritic()
char_critic = CharacterVoiceCritic()
char_critic.load_model("/kaggle/input/character-model/")

# In training loop
r_world = world_critic.score(dm_response, player_action)
r_char = char_critic.score(npc_name, npc_dialogue, context)
```

## Model Requirements

### World Consistency Critic
- **Model**: Flan-T5-Large (~780M parameters)
- **Memory**: ~3GB GPU
- **Inference**: ~0.5-1s per response on GPU
- **No training required**: Zero-shot state extraction

### Character Voice Critic
- **Model**: DeBERTa-v3-base (184M parameters)
- **Memory**: ~2.5GB GPU
- **Training**: 2-4 hours on single GPU
- **Inference**: ~50ms per dialogue on GPU
- **Requires**: CRD3 NPC dialogue data for training

## Limitations

### World Consistency Critic
1. Fantasy domain bias (trained on general text)
2. Cannot track implicit world knowledge
3. Keyword-based amnesia detection is simplistic
4. No forgetting mechanism for very long campaigns
5. Struggles with ambiguous descriptions

### Character Voice Critic
1. Requires substantial character dialogue (≥10 examples)
2. Assumes stable character personalities
3. Single-utterance context (no conversation history)
4. Returns neutral score for unknown characters
5. Domain-specific to Critical Role style

## Future Improvements

### World Critic
- Fine-tune Flan-T5 on D&D-specific state extraction
- Implement semantic similarity for amnesia detection
- Add state importance weighting
- Implement forgetting mechanism
- Add common-sense physical rules

### Character Critic
- Multi-turn context modeling
- Automatic personality trait extraction
- Character development tracking
- Cross-dataset transfer learning
- Contrastive learning for better embeddings

## Citation

```bibtex
@article{dropoutsquad2024director,
  title={The Director LLM: A Multi-Critic Reinforcement Learning Framework for Domain-Aware Narrative Generation},
  author={Dropout Squad},
  journal={International Institute of Information Technology, Hyderabad},
  year={2024}
}
```

## Contact

For questions or issues:
- Check README files in each critic directory
- Review the Kaggle implementation notebook
- Refer to the main paper for methodology details

---

**All components are ready for integration into your MCRL training pipeline!**
