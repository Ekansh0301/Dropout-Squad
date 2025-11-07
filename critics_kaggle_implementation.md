# Director LLM - World Consistency & Character Voice Critics
# Kaggle Implementation Notebook

This notebook demonstrates how to use the World Consistency Critic and Character Voice Critic in the Kaggle environment.

## Setup Instructions

1. **Add Input Data**:
   - Upload `world_consistency_critic.py` to Kaggle dataset
   - Upload `character_voice_critic.py` to Kaggle dataset
   - Add the dataset to this notebook's input

2. **File Structure in Kaggle Input**:
   ```
   /kaggle/input/director-llm-critics/
   ├── world_consistency_critic.py
   └── character_voice_critic.py
   ```

## Installation and Imports

Install required packages (if not already available)

```python
!pip install -q transformers torch scikit-learn
```

Add input directory to Python path and import modules

```python
import sys
sys.path.append('/kaggle/input/director-llm-critics/')

# Import the critics
from world_consistency_critic import WorldConsistencyCritic, score_world_consistency
from character_voice_critic import CharacterVoiceCritic, score_character_voice

import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
```

---

## Part 1: World Consistency Critic Demo

### Initialize the Critic

```python
# Initialize World Consistency Critic
print("Initializing World Consistency Critic...")
world_critic = WorldConsistencyCritic(
    model_name="google/flan-t5-large",
    device="cuda" if torch.cuda.is_available() else "cpu"
)
print("✓ World Consistency Critic ready!")
```

### Example 1: Consistent World State

```python
print("\n=== Example 1: Consistent World State ===\n")

# Player unlocks door
player_action_1 = "I use the rusty key to unlock the ancient door"
world_critic.update_world_state(player_action_1)
print(f"Player: {player_action_1}")

# DM responds consistently
dm_response_1 = "The key turns with a satisfying click. The ancient door swings open, revealing a dark corridor beyond."
result_1 = world_critic.evaluate_with_explanation(dm_response_1, player_action_1)
world_critic.update_world_state(dm_response_1)

print(f"DM: {dm_response_1}")
print(f"\n📊 Score: {result_1['score']:.2f}")
print(f"📝 Reason: {result_1['reason']}")
print(f"🌍 World State: {result_1['world_state']}")
```

### Example 2: Contradiction Detection

```python
print("\n=== Example 2: Contradiction Detection ===\n")

# Player tries to open the previously unlocked door
player_action_2 = "I walk through the open doorway"
world_critic.update_world_state(player_action_2)
print(f"Player: {player_action_2}")

# DM contradicts (door is suddenly locked again)
dm_response_2 = "You reach for the handle, but the door is locked tight."
result_2 = world_critic.evaluate_with_explanation(dm_response_2, player_action_2)

print(f"DM: {dm_response_2}")
print(f"\n📊 Score: {result_2['score']:.2f}")
print(f"📝 Reason: {result_2['reason']}")
print(f"⚠️ Issue: Door was unlocked and opened, now suddenly locked")
```

### Example 3: Hallucination Detection

```python
print("\n=== Example 3: Hallucination Detection ===\n")

# Reset for new scenario
world_critic.reset()

player_action_3 = "I enter the empty tavern"
world_critic.update_world_state(player_action_3)
print(f"Player: {player_action_3}")

# DM introduces many unmentioned entities
dm_response_3 = "The tavern is bustling with activity. The innkeeper greets you, while the bard plays music. Three merchants argue in the corner, and a mysterious hooded figure watches from the shadows."
result_3 = world_critic.evaluate_with_explanation(dm_response_3, player_action_3)
world_critic.update_world_state(dm_response_3)

print(f"DM: {dm_response_3}")
print(f"\n📊 Score: {result_3['score']:.2f}")
print(f"📝 Reason: {result_3['reason']}")
print(f"🌍 Entities introduced: innkeeper, bard, 3 merchants, hooded figure")
```

### Multi-Turn Conversation Test

```python
print("\n=== Multi-Turn Conversation Test ===\n")

# Reset critic
world_critic.reset()

conversation = [
    ("I examine the wooden chest in the corner", 
     "You approach the ornate wooden chest. It appears to be locked with a complex mechanism."),
    
    ("I search the room for a key", 
     "You find a small brass key hidden under the bed."),
    
    ("I use the brass key on the chest", 
     "The key fits perfectly. The chest unlocks with a satisfying click."),
    
    ("I open the chest", 
     "Inside the chest, you find 50 gold coins and a mysterious scroll."),
    
    ("I take the scroll and examine it", 
     "The scroll contains an ancient spell written in flowing script.")
]

scores = []
for i, (player, dm) in enumerate(conversation, 1):
    world_critic.update_world_state(player)
    score = world_critic.score(dm, player)
    scores.append(score)
    world_critic.update_world_state(dm)
    
    print(f"Turn {i}:")
    print(f"  Player: {player}")
    print(f"  DM: {dm}")
    print(f"  Consistency Score: {score:.2f}\n")

print(f"Average Consistency: {sum(scores)/len(scores):.2f}")
```

---

## Part 2: Character Voice Critic Demo

**Note**: Character Voice Critic requires training on CRD3 data first. This section shows the training and usage workflow.

### Option A: Training from Scratch (if you have CRD3 data)

```python
print("\n=== Training Character Voice Critic ===\n")

# Initialize critic
char_critic = CharacterVoiceCritic(
    model_name="microsoft/deberta-v3-base",
    num_characters=50
)

# Build training data from CRD3
# Note: You need to have crd3_npc_dialogues.json in your input
# Uncomment if you have the data:

# training_data = char_critic.build_training_data_from_crd3(
#     crd3_dialogue_file="/kaggle/input/crd3-data/crd3_npc_dialogues.json",
#     output_file="character_voice_training.json"
# )
# 
# print(f"Built {len(training_data)} training examples")
# 
# # Train the model
# char_critic.train(
#     training_data=training_data,
#     output_dir="./character_voice_model",
#     num_epochs=3,
#     batch_size=8,  # Smaller for Kaggle memory limits
#     learning_rate=2e-5
# )
# 
# print("✓ Training complete!")
```

### Option B: Load Pretrained Model (recommended for Kaggle)

```python
print("\n=== Loading Pretrained Character Voice Critic ===\n")

# If you have a pretrained model in input data
# char_critic = CharacterVoiceCritic()
# char_critic.load_model("/kaggle/input/character-voice-model/")
# print("✓ Pretrained model loaded!")

# For demonstration without pretrained model:
print("⚠️ Skipping - requires pretrained model")
print("To use: upload trained model to Kaggle dataset and uncomment above")
```

### Demo: Character Voice Scoring (with mock model)

```python
print("\n=== Character Voice Scoring Demo (Conceptual) ===\n")

# This demonstrates the API - requires trained model to actually run
print("Example usage once model is trained:\n")

example_code = '''
# Score how well dialogue matches character
score = char_critic.score(
    character_name="Scanlan Shorthalt",
    dialogue="Well, well! What a delightful surprise!",
    context="entering tavern"
)
print(f"Voice Match Score: {score:.2f}")  # Expected: ~0.85 (in-character)

# Compare with out-of-character dialogue
score_ooc = char_critic.score(
    character_name="Scanlan Shorthalt",
    dialogue="By my sacred oath, I shall smite thee with divine fury!",
    context="in battle"
)
print(f"Out-of-Character Score: {score_ooc:.2f}")  # Expected: ~0.25 (not Scanlan's style)

# Get detailed evaluation
result = char_critic.evaluate_with_explanation(
    character_name="Scanlan Shorthalt",
    dialogue="Let me charm our way out of this one!",
    context="social encounter"
)
print(f"Score: {result['score']:.2f}")
print(f"Interpretation: {result['interpretation']}")
'''

print(example_code)
```

---

## Part 3: Integration Example - Multi-Critic Evaluation

### Combining Critics for MCRL

```python
print("\n=== Multi-Critic Integration Example ===\n")

# Initialize both critics
world_critic = WorldConsistencyCritic()

# Mock scores for other critics (narrative, causal)
# In full system, these would be actual critic outputs
r_narrative = 0.72  # From narrative critic
r_causal = 0.68     # From causal critic

# Example scenario
print("Scenario: Player in combat with goblin")
player_action = "I attack the goblin with my sword"
world_critic.update_world_state(player_action)

dm_response = "Your blade strikes true! The goblin falls to the ground, defeated. You notice a small pouch on its belt."

# Get world consistency score
r_world = world_critic.score(dm_response, player_action)
world_critic.update_world_state(dm_response)

# Mock character voice score (would need trained model)
r_character = 1.0  # No NPC dialogue, neutral

print(f"Player Action: {player_action}")
print(f"DM Response: {dm_response}\n")
print("Individual Critic Scores:")
print(f"  📖 Narrative Quality: {r_narrative:.2f}")
print(f"  🔗 Causal Consistency: {r_causal:.2f}")
print(f"  🌍 World Consistency: {r_world:.2f}")
print(f"  🎭 Character Voice: {r_character:.2f}")

# Intent-based dynamic weighting (ACTION context)
weights = {
    'narrative': 0.3,
    'causal': 0.7,
    'world': 0.6,
    'character': 0.4
}

# Aggregate reward (simplified - actual MCRL normalizes weights)
R_total = (
    weights['narrative'] * r_narrative +
    weights['causal'] * r_causal +
    weights['world'] * r_world +
    weights['character'] * r_character
) / sum(weights.values())

print(f"\n🎯 Aggregated Reward: {R_total:.2f}")
print(f"   (Using ACTION intent weights: heavy on causal & world)")
```

---

## Part 4: Performance Testing

### World Critic Performance Test

```python
print("\n=== World Critic Performance Test ===\n")

import time

# Test inference speed
world_critic.reset()

test_responses = [
    "The ancient door swings open.",
    "A goblin emerges from the shadows.",
    "You find a glowing sword on the altar.",
    "The room is filled with treasure.",
    "A dragon roars in the distance."
]

start_time = time.time()
scores = []

for response in test_responses:
    score = world_critic.score(response)
    scores.append(score)
    world_critic.update_world_state(response)

end_time = time.time()

print(f"Processed {len(test_responses)} responses")
print(f"Total time: {end_time - start_time:.2f}s")
print(f"Average time per response: {(end_time - start_time) / len(test_responses):.2f}s")
print(f"Average score: {sum(scores) / len(scores):.2f}")
```

---

## Summary

This notebook demonstrates:

1. ✅ **World Consistency Critic**:
   - Detects contradictions, hallucinations, and amnesia
   - Maintains world state across conversations
   - Provides interpretable scores

2. ⚠️ **Character Voice Critic**:
   - Requires training on CRD3 NPC dialogue data
   - Learns character-specific embeddings
   - Evaluates personality consistency

3. 🎯 **Integration**:
   - Multi-critic reward aggregation
   - Intent-based dynamic weighting
   - Ready for MCRL pipeline

## Next Steps

To use in your MCRL training:

1. Train Character Voice Critic on CRD3 data
2. Integrate critics into PPO training loop
3. Implement dynamic weight selection based on intent classification
4. Monitor critic scores during policy optimization

## Files Generated

After running this notebook, you'll have:
- World state tracking demonstrations
- Consistency scoring examples
- (If trained) Character voice model checkpoint

Save outputs and use in your main training pipeline!
