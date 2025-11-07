# World Consistency Critic

## Overview
The World Consistency Critic maintains coherent narrative world state across extended D&D interactions by detecting and scoring three distinct types of consistency violations. Unlike simple fact-checking, this critic tracks the evolving state of entities, objects, locations, and narrative facts throughout a campaign, ensuring the Dungeon Master's responses respect previously established information.

**Core Innovation**: Hybrid architecture combining symbolic state tracking (explicit fact storage) with neural state extraction (understanding natural language), enabling robust consistency checking without manual state annotations.

## Problem Statement

**Challenge**: In multi-turn D&D campaigns, world consistency failures break player immersion:
- **Contradictions**: "The door is locked" → "You walk through the door" (without unlocking)
- **Hallucinations**: "You enter the empty room" → "Ten merchants, eight guards, and five wizards greet you" (where did they come from?)
- **Amnesia**: NPC says "I am Gregor the Innkeeper" → DM later refers to "the innkeeper, whose name you don't know"

Language models struggle with consistency because:
1. No explicit memory of prior narrative facts
2. Generate plausible-sounding but contradictory statements
3. Introduce entities without grounding in established context
4. Forget important information from earlier turns

**Solution**: Build a specialized critic that:
- Explicitly tracks world state (entities, objects, facts) as a symbolic knowledge base
- Automatically extracts state updates from natural language using neural models
- Detects three consistency failure types with differentiated scoring
- Provides interpretable explanations for detected inconsistencies

## Consistency Failure Taxonomy

### 1. Contradictions (Score: 0.0 - Complete Failure)
**Definition**: Violating previously established facts about entity/object states

**Examples**:
```
✗ Contradiction Detected:
   Turn 1: "The ancient door is locked. You'll need a key."
   Turn 3: "You push the door open and enter."
   Issue: Door state changed from "locked" to "open" without unlock action
   Score: 0.0
```

```
✗ Contradiction Detected:
   Turn 1: "You see a single goblin guarding the entrance."
   Turn 2: "The three goblins attack simultaneously."
   Issue: Goblin count changed from 1 to 3 without explanation
   Score: 0.0
```

**Detection Method**: 
- Track object states (locked/unlocked, open/closed)
- Detect conflicting state claims in new responses
- High severity → complete penalty

### 2. Hallucinations (Score: 0.3 - Moderate Failure)
**Definition**: Introducing excessive unmentioned entities/objects without contextual justification

**Examples**:
```
⚠ Hallucination Detected:
   Turn 1: "You enter the quiet, dimly lit tavern. A few patrons sit scattered."
   Turn 2: "The tavern explodes with activity: ten merchants arguing loudly, 
            eight armored guards patrol the room, five bards compete in song, 
            seven scholars debate at tables, and a dozen servants rush about."
   Issue: 42 new entities introduced (merchants, guards, bards, scholars, servants)
         in supposedly "quiet" tavern
   Score: 0.3
```

```
⚠ Hallucination Detected:
   Context: "exploring abandoned library"
   Response: "Three dragons, five giants, and a horde of demons surround you."
   Issue: 8+ major entities appear in "abandoned" location
   Score: 0.3
```

**Detection Method**:
- Count newly introduced entities not previously mentioned
- Threshold: >2 new entities triggers hallucination penalty
- Moderate severity → partial penalty (allows some creativity)

### 3. Amnesia (Score: 0.5 - Partial Failure)
**Definition**: Forgetting or ignoring important prior facts (NPC names, passwords, quest objectives)

**Examples**:
```
⚠ Amnesia Detected:
   Turn 1: "The innkeeper greets you warmly. 'I'm Gregor. Welcome to my tavern!'"
   Turn 5: "You approach the innkeeper, but can't recall his name."
   Issue: NPC introduced himself as "Gregor" - name forgotten
   Score: 0.5
```

```
⚠ Amnesia Detected:
   Turn 2: "The guard whispers: 'The password is NIGHTFALL. Don't forget.'"
   Turn 4: "You try to remember the password, but your mind is blank."
   Issue: Password explicitly stated - should be remembered
   Score: 0.5
```

**Detection Method**:
- Track important facts: NPC names, passwords, quest objectives
- Detect forgetting phrases ("can't recall", "don't remember")
- Moderate severity → partial penalty (some forgetting acceptable)

### 4. Consistent (Score: 1.0 - Success)
**Definition**: Respecting all established world state

**Examples**:
```
✓ Consistent:
   Turn 1: "The chest is locked."
   Turn 2: "You pick the lock successfully."
   Turn 3: "The chest opens, revealing a golden amulet inside."
   Score: 1.0
```

## Architecture

### Hybrid Neural-Symbolic System

The critic combines the strengths of two complementary approaches:

**1. Symbolic State Tracking** (Deterministic, Explainable)
- Explicit knowledge base storing world facts
- Precise state queries and updates
- Guaranteed consistency checking
- Interpretable failure explanations

**2. Neural State Extraction** (Flexible, Robust)
- Handles natural language variability
- Extracts structured information from unstructured text
- Generalizes to unseen entity types
- No manual annotation required

### Component Architecture

#### Component 1: World State Tracker (Symbolic)

**Data Structures**:
```python
class WorldStateTracker:
    entities: Dict[str, EntityState]     # NPCs, monsters, creatures
    objects: Dict[str, ObjectState]      # Items, doors, containers
    locations: Set[str]                  # Places visited
    facts: List[Fact]                    # Narrative truths (timestamped)
```

**Entity State**:
```python
@dataclass
class EntityState:
    name: str                    # "goblin guard"
    count: int                   # How many (1, 3, "horde")
    status: str                  # "alive", "dead", "fleeing"
    location: Optional[str]      # Where entity is
    attributes: Dict[str, Any]   # Custom properties
    first_mentioned: int         # Turn number
```

**Object State**:
```python
@dataclass  
class ObjectState:
    name: str                    # "iron door"
    state: str                   # "locked", "unlocked", "open", "broken"
    location: Optional[str]      # Where object is
    properties: List[str]        # ["rusty", "heavy", "ancient"]
    interactions: List[str]      # ["examined", "picked up"]
```

**Fact Tracking**:
```python
@dataclass
class Fact:
    content: str                 # "The password is NIGHTFALL"
    turn: int                    # When fact was established
    importance: float            # 0-1 relevance score
    fact_type: str              # "password", "npc_name", "quest_objective"
```

**Operations**:
- `add_entity(name, count, status)`: Register new entity
- `update_object_state(name, new_state)`: Change object state
- `check_contradiction(entity, claimed_state)`: Verify consistency
- `get_facts_matching(keywords)`: Retrieve relevant facts

#### Component 2: State Extractor (Neural - DeBERTa-v3-small)

**Model**: Fine-tuned DeBERTa-v3-small (86M parameters)
- **Training Data**: 38,436 examples from CRD3 with synthetic corruption
- **Task**: 4-class sequence classification
- **Classes**: Contradiction (0), Hallucination (1), Amnesia (2), Consistent (3)
- **Performance**: 98.39% test accuracy

**Training Strategy - Corruption Functions**:

To generate training data without manual labeling, we systematically corrupt consistent examples:

**Contradiction Generation** (11 template variations):
```python
Original: "The door is locked."
Corrupted: "The door is unlocked." (inverts state)
Corrupted: "The door no longer blocks your path." (subtle contradiction)
Corrupted: "The open door swings in the breeze." (obvious contradiction)
```

**Hallucination Generation** (10 templates, randomized quantities 2-12):
```python
Original: "You enter the room."
Corrupted: "You enter the room filled with seven merchants and five guards."
Corrupted: "The area contains ten priests, three wizards, and eight scholars."
Entity Groups: merchants, guards, priests, wizards, scholars, bards, soldiers, 
               monks, citizens, servants, kobolds, goblins, orcs, ogres, trolls
```

**Amnesia Generation** (8-11 templates per strategy):
```python
# NPC Name Amnesia
Original: "The innkeeper Gregor greets you."
Corrupted: "The innkeeper greets you, though you can't recall his name."

# Password Amnesia  
Original: "Remember the password: NIGHTFALL"
Corrupted: "You try to recall the password, but it escapes you."

# Object State Amnesia
Original: "The chest is locked."
Corrupted: "You can't quite remember the chest's condition."
```

**Training Results**:
- Test Accuracy: **98.39%**
- Macro F1: **98.37%**
- Contradiction F1: 96.82%
- Hallucination F1: 99.17% (highest - easiest to detect)
- Amnesia F1: 98.01%
- Consistent F1: 99.47%

**Confusion Matrix**:
```
True\Pred    Contra  Halluc  Amnes  Consis
Contradiction  892      5      19     14      (95.9% precision)
Hallucination    3    924       4      0      (99.2% precision)
Amnesia          6      4     912      8      (98.1% precision)  
Consistent       0      0       0    930      (100% precision)
```

**Key Insight**: No overfitting detected - train/val performance nearly identical (<1% gap), validating diverse corruption templates.

#### Component 3: Integration Layer

**Workflow**:
```
Player Action → Extract Entities/Objects → Update State Tracker
                                                    ↓
DM Response → Neural Classifier → Class Prediction (0/1/2/3)
            → State Extractor → New Entities/Objects
                                                    ↓
                         Check Against Tracker → Detect Contradictions
                                                    ↓
                         Count New Entities → Detect Hallucinations
                                                    ↓
                         Match Important Facts → Detect Amnesia
                                                    ↓
                         Compute Final Score (0.0/0.3/0.5/1.0)
```

**Scoring Logic**:
```python
def score(dm_response, player_action):
    # 1. Neural classification (primary signal)
    predicted_class = neural_model(dm_response)
    
    # 2. Symbolic verification (catches neural misses)
    contradictions = tracker.check_contradictions(dm_response)
    new_entities = tracker.count_new_entities(dm_response)
    forgotten_facts = tracker.check_amnesia(dm_response)
    
    # 3. Combine signals (neural + symbolic)
    if contradictions or predicted_class == 0:
        return 0.0  # Contradiction
    elif new_entities > 2 or predicted_class == 1:
        return 0.3  # Hallucination
    elif forgotten_facts or predicted_class == 2:
        return 0.5  # Amnesia
    else:
        return 1.0  # Consistent
```

## Detailed Implementation Examples

### Example 1: Contradiction Detection

**Scenario**: Door state violation

```python
from world_consistency_critic import WorldConsistencyCritic

critic = WorldConsistencyCritic()

# Turn 1: Establish door is locked
player_action_1 = "I approach the ancient wooden door"
dm_response_1 = "The door is locked tight. You'll need a key."

critic.update_world_state(player_action_1)
critic.update_world_state(dm_response_1)
# Internal state: objects["ancient door"] = {state: "locked"}

# Turn 2: Player tries to open without key
player_action_2 = "I push on the door"
dm_response_2 = "The door swings open easily, revealing a dark corridor."

score = critic.score(dm_response_2, player_action_2)
print(f"Score: {score}")  # 0.0 - CONTRADICTION
# Reason: Door was "locked", now "open" without unlock action

explanation = critic.evaluate_with_explanation(dm_response_2, player_action_2)
print(explanation['reason'])
# "Contradiction detected: object 'door' state changed from 'locked' to 'open' 
#  without documented state transition (unlock action missing)"
```

**Correct Sequence** (Score: 1.0):
```python
# Turn 2 (correct): Player gets key and unlocks
player_action_2 = "I use the rusty key on the door"
dm_response_2 = "The key turns with a satisfying click. The door is now unlocked."
score = critic.score(dm_response_2, player_action_2)  # 1.0 - Consistent

# Turn 3: Now opening is consistent
player_action_3 = "I push the door open"
dm_response_3 = "The door swings open, revealing a dark corridor."
score = critic.score(dm_response_3, player_action_3)  # 1.0 - Consistent
```

### Example 2: Hallucination Detection

**Scenario**: Excessive entity introduction

```python
critic = WorldConsistencyCritic()

# Turn 1: Establish setting
player_action_1 = "I enter the tavern"
dm_response_1 = "You push open the door to a quiet, dimly lit tavern. " \
                "A few patrons sit scattered at tables."

critic.update_world_state(player_action_1)
critic.update_world_state(dm_response_1)
# Internal state: entities=["patrons (few)"], location="tavern"

# Turn 2: Massive hallucination
player_action_2 = "I look around the room"
dm_response_2 = """The tavern explodes with activity! Ten merchants argue loudly 
                   near the bar, eight armored guards patrol between tables, 
                   five bards compete in a musical duel, seven scholars debate 
                   philosophy, and a dozen servants rush back and forth with drinks."""

score = critic.score(dm_response_2, player_action_2)
print(f"Score: {score}")  # 0.3 - HALLUCINATION

# Count new entities introduced
new_entities = ["10 merchants", "8 guards", "5 bards", "7 scholars", "12 servants"]
total_new = sum([10, 8, 5, 7, 12])  # 42 entities!
print(f"Entities introduced: {total_new}")
# Threshold violation: >2 new entities in supposedly "quiet" tavern

explanation = critic.evaluate_with_explanation(dm_response_2, player_action_2)
print(explanation['reason'])
# "Hallucination detected: 42 new entities introduced (merchants, guards, bards, 
#  scholars, servants) without contextual justification. Previously described 
#  as 'quiet tavern with few patrons'."
```

**Acceptable Variation** (Score: 1.0):
```python
# Introducing 1-2 entities is acceptable
dm_response_2_ok = "A hooded traveler sits alone in the corner, nursing an ale."
score = critic.score(dm_response_2_ok, player_action_2)  # 1.0 - Consistent
# Only 1 new entity (hooded traveler) - within acceptable range
```

### Example 3: Amnesia Detection

**Scenario**: Forgetting NPC name

```python
critic = WorldConsistencyCritic()

# Turn 1: NPC introduction
player_action_1 = "I approach the innkeeper"
dm_response_1 = "A cheerful halfling greets you warmly. 'Welcome, friend! " \
                "I'm Gregor Thorngage, proprietor of this fine establishment.'"

critic.update_world_state(player_action_1)
critic.update_world_state(dm_response_1)
# Internal state: facts=["NPC name: Gregor Thorngage", importance=0.9]

# Turn 5: Name forgotten
player_action_5 = "I ask the innkeeper about local rumors"
dm_response_5 = "The innkeeper leans close, his name escaping you for the moment."

score = critic.score(dm_response_5, player_action_5)
print(f"Score: {score}")  # 0.5 - AMNESIA

explanation = critic.evaluate_with_explanation(dm_response_5, player_action_5)
print(explanation['reason'])
# "Amnesia detected: NPC introduced himself as 'Gregor Thorngage' (turn 1), 
#  but response indicates name is forgotten ('name escaping you')"
```

**Password Amnesia Example**:
```python
# Turn 2: Password given
dm_response_2 = "The guard whispers urgently: 'The password is NIGHTFALL. " \
                "Don't forget - you'll need it to enter the vault.'"
critic.update_world_state(dm_response_2)
# Internal state: facts=["password: NIGHTFALL", importance=1.0]

# Turn 4: Password forgotten
dm_response_4 = "You approach the vault door, but can't recall the password."
score = critic.score(dm_response_4, player_action_4)  # 0.5 - AMNESIA

# Correct version
dm_response_4_ok = "You approach the vault door and whisper 'NIGHTFALL'. It opens."
score = critic.score(dm_response_4_ok, player_action_4)  # 1.0 - Consistent
```

### Example 4: Complex Multi-Turn Scenario

**Full Campaign Snippet** (combining all consistency checks):

```python
critic = WorldConsistencyCritic()

turns = [
    # Turn 1: Exploration
    ("I search the ancient library",
     "You enter a dusty library. The shelves are filled with crumbling tomes. " \
     "A single goblin guard patrols the far end.",
     1.0),  # Consistent - establishing state
    
    # Turn 2: Combat initiation
    ("I sneak up on the goblin",
     "You creep closer. The goblin doesn't notice you yet.",
     1.0),  # Consistent - respects established single goblin
    
    # Turn 3: Contradiction - guard count changes
    ("I attack the goblin",
     "You leap from hiding! The three goblins turn to face you in surprise.",
     0.0),  # CONTRADICTION - was 1 goblin, now 3
    
    # Turn 4: Hallucination - excessive entities
    ("I look for an escape route",
     "The room suddenly fills with ten orc warriors, five ogres, eight trolls, " \
     "and a dozen goblins!",
     0.3),  # HALLUCINATION - 35 entities appear in small library
    
    # Turn 5: Amnesia - forgotten locked door
    ("I try to leave through the main door",
     "You walk through the door into the hallway.",
     0.5),  # AMNESIA - door was never mentioned or unlocked
]

for i, (action, response, expected_score) in enumerate(turns, 1):
    critic.update_world_state(action)
    score = critic.score(response, action)
    
    print(f"\n=== Turn {i} ===")
    print(f"Player: {action}")
    print(f"DM: {response[:80]}...")
    print(f"Score: {score} (Expected: {expected_score})")
    print(f"Consistency: {'✓' if score == expected_score else '✗'}")
    
    critic.update_world_state(response)
```

### Example 5: Correct Consistent Sequence

**Ideal Gameplay** (all turns score 1.0):

```python
critic = WorldConsistencyCritic()

campaign = [
    # Setup
    ("I approach the locked chest",
     "The iron chest is locked with a complex mechanism."),
    
    # Skill check
    ("I examine the lock closely",
     "You notice the lock requires a key, but also has a tumbler you could pick."),
    
    # Action
    ("I attempt to pick the lock",
     "You carefully work your tools. After several tense moments, you hear a click."),
    
    # State update
    ("I open the chest",
     "The chest opens, revealing a golden amulet and three healing potions."),
    
    # Item interaction
    ("I take the amulet",
     "You take the golden amulet. It feels warm to the touch."),
    
    # Callback to earlier state
    ("I examine the chest again",
     "The chest is now empty except for the three potions."),
]

all_consistent = True
for action, response in campaign:
    critic.update_world_state(action)
    score = critic.score(response, action)
    
    if score != 1.0:
        all_consistent = False
        print(f"❌ Inconsistency: {response}")
    
    critic.update_world_state(response)

if all_consistent:
    print("✅ Entire sequence is perfectly consistent!")
    print(f"Final world state: {critic.get_world_state()}")
```

### Example 6: Threshold Analysis

**Testing Hallucination Thresholds**:

```python
critic = WorldConsistencyCritic()
critic.update_world_state("I enter the throne room")

# Test different entity counts
test_cases = [
    ("The king sits on his throne.", 1, 1.0),  # 1 entity - fine
    ("The king and queen sit together.", 2, 1.0),  # 2 entities - acceptable
    ("The king, queen, and advisor discuss matters.", 3, 0.3),  # 3 entities - hallucination
    ("Five nobles, three guards, and the king are present.", 9, 0.3),  # 9 entities - severe hallucination
]

print("Entity Count vs Score:")
for response, entity_count, expected_score in test_cases:
    score = critic.score(response, "I look around")
    status = "✓" if score == expected_score else "✗"
    print(f"{status} {entity_count} entities → score {score}")
```

**Output**:
```
Entity Count vs Score:
✓ 1 entities → score 1.0
✓ 2 entities → score 1.0
✓ 3 entities → score 0.3
✓ 9 entities → score 0.3
```

## Key Assumptions

1. **Partial State Tracking**: The tracker only maintains explicitly mentioned state; implicit world knowledge is not tracked
2. **Extraction Accuracy**: Flan-T5 extraction may miss nuanced state changes; few-shot prompting provides reasonable but not perfect extraction
3. **Hallucination Tolerance**: Some entity introduction is acceptable (DMing involves creating new NPCs); only excessive hallucination (>2 entities) is penalized
4. **Amnesia Heuristic**: Context relevance uses simple keyword matching; more sophisticated semantic similarity could improve detection
5. **State Persistence**: World state persists across conversation until explicit reset; suitable for single-session campaigns

## Usage

### Basic Scoring
```python
from world_consistency_critic import WorldConsistencyCritic

# Initialize critic
critic = WorldConsistencyCritic(model_name="google/flan-t5-large")

# Update state with player action
player_action = "I search the ancient library for clues"
critic.update_world_state(player_action)

# Score DM response
dm_response = "You find a locked chest in the corner. The ancient door is now open."
score = critic.score(dm_response, player_action)
print(f"Consistency Score: {score}")  # May be low if door wasn't unlocked

# Update state with response
critic.update_world_state(dm_response)
```

### Detailed Evaluation
```python
# Get explanation with score
result = critic.evaluate_with_explanation(dm_response, player_action)
print(f"Score: {result['score']}")
print(f"Reason: {result['reason']}")
print(f"World State: {result['world_state']}")
```

### Multi-Turn Conversations
```python
critic = WorldConsistencyCritic()

turns = [
    ("I unlock the door with the key", "The door swings open, revealing a dark corridor."),
    ("I enter the corridor", "You step into the corridor. The door slams shut behind you."),
    ("I try to open the door", "The door is locked from the outside.")  # Consistent!
]

for player_action, dm_response in turns:
    critic.update_world_state(player_action)
    score = critic.score(dm_response, player_action)
    print(f"Turn Score: {score}")
    critic.update_world_state(dm_response)
```

### Resetting State (New Conversation)
```python
# Reset for new campaign/session
critic.reset()
```

## Integration with MCRL Pipeline

In the full Director LLM framework:

```python
# During RL training iteration
world_critic = WorldConsistencyCritic()

for episode in training_episodes:
    # Player takes action
    player_action = hybrid_player.generate()
    world_critic.update_world_state(player_action)
    
    # Policy generates response
    dm_response = policy.generate(player_action)
    
    # Evaluate consistency
    r_world = world_critic.score(dm_response, player_action)
    
    # Update state for next turn
    world_critic.update_world_state(dm_response)
    
    # Combine with other critics
    R = w_narr * r_narr + w_caus * r_caus + w_world * r_world + w_char * r_char
```

## Model Requirements

- **Flan-T5-Large**: ~780M parameters, ~3GB GPU memory
- **Extraction Time**: ~0.5-1 second per response on GPU
- **State Tracking**: O(1) symbolic operations, negligible overhead

## Limitations

1. **Fantasy Domain Bias**: Extraction trained on general text; may miss fantasy-specific state nuances
2. **Implicit State**: Cannot track implicit world knowledge (e.g., "goblins are weak to fire")
3. **Ambiguity Handling**: Struggles with vague descriptions that could be interpreted multiple ways
4. **Long-Term Memory**: No mechanism for forgetting old, irrelevant state; may accumulate clutter in very long campaigns
5. **Semantic Understanding**: Keyword-based amnesia detection is simplistic; doesn't capture deep semantic relevance

## Future Improvements

- Fine-tune Flan-T5 on D&D-specific state extraction examples
- Implement semantic similarity for amnesia detection (e.g., using sentence embeddings)
- Add state importance weighting (recent facts matter more)
- Implement forgetting mechanism for very old state
- Add explicit world model for common-sense physical rules

## File Structure

```
world consistency critic/
├── world_consistency_critic.py    # Main implementation
├── README.md                        # This file
└── requirements.txt                 # Dependencies
```

## Dependencies

```
torch>=2.0.0
transformers>=4.30.0
numpy>=1.24.0
```

Install with:
```bash
pip install torch transformers numpy
```

## Citation

If you use this critic in your work, please cite:

```
@article{dropoutsquad2024director,
  title={The Director LLM: A Multi-Critic Reinforcement Learning Framework for Domain-Aware Narrative Generation},
  author={Dropout Squad},
  journal={IIIT Hyderabad},
  year={2024}
}
```
