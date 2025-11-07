"""
World Consistency Critic - Data Preparation Pipeline
Extracts multi-turn sequences from CRD3 and applies corruption functions
to create balanced training data for consistency detection.

Corruption Strategy (like Character Voice Critic negative sampling):
- 25% Consistent (original CRD3 sequences)
- 25% Contradiction (flip object/entity states)
- 25% Hallucination (inject excessive entities)
- 25% Amnesia (remove tracked information)
"""

import json
import random
import re
from pathlib import Path
from typing import Dict, List, Tuple, Set
from collections import defaultdict
from tqdm import tqdm


class WorldStateExtractor:
    """Extract world state from text using regex patterns"""
    
    def __init__(self):
        # NPC/Entity patterns
        self.npc_patterns = [
            r'\b(goblin|orc|dragon|wizard|innkeeper|guard|merchant|bard|chieftain|dwarf|elf|scholar|librarian|student|monk|child|vendor|villager|blacksmith|servant|figure)\b',
            r'\b(five|three|two|ten|\d+)\s+(merchants|guards|scholars|servants|villagers|children|students|monks|vendors|figures)\b',
        ]
        
        # Object patterns
        self.object_patterns = [
            r'\b(door|chest|key|sword|shield|potion|book|gem|ring|amulet|scroll|bag|box|torch|candle|chalice|cup|ward|gate|coin|dagger|map|artifact)\b',
        ]
        
        # State patterns
        self.state_patterns = {
            'locked': r'\b(locked|securely locked)\b',
            'unlocked': r'\b(unlocked|unlock)\b',
            'open': r'\b(open|swings open|opened)\b',
            'closed': r'\b(closed|shut)\b',
            'lit': r'\b(lit|ignite|burning|burns)\b',
            'unlit': r'\b(unlit|extinguish)\b',
            'destroyed': r'\b(shatter|broken|destroyed|dissipate)\b',
            'consumed': r'\b(drink|consume|drunk)\b',
        }
        
        # NPC name pattern
        self.name_pattern = r'\b(?:I am|my name is|named)\s+([A-Z][a-z]+)\b'
    
    def extract(self, text: str) -> Dict:
        """Extract entities, objects, and states from text"""
        text_lower = text.lower()
        
        result = {
            'entities': set(),
            'objects': {},  # object_name -> state
            'npc_names': {},  # entity_type -> name
            'passwords': set(),
        }
        
        # Extract entities
        for pattern in self.npc_patterns:
            matches = re.findall(pattern, text_lower)
            for match in matches:
                if isinstance(match, tuple):
                    entity = match[-1] if match[-1] else match[0]
                else:
                    entity = match
                if entity:
                    result['entities'].add(entity.strip())
        
        # Extract objects with states
        for pattern in self.object_patterns:
            matches = re.findall(pattern, text_lower)
            for match in matches:
                obj_name = match if isinstance(match, str) else match[0]
                
                # Detect state
                state = None
                for state_name, state_pattern in self.state_patterns.items():
                    if re.search(state_pattern, text_lower):
                        # Check if state applies to this object (within 50 chars)
                        obj_pos = text_lower.find(obj_name)
                        state_match = re.search(state_pattern, text_lower)
                        if state_match and abs(state_match.start() - obj_pos) < 50:
                            state = state_name
                            break
                
                result['objects'][obj_name] = state
        
        # Extract NPC names
        name_matches = re.findall(self.name_pattern, text)
        for name in name_matches:
            # Try to associate with an entity mentioned nearby
            for entity in result['entities']:
                if entity in text_lower:
                    result['npc_names'][entity] = name
                    break
        
        # Extract passwords
        password_patterns = [
            r"password is ['\"]?(\w+)['\"]?",
            r"passphrase is ['\"]?(\w+)['\"]?",
            r"secret word is ['\"]?(\w+)['\"]?",
        ]
        for pattern in password_patterns:
            matches = re.findall(pattern, text_lower)
            result['passwords'].update(matches)
        
        return result


class CorruptionFunctions:
    """Apply corruption functions to create negative examples"""
    
    @staticmethod
    def inject_contradiction(example: Dict) -> Dict:
        """Violate established object/entity states"""
        corrupted = example.copy()
        
        response = corrupted['dm_response']
        world_state = corrupted['world_state']
        
        # Get tracked objects with states
        objects_with_states = [(obj, state) for obj, state in world_state['objects'].items() if state]
        
        if not objects_with_states:
            # No objects to contradict, return unchanged
            return None
        
        # Pick random object to contradict
        obj_name, current_state = random.choice(objects_with_states)
        
        # Define contradictions
        contradictions = {
            'locked': ['open', 'unlocked', 'swings open'],
            'unlocked': ['locked', 'securely locked'],
            'open': ['locked', 'closed'],
            'closed': ['open', 'swings open'],
            'lit': ['unlit', 'extinguished'],
            'unlit': ['lit', 'burning'],
            'destroyed': ['intact', 'whole', 'undamaged'],
        }
        
        if current_state in contradictions:
            new_state = random.choice(contradictions[current_state])
            
            # Create contradictory response
            contradiction_templates = [
                f"The {obj_name} is {new_state}.",
                f"You notice the {obj_name} is still {new_state}.",
                f"The {obj_name} remains {new_state}, blocking your path.",
                f"Despite your efforts, the {obj_name} is {new_state}.",
            ]
            
            corrupted['dm_response'] = response + " " + random.choice(contradiction_templates)
            corrupted['label'] = 'contradiction'
            corrupted['score'] = 0.0
            
            return corrupted
        
        return None
    
    @staticmethod
    def inject_hallucination(example: Dict) -> Dict:
        """Add excessive new entities"""
        corrupted = example.copy()
        
        hallucination_templates = [
            " Five merchants approach from the shadows, along with three guards and two scholars.",
            " Ten goblins suddenly appear, accompanied by a dragon and five wizards.",
            " The room fills with seven bards, four innkeepers, and six hooded figures.",
            " Eight merchants surround you, joined by five guards and three vendors.",
            " A dozen villagers emerge, along with five blacksmiths and four children.",
            " Nine scholars rush in, followed by six students and five monks.",
            " Seven guards arrive, accompanied by ten merchants and four servants.",
            " The tavern explodes with activity: eight bards, six merchants, and seven guards.",
        ]
        
        corrupted['dm_response'] = corrupted['dm_response'] + random.choice(hallucination_templates)
        corrupted['label'] = 'hallucination'
        corrupted['score'] = 0.3
        
        return corrupted
    
    @staticmethod
    def inject_amnesia(example: Dict) -> Dict:
        """Remove tracked information from response"""
        corrupted = example.copy()
        
        response = corrupted['dm_response']
        world_state = corrupted['world_state']
        
        amnesia_type = random.choice(['npc_name', 'password', 'object_location', 'destroyed_object'])
        
        if amnesia_type == 'npc_name' and world_state['npc_names']:
            # Forget NPC name
            entity_type, name = random.choice(list(world_state['npc_names'].items()))
            
            # Replace name with generic reference
            if name in response:
                corrupted['dm_response'] = response.replace(name, f"the {entity_type}")
            else:
                # Add amnesia statement
                amnesia_templates = [
                    f" The {entity_type} looks at you, but you can't recall their name.",
                    f" You try to remember the {entity_type}'s name, but it escapes you.",
                    f" The {entity_type} greets you, though their name is a blur.",
                ]
                corrupted['dm_response'] = response + random.choice(amnesia_templates)
        
        elif amnesia_type == 'password' and world_state['passwords']:
            # Forget password
            amnesia_templates = [
                " You try to recall the password, but your mind is blank.",
                " The password... what was it? You can't seem to remember.",
                " You don't recall hearing any password.",
                " The secret word eludes your memory.",
            ]
            corrupted['dm_response'] = response + random.choice(amnesia_templates)
        
        elif amnesia_type == 'object_location':
            # Forget where object is
            objects = list(world_state['objects'].keys())
            if objects:
                obj = random.choice(objects)
                amnesia_templates = [
                    f" You look around for the {obj}, but can't remember where it is.",
                    f" The {obj}'s location escapes you.",
                    f" You're not sure where you left the {obj}.",
                ]
                corrupted['dm_response'] = response + random.choice(amnesia_templates)
        
        elif amnesia_type == 'destroyed_object':
            # Reference destroyed object as still existing
            destroyed_objects = [obj for obj, state in world_state['objects'].items() 
                               if state in ['destroyed', 'consumed']]
            if destroyed_objects:
                obj = random.choice(destroyed_objects)
                amnesia_templates = [
                    f" The {obj} sits before you, intact.",
                    f" You notice the {obj} on the pedestal.",
                    f" The {obj} gleams in the light.",
                ]
                corrupted['dm_response'] = response + random.choice(amnesia_templates)
        
        corrupted['label'] = 'amnesia'
        corrupted['score'] = 0.5
        
        return corrupted


class CRD3SequenceExtractor:
    """Extract multi-turn sequences from CRD3 data"""
    
    def __init__(self, window_size: int = 5):
        self.window_size = window_size
        self.state_extractor = WorldStateExtractor()
        
        # DM identifier
        self.dm_names = ['MATT', 'MATTHEW', 'DM']
        
        # Player characters to exclude
        self.player_characters = {
            'GROG', 'KEYLETH', 'PERCY', 'SCANLAN', 'TIBERIUS', 
            'VAX', "VAX'ILDAN", 'VEX', "VEX'AHLIA", 'PIKE',
            'TARYON', 'FJORD', 'JESTER', 'CALEB', 'BEAUREGARD', 
            'BEAU', 'NOTT', 'MOLLYMAUK', 'MOLLY', 'YASHA', 'CADUCEUS',
            'TRAVIS', 'MARISHA', 'TALIESIN', 'SAM', 'LIAM', 
            'LAURA', 'ASHLEY', 'ORION'
        }
    
    def extract_sequences(self, crd3_data: List[Dict]) -> List[Dict]:
        """Extract multi-turn sequences with world state tracking"""
        sequences = []
        
        # Group by episode
        episodes = defaultdict(list)
        for turn in crd3_data:
            episodes[turn['episode']].append(turn)
        
        print(f"Processing {len(episodes)} episodes...")
        
        for episode_id, turns in tqdm(episodes.items()):
            # Sort by turn number
            turns = sorted(turns, key=lambda x: x['turn_number'])
            
            # Extract DM turns only
            dm_turns = [t for t in turns if t['character'].upper() in self.dm_names]
            
            # Create sliding windows
            for i in range(len(dm_turns) - self.window_size + 1):
                window = dm_turns[i:i + self.window_size]
                
                # Build world state from all turns except the last one
                world_state = self._build_world_state(window[:-1])
                
                # Last turn is the response to evaluate
                dm_response = window[-1]['text']
                
                # Build history context
                history = [f"{t['character']}: {t['text']}" for t in window[:-1]]
                
                # Create sequence
                sequence = {
                    'episode': episode_id,
                    'history': history,
                    'world_state': world_state,
                    'dm_response': dm_response,
                    'label': 'consistent',
                    'score': 1.0,
                }
                
                sequences.append(sequence)
        
        return sequences
    
    def _build_world_state(self, turns: List[Dict]) -> Dict:
        """Build cumulative world state from turns"""
        state = {
            'entities': set(),
            'objects': {},
            'npc_names': {},
            'passwords': set(),
        }
        
        for turn in turns:
            extracted = self.state_extractor.extract(turn['text'])
            
            # Merge entities
            state['entities'].update(extracted['entities'])
            
            # Merge objects (later states override)
            state['objects'].update(extracted['objects'])
            
            # Merge NPC names
            state['npc_names'].update(extracted['npc_names'])
            
            # Merge passwords
            state['passwords'].update(extracted['passwords'])
        
        # Convert sets to lists for JSON serialization
        return {
            'entities': list(state['entities']),
            'objects': state['objects'],
            'npc_names': state['npc_names'],
            'passwords': list(state['passwords']),
        }


def prepare_training_data(
    crd3_file: str,
    output_file: str,
    num_examples: int = 100000,
    window_size: int = 5,
    seed: int = 42
):
    """
    Prepare balanced training data from CRD3.
    
    Args:
        crd3_file: Path to crd3_npc_dialogues.json
        output_file: Output path for training data
        num_examples: Total number of examples to generate
        window_size: Number of turns in sequence window
        seed: Random seed
    """
    random.seed(seed)
    
    print("Loading CRD3 data...")
    with open(crd3_file, 'r', encoding='utf-8') as f:
        crd3_data = json.load(f)
    
    print(f"Loaded {len(crd3_data)} turns from CRD3")
    
    # Extract sequences
    extractor = CRD3SequenceExtractor(window_size=window_size)
    sequences = extractor.extract_sequences(crd3_data)
    
    print(f"Extracted {len(sequences)} consistent sequences")
    
    # Sample to get desired number of consistent examples
    num_per_class = num_examples // 4
    
    if len(sequences) > num_per_class:
        consistent_examples = random.sample(sequences, num_per_class)
    else:
        consistent_examples = sequences
        print(f"Warning: Only {len(sequences)} sequences available, using all")
    
    print(f"\nGenerating corrupted examples...")
    
    # Apply corruption functions
    corruption = CorruptionFunctions()
    
    # Generate corruptions
    contradictions = []
    hallucinations = []
    amnesias = []
    
    # We need num_per_class of each type
    attempts = 0
    max_attempts = len(sequences) * 3  # Try up to 3x the sequences
    
    source_pool = sequences.copy()
    random.shuffle(source_pool)
    
    for example in tqdm(source_pool, desc="Generating corruptions"):
        if len(contradictions) >= num_per_class and \
           len(hallucinations) >= num_per_class and \
           len(amnesias) >= num_per_class:
            break
        
        # Try contradiction
        if len(contradictions) < num_per_class:
            corrupted = corruption.inject_contradiction(example)
            if corrupted:
                contradictions.append(corrupted)
        
        # Try hallucination
        if len(hallucinations) < num_per_class:
            corrupted = corruption.inject_hallucination(example)
            if corrupted:
                hallucinations.append(corrupted)
        
        # Try amnesia
        if len(amnesias) < num_per_class:
            corrupted = corruption.inject_amnesia(example)
            if corrupted:
                amnesias.append(corrupted)
    
    # Combine all examples
    all_examples = consistent_examples + contradictions + hallucinations + amnesias
    
    # Shuffle
    random.shuffle(all_examples)
    
    print(f"\nDataset Statistics:")
    print(f"  Consistent: {len(consistent_examples)}")
    print(f"  Contradiction: {len(contradictions)}")
    print(f"  Hallucination: {len(hallucinations)}")
    print(f"  Amnesia: {len(amnesias)}")
    print(f"  Total: {len(all_examples)}")
    
    # Save
    print(f"\nSaving to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_examples, f, indent=2, ensure_ascii=False)
    
    print("Done!")
    
    return all_examples


if __name__ == "__main__":
    # Example usage
    prepare_training_data(
        crd3_file="crd3_npc_dialogues.json",
        output_file="world_consistency_training_data.json",
        num_examples=100000,
        window_size=5,
        seed=42
    )
