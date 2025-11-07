"""
Enhanced data preparation for 3-class causal consistency training.
Creates entailment, contradiction, AND neutral examples.
"""
import json
import random
from pathlib import Path
from datasets import Dataset
from tqdm import tqdm

def extract_causal_pairs_from_crd3(crd3_dir="../data/crd3"):
    """Extract sequential player-DM interaction pairs from CRD3 dataset."""
    print("\n" + "="*60)
    print("EXTRACTING CAUSAL PAIRS FROM CRD3")
    print("="*60)
    
    json_files = list(Path(crd3_dir).glob("*.json"))
    print(f"\nProcessing {len(json_files)} CRD3 files...")
    
    all_turns = []  # Store all turns for later pairing
    positive_pairs = []
    
    for file in tqdm(json_files, desc="Extracting pairs"):
        try:
            with open(file, 'r') as f:
                data = json.load(f)
            
            if isinstance(data, list):
                for chunk in data:
                    turns = chunk.get('TURNS', [])
                    
                    # Extract Player → DM sequences for ENTAILMENT
                    for i in range(len(turns) - 1):
                        curr_turn = turns[i]
                        next_turn = turns[i + 1]
                        
                        curr_names = curr_turn.get('NAMES', [])
                        next_names = next_turn.get('NAMES', [])
                        
                        curr_text = ' '.join(curr_turn.get('UTTERANCES', [])).strip()
                        next_text = ' '.join(next_turn.get('UTTERANCES', [])).strip()
                        
                        if len(curr_text) < 20 or len(next_text) < 20:
                            continue
                        
                        # Identify DM (Matthew/Matt) vs Player speakers
                        curr_is_dm = any(n.upper() in ['MATT', 'MATTHEW'] for n in curr_names)
                        next_is_dm = any(n.upper() in ['MATT', 'MATTHEW'] for n in next_names)
                        
                        # Store all turns for neutral pair generation
                        all_turns.append({
                            'text': curr_text[:512],
                            'is_dm': curr_is_dm
                        })
                        
                        # Collect Player action → DM response pairs (ENTAILMENT)
                        if not curr_is_dm and next_is_dm:
                            positive_pairs.append({
                                'premise': curr_text[:512],
                                'hypothesis': next_text[:512],
                                'label': 2,  # Entailment
                                'label_text': 'entailment'
                            })
        
        except Exception as e:
            continue
    
    print(f"\n✓ Extracted {len(positive_pairs):,} entailment pairs")
    print(f"✓ Collected {len(all_turns):,} total turns for neutral generation")
    return positive_pairs, all_turns

def create_contradiction_pairs(positive_pairs, n_contradictions=None):
    """Generate contradiction examples by mismatching contexts with unrelated responses."""
    if n_contradictions is None:
        n_contradictions = len(positive_pairs)
    
    print(f"\nCreating {n_contradictions:,} contradiction pairs...")
    
    contradiction_pairs = []
    random.seed(42)
    
    for i in range(min(n_contradictions, len(positive_pairs))):
        # Mismatch premise from one pair with hypothesis from a distant pair
        pair_i = positive_pairs[i]
        # Choose a random different pair
        j = random.randint(0, len(positive_pairs) - 1)
        while j == i and len(positive_pairs) > 1:
            j = random.randint(0, len(positive_pairs) - 1)
        pair_j = positive_pairs[j]
        
        contradiction_pairs.append({
            'premise': pair_i['premise'],
            'hypothesis': pair_j['hypothesis'],  # Unrelated response
            'label': 0,  # Contradiction
            'label_text': 'contradiction'
        })
    
    print(f"✓ Created {len(contradiction_pairs):,} contradiction pairs")
    return contradiction_pairs

def create_neutral_pairs(all_turns, n_neutral=None):
    """
    Generate neutral examples by pairing statements from similar contexts
    but without direct causal relationship.
    
    Strategy: Pair DM statements with other DM statements, or player statements
    with player statements from different contexts (same speaker type but unrelated).
    """
    if n_neutral is None:
        n_neutral = len(all_turns) // 3  # About 1/3 of total
    
    print(f"\nCreating {n_neutral:,} neutral pairs...")
    
    # Separate DM and player turns
    dm_turns = [t for t in all_turns if t['is_dm']]
    player_turns = [t for t in all_turns if not t['is_dm']]
    
    print(f"  DM turns: {len(dm_turns):,}")
    print(f"  Player turns: {len(player_turns):,}")
    
    neutral_pairs = []
    random.seed(42)
    
    # Create neutral pairs: same speaker type but unrelated context
    for i in range(n_neutral):
        if random.random() < 0.5 and len(dm_turns) >= 2:
            # Pair two DM statements
            idx1, idx2 = random.sample(range(len(dm_turns)), 2)
            neutral_pairs.append({
                'premise': dm_turns[idx1]['text'],
                'hypothesis': dm_turns[idx2]['text'],
                'label': 1,  # Neutral
                'label_text': 'neutral'
            })
        elif len(player_turns) >= 2:
            # Pair two player statements
            idx1, idx2 = random.sample(range(len(player_turns)), 2)
            neutral_pairs.append({
                'premise': player_turns[idx1]['text'],
                'hypothesis': player_turns[idx2]['text'],
                'label': 1,  # Neutral
                'label_text': 'neutral'
            })
        else:
            # Fallback: mix if one type is insufficient
            if dm_turns and player_turns:
                neutral_pairs.append({
                    'premise': random.choice(dm_turns)['text'],
                    'hypothesis': random.choice(player_turns)['text'],
                    'label': 1,
                    'label_text': 'neutral'
                })
    
    print(f"✓ Created {len(neutral_pairs):,} neutral pairs")
    return neutral_pairs

def prepare_causal_training_data_3class():
    """Create complete 3-class training dataset: entailment, contradiction, neutral."""
    print("\n" + "="*60)
    print("PREPARING 3-CLASS CAUSAL CRITIC TRAINING DATA")
    print("="*60)
    
    # Extract entailment examples from CRD3
    entailment_pairs, all_turns = extract_causal_pairs_from_crd3()
    
    # Generate contradiction examples (same count as entailment)
    contradiction_pairs = create_contradiction_pairs(entailment_pairs, 
                                                     n_contradictions=len(entailment_pairs))
    
    # Generate neutral examples (same count to balance dataset)
    neutral_pairs = create_neutral_pairs(all_turns, n_neutral=len(entailment_pairs))
    
    # Combine all three classes
    all_pairs = entailment_pairs + contradiction_pairs + neutral_pairs
    random.seed(42)
    random.shuffle(all_pairs)
    
    print(f"\n" + "="*60)
    print("DATASET STATISTICS")
    print("="*60)
    print(f"\nTotal pairs: {len(all_pairs):,}")
    print(f"  Entailment (label 2): {len(entailment_pairs):,}")
    print(f"  Contradiction (label 0): {len(contradiction_pairs):,}")
    print(f"  Neutral (label 1): {len(neutral_pairs):,}")
    
    # Verify label distribution
    label_counts = {}
    for pair in all_pairs:
        label = pair['label']
        label_counts[label] = label_counts.get(label, 0) + 1
    print(f"\nLabel distribution verification:")
    for label in sorted(label_counts.keys()):
        print(f"  Label {label}: {label_counts[label]:,}")
    
    # Create train/val/test splits (80/10/10)
    train_idx = int(len(all_pairs) * 0.8)
    val_idx = int(len(all_pairs) * 0.9)
    
    train_data = all_pairs[:train_idx]
    val_data = all_pairs[train_idx:val_idx]
    test_data = all_pairs[val_idx:]
    
    print(f"\nSplits:")
    print(f"  Train: {len(train_data):,}")
    print(f"  Val: {len(val_data):,}")
    print(f"  Test: {len(test_data):,}")
    
    # Save processed datasets
    output_dir = Path("../data/causal_critic_training_3class")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    Dataset.from_list(train_data).save_to_disk(output_dir / "train")
    Dataset.from_list(val_data).save_to_disk(output_dir / "val")
    Dataset.from_list(test_data).save_to_disk(output_dir / "test")
    
    # Save summary
    summary = {
        "total_examples": len(all_pairs),
        "train_size": len(train_data),
        "val_size": len(val_data),
        "test_size": len(test_data),
        "label_distribution": label_counts,
        "class_mapping": {
            "0": "contradiction",
            "1": "neutral",
            "2": "entailment"
        }
    }
    
    with open(output_dir / "dataset_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n✓ Saved to: {output_dir}")
    print(f"✓ Dataset summary saved to: {output_dir / 'dataset_summary.json'}")
    return True

if __name__ == "__main__":
    prepare_causal_training_data_3class()
