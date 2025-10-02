"""
Prepare causal consistency training data from CRD3
Extract player action → DM response pairs
"""
import json
import random
from pathlib import Path
from datasets import Dataset, load_from_disk
from tqdm import tqdm

def extract_causal_pairs_from_crd3(crd3_dir="data/crd3"):
    """Extract player action → DM response pairs from CRD3"""
    print("\n" + "="*60)
    print("EXTRACTING CAUSAL PAIRS FROM CRD3")
    print("="*60)
    
    json_files = list(Path(crd3_dir).glob("*.json"))
    print(f"\nProcessing {len(json_files)} CRD3 files...")
    
    positive_pairs = []
    
    for file in tqdm(json_files, desc="Extracting pairs"):
        try:
            with open(file, 'r') as f:
                data = json.load(f)
            
            if isinstance(data, list):
                for chunk in data:
                    turns = chunk.get('TURNS', [])
                    
                    # Look for Player → DM sequences
                    for i in range(len(turns) - 1):
                        curr_turn = turns[i]
                        next_turn = turns[i + 1]
                        
                        curr_names = curr_turn.get('NAMES', [])
                        next_names = next_turn.get('NAMES', [])
                        
                        curr_text = ' '.join(curr_turn.get('UTTERANCES', [])).strip()
                        next_text = ' '.join(next_turn.get('UTTERANCES', [])).strip()
                        
                        if len(curr_text) < 20 or len(next_text) < 20:
                            continue
                        
                        # Check if Player → DM sequence
                        curr_is_dm = any(n.upper() in ['MATT', 'MATTHEW'] for n in curr_names)
                        next_is_dm = any(n.upper() in ['MATT', 'MATTHEW'] for n in next_names)
                        
                        # Player action → DM response
                        if not curr_is_dm and next_is_dm:
                            positive_pairs.append({
                                'premise': curr_text[:512],
                                'hypothesis': next_text[:512],
                                'label': 2,  # Entailment
                                'label_float': 1.0
                            })
        
        except Exception as e:
            continue
    
    print(f"\n✓ Extracted {len(positive_pairs):,} causal pairs")
    return positive_pairs

def create_negative_pairs(positive_pairs, n_negatives=None):
    """Create negative examples by mismatching contexts and responses"""
    if n_negatives is None:
        n_negatives = len(positive_pairs)
    
    print(f"\nCreating {n_negatives:,} negative pairs...")
    
    negative_pairs = []
    random.seed(42)
    
    for i in range(min(n_negatives, len(positive_pairs))):
        # Take premise from one pair, hypothesis from random other pair
        pair_i = positive_pairs[i]
        pair_j = positive_pairs[random.randint(0, len(positive_pairs) - 1)]
        
        # Ensure they're actually different
        if pair_i != pair_j:
            negative_pairs.append({
                'premise': pair_i['premise'],
                'hypothesis': pair_j['hypothesis'],  # Mismatched
                'label': 0,  # Contradiction/Not entailment
                'label_float': 0.0
            })
    
    print(f"✓ Created {len(negative_pairs):,} negative pairs")
    return negative_pairs

def prepare_causal_training_data():
    print("\n" + "="*60)
    print("PREPARING CAUSAL CRITIC TRAINING DATA")
    print("="*60)
    
    # Extract positive pairs
    positive_pairs = extract_causal_pairs_from_crd3()
    
    # Create negative pairs
    negative_pairs = create_negative_pairs(positive_pairs)
    
    # Combine
    all_pairs = positive_pairs + negative_pairs
    random.seed(42)
    random.shuffle(all_pairs)
    
    print(f"\n" + "="*60)
    print("DATASET STATISTICS")
    print("="*60)
    print(f"\nTotal pairs: {len(all_pairs):,}")
    print(f"  Positive (causal): {len(positive_pairs):,}")
    print(f"  Negative (non-causal): {len(negative_pairs):,}")
    
    # Split
    split_idx = int(len(all_pairs) * 0.9)
    train_data = all_pairs[:split_idx]
    val_data = all_pairs[split_idx:]
    
    print(f"\nSplits:")
    print(f"  Train: {len(train_data):,}")
    print(f"  Val: {len(val_data):,}")
    
    # Save
    output_dir = Path("data/causal_critic_training")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    Dataset.from_list(train_data).save_to_disk(output_dir / "train")
    Dataset.from_list(val_data).save_to_disk(output_dir / "val")
    
    print(f"\n✓ Saved to: {output_dir}")
    return True

if __name__ == "__main__":
    prepare_causal_training_data()