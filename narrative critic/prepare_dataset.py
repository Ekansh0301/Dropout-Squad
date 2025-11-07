"""
Dataset Preparation Script for Narrative Critic Training
Converts ROCStories CSV to JSON format with quality labels for training.

This script creates:
- Coherent examples (original stories)
- Shuffled examples (scrambled sentences)
- Repetitive examples (repeated sentences)
- Truncated examples (incomplete stories)
"""

import pandas as pd
import json
import random
from pathlib import Path
from typing import List, Dict
import numpy as np

# Set seed for reproducibility
random.seed(42)
np.random.seed(42)


def combine_story_sentences(row: pd.Series) -> str:
    """Combine the 5 sentences of a ROCStory into a single narrative."""
    sentences = [
        row['sentence1'],
        row['sentence2'],
        row['sentence3'],
        row['sentence4'],
        row['sentence5']
    ]
    return ' '.join(sentences)


def create_shuffled_story(row: pd.Series) -> str:
    """Create a shuffled version by randomizing sentence order."""
    sentences = [
        row['sentence1'],
        row['sentence2'],
        row['sentence3'],
        row['sentence4'],
        row['sentence5']
    ]
    random.shuffle(sentences)
    return ' '.join(sentences)


def create_repetitive_story(row: pd.Series) -> str:
    """Create a repetitive version by repeating sentences."""
    sentences = [
        row['sentence1'],
        row['sentence2'],
        row['sentence3'],
        row['sentence4'],
        row['sentence5']
    ]
    # Randomly select 2-3 sentences to repeat
    num_repeats = random.randint(2, 3)
    selected = random.sample(sentences, num_repeats)
    
    # Insert repetitions
    result = []
    for sent in sentences:
        result.append(sent)
        if sent in selected:
            result.append(sent)  # Repeat it
    
    return ' '.join(result)


def create_truncated_story(row: pd.Series) -> str:
    """Create a truncated version by cutting off the story early."""
    sentences = [
        row['sentence1'],
        row['sentence2'],
        row['sentence3'],
        row['sentence4'],
        row['sentence5']
    ]
    # Keep only 2-4 sentences
    num_keep = random.randint(2, 4)
    return ' '.join(sentences[:num_keep])


def generate_quality_labels(quality_type: str) -> float:
    """
    Generate quality scores for different narrative types.
    
    Quality scoring:
    - Coherent: 0.7-1.0 (high quality)
    - Shuffled: 0.0-0.3 (poor coherence)
    - Repetitive: 0.2-0.4 (repetitive/low quality)
    - Truncated: 0.3-0.5 (incomplete)
    """
    if quality_type == 'coherent':
        return round(random.uniform(0.7, 1.0), 3)
    elif quality_type == 'shuffled':
        return round(random.uniform(0.0, 0.3), 3)
    elif quality_type == 'repetitive':
        return round(random.uniform(0.2, 0.4), 3)
    elif quality_type == 'truncated':
        return round(random.uniform(0.3, 0.5), 3)
    else:
        return 0.5


def create_dataset_examples(df: pd.DataFrame, max_per_type: int = None) -> List[Dict]:
    """
    Create training examples from ROCStories dataset.
    
    Args:
        df: DataFrame with ROCStories
        max_per_type: Maximum examples per quality type (None = use all)
    
    Returns:
        List of training examples with text and quality labels
    """
    examples = []
    
    # Determine how many examples to create per type
    total_stories = len(df)
    if max_per_type is None:
        max_per_type = total_stories
    
    # Limit to available data
    num_examples = min(max_per_type, total_stories)
    
    print(f"\nGenerating {num_examples} examples per quality type...")
    
    # Coherent stories (original)
    print("  - Coherent stories...")
    for idx, row in df.head(num_examples).iterrows():
        examples.append({
            'text': combine_story_sentences(row),
            'label_float': generate_quality_labels('coherent'),  # ONLY regression label
            'source': 'rocstories',
            'type': 'coherent',
            'story_id': row['storyid']
        })
    
    # Shuffled stories
    print("  - Shuffled stories...")
    for idx, row in df.head(num_examples).iterrows():
        examples.append({
            'text': create_shuffled_story(row),
            'label_float': generate_quality_labels('shuffled'),  # ONLY regression label
            'source': 'rocstories',
            'type': 'shuffled',
            'story_id': row['storyid']
        })
    
    # Repetitive stories
    print("  - Repetitive stories...")
    for idx, row in df.head(num_examples).iterrows():
        examples.append({
            'text': create_repetitive_story(row),
            'label_float': generate_quality_labels('repetitive'),  # ONLY regression label
            'source': 'rocstories',
            'type': 'repetitive',
            'story_id': row['storyid']
        })
    
    # Truncated stories
    print("  - Truncated stories...")
    for idx, row in df.head(num_examples).iterrows():
        examples.append({
            'text': create_truncated_story(row),
            'label_float': generate_quality_labels('truncated'),  # ONLY regression label
            'source': 'rocstories',
            'type': 'truncated',
            'story_id': row['storyid']
        })
    
    print(f"\n  Created {len(examples)} total examples")
    
    # CRITICAL FIX: Remove duplicates
    print("  Removing duplicate texts...")
    examples_df = pd.DataFrame(examples)
    before_dedup = len(examples_df)
    examples_df = examples_df.drop_duplicates(subset=['text'], keep='first')
    after_dedup = len(examples_df)
    
    if before_dedup != after_dedup:
        print(f"  ⚠️  Removed {before_dedup - after_dedup} duplicate texts")
    else:
        print(f"  ✓ No duplicates found")
    
    # Shuffle all examples
    examples = examples_df.to_dict('records')
    random.shuffle(examples)
    
    return examples


def train_val_split(examples: List[Dict], val_ratio: float = 0.1) -> tuple:
    """Split examples into train and validation sets."""
    random.shuffle(examples)
    split_idx = int(len(examples) * (1 - val_ratio))
    return examples[:split_idx], examples[split_idx:]


def save_to_json(examples: List[Dict], output_path: str):
    """Save examples to JSON file."""
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(examples, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ Saved {len(examples)} examples to: {output_path}")


def print_dataset_stats(examples: List[Dict], split_name: str):
    """Print statistics about the dataset."""
    print(f"\n{'='*60}")
    print(f"{split_name.upper()} DATASET STATISTICS")
    print(f"{'='*60}")
    
    print(f"\nTotal examples: {len(examples):,}")
    
    # Count by type
    types = {}
    for ex in examples:
        type_key = ex['type']
        types[type_key] = types.get(type_key, 0) + 1
    
    print("\nExamples by type:")
    for type_name, count in sorted(types.items()):
        print(f"  {type_name.capitalize()}: {count:,}")
    
    # Quality score distribution
    scores = [ex['label_float'] for ex in examples]
    print(f"\nQuality scores:")
    print(f"  Min: {min(scores):.3f}")
    print(f"  Max: {max(scores):.3f}")
    print(f"  Mean: {np.mean(scores):.3f}")
    print(f"  Median: {np.median(scores):.3f}")
    
    # Show sample
    print(f"\nSample examples:")
    for i in range(min(3, len(examples))):
        ex = examples[i]
        print(f"\n  Example {i+1} ({ex['type']}):")
        print(f"    Text: {ex['text'][:100]}...")
        print(f"    Quality Score: {ex['label_float']:.3f}")


def main():
    """Main execution pipeline."""
    print("="*60)
    print("NARRATIVE CRITIC DATASET PREPARATION")
    print("="*60)
    
    # Configuration
    csv_path = "ROCStories__spring2016 - ROCStories_spring2016.csv"
    output_dir = "dataset_json"
    max_stories_per_type = 7500  # 7500 * 4 types = 30,000 examples
    val_ratio = 0.1
    
    # Load CSV
    print(f"\nLoading ROCStories from: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"✓ Loaded {len(df):,} stories")
    
    # Show sample
    print(f"\nSample story:")
    sample = df.iloc[0]
    print(f"  Title: {sample['storytitle']}")
    print(f"  Story: {combine_story_sentences(sample)}")
    
    # Generate examples
    print(f"\n{'='*60}")
    print("GENERATING TRAINING EXAMPLES")
    print(f"{'='*60}")
    
    all_examples = create_dataset_examples(df, max_per_type=max_stories_per_type)
    
    print(f"\n✓ Generated {len(all_examples):,} total examples")
    
    # Split into train/val
    print(f"\nSplitting into train/val (val_ratio={val_ratio})...")
    train_examples, val_examples = train_val_split(all_examples, val_ratio)
    
    print(f"✓ Train: {len(train_examples):,} examples")
    print(f"✓ Val: {len(val_examples):,} examples")
    
    # Print statistics
    print_dataset_stats(train_examples, "train")
    print_dataset_stats(val_examples, "validation")
    
    # Save to JSON
    print(f"\n{'='*60}")
    print("SAVING TO JSON")
    print(f"{'='*60}")
    
    train_path = f"{output_dir}/rocstoriestrain.json"
    val_path = f"{output_dir}/rocstoriesval.json"
    
    save_to_json(train_examples, train_path)
    save_to_json(val_examples, val_path)
    
    # CRITICAL: Verify no 'label' field exists
    print(f"\n{'='*60}")
    print("VERIFICATION")
    print(f"{'='*60}")
    
    print("\nVerifying dataset structure...")
    
    # Check train examples
    if train_examples:
        sample = train_examples[0]
        print(f"\n✓ Sample train example keys: {list(sample.keys())}")
        
        # Critical checks
        if 'label' in sample:
            print("  ❌ ERROR: 'label' field found (classification)!")
            print("     This will cause model collapse!")
        else:
            print("  ✅ No 'label' field (good)")
        
        if 'label_float' in sample:
            print("  ✅ 'label_float' field present (regression)")
        else:
            print("  ❌ ERROR: 'label_float' field missing!")
        
        # Show value ranges
        labels = [ex['label_float'] for ex in train_examples]
        print(f"\n  Label range: {min(labels):.3f} to {max(labels):.3f}")
        print(f"  Mean: {np.mean(labels):.3f}, Std: {np.std(labels):.3f}")
    
    # Final summary
    print(f"\n{'='*60}")
    print("✓ DATASET PREPARATION COMPLETE")
    print(f"{'='*60}")
    print(f"\nOutput files:")
    print(f"  Train: {train_path} ({len(train_examples):,} examples)")
    print(f"  Val: {val_path} ({len(val_examples):,} examples)")
    print(f"\nDataset composition:")
    print(f"  - Coherent stories: High quality (0.7-1.0)")
    print(f"  - Shuffled stories: Poor coherence (0.0-0.3)")
    print(f"  - Repetitive stories: Repetitive text (0.2-0.4)")
    print(f"  - Truncated stories: Incomplete (0.3-0.5)")
    print(f"\n✅ CRITICAL FIXES APPLIED:")
    print(f"  ✓ Removed 'label' field (classification)")
    print(f"  ✓ Using only 'label_float' (regression)")
    print(f"  ✓ Deduplicated texts")
    print(f"\nReady for Kaggle upload!")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
