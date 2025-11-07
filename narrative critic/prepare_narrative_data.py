import json
import random
from datasets import Dataset
import os

def create_narrative_critic_dataset():
    """Create training data for narrative critic using ROCStories and synthetic negatives"""
    
    # Create data directory if it doesn't exist
    os.makedirs('data', exist_ok=True)
    
    # Load ROCStories data (you'll need to provide this)
    try:
        with open('data/rocstories_raw.json', 'r') as f:
            roc_data = json.load(f)
    except FileNotFoundError:
        print("ROCStories raw data not found. Creating dummy data for testing...")
        roc_data = [{"sentences": [f"This is story {i} sentence 1.", f"This is story {i} sentence 2."]} for i in range(100)]
    
    # Load D&D responses (you'll need to provide this)
    try:
        with open('data/dnd_responses.json', 'r') as f:
            dnd_data = json.load(f)
    except FileNotFoundError:
        print("D&D responses not found. Creating dummy data...")
        dnd_data = [{"text": f"D&D response {i}"} for i in range(100)]
    
    positive_examples = []
    
    # Add ROCStories as high-quality examples
    for story in roc_data[:min(30000, len(roc_data))]:
        text = " ".join(story['sentences']) if isinstance(story['sentences'], list) else story['text']
        positive_examples.append({'text': text, 'label': 1.0})
    
    # Add D&D responses as high-quality examples
    for response in dnd_data[:min(10906, len(dnd_data))]:
        positive_examples.append({'text': response['text'], 'label': 1.0})
    
    # Generate synthetic negatives
    synthetic_negatives = []
    
    # Shuffled examples (label 0.0)
    for example in positive_examples[:min(10571, len(positive_examples))]:
        sentences = example['text'].split('. ')
        random.shuffle(sentences)
        shuffled_text = '. '.join(sentences)
        synthetic_negatives.append({'text': shuffled_text, 'label': 0.0})
    
    # Repetitive examples (label 0.2)
    start_idx = min(10571, len(positive_examples))
    end_idx = min(21142, len(positive_examples))
    for example in positive_examples[start_idx:end_idx]:
        sentences = example['text'].split('. ')
        if len(sentences) > 1:
            repeated_sentences = sentences + [sentences[0]] * 2
            repetitive_text = '. '.join(repeated_sentences)
            synthetic_negatives.append({'text': repetitive_text, 'label': 0.2})
    
    # Truncated examples (label 0.4)
    start_idx = min(21142, len(positive_examples))
    end_idx = min(31935, len(positive_examples))
    for example in positive_examples[start_idx:end_idx]:
        sentences = example['text'].split('. ')
        if len(sentences) > 2:
            truncated_text = '. '.join(sentences[:2])
            synthetic_negatives.append({'text': truncated_text, 'label': 0.4})
    
    # Combine all examples
    all_examples = positive_examples + synthetic_negatives
    random.shuffle(all_examples)
    
    print(f"Total examples: {len(all_examples)}")
    print(f"Positive (1.0): {len([x for x in all_examples if x['label'] == 1.0])}")
    print(f"Shuffled (0.0): {len([x for x in all_examples if x['label'] == 0.0])}")
    print(f"Repetitive (0.2): {len([x for x in all_examples if x['label'] == 0.2])}")
    print(f"Truncated (0.4): {len([x for x in all_examples if x['label'] == 0.4])}")
    
    # Split into train/validation
    split_idx = int(0.9 * len(all_examples))
    train_data = all_examples[:split_idx]
    val_data = all_examples[split_idx:]
    
    # Save datasets
    with open('data/rocstoriestrain.json', 'w') as f:
        json.dump(train_data, f, indent=2)
    
    with open('data/rocstoriesval.json', 'w') as f:
        json.dump(val_data, f, indent=2)
    
    print(f"Saved {len(train_data)} training examples and {len(val_data)} validation examples")
    
    return train_data, val_data

if __name__ == "__main__":
    create_narrative_critic_dataset()