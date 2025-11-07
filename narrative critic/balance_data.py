# balance_data.py
import json
import random
import numpy as np

def balance_dataset():
    """Balance the dataset to have equal high and low quality examples"""
    
    with open('data/rocstoriestrain.json', 'r') as f:
        train_data = json.load(f)
    
    with open('data/rocstoriesval.json', 'r') as f:
        val_data = json.load(f)
    
    # Separate high and low quality
    high_quality_train = [ex for ex in train_data if ex['label'] >= 0.5]
    low_quality_train = [ex for ex in train_data if ex['label'] < 0.5]
    
    high_quality_val = [ex for ex in val_data if ex['label'] >= 0.5]
    low_quality_val = [ex for ex in val_data if ex['label'] < 0.5]
    
    print(f"Before balancing:")
    print(f"Train - High: {len(high_quality_train)}, Low: {len(low_quality_train)}")
    print(f"Val - High: {len(high_quality_val)}, Low: {len(low_quality_val)}")
    
    # Balance by oversampling minority class (high quality)
    if len(high_quality_train) < len(low_quality_train):
        # Need to add more high quality examples
        oversample_factor = len(low_quality_train) // len(high_quality_train)
        additional_high_quality = []
        
        for i in range(oversample_factor - 1):
            additional_high_quality.extend(high_quality_train.copy())
        
        # Add remaining to make equal
        remaining = len(low_quality_train) - (len(high_quality_train) + len(additional_high_quality))
        if remaining > 0:
            additional_high_quality.extend(random.sample(high_quality_train, remaining))
        
        balanced_train = high_quality_train + additional_high_quality + low_quality_train
    else:
        balanced_train = train_data  # Already balanced or high quality is majority
    
    # Same for validation
    if len(high_quality_val) < len(low_quality_val):
        oversample_factor = len(low_quality_val) // len(high_quality_val)
        additional_high_quality_val = []
        
        for i in range(oversample_factor - 1):
            additional_high_quality_val.extend(high_quality_val.copy())
        
        remaining = len(low_quality_val) - (len(high_quality_val) + len(additional_high_quality_val))
        if remaining > 0:
            additional_high_quality_val.extend(random.sample(high_quality_val, remaining))
        
        balanced_val = high_quality_val + additional_high_quality_val + low_quality_val
    else:
        balanced_val = val_data
    
    # Shuffle the balanced datasets
    random.shuffle(balanced_train)
    random.shuffle(balanced_val)
    
    # Save balanced datasets
    with open('data/balanced_rocstoriestrain.json', 'w') as f:
        json.dump(balanced_train, f, indent=2)
    
    with open('data/balanced_rocstoriesval.json', 'w') as f:
        json.dump(balanced_val, f, indent=2)
    
    # Verify balance
    high_quality_balanced_train = len([ex for ex in balanced_train if ex['label'] >= 0.5])
    low_quality_balanced_train = len([ex for ex in balanced_train if ex['label'] < 0.5])
    
    print(f"\nAfter balancing:")
    print(f"Train - High: {high_quality_balanced_train}, Low: {low_quality_balanced_train}")
    print(f"Total train samples: {len(balanced_train)}")
    print(f"Total val samples: {len(balanced_val)}")
    
    return balanced_train, balanced_val

if __name__ == "__main__":
    balance_dataset()