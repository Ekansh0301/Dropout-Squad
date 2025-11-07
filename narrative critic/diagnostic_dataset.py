"""Quick dataset diagnostic to find issues"""
import json
import numpy as np

# Load data
with open('./dataset_json/rocstoriestrain.json', 'r') as f:
    train_data = json.load(f)

with open('./dataset_json/rocstoriesval.json', 'r') as f:
    val_data = json.load(f)

print("="*80)
print("DATASET DIAGNOSTIC")
print("="*80)

# Check labels
train_labels = [x['label_float'] for x in train_data]
val_labels = [x['label_float'] for x in val_data]

print(f"\nTraining Data: {len(train_labels)} examples")
print(f"  Mean: {np.mean(train_labels):.4f}")
print(f"  Std:  {np.std(train_labels):.4f}")
print(f"  Min:  {np.min(train_labels):.4f}")
print(f"  Max:  {np.max(train_labels):.4f}")

print(f"\nValidation Data: {len(val_labels)} examples")
print(f"  Mean: {np.mean(val_labels):.4f}")
print(f"  Std:  {np.std(val_labels):.4f}")
print(f"  Min:  {np.min(val_labels):.4f}")
print(f"  Max:  {np.max(val_labels):.4f}")

# Check per type
print("\n" + "="*80)
print("PER-TYPE ANALYSIS")
print("="*80)

for dataset_name, data in [("TRAIN", train_data), ("VAL", val_data)]:
    print(f"\n{dataset_name}:")
    types = {}
    for x in data:
        types.setdefault(x['type'], []).append(x['label_float'])
    
    for qtype in sorted(types.keys()):
        labels = types[qtype]
        print(f"  {qtype:12s}: mean={np.mean(labels):.4f}, std={np.std(labels):.4f}, n={len(labels):5d}, range=[{np.min(labels):.3f}, {np.max(labels):.3f}]")

# Check for issues
print("\n" + "="*80)
print("POTENTIAL ISSUES")
print("="*80)

# Issue 1: Check if 'label' field exists and is different from label_float
if 'label' in train_data[0]:
    train_label_ints = [x['label'] for x in train_data]
    unique_ints = set(train_label_ints)
    print(f"\n❌ ISSUE 1: 'label' field exists with values: {unique_ints}")
    print("   This is a CLASSIFICATION field (0/1), NOT regression!")
    print("   Model might be seeing wrong labels during training!")
else:
    print("\n✅ No 'label' field (good)")

# Issue 2: Check label ranges
coherent_labels = [x['label_float'] for x in train_data if x['type'] == 'coherent']
shuffled_labels = [x['label_float'] for x in train_data if x['type'] == 'shuffled']

if np.mean(coherent_labels) < 0.7:
    print(f"\n⚠️  ISSUE 2: Coherent mean ({np.mean(coherent_labels):.4f}) lower than expected (0.7-1.0)")
    
if np.mean(shuffled_labels) > 0.3:
    print(f"\n⚠️  ISSUE 2: Shuffled mean ({np.mean(shuffled_labels):.4f}) higher than expected (0.0-0.3)")

# Issue 3: Check for duplicate texts
print("\n\nChecking for duplicates...")
texts = [x['text'] for x in train_data]
unique_texts = set(texts)
if len(texts) != len(unique_texts):
    print(f"⚠️  ISSUE 3: Found {len(texts) - len(unique_texts)} duplicate texts")
else:
    print("✅ No duplicates")

# Issue 4: Check text lengths
text_lengths = [len(x['text'].split()) for x in train_data]
print(f"\n\nText lengths (words):")
print(f"  Mean: {np.mean(text_lengths):.1f}")
print(f"  Min:  {np.min(text_lengths)}")
print(f"  Max:  {np.max(text_lengths)}")

if np.min(text_lengths) < 5:
    print("⚠️  ISSUE 4: Some texts are very short")

# Issue 5: Sample examples
print("\n" + "="*80)
print("SAMPLE EXAMPLES")
print("="*80)

for qtype in ['coherent', 'shuffled', 'repetitive', 'truncated']:
    examples = [x for x in train_data if x['type'] == qtype][:2]
    print(f"\n{qtype.upper()}:")
    for ex in examples:
        print(f"  Label: {ex['label_float']:.3f}")
        print(f"  Text: {ex['text'][:150]}...")
        print()
