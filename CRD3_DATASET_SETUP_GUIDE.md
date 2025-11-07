# CRD3 Dataset Setup Guide for Character Voice Critic

## Overview

The CRD3 (Critical Role Dungeons & Dragons Dataset) contains transcribed dialogues from 159 episodes with 398,682 turns. For the Character Voice Critic, we need to extract NPC dialogue with character attribution to train the model on character-specific speech patterns.

**Repository**: https://github.com/RevanthRameshkumar/CRD3

---

## Quick Start (TL;DR)

```bash
# 1. Clone CRD3 repository
git clone https://github.com/RevanthRameshkumar/CRD3.git

# 2. Update the path in extract_crd3_npc_dialogues.py
# Edit line 18: CRD3_DIR = "./CRD3/data/aligned data/c=3/"

# 3. Run extraction
python extract_crd3_npc_dialogues.py

# 4. Use extracted data in Character Voice Critic
# See character voice critic/README.md for training instructions
```

---

## Step 1: Download the CRD3 Dataset

### Option A: Clone the Repository (Recommended)

```bash
# Clone the repository
git clone https://github.com/RevanthRameshkumar/CRD3.git
cd CRD3
```

### Option B: Download Specific Data Files

The aligned dialogue data is located in: `data/aligned data/c=3/`

You can download individual JSON files or the entire folder from:
```
https://github.com/RevanthRameshkumar/CRD3/tree/master/data/aligned%20data/c%3D3
```

**Why c=3?** 
- Chunk size 3 provides good context (2 previous turns)
- Matches the specification in your report: "c=3 chunk format (2 previous turns context)"

---

## Step 2: Extract NPC Dialogue Data

Create a Python script to extract NPC dialogues with character attribution:

### Script: `extract_crd3_npc_dialogues.py`

```python
import json
import os
from pathlib import Path
from collections import defaultdict

def extract_npc_dialogues(crd3_data_dir, output_file):
    """
    Extract NPC dialogues from CRD3 aligned data.
    
    Args:
        crd3_data_dir: Path to CRD3/data/aligned data/c=3/
        output_file: Output JSON file path
    """
    
    # DM name (Matthew Mercer)
    DM_NAMES = ['MATT', 'MATTHEW', 'DM']
    
    # Player character names (to exclude)
    PLAYER_CHARACTERS = [
        'GROG', 'KEYLETH', 'PERCY', 'SCANLAN', 'TIBERIUS', 
        'VAX', "VAX'ILDAN", 'VEX', "VEX'AHLIA", 'PIKE',
        'TARYON', 'FJORD', 'JESTER', 'CALEB', 'BEAUREGARD', 
        'BEAU', 'NOTT', 'MOLLYMAUK', 'MOLLY', 'YASHA', 'CADUCEUS',
        'TRAVIS', 'MARISHA', 'TALIESIN', 'SAM', 'LIAM', 
        'LAURA', 'ASHLEY', 'ORION'
    ]
    
    npc_dialogues = []
    character_counts = defaultdict(int)
    
    # Process all JSON files in c=3 directory
    json_files = list(Path(crd3_data_dir).glob('*.json'))
    print(f"Found {len(json_files)} episode files")
    
    for json_file in json_files:
        print(f"Processing {json_file.name}...")
        
        with open(json_file, 'r', encoding='utf-8') as f:
            episode_data = json.load(f)
        
        # Extract episode info from filename (e.g., C1E001_3_1.json)
        episode_id = json_file.stem.split('_')[0]  # e.g., C1E001
        
        for chunk in episode_data:
            turns = chunk.get('TURNS', [])
            
            for turn in turns:
                names = turn.get('NAMES', [])
                utterances = turn.get('UTTERANCES', [])
                
                # Skip if no names or utterances
                if not names or not utterances:
                    continue
                
                # Process each name in the turn
                for name in names:
                    name_upper = name.upper().strip()
                    
                    # Skip DM and player characters
                    if name_upper in DM_NAMES:
                        continue
                    if any(pc in name_upper for pc in PLAYER_CHARACTERS):
                        continue
                    
                    # This is likely an NPC!
                    dialogue_text = ' '.join(utterances)
                    
                    # Skip very short utterances (likely crosstalk/reactions)
                    if len(dialogue_text.split()) < 3:
                        continue
                    
                    # Get context from previous turn if available
                    turn_idx = turns.index(turn)
                    context = ""
                    if turn_idx > 0:
                        prev_turn = turns[turn_idx - 1]
                        prev_utterances = prev_turn.get('UTTERANCES', [])
                        context = ' '.join(prev_utterances[:2])  # First 2 utterances
                    
                    npc_dialogues.append({
                        'character': name.strip(),
                        'text': dialogue_text,
                        'context': context,
                        'episode': episode_id,
                        'turn_number': turn.get('NUMBER', -1)
                    })
                    
                    character_counts[name.strip()] += 1
    
    print(f"\nExtracted {len(npc_dialogues)} NPC dialogue turns")
    print(f"Found {len(character_counts)} unique NPC characters")
    
    # Show top characters
    print("\nTop 20 NPCs by dialogue count:")
    for char, count in sorted(character_counts.items(), key=lambda x: x[1], reverse=True)[:20]:
        print(f"  {char}: {count} turns")
    
    # Save to JSON
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(npc_dialogues, f, indent=2, ensure_ascii=False)
    
    print(f"\nSaved to {output_file}")
    
    return npc_dialogues, character_counts


if __name__ == "__main__":
    # Update these paths
    CRD3_DIR = "./CRD3/data/aligned data/c=3/"
    OUTPUT_FILE = "crd3_npc_dialogues.json"
    
    npc_dialogues, char_counts = extract_npc_dialogues(CRD3_DIR, OUTPUT_FILE)
    
    # Also save character statistics
    with open("crd3_character_stats.json", 'w') as f:
        json.dump(dict(char_counts), f, indent=2)
    
    print("\n✓ Extraction complete!")
```

### Run the Extraction Script

```bash
python extract_crd3_npc_dialogues.py
```

**Expected Output:**
- `crd3_npc_dialogues.json` - Main NPC dialogue data (~10-50MB depending on extraction)
- `crd3_character_stats.json` - Character dialogue counts

---

## Step 3: Verify Data Quality

### Quality Check Script: `verify_crd3_data.py`

```python
import json
from collections import Counter

def verify_npc_data(filepath):
    """Verify extracted NPC dialogue data quality"""
    
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"Total dialogue examples: {len(data)}")
    
    # Character distribution
    characters = [d['character'] for d in data]
    char_counts = Counter(characters)
    
    print(f"\nUnique characters: {len(char_counts)}")
    print(f"Characters with ≥10 examples: {sum(1 for c in char_counts.values() if c >= 10)}")
    print(f"Characters with ≥50 examples: {sum(1 for c in char_counts.values() if c >= 50)}")
    
    # Text length distribution
    text_lengths = [len(d['text'].split()) for d in data]
    print(f"\nDialogue length (words):")
    print(f"  Mean: {sum(text_lengths) / len(text_lengths):.1f}")
    print(f"  Min: {min(text_lengths)}")
    print(f"  Max: {max(text_lengths)}")
    
    # Show sample
    print("\nSample entries:")
    for i in range(min(3, len(data))):
        print(f"\n{i+1}. Character: {data[i]['character']}")
        print(f"   Text: {data[i]['text'][:100]}...")
        print(f"   Context: {data[i]['context'][:80]}...")
    
    return char_counts

if __name__ == "__main__":
    char_counts = verify_npc_data("crd3_npc_dialogues.json")
    
    print("\n✓ Data verification complete!")
```

Run verification:
```bash
python verify_crd3_data.py
```

---

## Step 4: Upload to Kaggle Dataset

### Create Kaggle Dataset

1. **Prepare files:**
   ```
   crd3-npc-dialogues/
   ├── crd3_npc_dialogues.json
   ├── crd3_character_stats.json
   └── README.md (optional description)
   ```

2. **Upload to Kaggle:**
   - Go to https://www.kaggle.com/datasets
   - Click "New Dataset"
   - Upload the files
   - Title: "CRD3 NPC Dialogues for Character Voice Training"
   - Description: "Extracted NPC dialogues from Critical Role D&D Dataset with character attribution"

3. **Make it public or private** (your choice)

---

## Step 5: Use in Character Voice Critic Training

### Update your training code:

```python
from character_voice_critic import CharacterVoiceCritic

# Initialize critic
char_critic = CharacterVoiceCritic(
    model_name="microsoft/deberta-v3-base",
    num_characters=100
)

# Build training data from extracted CRD3 dialogues
training_data = char_critic.build_training_data_from_crd3(
    crd3_dialogue_file="/kaggle/input/crd3-npc-dialogues/crd3_npc_dialogues.json",
    output_file="character_voice_training.json"
)

print(f"Built {len(training_data)} training examples")

# Train the model
char_critic.train(
    training_data=training_data,
    output_dir="./character_voice_model",
    num_epochs=3,
    batch_size=8,  # Adjust for GPU memory
    learning_rate=2e-5
)

print("✓ Training complete!")
```

---

## Alternative: Download Pre-processed Data

If someone has already processed the data, you might find it on:
- **Kaggle Datasets**: Search for "CRD3 NPC" or "Critical Role Dialogue"
- **Hugging Face Datasets**: https://huggingface.co/datasets

However, it's better to extract it yourself to ensure data quality and format compatibility.

---

## Expected Data Statistics

Based on the paper and dataset:
- **Total episodes**: 159 (Campaigns 1 & 2)
- **Total turns**: ~398,682
- **Expected NPC dialogues**: 10,000 - 50,000 (after filtering)
- **Unique NPCs**: 100-500 (varying by episode frequency)
- **Major NPCs with ≥10 examples**: 50-100 characters

### Major NPCs to Expect:
- **Campaign 1**: Gilmore, Allura Vysoren, Cassandra de Rolo, Kima, etc.
- **Campaign 2**: Pumat Sol, Marion Lavorre (The Ruby), Essek Thelyss, etc.

---

## Troubleshooting

### Issue: Too many player character lines included

**Solution**: Update the `PLAYER_CHARACTERS` list in the extraction script with more variations:
```python
PLAYER_CHARACTERS = [
    'GROG', 'TRAVIS',
    'KEYLETH', 'MARISHA',
    'PERCY', 'PERCIVAL', 'TALIESIN',
    # ... add more variations
]
```

### Issue: Some NPC names are inconsistent

**Solution**: Add name normalization:
```python
def normalize_character_name(name):
    """Normalize character names for consistency"""
    name = name.strip().upper()
    
    # Handle common variations
    name_map = {
        'GILMORE': 'SHAUN GILMORE',
        'PUMAT': 'PUMAT SOL',
        # Add more mappings
    }
    
    return name_map.get(name, name)
```

### Issue: Not enough data per character

**Solution**: Lower the minimum dialogue threshold in `build_training_data_from_crd3()`:
```python
# In character_voice_critic.py, change from:
if len(dialogues) >= 10  # to:
if len(dialogues) >= 5   # for more characters with less data
```

---

## File Locations Summary

```
Your Project/
├── CRD3/                                    # Cloned repository
│   └── data/aligned data/c=3/*.json        # Source data
│
├── extract_crd3_npc_dialogues.py           # Extraction script
├── verify_crd3_data.py                     # Verification script
│
├── crd3_npc_dialogues.json                 # Extracted NPC data
├── crd3_character_stats.json               # Character statistics
│
└── character voice critic/
    ├── character_voice_critic.py
    └── README.md
```

---

## Next Steps

1. ✅ Clone CRD3 repository
2. ✅ Run extraction script
3. ✅ Verify data quality
4. ✅ Upload to Kaggle (optional)
5. ✅ Train Character Voice Critic
6. ✅ Integrate with MCRL pipeline

---

## Citation

When using CRD3 data, please cite:

```bibtex
@inproceedings{rameshkumar-bailey-2020-storytelling,
    title = "Storytelling with Dialogue: A Critical Role Dungeons and Dragons Dataset",
    author = "Rameshkumar, Revanth and Bailey, Peter",
    booktitle = "Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics",
    year = "2020",
    publisher = "Association for Computational Linguistics",
    pages = "5121--5134",
}
```

**License**: Creative Commons Attribution-ShareAlike 4.0 International License (CC-BY-SA 4.0)

---

## Support

For issues with the CRD3 dataset itself, open an issue at:
https://github.com/RevanthRameshkumar/CRD3/issues

For issues with the Character Voice Critic implementation, refer to:
`character voice critic/README.md`
