"""
CRD3 NPC Dialogue Extraction Script

Extracts NPC dialogues with character attribution from the Critical Role 
Dungeons & Dragons Dataset for training the Character Voice Critic.

Usage:
    python extract_crd3_npc_dialogues.py

Requirements:
    - CRD3 repository cloned locally
    - Update CRD3_DIR path below to point to your CRD3/data/aligned data/c=3/ directory
"""

import json
import os
from pathlib import Path
from collections import defaultdict, Counter


# Configuration
CRD3_DIR = "../CRD3/data/aligned data/c=3/"  # Update this path!
OUTPUT_FILE = "crd3_npc_dialogues.json"
STATS_FILE = "crd3_character_stats.json"

# DM names (to exclude)
DM_NAMES = ['MATT', 'MATTHEW', 'DM', 'MERCER']

# Player character names (to exclude) - Campaign 1 & 2
PLAYER_CHARACTERS = [
    # Campaign 1 - Vox Machina
    'GROG', 'STRONGJAW',
    'KEYLETH',
    'PERCY', 'PERCIVAL', 'DE ROLO',
    'SCANLAN', 'SHORTHALT',
    'TIBERIUS', 'STORMWIND',
    'VAX', "VAX'ILDAN",
    'VEX', "VEX'AHLIA",
    'PIKE', 'TRICKFOOT',
    'TARYON', 'DARRINGTON',
    
    # Campaign 2 - Mighty Nein
    'FJORD', 'STONE',
    'JESTER', 'LAVORRE',
    'CALEB', 'WIDOGAST',
    'BEAUREGARD', 'BEAU', 'LIONETT',
    'NOTT', 'THE BRAVE',
    'MOLLYMAUK', 'MOLLY', 'TEALEAF',
    'YASHA', 'NYDOORIN',
    'CADUCEUS', 'CLAY',
    
    # Player names (real people)
    'TRAVIS', 'WILLINGHAM',
    'MARISHA', 'RAY',
    'TALIESIN', 'JAFFE',
    'SAM', 'RIEGEL',
    'LIAM', "O'BRIEN",
    'LAURA', 'BAILEY',
    'ASHLEY', 'JOHNSON',
    'ORION', 'ACABA'
]


def is_player_or_dm(name):
    """Check if a name is a DM or player character"""
    name_upper = name.upper().strip()
    
    # Check DM
    if name_upper in DM_NAMES:
        return True
    
    # Check player characters
    if any(pc in name_upper for pc in PLAYER_CHARACTERS):
        return True
    
    return False


def extract_npc_dialogues(crd3_data_dir, output_file, min_words=3):
    """
    Extract NPC dialogues from CRD3 aligned data.
    
    Args:
        crd3_data_dir: Path to CRD3/data/aligned data/c=3/
        output_file: Output JSON file path
        min_words: Minimum words in dialogue to include
        
    Returns:
        Tuple of (npc_dialogues list, character_counts dict)
    """
    
    npc_dialogues = []
    character_counts = defaultdict(int)
    
    # Check if directory exists
    data_path = Path(crd3_data_dir)
    if not data_path.exists():
        raise FileNotFoundError(
            f"CRD3 directory not found: {crd3_data_dir}\n"
            f"Please clone the CRD3 repository and update CRD3_DIR path."
        )
    
    # Process all JSON files in c=3 directory
    json_files = list(data_path.glob('*.json'))
    
    if not json_files:
        raise FileNotFoundError(
            f"No JSON files found in {crd3_data_dir}\n"
            f"Make sure you're pointing to the 'aligned data/c=3/' directory."
        )
    
    print(f"Found {len(json_files)} episode files in {crd3_data_dir}")
    print(f"Processing episodes...\n")
    
    processed_episodes = 0
    
    for json_file in sorted(json_files):
        # Extract episode info from filename (e.g., C1E001_3_1.json)
        episode_id = json_file.stem.split('_')[0]  # e.g., C1E001
        
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                episode_data = json.load(f)
        except Exception as e:
            print(f"⚠️  Error reading {json_file.name}: {e}")
            continue
        
        episode_npc_count = 0
        
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
                    # Skip DM and player characters
                    if is_player_or_dm(name):
                        continue
                    
                    # This is likely an NPC!
                    dialogue_text = ' '.join(utterances).strip()
                    
                    # Skip very short utterances (likely crosstalk/reactions)
                    if len(dialogue_text.split()) < min_words:
                        continue
                    
                    # Get context from previous turn if available
                    turn_idx = turns.index(turn)
                    context = ""
                    if turn_idx > 0:
                        prev_turn = turns[turn_idx - 1]
                        prev_utterances = prev_turn.get('UTTERANCES', [])
                        # Take first 2 utterances from previous turn as context
                        context = ' '.join(prev_utterances[:2]).strip()
                    
                    # Clean character name
                    character_name = name.strip()
                    
                    npc_dialogues.append({
                        'character': character_name,
                        'text': dialogue_text,
                        'context': context,
                        'episode': episode_id,
                        'turn_number': turn.get('NUMBER', -1)
                    })
                    
                    character_counts[character_name] += 1
                    episode_npc_count += 1
        
        processed_episodes += 1
        if processed_episodes % 10 == 0:
            print(f"Processed {processed_episodes}/{len(json_files)} episodes... "
                  f"({len(npc_dialogues)} NPC turns so far)")
    
    print(f"\n{'='*60}")
    print(f"Extraction Complete!")
    print(f"{'='*60}")
    print(f"Total episodes processed: {processed_episodes}")
    print(f"Total NPC dialogue turns: {len(npc_dialogues)}")
    print(f"Unique NPC characters: {len(character_counts)}")
    
    # Show top characters
    print(f"\n{'='*60}")
    print("Top 30 NPCs by Dialogue Count:")
    print(f"{'='*60}")
    for i, (char, count) in enumerate(
        sorted(character_counts.items(), key=lambda x: x[1], reverse=True)[:30], 1
    ):
        print(f"{i:2d}. {char:30s} : {count:4d} turns")
    
    # Character distribution statistics
    counts_list = list(character_counts.values())
    chars_with_10_plus = sum(1 for c in counts_list if c >= 10)
    chars_with_50_plus = sum(1 for c in counts_list if c >= 50)
    
    print(f"\n{'='*60}")
    print("Character Distribution:")
    print(f"{'='*60}")
    print(f"Characters with ≥10 examples: {chars_with_10_plus}")
    print(f"Characters with ≥50 examples: {chars_with_50_plus}")
    print(f"Mean dialogues per character: {sum(counts_list) / len(counts_list):.1f}")
    
    # Save to JSON
    print(f"\n{'='*60}")
    print("Saving files...")
    print(f"{'='*60}")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(npc_dialogues, f, indent=2, ensure_ascii=False)
    print(f"✓ Saved NPC dialogues to: {output_file}")
    print(f"  File size: {os.path.getsize(output_file) / 1024 / 1024:.2f} MB")
    
    # Save character statistics
    stats = {
        'total_dialogues': len(npc_dialogues),
        'unique_characters': len(character_counts),
        'characters_with_10_plus': chars_with_10_plus,
        'characters_with_50_plus': chars_with_50_plus,
        'character_counts': dict(sorted(
            character_counts.items(), 
            key=lambda x: x[1], 
            reverse=True
        ))
    }
    
    with open(STATS_FILE, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(f"✓ Saved statistics to: {STATS_FILE}")
    
    return npc_dialogues, character_counts


def verify_extraction(output_file):
    """Verify the extracted data"""
    print(f"\n{'='*60}")
    print("Verification")
    print(f"{'='*60}")
    
    with open(output_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Text length distribution
    text_lengths = [len(d['text'].split()) for d in data]
    print(f"\nDialogue Length Statistics (words):")
    print(f"  Mean: {sum(text_lengths) / len(text_lengths):.1f}")
    print(f"  Min: {min(text_lengths)}")
    print(f"  Max: {max(text_lengths)}")
    
    # Show samples
    print(f"\n{'='*60}")
    print("Sample Dialogues:")
    print(f"{'='*60}")
    
    # Show diverse samples
    sample_indices = [0, len(data)//4, len(data)//2, len(data)*3//4, -1]
    
    for i in sample_indices[:3]:  # Show 3 samples
        sample = data[i]
        print(f"\nCharacter: {sample['character']}")
        print(f"Episode: {sample['episode']}")
        print(f"Context: {sample['context'][:80]}{'...' if len(sample['context']) > 80 else ''}")
        print(f"Dialogue: {sample['text'][:120]}{'...' if len(sample['text']) > 120 else ''}")
        print("-" * 60)


if __name__ == "__main__":
    print("="*60)
    print("CRD3 NPC Dialogue Extraction")
    print("="*60)
    print(f"\nConfiguration:")
    print(f"  Input directory: {CRD3_DIR}")
    print(f"  Output file: {OUTPUT_FILE}")
    print(f"  Stats file: {STATS_FILE}")
    print(f"  Minimum words per dialogue: 3")
    print()
    
    try:
        # Extract dialogues
        npc_dialogues, char_counts = extract_npc_dialogues(
            CRD3_DIR, 
            OUTPUT_FILE,
            min_words=3
        )
        
        # Verify extraction
        verify_extraction(OUTPUT_FILE)
        
        print(f"\n{'='*60}")
        print("✓ Extraction Complete!")
        print(f"{'='*60}")
        print(f"\nNext steps:")
        print(f"1. Review {OUTPUT_FILE} to verify data quality")
        print(f"2. Upload to Kaggle dataset (optional)")
        print(f"3. Use with Character Voice Critic training:")
        print(f"   char_critic.build_training_data_from_crd3('{OUTPUT_FILE}')")
        
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print(f"\nTo fix:")
        print(f"1. Clone CRD3 repository:")
        print(f"   git clone https://github.com/RevanthRameshkumar/CRD3.git")
        print(f"2. Update CRD3_DIR variable in this script to point to:")
        print(f"   CRD3/data/aligned data/c=3/")
        
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
