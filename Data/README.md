# Dropout Squad - Project Data Documentation


## Overview

The Dropout Squad project utilizes multiple datasets for training different components of the multi-critic reinforcement learning system. Each dataset serves specific purposes in the training pipeline, from supervised fine-tuning to critic training and player simulation.

## Required Datasets

### 1. CRD3 (Critical Role Dungeons & Dragons Dataset)

**Purpose**: Primary training data for D&D narrative generation
**Location**: `../data/crd3/`
**Format**: JSON files containing episodic D&D session transcripts

**Structure**:

- **Files**: `C{season}E{episode}_{chunk}_{segment}.json`
- **Content**: Structured dialogue data with speaker identification, turn management, and narrative chunks
- **Coverage**: Campaign 1 & 2 episodes with 3-chunk segmentation

**Data Fields**:

```json
{
  "CHUNK": "Narrative summary",
  "ALIGNMENT": {
    "CHUNK ID": 0,
    "TURN START": 0,
    "TURN END": 11,
    "ALIGNMENT SCORE": 0.462
  },
  "TURNS": [
    {
      "NAMES": ["SPEAKER"],
      "UTTERANCES": ["dialogue text"],
      "NUMBER": 0
    }
  ]
}
```

**Usage**:

- DM-SFT baseline training
- Context understanding for narrative generation
- Multi-turn dialogue modeling

### 2. LIGHT (Learning in Interactive Games with Humans and Text)

**Purpose**: Fantasy dialogue and action simulation training
**Location**: `../data/light_dialogue_processed/`
**Format**: Text files with structured dialogue and action sequences

**Data Types**:

- **Action files**: `action_{split}.txt` - Character action prediction
- **Speech files**: `speech_{split}.txt` - Dialogue generation
- **Emote files**: `emote_{split}.txt` - Emotional expression
- **Which files**: `which_{split}.txt` - Action selection

**Splits Available**:

- `train`: Training data
- `valid`: Validation data
- `test`: Test data
- `test_unseen`: Unseen scenarios

**Structure Example**:

```
text:_task_action
_setting_name Location Name
_setting_desc Detailed environment description
_partner_name character_name
_self_name character_name
_self_persona Character background and personality
_object_desc item : Description of interactive objects
_partner_say Dialogue from other character
_self_say Your character's response
labels:action_label
label_candidates:option1|option2|option3
```

**Usage**:

- Hybrid player training (both generation and classification)
- Fantasy dialogue understanding
- Character behavior simulation
- Intent classification training

### 3. Causal Critic Training Data

**Purpose**: Training data for causal consistency evaluation
**Location**: `../data/causal_critic_training/`
**Structure**:

- `train/`: Training examples for premise-hypothesis pairs
- `val/`: Validation data for model evaluation

**Usage**:

- Training causal consistency critic
- NLI-based evaluation model development
- Logical reasoning assessment

### 4. Critic Training Data

**Purpose**: Comprehensive narrative quality assessment training data
**Location**: `../data/critic_training/`
**Format**: Arrow files with structured narrative examples and quality labels

**Structure**:

- `train/`: Training examples (36,815 samples)
- `val/`: Validation examples (4,091 samples)
- `data_summary.json`: Dataset composition and statistics

**Data Composition** (Total: 40,906 examples):

- **ROCStories source**: 30,000 examples for narrative coherence
- **D&D pairs source**: 10,906 examples from DM responses
- **Quality types**:
  - Coherent: 10,571 examples (high-quality narratives)
  - Shuffled: 10,571 examples (scrambled for contrast)
  - Repetitive: 10,571 examples (repetitive text detection)
  - Truncated: 9,193 examples (incomplete narratives)

**Usage**:

- Training narrative critic for quality assessment
- Learning to distinguish coherent vs. problematic text
- Regression training for narrative scoring

### 5. ROCStories (TinyStories)

**Purpose**: Large-scale narrative coherence and story completion training
**Location**: `../data/rocstories/`
**Format**: Arrow files from HuggingFace TinyStories dataset

**Structure**:

- `train/`: Training stories (4 arrow files)
- `validation/`: Validation stories
- `dataset_dict.json`: Split configuration

**Dataset Details**:

- **Source**: roneneldan/TinyStories from HuggingFace
- **Size**: ~1.9GB of story data
- **Content**: Short, coherent stories for narrative understanding
- **Usage in Project**: Primary source for narrative critic training (30,000 samples extracted)

**Usage**:

- Narrative coherence training
- Story completion and generation
- Baseline narrative quality understanding
- Feeding into critic_training dataset

### 6. Data Splits (DM-SFT Training Dataset)

**Purpose**: Combined and preprocessed training data for DM supervised fine-tuning
**Location**: `../data/splits/`
**Format**: Arrow files and JSON samples combining LIGHT and CRD3 data

**Structure**:

- `train/`: Main training split
- `validation/`: Validation split
- `test/`: Test split
- `*_sample.json`: Sample files for quick inspection
- `player_utterances.json`: Large collection of player responses

**Data Integration**:

- **CRD3 Integration**: DM responses with context from D&D sessions
- **LIGHT Integration**: Fantasy dialogue responses and scenarios
- **Format**: Instruction-tuned format with system prompts for DM role

**Sample Structure**:

```json
{
  "text": "<s>[INST] <<SYS>>\\nYou are an expert Dungeon Master...\\n<</SYS>>\\n\\nAs the Dungeon Master, describe a scene or respond to the player: [/INST] DM response </s>",
  "response": "DM response text",
  "source": "crd3" | "light"
}
```

**Usage**:

- Primary dataset for DM-SFT baseline training
- Combines fantasy dialogue understanding with D&D-specific content
- Instruction-tuned format for consistent DM behavior
- Cross-validation and testing for model evaluation

- Consistent evaluation across experiments
- Reproducible results


## Data Usage by Module

### DM-SFT Module

- **Primary**: Data Splits dataset (combined LIGHT + CRD3)
- **Processing**: Uses instruction-tuned format for supervised fine-tuning baseline
- **Sources**: Both CRD3 and LIGHT data integrated with DM system prompts

### PPO Module

- **Primary**: CRD3 for environment setup
- **Secondary**: Critic outputs for reward signals
- **Processing**: Dynamic episode sampling with critic feedback

### Causal Critic

- **Primary**: Causal critic training data
- **Secondary**: CRD3 for additional examples
- **Processing**: Premise-hypothesis pair extraction for NLI training

### Narrative Critic

- **Primary**: Critic Training Data (30K ROCStories + 10.9K DM pairs)
- **Secondary**: ROCStories for baseline narrative understanding
- **Processing**: Regression training on quality types (coherent, shuffled, repetitive, truncated)
- **Training**: 40,906 total examples for comprehensive quality assessment

### Hybrid Player

- **Primary**: LIGHT dialogue processed data
- **Secondary**: CRD3 for D&D-specific examples
- **Processing**: Dual training for generation and intent classification

### Evaluation

- **All datasets**: Comprehensive evaluation across all data sources
- **Processing**: Statistical analysis and metric computation

## Data Statistics

### CRD3 Dataset

- **Episodes**: ~200+ episode files
- **Campaigns**: 2 major campaigns
- **Speakers**: DM + 6-8 players per session
- **Content**: ~500+ hours of D&D gameplay

### LIGHT Dataset

- **Training samples**: ~20,000+ action sequences
- **Validation samples**: ~2,000+ examples
- **Test samples**: ~2,000+ examples
- **Scenarios**: Fantasy RPG environments
- **Actions**: Physical actions, dialogue, emotes

### ROCStories Dataset

- **Total size**: ~1.9GB story data
- **Source**: TinyStories (roneneldan/TinyStories)
- **Content**: Short, coherent narrative stories
- **Usage**: 30,000 samples extracted for critic training

### Critic Training Dataset

- **Total examples**: 40,906
- **Training split**: 36,815 samples
- **Validation split**: 4,091 samples
- **ROCStories contribution**: 30,000 examples
- **DM pairs contribution**: 10,906 examples
- **Quality categories**: 4 types (coherent, shuffled, repetitive, truncated)

### Data Splits (DM-SFT)

- **Format**: Instruction-tuned with system prompts
- **Sources**: Combined LIGHT and CRD3 data
- **Structure**: Train/validation/test splits
- **Integration**: Both fantasy dialogue and D&D-specific content

## File Naming Conventions

### CRD3 Files

- Format: `C{campaign}E{episode}_{chunk}_{segment}.json`
- Example: `C1E001_3_0.json` = Campaign 1, Episode 1, Chunk 3, Segment 0

### LIGHT Files

- Format: `{task}_{split}[_unseen][_cands].txt`
- Examples:
  - `action_train.txt` - Training action data
  - `speech_test_unseen.txt` - Unseen test speech data
  - `emote_valid_cands.txt` - Validation emote candidates

## Data Quality and Preprocessing

### CRD3 Preprocessing

- Speaker name normalization
- Turn boundary detection
- Narrative chunk alignment
- Content filtering for appropriate material

### LIGHT Preprocessing

- Action label standardization
- Setting description normalization
- Character persona extraction
- Dialogue context preservation

## Storage Requirements

- **CRD3**: ~2GB (JSON transcripts)
- **LIGHT**: ~500MB (processed text files)
- **ROCStories**: ~1.9GB (TinyStories arrow files)
- **Critic Training**: ~100MB (processed training data)
- **Data Splits**: ~200MB (combined DM training data)
- **Causal Critic Training**: ~50MB (NLI training data)
- **Total**: ~5.25GB recommended free space

## Important Notes

1. **Data Privacy**: All datasets are publicly available research datasets
2. **Licensing**: Ensure compliance with individual dataset licenses
3. **Updates**: Check for dataset updates and version compatibility
4. **Preprocessing**: Some modules may require additional preprocessing steps
5. **Memory**: Large datasets may require memory-efficient loading strategies


---