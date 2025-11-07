# Narrative Critic: Visual Architecture Guide

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    NARRATIVE CRITIC SYSTEM                       │
└─────────────────────────────────────────────────────────────────┘

┌──────────────────┐
│  ROCStories CSV  │  45,000 five-sentence stories
│  Source Dataset  │  Format: storyid, title, sentence1-5
└────────┬─────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────┐
│              prepare_dataset.py - DATA TRANSFORMATION            │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  For each story, create 4 variants:                             │
│                                                                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │  COHERENT    │  │  SHUFFLED    │  │  REPETITIVE  │          │
│  │  Original    │  │  Random      │  │  Repeated    │          │
│  │  Story       │  │  Sentence    │  │  Sentences   │          │
│  │              │  │  Order       │  │              │          │
│  │  Score:      │  │  Score:      │  │  Score:      │          │
│  │  0.7 - 1.0   │  │  0.0 - 0.3   │  │  0.2 - 0.4   │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
│                                                                  │
│  ┌──────────────┐                                               │
│  │  TRUNCATED   │                                               │
│  │  Incomplete  │                                               │
│  │  Story       │                                               │
│  │              │                                               │
│  │  Score:      │                                               │
│  │  0.3 - 0.5   │                                               │
│  └──────────────┘                                               │
│                                                                  │
└────────┬────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────┐
│                    OUTPUT: JSON FILES                            │
├─────────────────────────────────────────────────────────────────┤
│  rocstoriestrain.json  (~27,000 examples)                       │
│  rocstoriesval.json    (~3,000 examples)                        │
│                                                                  │
│  Format: {text, label, label_float, source, type, story_id}     │
└────────┬────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────┐
│               MODEL TRAINING (narrative_critic.py)               │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  1. TOKENIZATION                                                │
│     Text → Tokens (max 128)                                     │
│                                                                  │
│  2. MODEL INITIALIZATION                                        │
│     DeBERTa-v3-base (139M parameters)                           │
│     + Regression Head (1 output)                                │
│                                                                  │
│  3. TRAINING LOOP                                               │
│     Epochs: 3                                                   │
│     Optimizer: AdamW                                            │
│     Loss: MSE                                                   │
│     Scheduler: Cosine with warmup                               │
│                                                                  │
│  4. EVALUATION                                                  │
│     Metrics: MSE, MAE, RMSE, R², Correlation                    │
│                                                                  │
└────────┬────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────┐
│                  TRAINED MODEL OUTPUT                            │
├─────────────────────────────────────────────────────────────────┤
│  models/narrative_critic/                                       │
│  ├── pytorch_model.bin      (Trained weights)                   │
│  ├── config.json            (Model configuration)               │
│  ├── tokenizer.json         (Tokenizer settings)                │
│  ├── training_history.json  (Training logs)                     │
│  └── eval_metrics.json      (Performance metrics)               │
└────────┬────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────┐
│                     INFERENCE PIPELINE                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Input Text                                                     │
│      │                                                           │
│      ▼                                                           │
│  Tokenizer  →  [101, 2023, 2003, ..., 102]                      │
│      │                                                           │
│      ▼                                                           │
│  DeBERTa Encoder  →  Hidden States [768-dim]                    │
│      │                                                           │
│      ▼                                                           │
│  Pooler Layer  →  Sentence Embedding                            │
│      │                                                           │
│      ▼                                                           │
│  Regression Head  →  Raw Logit                                  │
│      │                                                           │
│      ▼                                                           │
│  Sigmoid  →  Quality Score [0.0 - 1.0]                          │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## 📊 Data Flow Diagram

```
┌────────────────────────────────────────────────────────────────┐
│                    DATA TRANSFORMATION                          │
└────────────────────────────────────────────────────────────────┘

ORIGINAL STORY (from CSV)
┌──────────────────────────────────────────────────────────────┐
│ Sentence 1: Dan's parents were overweight.                   │
│ Sentence 2: Dan was overweight as well.                      │
│ Sentence 3: The doctors told his parents it was unhealthy.   │
│ Sentence 4: His parents understood and decided to change.    │
│ Sentence 5: They got themselves and Dan on a diet.           │
└──────────────────────────────────────────────────────────────┘
                        │
        ┌───────────────┼───────────────┬──────────────┐
        │               │               │              │
        ▼               ▼               ▼              ▼
┌──────────────┐ ┌─────────────┐ ┌─────────────┐ ┌──────────────┐
│  COHERENT    │ │  SHUFFLED   │ │ REPETITIVE  │ │  TRUNCATED   │
├──────────────┤ ├─────────────┤ ├─────────────┤ ├──────────────┤
│ S1 S2 S3 S4  │ │ S5 S2 S4 S3 │ │ S1 S1 S2 S3 │ │ S1 S2 S3     │
│ S5           │ │ S1          │ │ S3 S4 S5    │ │              │
│              │ │             │ │             │ │              │
│ Score: 0.85  │ │ Score: 0.15 │ │ Score: 0.35 │ │ Score: 0.42  │
└──────────────┘ └─────────────┘ └─────────────┘ └──────────────┘
```

## 🎯 Quality Score Distribution

```
       Quality Score Distribution by Type
       
1.0 │                                        ┌──┐
    │                                    ┌───┤██│
0.9 │                                ┌───┤███│██│
    │                            ┌───┤███│███│██│
0.8 │                        ┌───┤███│███│███│██│
    │                    ┌───┤███│███│███│███│██│
0.7 │ COHERENT       ┌───┤███│███│███│███│███│██│
    │ (High Quality) │███│███│███│███│███│███│██│
0.6 │ ───────────────┴───┴───┴───┴───┴───┴───┴──┴───
    │
0.5 │                                            ┌──┐
    │                                        ┌───┤██│
0.4 │ TRUNCATED/REPETITIVE               ┌───┤███│██│
    │ (Medium Quality)               ┌───┤███│███│██│
0.3 │ ───────────────────────────────┴───┴───┴───┴──┴───
    │
0.2 │                                                ┌──┐
    │                                            ┌───┤██│
0.1 │ SHUFFLED                               ┌───┤███│██│
    │ (Low Quality)                      ┌───┤███│███│██│
0.0 │ ───────────────────────────────────┴───┴───┴───┴──┴───
    └────────────────────────────────────────────────────────
         Clear quality separation enables effective learning
```

## 🧠 Model Architecture Detail

```
┌─────────────────────────────────────────────────────────────────┐
│              DeBERTa-v3-base ARCHITECTURE                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  INPUT LAYER                                                    │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ Text: "You enter a dimly lit tavern..."                  │  │
│  │ Tokens: [101, 2017, 4607, 1037, 9737, ...]              │  │
│  │ Length: max 128 tokens                                   │  │
│  └──────────────────────────────────────────────────────────┘  │
│                           │                                     │
│                           ▼                                     │
│  EMBEDDING LAYER                                                │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ Token Embeddings      [128 × 768]                        │  │
│  │ Position Embeddings   [128 × 768]                        │  │
│  │ → Combined           [128 × 768]                         │  │
│  └──────────────────────────────────────────────────────────┘  │
│                           │                                     │
│                           ▼                                     │
│  TRANSFORMER LAYERS (×12)                                       │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ Layer 1:  Disentangled Self-Attention                    │  │
│  │           LayerNorm                                       │  │
│  │           Feed-Forward Network                            │  │
│  │           LayerNorm                                       │  │
│  │ ─────────────────────────────────────────────            │  │
│  │ Layer 2-11: Same structure                               │  │
│  │ ─────────────────────────────────────────────            │  │
│  │ Layer 12: Final transformer layer                        │  │
│  │           Output: [128 × 768]                            │  │
│  └──────────────────────────────────────────────────────────┘  │
│                           │                                     │
│                           ▼                                     │
│  POOLING LAYER                                                  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ Take [CLS] token representation                          │  │
│  │ → [1 × 768]                                              │  │
│  └──────────────────────────────────────────────────────────┘  │
│                           │                                     │
│                           ▼                                     │
│  REGRESSION HEAD                                                │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ Linear Layer: [768 → 1]                                  │  │
│  │ → Raw Logit                                              │  │
│  └──────────────────────────────────────────────────────────┘  │
│                           │                                     │
│                           ▼                                     │
│  ACTIVATION                                                     │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ Sigmoid: logit → [0.0, 1.0]                              │  │
│  │ → Quality Score: 0.78                                    │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘

Parameters: 139,082,753 total
            139,082,753 trainable
```

## 🔄 Training Loop Visualization

```
EPOCH 1
┌─────────────────────────────────────────────────────────────┐
│ Batch 1    →  Forward Pass  →  Loss: 0.145  →  Backward    │
│ Batch 2    →  Forward Pass  →  Loss: 0.138  →  Backward    │
│ Batch 3    →  Forward Pass  →  Loss: 0.132  →  Backward    │
│ ...                                                         │
│ Batch 843  →  Forward Pass  →  Loss: 0.089  →  Backward    │
│                                                             │
│ → Evaluation: Val Loss = 0.092, MAE = 0.124                │
└─────────────────────────────────────────────────────────────┘

EPOCH 2
┌─────────────────────────────────────────────────────────────┐
│ Batch 844  →  Forward Pass  →  Loss: 0.085  →  Backward    │
│ Batch 845  →  Forward Pass  →  Loss: 0.082  →  Backward    │
│ ...                                                         │
│                                                             │
│ → Evaluation: Val Loss = 0.078, MAE = 0.105                │
└─────────────────────────────────────────────────────────────┘

EPOCH 3
┌─────────────────────────────────────────────────────────────┐
│ Batch ...  →  Forward Pass  →  Loss: 0.075  →  Backward    │
│ ...                                                         │
│                                                             │
│ → Final Evaluation: Val Loss = 0.072, MAE = 0.098          │
│ → Best Model Saved! ✓                                      │
└─────────────────────────────────────────────────────────────┘
```

## 📈 Performance Visualization

```
Training Progress
─────────────────────────────────────────────────────────────

Loss
0.20 │ ●
     │  ●
0.15 │   ●●
     │     ●●                                Train Loss
0.10 │       ●●●                          ───────────────
     │          ●●●●                      Val Loss
0.05 │              ●●●●●■■■              ●●●●●●●●●●●●●●
     │                    ■■■■■
0.00 │────────────────────────────■■■■■■■
     0    500   1000  1500  2000  2500  3000 (steps)


Metrics Over Time
─────────────────────────────────────────────────────────────

R² Score
1.00 │                              ■■■■■■■
     │                         ■■■■■
0.80 │                    ■■■■■
     │               ■■■■
0.60 │          ■■■■
     │      ■■■■
0.40 │  ■■■■
     │
0.00 │────────────────────────────────────
     0         500        1000       1500 (steps)
```

## 🎮 D&D Integration Flow

```
┌─────────────────────────────────────────────────────────────────┐
│              D&D REINFORCEMENT LEARNING PIPELINE                 │
└─────────────────────────────────────────────────────────────────┘

┌────────────────┐
│ Player Action  │  "I search the room for treasure"
└───────┬────────┘
        │
        ▼
┌────────────────────────────┐
│ DM Response Generator      │  GPT-based model generates response
│ (GPT-2 / LLaMA)           │
└───────┬────────────────────┘
        │
        ▼
┌──────────────────────────────────────────────────────┐
│ Generated Response:                                   │
│ "You carefully search the ancient chamber. Among the │
│  dusty tomes and broken furniture, you discover a    │
│  small wooden chest hidden beneath the floorboards." │
└───────┬──────────────────────────────────────────────┘
        │
        ├─────────────────┬─────────────────┬──────────────┐
        │                 │                 │              │
        ▼                 ▼                 ▼              ▼
┌──────────────┐  ┌──────────────┐  ┌─────────────┐  ┌─────────┐
│  NARRATIVE   │  │   CAUSAL     │  │  ENGAGEMENT │  │  OTHER  │
│   CRITIC     │  │   CRITIC     │  │   SCORER    │  │ METRICS │
├──────────────┤  ├──────────────┤  ├─────────────┤  ├─────────┤
│ Score: 0.78  │  │ Score: 0.85  │  │ Score: 0.72 │  │   ...   │
└──────┬───────┘  └──────┬───────┘  └──────┬──────┘  └────┬────┘
       │                 │                 │              │
       └─────────────────┴─────────────────┴──────────────┘
                                │
                                ▼
                        ┌───────────────┐
                        │ COMBINE       │
                        │ REWARDS       │
                        ├───────────────┤
                        │ 0.6 × 0.78 + │
                        │ 0.2 × 0.85 + │
                        │ 0.2 × 0.72   │
                        │ = 0.782      │
                        └───────┬───────┘
                                │
                                ▼
                        ┌───────────────┐
                        │ PPO UPDATE    │
                        │ Improve model │
                        │ based on      │
                        │ reward signal │
                        └───────────────┘
```

## 📊 Quality Assessment Examples

```
┌─────────────────────────────────────────────────────────────────┐
│                    PREDICTION EXAMPLES                           │
└─────────────────────────────────────────────────────────────────┘

HIGH QUALITY (Score: 0.82)
┌──────────────────────────────────────────────────────────────┐
│ "The ancient library stretched endlessly before you, its     │
│  towering shelves groaning under countless leather-bound     │
│  tomes. Dust motes danced in golden sunlight filtering      │
│  through stained glass windows, casting rainbow patterns    │
│  across worn stone floors."                                  │
└──────────────────────────────────────────────────────────────┘
Features: Descriptive, coherent, engaging, complete

LOW QUALITY (Score: 0.28)
┌──────────────────────────────────────────────────────────────┐
│ "You see a room. There is a door. There is a table."        │
└──────────────────────────────────────────────────────────────┘
Features: Simple, repetitive structure, lacks detail

REPETITIVE (Score: 0.35)
┌──────────────────────────────────────────────────────────────┐
│ "The dragon roars. The dragon breathes fire. The dragon     │
│  roars again. The dragon breathes more fire."                │
└──────────────────────────────────────────────────────────────┘
Features: Excessive repetition, monotonous

TRUNCATED (Score: 0.45)
┌──────────────────────────────────────────────────────────────┐
│ "Your blade finds its mark with a satisfying thud. The      │
│  orc's eyes widen in surprise before it crumples to"         │
└──────────────────────────────────────────────────────────────┘
Features: Incomplete, abrupt ending
```

## 🎯 Decision Flow

```
┌─────────────────────────────────────────────────────────────────┐
│            QUALITY-BASED RESPONSE SELECTION                      │
└─────────────────────────────────────────────────────────────────┘

Input: Player requests exploration

              ┌─────────────────┐
              │ Generate 5      │
              │ Candidate       │
              │ Responses       │
              └────────┬────────┘
                       │
        ┌──────────────┼──────────────┐
        │              │              │
        ▼              ▼              ▼
   Candidate 1    Candidate 2    Candidate 3    ...
   Score: 0.65    Score: 0.82    Score: 0.58
        │              │              │
        └──────────────┼──────────────┘
                       │
                       ▼
              ┌─────────────────┐
              │ Select Highest  │
              │ Score: 0.82     │
              └────────┬────────┘
                       │
                       ▼
              ┌─────────────────┐
              │ Return          │
              │ Candidate 2     │
              │ to Player       │
              └─────────────────┘
```

---

This visual guide provides a complete architectural overview of the Narrative Critic system! 🎨
