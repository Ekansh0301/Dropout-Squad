# Kaggle Training Setup for Narrative Critic

This guide explains how to train the Narrative Critic model on Kaggle using the prepared ROCStories dataset.

## 📋 Overview

The Narrative Critic is a **DeBERTa-v3-base** regression model that:
- Evaluates narrative quality on a scale of 0.0 to 1.0
- Trained on ROCStories dataset with 4 quality categories
- Used to provide reward signals for D&D response generation in reinforcement learning
- Assesses coherence, descriptiveness, and overall narrative quality

## 🎯 Dataset Context

### Original Dataset Structure
The ROCStories dataset contains short 5-sentence narratives with:
- `storyid`: Unique identifier
- `storytitle`: Story title
- `sentence1-5`: Five sequential sentences forming a coherent story

### Processed Training Data
From each story, we create **4 types of examples**:

| Type | Quality Range | Description | Purpose |
|------|---------------|-------------|---------|
| **Coherent** | 0.7 - 1.0 | Original well-structured stories | High-quality narrative examples |
| **Shuffled** | 0.0 - 0.3 | Randomly reordered sentences | Poor coherence detection |
| **Repetitive** | 0.2 - 0.4 | Stories with repeated sentences | Repetition detection |
| **Truncated** | 0.3 - 0.5 | Incomplete stories (2-4 sentences) | Incompleteness detection |

### Example Transformations

**Original Story (Coherent - Score: 0.85)**
```
Dan's parents were overweight. Dan was overweight as well. The doctors told his parents it was unhealthy. His parents understood and decided to make a change. They got themselves and Dan on a diet.
```

**Shuffled Version (Score: 0.15)**
```
They got themselves and Dan on a diet. Dan was overweight as well. His parents understood and decided to make a change. The doctors told his parents it was unhealthy. Dan's parents were overweight.
```

**Repetitive Version (Score: 0.35)**
```
Dan's parents were overweight. Dan's parents were overweight. Dan was overweight as well. The doctors told his parents it was unhealthy. The doctors told his parents it was unhealthy. His parents understood and decided to make a change.
```

**Truncated Version (Score: 0.42)**
```
Dan's parents were overweight. Dan was overweight as well. The doctors told his parents it was unhealthy.
```

## 🚀 Step-by-Step Setup

### Step 1: Prepare Dataset

1. **Run the dataset preparation script**:
   ```bash
   python prepare_dataset.py
   ```

2. **Expected output**:
   - `dataset_json/rocstoriestrain.json` (~27,000 examples)
   - `dataset_json/rocstoriesval.json` (~3,000 examples)

3. **Dataset statistics**:
   - Total examples: ~30,000
   - Train/Val split: 90/10
   - Balanced across 4 quality types
   - Quality scores: Continuous values 0.0-1.0

### Step 2: Upload to Kaggle

1. **Create a Kaggle Dataset**:
   - Go to [Kaggle Datasets](https://www.kaggle.com/datasets)
   - Click "New Dataset"
   - Upload both JSON files
   - Name it: `ROCStoriesData` (or your preferred name)

2. **Dataset structure on Kaggle**:
   ```
   /kaggle/input/ROCStoriesData/
   ├── rocstoriestrain.json
   └── rocstoriesval.json
   ```

### Step 3: Create Kaggle Notebook

1. **Upload the notebook**:
   - Upload `kaggle_narrative_critic_training.ipynb`
   - Or create a new notebook and copy the cells

2. **Add the dataset**:
   - Click "Add Data" → "Your Datasets"
   - Select your `ROCStoriesData` dataset
   - Verify paths match: `/kaggle/input/ROCStoriesData/`

3. **Enable GPU**:
   - Settings → Accelerator → GPU T4 x2 (or P100)
   - This significantly speeds up training (3 hours → 1 hour)

### Step 4: Run Training

1. **Execute all cells** in order
2. **Expected training time**:
   - With GPU: ~1-2 hours
   - With CPU: ~4-6 hours

3. **Monitor progress**:
   - Training loss should decrease steadily
   - Validation metrics updated every 500 steps
   - Best model saved automatically

## 📊 Expected Results

### Performance Metrics

After training, you should see metrics similar to:

| Metric | Expected Value | Interpretation |
|--------|----------------|----------------|
| **MSE** | 0.015 - 0.025 | Mean Squared Error |
| **MAE** | 0.08 - 0.12 | Mean Absolute Error |
| **RMSE** | 0.12 - 0.16 | Root Mean Squared Error |
| **R² Score** | 0.75 - 0.85 | Model fit quality |
| **Correlation** | 0.85 - 0.92 | Prediction correlation |
| **Accuracy (±0.2)** | 0.85 - 0.92 | Within threshold accuracy |

### Performance by Narrative Type

| Type | MAE | Interpretation |
|------|-----|----------------|
| Coherent | ~0.08 | Best performance on high-quality text |
| Shuffled | ~0.10 | Good detection of poor coherence |
| Repetitive | ~0.12 | Moderate detection of repetition |
| Truncated | ~0.11 | Good detection of incompleteness |

## 📁 Output Files

The notebook generates:

### Model Files
- `narrative_critic_model/pytorch_model.bin` - Trained weights
- `narrative_critic_model/config.json` - Model configuration
- `narrative_critic_model/tokenizer.json` - Tokenizer config

### Analysis Files
- `training_progress.png` - Loss curves
- `model_analysis.png` - Comprehensive analysis plots
- `confusion_matrix.png` - Quality category confusion matrix
- `validation_predictions.csv` - All predictions on validation set
- `model_summary.json` - Complete metrics summary

## 🧪 Model Testing

The notebook includes comprehensive testing:

### 1. Validation Set Evaluation
- Full predictions on ~3,000 validation examples
- Per-type performance breakdown
- Error distribution analysis

### 2. Custom D&D Examples
Tests on hand-crafted examples:
- High-quality descriptive text
- Low-quality simple descriptions
- Repetitive combat text
- Truncated narratives
- Typical D&D scenes

### 3. Visual Analysis
- Scatter plot: Predicted vs True scores
- Error distribution histogram
- Performance by narrative type
- Box plots of score distributions
- Confusion matrix for quality bins

## 💡 Usage After Training

### Load the Trained Model

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

# Load model
model = AutoModelForSequenceClassification.from_pretrained(
    "/kaggle/working/narrative_critic_model"
)
tokenizer = AutoTokenizer.from_pretrained(
    "/kaggle/working/narrative_critic_model"
)

# Predict quality
def evaluate_narrative(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
        score = torch.sigmoid(outputs.logits).item()
    return score

# Test
dm_response = "You enter a dimly lit tavern filled with rowdy patrons..."
quality_score = evaluate_narrative(dm_response)
print(f"Quality Score: {quality_score:.3f}")
```

### Integration with RL Training

```python
# Used in PPO training for reward computation
class NarrativeCritic:
    def __init__(self, model_path):
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        
    def get_reward(self, texts):
        """Compute reward scores for batch of DM responses"""
        inputs = self.tokenizer(texts, return_tensors="pt", 
                               truncation=True, padding=True, max_length=128)
        with torch.no_grad():
            outputs = self.model(**inputs)
            rewards = torch.sigmoid(outputs.logits).squeeze()
        return rewards

# In PPO training
narrative_critic = NarrativeCritic("/path/to/model")
rewards = narrative_critic.get_reward(dm_responses)
```

## 🔧 Troubleshooting

### Common Issues

**1. Out of Memory (OOM)**
- Reduce batch size: `CONFIG['batch_size'] = 8`
- Reduce max_length: `CONFIG['max_length'] = 96`
- Enable gradient checkpointing

**2. Dataset Not Found**
- Verify dataset path: `/kaggle/input/ROCStoriesData/`
- Check dataset name matches exactly
- Ensure files are `rocstoriestrain.json` and `rocstoriesval.json`

**3. Slow Training**
- Enable GPU in settings
- Increase batch size if memory allows
- Reduce number of epochs for testing

**4. Poor Performance**
- Check data quality and balance
- Verify quality score distributions
- Increase training epochs
- Adjust learning rate

## 📈 Optimization Tips

### For Faster Training
1. Use GPU T4 x2 or P100
2. Increase batch size to 32 (if memory allows)
3. Reduce max_length to 96 tokens
4. Use fewer epochs (2 instead of 3) for initial testing

### For Better Performance
1. Train for more epochs (4-5)
2. Use learning rate scheduling
3. Add more training data
4. Fine-tune on domain-specific D&D text
5. Experiment with different quality score ranges

## 🎓 Understanding the Model

### Architecture
- **Base**: DeBERTa-v3-base (139M parameters)
- **Task Head**: Single linear layer for regression
- **Output**: Raw logit → Sigmoid → Score (0.0-1.0)

### Training Objective
- **Loss**: Mean Squared Error (MSE)
- **Optimization**: AdamW with cosine LR scheduling
- **Regularization**: Weight decay, gradient clipping

### Quality Assessment Factors
The model learns to recognize:
- ✅ Coherent narrative flow
- ✅ Descriptive language
- ✅ Proper sentence structure
- ✅ Story completeness
- ❌ Shuffled/incoherent text
- ❌ Excessive repetition
- ❌ Incomplete narratives

## 📚 References

- **DeBERTa Paper**: [DeBERTa: Decoding-enhanced BERT with Disentangled Attention](https://arxiv.org/abs/2006.03654)
- **ROCStories Dataset**: [A Corpus and Cloze Evaluation for Deeper Understanding of Commonsense Stories](https://cs.rochester.edu/nlp/rocstories/)
- **HuggingFace Transformers**: [Documentation](https://huggingface.co/docs/transformers/)

## 🤝 Contributing

To improve the model:
1. Add more diverse training examples
2. Include D&D-specific narratives
3. Experiment with different model architectures
4. Fine-tune quality score ranges
5. Add multi-aspect scoring (coherence, creativity, etc.)

## 📝 Notes

- Model is designed for **English text** only
- Optimal for **narrative text** (50-200 tokens)
- Quality scores are **relative** to training data
- Best used as **comparative metric** rather than absolute
- Can be fine-tuned on specific domains (D&D, fantasy, etc.)

---

**Ready to train?** Upload your dataset to Kaggle and run the notebook! 🚀
