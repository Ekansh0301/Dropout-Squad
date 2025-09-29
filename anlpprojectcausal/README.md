# Causal Responsiveness Critic - Director LLM System

## Overview

This repository implements the **Causal Responsiveness Critic** component of the Director LLM system, as described in "The Director LLM: A Multi-Critic Reinforcement Learning Framework for Domain-Aware Narrative Generation" by Team Dropout Squad.

The Causal Responsiveness Critic evaluates whether narrative responses logically follow from and causally respond to player actions using a pre-trained Natural Language Inference (NLI) model.

## 🎯 Key Features

- **Zero-shot evaluation** using pre-trained NLI model (`MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli`)
- **Batch processing** for efficient evaluation of multiple interactions
- **Detailed explanations** for causality scores with probability breakdowns
- **MCRL integration** ready for PPO training pipeline
- **Configurable** scoring thresholds and dynamic weighting
- **Comprehensive testing** suite with edge cases and integration tests

## 🏗️ Architecture

```
┌─────────────────────────────────────────────┐
│              Director LLM System             │
├─────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌─────────────────┐   │
│  │  Player Input   │  │ Director Output │   │
│  └─────────────────┘  └─────────────────┘   │
│           │                     │           │
│           └──────────┬──────────┘           │
│                      │                      │
│  ┌─────────────────────────────────────────┐ │
│  │     Causal Responsiveness Critic        │ │
│  │  ┌─────────────────────────────────┐    │ │
│  │  │    Pre-trained NLI Model        │    │ │
│  │  │ (DeBERTa-v3-base-mnli-fever)    │    │ │
│  │  └─────────────────────────────────┘    │ │
│  │                                         │ │
│  │  Output: Causal Score (0.0 - 1.0)      │ │
│  └─────────────────────────────────────────┘ │
│                      │                      │
│  ┌─────────────────────────────────────────┐ │
│  │          MCRL Reward Vector             │ │
│  │  [causal, narrative, world, character]  │ │
│  └─────────────────────────────────────────┘ │
└─────────────────────────────────────────────┘
```

## 🚀 Quick Start

### Installation

```bash
# Clone or download the repository
cd anlpprojectcausal

# Install dependencies
pip install -r requirements.txt
```

### Basic Usage

```python
from causal_critic import CausalResponsivenessCritic

# Initialize the critic
critic = CausalResponsivenessCritic()

# Evaluate a player-director interaction
score = critic.evaluate_causality(
    player_action="I cast a fireball at the goblin",
    director_response="The fireball streaks through the air and explodes against the goblin, dealing 8 points of fire damage.",
    context="You are in combat with a hostile goblin in a dungeon corridor."
)

print(f"Causal Score: {score:.3f}")  # Output: Causal Score: 0.876
```

### Detailed Evaluation with Explanations

```python
# Get detailed analysis
result = critic.evaluate_with_explanations(
    "I try to pick the lock",
    "You carefully work with your lockpicks and hear a satisfying click as the lock opens."
)

print(f"Score: {result['causal_score']:.3f}")
print(f"Explanation: {result['explanation']}")
print(f"Entailment: {result['entailment_prob']:.3f}")
```

## 📁 Project Structure

```
anlpprojectcausal/
├── causal_critic.py          # Main critic implementation
├── config.py                 # Configuration settings
├── test_causal_critic.py     # Comprehensive test suite
├── usage_examples.py         # Examples and integration guide
├── requirements.txt          # Python dependencies
├── README.md                # This file
└── Dropout_Squad-Outline.pdf # Original project specification
```

## 🧪 Running Tests

```bash
# Run the comprehensive test suite
python test_causal_critic.py

# Run usage examples and demonstrations
python usage_examples.py

# Run specific pytest tests (if pytest is installed)
pytest test_causal_critic.py -v
```

## ⚙️ Configuration

The system supports multiple configuration presets:

```python
from config import get_config

# Available configurations: 'default', 'strict', 'lenient', 'balanced', 'fast'
config = get_config('strict')  # Higher causality requirements
config = get_config('lenient')  # More forgiving causality scoring
```

### Custom Configuration

```python
from config import CausalCriticConfig

custom_config = CausalCriticConfig(
    model_name="MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli",
    entailment_weight=1.0,
    neutral_weight=0.3,
    strong_causality_threshold=0.8
)
```

## 🔄 MCRL Integration

The critic integrates seamlessly with the Multi-Critic Reinforcement Learning pipeline:

```python
from causal_critic import CausalCriticRewardModel

# Wrap critic for MCRL training
reward_model = CausalCriticRewardModel(critic)

# Use in PPO training loop
reward = reward_model(player_input, director_output, context)

# Dynamic weighting based on player intent
intent_weights = {
    "EXPLORE": 0.6,
    "ACTION": 1.0,
    "DIALOGUE": 0.8
}
```

## 📊 Performance Metrics

The critic evaluates causality across several dimensions:

- **Entailment Probability**: How well the response logically follows from the action
- **Contradiction Probability**: Whether the response contradicts the action
- **Neutral Probability**: Whether the response is unrelated to the action
- **Final Causal Score**: Weighted combination optimized for narrative coherence

### Scoring Interpretation

| Score Range | Quality Level | Description |
|-------------|---------------|-------------|
| 0.8 - 1.0   | Excellent     | Strong causal relationship, response directly follows from action |
| 0.6 - 0.8   | Good          | Clear causal connection with minor gaps |
| 0.4 - 0.6   | Moderate      | Some causal connection, but could be stronger |
| 0.2 - 0.4   | Poor          | Weak causal relationship |
| 0.0 - 0.2   | Very Poor     | No clear causal connection or contradictory |

## 🎮 Use Cases

### Tabletop RPG AI Dungeon Masters
- Evaluate DM responses for logical consistency
- Ensure player actions have meaningful consequences
- Maintain narrative coherence across long campaigns

### Interactive Fiction Systems
- Score dialogue responses for causality
- Improve branching narrative quality
- Validate story progression logic

### Training Data Quality Assessment
- Evaluate training datasets for causal consistency
- Filter low-quality player-DM interaction pairs
- Benchmark different narrative generation models

## 🔬 Technical Details

### Model Architecture
- **Base Model**: DeBERTa-v3-base (180M parameters)
- **Training**: Pre-trained on MNLI, FEVER, and ANLI datasets
- **Task**: 3-way classification (entailment, neutral, contradiction)
- **Input**: Premise-hypothesis pairs formatted for narrative evaluation

### Premise-Hypothesis Formatting
```
Premise: "Context: [story context] Player Action: [player action]"
Hypothesis: "The following narrative response logically follows: [director response]"
```

### Computational Requirements
- **GPU Memory**: ~2GB VRAM (recommended)
- **CPU**: Works on CPU but slower (~10x)
- **Batch Size**: Configurable (default: 8)
- **Inference Speed**: ~50ms per evaluation on GPU

## 🚧 Future Enhancements

### Planned Features
- **Caching System**: Cache frequent evaluations for speed
- **Multi-turn Context**: Better handling of long conversation history
- **Domain Adaptation**: Fine-tuning for specific narrative domains
- **Uncertainty Quantification**: Confidence scores for predictions

### Integration Opportunities
- **Reward Shaping**: More sophisticated reward combination strategies
- **Active Learning**: Use low-confidence cases for model improvement
- **Human Feedback**: Incorporate human evaluations for calibration

## 📖 Citation

If you use this implementation in your research, please cite:

```bibtex
@article{dropout_squad_2025,
  title={The Director LLM: A Multi-Critic Reinforcement Learning Framework for Domain-Aware Narrative Generation},
  author={Dropout Squad},
  journal={ANLP Project},
  year={2025},
  note={Causal Responsiveness Critic Implementation}
}
```

## 🤝 Contributing

We welcome contributions! Please:

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

### Development Setup

```bash
# Install development dependencies
pip install -r requirements.txt
pip install pytest black flake8

# Run code formatting
black causal_critic.py

# Run linting
flake8 causal_critic.py

# Run tests
pytest test_causal_critic.py
```

## 📄 License

This project is part of the Director LLM research initiative. Please refer to your institution's policies regarding academic code sharing and usage.

## 🆘 Support

For questions or issues:

1. Check the test suite for usage examples
2. Review the configuration options in `config.py`
3. Run the examples in `usage_examples.py`
4. Consult the original paper for theoretical background

## 🎯 Related Work

This implementation builds upon:

- **Reinforcement Learning from Human Feedback (RLHF)**
- **Multi-objective Reinforcement Learning**
- **Natural Language Inference for Text Generation Evaluation**
- **Grounded Language Learning in Interactive Environments**

---

*Built with ❤️ by Team Dropout Squad for principled, controllable AI narrative generation.*