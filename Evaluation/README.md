# Evaluation: Comprehensive Analysis Pipeline

This module provides comprehensive evaluation tools for the Director LLM project, including statistical analysis, visualization, and performance metrics for all model components.

## Purpose

Evaluates the complete multi-critic reinforcement learning system with research-grade metrics, statistical analysis, and publication-ready visualizations. Provides both automated evaluation and detailed manual testing capabilities.

## Files Overview

### Core Files

#### `full_eval.py`

Comprehensive evaluation pipeline with advanced metrics and statistical analysis.

**Main Class: `ResearchReportGenerator`**

- Complete evaluation orchestrator with 9-step pipeline
- Loads all trained components (baseline model, critics, hybrid player)
- Generates statistical analysis and visualizations
- Produces research-grade evaluation reports

**Key Methods:**

- `load_all_components()`: Loads baseline model with LoRA adapters, critics, and hybrid player
- `_generate_baseline_responses()`: Generates model outputs for evaluation
- `_evaluate_critics()`: Tests both narrative and causal critics
- `_calculate_linguistic_metrics()`: Computes diversity, repetition, and vocabulary metrics
- `_analyze_hybrid_player()`: Evaluates player simulation and intent classification
- `_generate_response_samples()`: Creates example outputs for qualitative analysis
- `_create_comprehensive_visualizations()`: Generates publication-ready plots
- `_run_statistical_analysis()`: Performs correlation and significance tests
- `generate_full_report()`: Orchestrates complete evaluation pipeline

**Evaluation Pipeline (9 Steps):**

1. Initialize evaluation system
2. Load all trained components
3. Generate baseline model outputs
4. Evaluate critic performance
5. Calculate linguistic diversity metrics
6. Analyze hybrid player behavior
7. Generate response samples
8. Create comprehensive visualizations
9. Run statistical analysis

**Metrics Computed:**

- **Narrative Quality**: Critic scores, distribution analysis
- **Causal Consistency**: Logic evaluation, correlation analysis
- **Linguistic Diversity**: Type-token ratio, n-gram diversity, repetition analysis
- **Intent Classification**: Accuracy, distribution of predicted intents
- **Response Quality**: Length analysis, vocabulary richness

#### `evalc.py`

Standalone critic evaluation with curated test cases.

**Purpose**: Quick verification of critic model performance on known examples

**Key Functions:**

- `test_narrative_critic()`: Evaluates narrative critic on high/low quality examples
- `test_causal_critic()`: Tests causal critic on premise-hypothesis pairs
- `main()`: Orchestrates critic testing and generates markdown report

**Test Categories:**

- **High Quality**: Descriptive, engaging narrative examples
- **Low Quality**: Simple, repetitive, or poorly written text
- **Repetitive**: Examples with excessive repetition
- **Causal Logic**: Action-consequence pairs for consistency testing

**Output**: Markdown-formatted evaluation report with scores and analysis

#### `eval_config.yaml`

Configuration file for evaluation parameters and model paths.

**Key Sections:**

- `model_paths`: Paths to baseline model, critics, and hybrid player
- `data_path`: Test dataset location
- `evaluation_settings`: Number of samples, output directory, random seed
- `generation_settings`: Text generation parameters for evaluation

**Critical Parameters:**

```yaml
evaluation_settings:
  num_samples: 100 # Number of test samples
  output_dir: "evaluation_results"
  seed: 42

generation_settings:
  max_new_tokens: 150 # Response length
  temperature: 0.7 # Generation creativity
  top_p: 0.95 # Nucleus sampling
```

## Technical Implementation

### Evaluation Architecture

- **Multi-Component Loading**: Integrates baseline model, critics, and hybrid player
- **Statistical Analysis**: Scipy-based correlation and significance testing
- **Visualization**: High-quality matplotlib/seaborn plots for publication
- **Memory Management**: Efficient GPU memory usage across multiple models

### Output Organization

```
evaluation_results/
├── figures/              # Publication-ready visualizations
├── data/                # Raw evaluation data and metrics
├── samples/             # Example responses for qualitative analysis
└── evaluation_report.md # Comprehensive results summary
```

## Usage

### Prerequisites

```bash
pip install torch transformers peft
pip install matplotlib seaborn pandas numpy scipy
pip install scikit-learn datasets tqdm
```

### Full Evaluation Pipeline

```bash
# Run comprehensive evaluation
python full_eval.py

# Or with custom config
python full_eval.py --config custom_eval_config.yaml
```

### Quick Critic Testing

```bash
# Test critics on curated examples
python evalc.py
```

### Configuration

1. Edit `eval_config.yaml` to specify model paths
2. Adjust evaluation parameters (number of samples, output settings)
3. Configure generation parameters for response quality

## Evaluation Outputs

### Quantitative Metrics

- **Narrative Scores**: Distribution, mean, variance of narrative quality
- **Causal Scores**: Consistency evaluation across response types
- **Linguistic Diversity**: Token diversity, n-gram analysis, repetition metrics
- **Intent Distribution**: Classification accuracy and intent type distribution

### Qualitative Analysis

- **Response Samples**: Categorized examples (high/low quality, different intents)
- **Comparison Tables**: Side-by-side baseline vs. trained model outputs
- **Error Analysis**: Common failure modes and improvement opportunities

### Visualizations

- **Score Distributions**: Histograms of critic scores
- **Correlation Plots**: Relationships between different metrics
- **Intent Analysis**: Visualization of player intent classification
- **Quality Trends**: Performance across different response categories

### Statistical Analysis

- **Significance Testing**: Statistical validation of improvements
- **Correlation Analysis**: Relationships between narrative and causal scores
- **Distribution Comparisons**: Baseline vs. trained model differences

## Integration

This evaluation pipeline works with:

- **Baseline Models**: From `../DM-SFT/` training
- **Critic Models**: From `../narrative critic/` and `../causal critic/` training
- **Hybrid Player**: From `../hybrid_player/` training
- **PPO Models**: From `../PPO/` multi-critic RL training

## Performance Characteristics

- **Evaluation Time**: ~30 minutes for 100 samples
- **Memory Usage**: ~8GB GPU memory for all models
- **Output Size**: ~50MB for complete evaluation with visualizations
- **Reproducibility**: Seeded random generation for consistent results

## Research Applications

This evaluation framework supports:

- **Academic Research**: Publication-ready metrics and visualizations
- **Model Comparison**: Systematic comparison between different training approaches
- **Ablation Studies**: Component-wise performance analysis
- **Error Analysis**: Detailed investigation of model limitations
