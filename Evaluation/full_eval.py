"""
Comprehensive evaluation pipeline for Director LLM project.
Generates advanced metrics, statistical analysis, and publication-ready visualizations.
"""
import torch
import yaml
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from tqdm import tqdm
from collections import Counter, defaultdict
from scipy import stats
from sklearn.metrics import confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# Import project components
from critics import NarrativeCritic, CausalCritic
from hybrid_player import HybridPlayer

class ResearchReportGenerator:
    """Comprehensive evaluation and report generation system."""

    def __init__(self, config_path="configs/evaluation_config.yaml"):
        print("=" * 80)
        print("DIRECTOR LLM - COMPREHENSIVE EVALUATION PIPELINE")
        print("=" * 80)
        print("\n[1/9] Initializing Research Report Generator")
        
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.output_dir = Path(self.config['evaluation_settings']['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create organized output subdirectories
        (self.output_dir / 'figures').mkdir(exist_ok=True)
        (self.output_dir / 'data').mkdir(exist_ok=True)
        (self.output_dir / 'samples').mkdir(exist_ok=True)
        
        print(f"✓ Device: {self.device}")
        print(f"✓ Output directory: {self.output_dir}")
        
        # Configure matplotlib for high-quality output
        sns.set_style("whitegrid")
        plt.rcParams['figure.dpi'] = 300
        plt.rcParams['savefig.dpi'] = 300

    def load_all_components(self):
        """Load all trained models and components for evaluation."""
        print("\n[2/9] Loading All Trained Components")
        paths = self.config['model_paths']

        # Load baseline DM model with LoRA adapters
        base_model_name = "microsoft/phi-2"
        print(f"  → Loading base model: {base_model_name}")
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name, torch_dtype=torch.bfloat16, trust_remote_code=True
        )
        self.baseline_model = PeftModel.from_pretrained(base_model, paths['baseline_model_path'])
        self.baseline_model = self.baseline_model.merge_and_unload().to(self.device).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(paths['baseline_model_path'])
        print(f"✓ Baseline DM loaded from: {paths['baseline_model_path']}")

        # Load trained critic models
        print("  → Loading critic models")
        self.narrative_critic = NarrativeCritic(paths['narrative_critic_path'], self.device)
        self.causal_critic = CausalCritic(paths['causal_critic_model_name'], self.device)
        print("✓ Critics loaded successfully")

        # Load hybrid player for simulation
        print("  → Loading hybrid player")
        self.hybrid_player = HybridPlayer(
            paths['player_generator_path'],
            paths['player_classifier_path'],
            self.device
        )
        print("✓ Hybrid player loaded successfully")

    def _generate_baseline_responses(self):
        """Generate baseline model responses for evaluation."""
        print("\n[3/9] Generating Baseline Model Outputs")
        from datasets import load_from_disk
        test_dataset = load_from_disk(self.config['data_path'])
        num_samples = self.config['evaluation_settings']['num_samples']
        outputs = []
        gen_kwargs = self.config['generation_settings']

        print(f"  → Generating {num_samples} responses")
        for i in tqdm(range(min(num_samples, len(test_dataset))), desc="Generating"):
            full_prompt = test_dataset[i]['text']
            prompt_for_model = full_prompt.split("</s>")[0]
            context_part = full_prompt.split("<</SYS>>\n\n")[1].split("[/INST]")[0].strip()
            
            inputs = self.tokenizer(prompt_for_model, return_tensors="pt").to(self.device)
            with torch.no_grad():
                generated_ids = self.baseline_model.generate(**inputs, **gen_kwargs)
            
            full_text = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            response = full_text.split("[/INST]")[-1].strip()
            
            # Additional linguistic metrics
            outputs.append({
                'id': i,
                'prompt_context': context_part,
                'dm_response': response,
                'response_length': len(response.split()),
                'response_chars': len(response),
                'num_sentences': response.count('.') + response.count('!') + response.count('?')
            })
        
        print(f"✓ Generated {len(outputs)} responses")
        return outputs

    def _score_with_critics(self, outputs):
        print("\n[4/9] Scoring Outputs with Critics")
        prompts = [o['prompt_context'] for o in outputs]
        responses = [o['dm_response'] for o in outputs]
        
        print("  → Computing narrative scores")
        narrative_scores = self.narrative_critic.get_reward(responses).cpu().numpy()
        
        print("  → Computing causal scores")
        causal_scores = self.causal_critic.get_reward(prompts, responses).cpu().numpy()
        
        for i, output in enumerate(outputs):
            output['narrative_score'] = float(narrative_scores[i])
            output['causal_score'] = float(causal_scores[i])
            output['combined_score'] = (narrative_scores[i] + causal_scores[i]) / 2
        
        print("✓ Critic scoring complete")
        return outputs

    def _compute_linguistic_metrics(self, outputs):
        """Compute advanced linguistic and diversity metrics."""
        print("\n[5/9] Computing Linguistic & Diversity Metrics")
        
        responses = [o['dm_response'] for o in outputs]
        
        # Lexical diversity
        all_tokens = []
        unique_tokens_per_response = []
        for resp in responses:
            tokens = resp.lower().split()
            all_tokens.extend(tokens)
            unique_tokens_per_response.append(len(set(tokens)) / max(len(tokens), 1))
        
        type_token_ratio = len(set(all_tokens)) / len(all_tokens)
        avg_unique_ratio = np.mean(unique_tokens_per_response)
        
        # N-gram diversity
        bigrams = [tuple(all_tokens[i:i+2]) for i in range(len(all_tokens)-1)]
        trigrams = [tuple(all_tokens[i:i+3]) for i in range(len(all_tokens)-2)]
        
        bigram_diversity = len(set(bigrams)) / max(len(bigrams), 1)
        trigram_diversity = len(set(trigrams)) / max(len(trigrams), 1)
        
        # Repetition analysis
        repetition_scores = []
        for resp in responses:
            words = resp.lower().split()
            if len(words) > 1:
                word_counts = Counter(words)
                repetition = sum(c - 1 for c in word_counts.values()) / len(words)
                repetition_scores.append(repetition)
        
        # Vocabulary richness (unique words per 100 tokens)
        vocab_richness = (len(set(all_tokens)) / len(all_tokens)) * 100
        
        metrics = {
            'type_token_ratio': type_token_ratio,
            'avg_unique_token_ratio': avg_unique_ratio,
            'bigram_diversity': bigram_diversity,
            'trigram_diversity': trigram_diversity,
            'avg_repetition': np.mean(repetition_scores) if repetition_scores else 0,
            'vocab_richness': vocab_richness,
            'total_unique_tokens': len(set(all_tokens)),
            'total_tokens': len(all_tokens)
        }
        
        print("✓ Linguistic metrics computed")
        for k, v in metrics.items():
            print(f"  • {k}: {v:.4f}")
        
        return metrics

    def _analyze_hybrid_player(self):
        """Enhanced hybrid player analysis with more metrics."""
        print("\n[6/9] Analyzing Hybrid Player Behavior")
        
        num_samples = 500
        print(f"  → Generating {num_samples} player prompts")
        results = self.hybrid_player.generate_prompts(batch_size=num_samples, max_length=50)
        prompts, intents = zip(*results)
        
        # Intent distribution
        intent_counts = Counter(intents)
        
        # Diversity metrics
        all_tokens = " ".join(prompts).split()
        diversity = len(set(all_tokens)) / len(all_tokens)
        
        # Length statistics
        lengths = [len(p.split()) for p in prompts]
        
        # Intent-specific analysis
        intent_prompts = defaultdict(list)
        for prompt, intent in results:
            intent_prompts[intent].append(prompt)
        
        intent_diversity = {}
        for intent, prompt_list in intent_prompts.items():
            tokens = " ".join(prompt_list).split()
            intent_diversity[intent] = len(set(tokens)) / len(tokens) if tokens else 0
        
        analysis = {
            "intent_counts": intent_counts,
            "diversity": diversity,
            "samples": results[:15],
            "length_stats": {
                "mean": np.mean(lengths),
                "std": np.std(lengths),
                "min": np.min(lengths),
                "max": np.max(lengths)
            },
            "intent_diversity": intent_diversity,
            "all_prompts": prompts
        }
        
        print("✓ Player analysis complete")
        print(f"  • Overall diversity: {diversity:.4f}")
        print(f"  • Avg prompt length: {np.mean(lengths):.1f} tokens")
        
        return analysis

    def _statistical_analysis(self, df):
        """Perform statistical tests and correlation analysis."""
        print("\n[7/9] Performing Statistical Analysis")
        
        stats_results = {}
        
        # Correlation analysis
        numeric_cols = ['narrative_score', 'causal_score', 'response_length', 
                       'response_chars', 'num_sentences']
        correlation_matrix = df[numeric_cols].corr()
        stats_results['correlations'] = correlation_matrix
        
        # Score distributions normality test
        _, narrative_p = stats.normaltest(df['narrative_score'])
        _, causal_p = stats.normaltest(df['causal_score'])
        stats_results['normality'] = {
            'narrative_pvalue': narrative_p,
            'causal_pvalue': causal_p
        }
        
        # Identify outliers (z-score method)
        narrative_z = np.abs(stats.zscore(df['narrative_score']))
        causal_z = np.abs(stats.zscore(df['causal_score']))
        
        stats_results['outliers'] = {
            'narrative_outliers': len(df[narrative_z > 3]),
            'causal_outliers': len(df[causal_z > 3])
        }
        
        # Quartile analysis
        stats_results['quartiles'] = {
            'narrative': df['narrative_score'].quantile([0.25, 0.5, 0.75]).to_dict(),
            'causal': df['causal_score'].quantile([0.25, 0.5, 0.75]).to_dict()
        }
        
        print("✓ Statistical analysis complete")
        return stats_results

    def _create_visualizations(self, scored_outputs, player_analysis, linguistic_metrics, stats_results):
        """Create comprehensive publication-quality visualizations."""
        print("\n[8/9] Creating Visualizations")
        df = pd.DataFrame(scored_outputs)
        fig_dir = self.output_dir / 'figures'
        
        # Set color palette
        colors = sns.color_palette("husl", 8)
        
        # 1. Score Distribution (Dual Histogram)
        print("  → Creating score distributions")
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        axes[0].hist(df['narrative_score'], bins=30, alpha=0.7, color=colors[0], edgecolor='black')
        axes[0].axvline(df['narrative_score'].mean(), color='red', linestyle='--', 
                       label=f'Mean: {df["narrative_score"].mean():.3f}')
        axes[0].set_xlabel('Narrative Score', fontsize=12)
        axes[0].set_ylabel('Frequency', fontsize=12)
        axes[0].set_title('Narrative Quality Distribution', fontsize=14, fontweight='bold')
        axes[0].legend()
        axes[0].grid(alpha=0.3)
        
        axes[1].hist(df['causal_score'], bins=30, alpha=0.7, color=colors[1], edgecolor='black')
        axes[1].axvline(df['causal_score'].mean(), color='red', linestyle='--',
                       label=f'Mean: {df["causal_score"].mean():.3f}')
        axes[1].set_xlabel('Causal Score', fontsize=12)
        axes[1].set_ylabel('Frequency', fontsize=12)
        axes[1].set_title('Causal Responsiveness Distribution', fontsize=14, fontweight='bold')
        axes[1].legend()
        axes[1].grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(fig_dir / 'score_distributions.png', bbox_inches='tight')
        plt.close()
        
        # 2. Scatter Plot: Narrative vs Causal
        print("  → Creating correlation scatter plot")
        fig, ax = plt.subplots(figsize=(10, 8))
        scatter = ax.scatter(df['narrative_score'], df['causal_score'], 
                           alpha=0.6, c=df['combined_score'], cmap='viridis', s=50)
        
        # Add regression line
        z = np.polyfit(df['narrative_score'], df['causal_score'], 1)
        p = np.poly1d(z)
        ax.plot(df['narrative_score'], p(df['narrative_score']), "r--", alpha=0.8, 
               label=f'Linear fit (r={df["narrative_score"].corr(df["causal_score"]):.3f})')
        
        ax.set_xlabel('Narrative Quality Score', fontsize=12)
        ax.set_ylabel('Causal Responsiveness Score', fontsize=12)
        ax.set_title('Critic Score Correlation Analysis', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)
        
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Combined Score', fontsize=11)
        
        plt.tight_layout()
        plt.savefig(fig_dir / 'score_correlation.png', bbox_inches='tight')
        plt.close()
        
        # 3. Box Plot: Score Comparison
        print("  → Creating box plots")
        fig, ax = plt.subplots(figsize=(10, 6))
        box_data = [df['narrative_score'], df['causal_score'], df['combined_score']]
        bp = ax.boxplot(box_data, labels=['Narrative', 'Causal', 'Combined'],
                       patch_artist=True, notch=True, showmeans=True)
        
        for patch, color in zip(bp['boxes'], colors[:3]):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title('Critic Score Distributions (Boxplot)', fontsize=14, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(fig_dir / 'score_boxplots.png', bbox_inches='tight')
        plt.close()
        
        # 4. Correlation Heatmap
        print("  → Creating correlation heatmap")
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(stats_results['correlations'], annot=True, fmt='.3f', 
                   cmap='coolwarm', center=0, square=True, ax=ax,
                   cbar_kws={'label': 'Correlation Coefficient'})
        ax.set_title('Feature Correlation Matrix', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(fig_dir / 'correlation_heatmap.png', bbox_inches='tight')
        plt.close()
        
        # 5. Response Length vs Score
        print("  → Creating length analysis plots")
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        axes[0].scatter(df['response_length'], df['narrative_score'], 
                       alpha=0.5, c=colors[2], s=30)
        axes[0].set_xlabel('Response Length (words)', fontsize=12)
        axes[0].set_ylabel('Narrative Score', fontsize=12)
        axes[0].set_title('Response Length vs Narrative Quality', fontsize=13, fontweight='bold')
        axes[0].grid(alpha=0.3)
        
        axes[1].scatter(df['response_length'], df['causal_score'], 
                       alpha=0.5, c=colors[3], s=30)
        axes[1].set_xlabel('Response Length (words)', fontsize=12)
        axes[1].set_ylabel('Causal Score', fontsize=12)
        axes[1].set_title('Response Length vs Causal Score', fontsize=13, fontweight='bold')
        axes[1].grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(fig_dir / 'length_analysis.png', bbox_inches='tight')
        plt.close()
        
        # 6. Player Intent Distribution
        print("  → Creating player analysis plots")
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        intents = list(player_analysis['intent_counts'].keys())
        counts = list(player_analysis['intent_counts'].values())
        
        axes[0].bar(intents, counts, color=colors[:len(intents)], alpha=0.8, edgecolor='black')
        axes[0].set_xlabel('Intent Type', fontsize=12)
        axes[0].set_ylabel('Count', fontsize=12)
        axes[0].set_title('Hybrid Player Intent Distribution', fontsize=13, fontweight='bold')
        axes[0].grid(axis='y', alpha=0.3)
        
        # Pie chart
        axes[1].pie(counts, labels=intents, autopct='%1.1f%%', colors=colors[:len(intents)],
                   startangle=90, textprops={'fontsize': 11})
        axes[1].set_title('Intent Proportion', fontsize=13, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(fig_dir / 'player_intent_distribution.png', bbox_inches='tight')
        plt.close()
        
        # 7. Linguistic Diversity Metrics
        print("  → Creating diversity metrics visualization")
        fig, ax = plt.subplots(figsize=(12, 6))
        
        diversity_metrics = {
            'Type-Token\nRatio': linguistic_metrics['type_token_ratio'],
            'Bigram\nDiversity': linguistic_metrics['bigram_diversity'],
            'Trigram\nDiversity': linguistic_metrics['trigram_diversity'],
            'Vocab\nRichness': linguistic_metrics['vocab_richness'] / 100,
            'Avg Unique\nToken Ratio': linguistic_metrics['avg_unique_token_ratio']
        }
        
        bars = ax.bar(diversity_metrics.keys(), diversity_metrics.values(), 
                     color=colors[:5], alpha=0.8, edgecolor='black')
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title('Linguistic Diversity Metrics', fontsize=14, fontweight='bold')
        ax.set_ylim(0, 1.0)
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}', ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(fig_dir / 'diversity_metrics.png', bbox_inches='tight')
        plt.close()
        
        # 8. Quartile Performance Analysis
        print("  → Creating quartile analysis")
        fig, ax = plt.subplots(figsize=(10, 6))
        
        df['score_quartile'] = pd.qcut(df['combined_score'], q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
        quartile_stats = df.groupby('score_quartile')[['narrative_score', 'causal_score']].mean()
        
        x = np.arange(len(quartile_stats.index))
        width = 0.35
        
        ax.bar(x - width/2, quartile_stats['narrative_score'], width, 
              label='Narrative', color=colors[0], alpha=0.8)
        ax.bar(x + width/2, quartile_stats['causal_score'], width,
              label='Causal', color=colors[1], alpha=0.8)
        
        ax.set_xlabel('Performance Quartile', fontsize=12)
        ax.set_ylabel('Average Score', fontsize=12)
        ax.set_title('Score Breakdown by Performance Quartile', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(quartile_stats.index)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(fig_dir / 'quartile_analysis.png', bbox_inches='tight')
        plt.close()
        
        print("✓ All visualizations created successfully")

    def generate_report(self, scored_outputs, player_analysis, linguistic_metrics, stats_results):
        """Generate comprehensive research report."""
        print("\n[9/9] Synthesizing Research Report")
        
        df = pd.DataFrame(scored_outputs)
        
        # Sample selection for qualitative analysis
        top_narrative = df.nlargest(5, 'narrative_score').to_dict('records')
        bottom_narrative = df.nsmallest(5, 'narrative_score').to_dict('records')
        top_causal = df.nlargest(5, 'causal_score').to_dict('records')
        bottom_causal = df.nsmallest(5, 'causal_score').to_dict('records')
        top_combined = df.nlargest(5, 'combined_score').to_dict('records')
        
        # Calculate percentiles
        narrative_percentiles = df['narrative_score'].quantile([0.25, 0.5, 0.75])
        causal_percentiles = df['causal_score'].quantile([0.25, 0.5, 0.75])
        
        report = f""" """
        
        # Save report
        report_path = self.output_dir / "research_report.md"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        # Save raw data
        df.to_csv(self.output_dir / 'data' / 'scored_outputs.csv', index=False)
        
        with open(self.output_dir / 'data' / 'player_analysis.json', 'w') as f:
            # Convert Counter to dict for JSON serialization
            player_data = player_analysis.copy()
            player_data['intent_counts'] = dict(player_data['intent_counts'])
            json.dump(player_data, f, indent=2, default=str)
        
        with open(self.output_dir / 'data' / 'linguistic_metrics.json', 'w') as f:
            json.dump(linguistic_metrics, f, indent=2)
        
        with open(self.output_dir / 'data' / 'statistical_results.json', 'w') as f:
            stats_data = {
                'correlations': stats_results['correlations'].to_dict(),
                'normality': stats_results['normality'],
                'outliers': stats_results['outliers'],
                'quartiles': stats_results['quartiles']
            }
            json.dump(stats_data, f, indent=2)
        
        # Save sample outputs
        with open(self.output_dir / 'samples' / 'top_narrative.json', 'w') as f:
            json.dump(top_narrative, f, indent=2)
        with open(self.output_dir / 'samples' / 'top_causal.json', 'w') as f:
            json.dump(top_causal, f, indent=2)
        with open(self.output_dir / 'samples' / 'top_combined.json', 'w') as f:
            json.dump(top_combined, f, indent=2)
        
        print(f"✓ Research report saved to: {report_path}")
        print(f"✓ All data exported to: {self.output_dir}")

    def _analyze_response_quality(self, example, category):
        """Generate analysis text for qualitative examples."""
        analyses = {
            'high_narrative': "This response demonstrates strong narrative quality through vivid description, engaging prose, and clear story progression. The Narrative Critic correctly identifies these elements.",
            'low_narrative': "This response lacks descriptive detail, uses repetitive language, or has unclear structure. The Narrative Critic appropriately assigns a low score.",
            'high_causal': "The DM response logically follows from the player's action, showing clear cause-and-effect reasoning. The Causal Critic correctly recognizes this consistency.",
            'low_causal': "The response shows weak logical connection to the player's action, potentially introducing non-sequiturs or ignoring key context. The Causal Critic identifies this inconsistency."
        }
        return analyses.get(category, "Response demonstrates the critic's evaluation capabilities.")

    def run(self):
        """Execute the complete evaluation pipeline."""
        print("\n" + "="*80)
        print("STARTING COMPREHENSIVE EVALUATION")
        print("="*80)
        
        self.load_all_components()
        outputs = self._generate_baseline_responses()
        scored_outputs = self._score_with_critics(outputs)
        linguistic_metrics = self._compute_linguistic_metrics(scored_outputs)
        player_analysis = self._analyze_hybrid_player()
        
        df = pd.DataFrame(scored_outputs)
        stats_results = self._statistical_analysis(df)
        
        self._create_visualizations(scored_outputs, player_analysis, 
                                   linguistic_metrics, stats_results)
        self.generate_report(scored_outputs, player_analysis, 
                           linguistic_metrics, stats_results)
        
        print("\n" + "="*80)
        print("✅ EVALUATION COMPLETE!")
        print("="*80)
        print(f"\n📊 Report: {self.output_dir}/research_report.md")
        print(f"📈 Figures: {self.output_dir}/figures/")
        print(f"💾 Data: {self.output_dir}/data/")
        print(f"📝 Samples: {self.output_dir}/samples/")
        print("\nAll artifacts ready for interim submission!")

if __name__ == "__main__":
    generator = ResearchReportGenerator()
    generator.run()