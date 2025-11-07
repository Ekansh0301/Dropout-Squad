#!/usr/bin/env python3
"""
Quick viewer for evaluation results
Usage: python3 view_results.py
"""

import json
from pathlib import Path
import sys

def print_header(text):
    print("\n" + "=" * 70)
    print(f" {text}")
    print("=" * 70 + "\n")

def view_summary():
    """Display summary statistics"""
    print_header("📊 SUMMARY STATISTICS")
    
    json_path = Path("evaluation_results/metrics/summary_statistics.json")
    if not json_path.exists():
        print("❌ No evaluation results found. Run evaluation first:")
        print("   python3 evaluate_sft_model.py")
        return False
    
    with open(json_path) as f:
        stats = json.load(f)
    
    print(f"Total Samples: {stats['total_samples']}")
    print(f"Categories: {stats['categories']}")
    print()
    
    print("📝 Word Count:")
    wc = stats['word_count']
    print(f"   Mean: {wc['mean']:.1f} ± {wc['std']:.1f}")
    print(f"   Range: {wc['min']} - {wc['max']}")
    print(f"   Median: {wc['median']:.1f}")
    print()
    
    print("🎯 Expected Elements Coverage:")
    ee = stats['expected_elements']
    print(f"   Mean: {ee['mean_ratio']:.1%}")
    print(f"   All elements found: {ee['samples_with_all_elements']} samples")
    print()
    
    print("🎲 D&D Terminology:")
    dnd = stats['dnd_terms']
    print(f"   Mean: {dnd['mean_count']:.1f} ± {dnd['std']:.1f} terms/response")
    print(f"   Max: {dnd['max']} terms")
    print()
    
    print("✅ Quality Indicators:")
    qi = stats['quality_indicators']
    print(f"   Repetition: {qi['repetition_score_mean']:.3f} (lower is better)")
    print(f"   Complete: {qi['complete_responses']}/{stats['total_samples']} ({qi['complete_responses']/stats['total_samples']:.1%})")
    
    return True

def view_category_breakdown():
    """Display per-category statistics"""
    print_header("📂 CATEGORY BREAKDOWN")
    
    json_path = Path("evaluation_results/metrics/summary_statistics.json")
    with open(json_path) as f:
        stats = json.load(f)
    
    categories = stats['by_category']
    
    # Sort by count
    sorted_cats = sorted(categories.items(), key=lambda x: x[1]['count'], reverse=True)
    
    for cat_name, cat_stats in sorted_cats:
        print(f"\n{cat_name}:")
        print(f"   Samples: {cat_stats['count']}")
        print(f"   Avg Words: {cat_stats['avg_word_count']:.1f}")
        print(f"   Expected Elements: {cat_stats['avg_expected_elements_ratio']:.1%}")
        print(f"   D&D Terms: {cat_stats['avg_dnd_terms']:.1f}")
        print(f"   Repetition: {cat_stats['avg_repetition']:.3f}")

def view_best_examples():
    """Display a few best examples"""
    print_header("⭐ BEST EXAMPLES")
    
    samples_path = Path("evaluation_results/samples/generated_samples.json")
    with open(samples_path) as f:
        samples = json.load(f)
    
    # Sort by quality score (expected_elements_ratio - repetition_score)
    scored = [
        (s, s['expected_elements_ratio'] - s['repetition_score'])
        for s in samples
        if not s.get('likely_cutoff', False)  # Only complete responses
    ]
    scored.sort(key=lambda x: x[1], reverse=True)
    
    # Show top 3
    for i, (sample, score) in enumerate(scored[:3], 1):
        print(f"\n{'─' * 70}")
        print(f"Example {i}: {sample['category']} (Quality Score: {score:.3f})")
        print(f"{'─' * 70}")
        
        # Truncate prompt for display
        prompt = sample['prompt'].split('\n')[-1][:60]
        print(f"\nPrompt: {prompt}...")
        
        # Show first 200 chars of response
        response = sample['response'][:200]
        if len(sample['response']) > 200:
            response += "..."
        print(f"\nResponse:\n{response}")
        
        print(f"\nMetrics:")
        print(f"   Words: {sample['word_count']}")
        print(f"   Expected Elements: {sample['expected_elements_ratio']:.1%}")
        print(f"   D&D Terms: {sample['dnd_terms_count']}")
        print(f"   Repetition: {sample['repetition_score']:.3f}")

def view_plots():
    """Show available plots"""
    print_header("📈 AVAILABLE PLOTS")
    
    plots_dir = Path("evaluation_results/plots")
    if not plots_dir.exists():
        print("❌ No plots found")
        return
    
    plots = list(plots_dir.glob("*.png"))
    
    if not plots:
        print("❌ No plot files found")
        return
    
    print("Generated plots:\n")
    for plot in sorted(plots):
        size_kb = plot.stat().st_size / 1024
        print(f"   ✅ {plot.name} ({size_kb:.1f} KB)")
    
    print(f"\nTotal plots: {len(plots)}")
    print(f"\nLocation: {plots_dir.absolute()}")
    print("\n💡 Open these images to view the visualizations")

def view_report():
    """Display the evaluation report"""
    print_header("📄 EVALUATION REPORT")
    
    report_path = Path("evaluation_results/metrics/evaluation_report.txt")
    if not report_path.exists():
        print("❌ No report found")
        return
    
    with open(report_path) as f:
        content = f.read()
    
    # Show first 1500 chars
    if len(content) > 1500:
        print(content[:1500])
        print("\n... (truncated)")
        print(f"\n📖 Full report: {report_path.absolute()}")
    else:
        print(content)

def main():
    """Main viewer function"""
    
    # Check if results exist
    if not Path("evaluation_results").exists():
        print("\n❌ No evaluation results found!")
        print("\nRun evaluation first:")
        print("   python3 evaluate_sft_model.py")
        print()
        sys.exit(1)
    
    print("\n" + "╔" + "═" * 68 + "╗")
    print("║" + " " * 20 + "EVALUATION RESULTS VIEWER" + " " * 23 + "║")
    print("╚" + "═" * 68 + "╝")
    
    # Display all sections
    if view_summary():
        view_category_breakdown()
        view_best_examples()
        view_plots()
    
    print("\n" + "═" * 70)
    print("\n💡 For more details:")
    print("   • Full report: evaluation_results/metrics/evaluation_report.txt")
    print("   • All samples: evaluation_results/samples/generated_samples.txt")
    print("   • Plots: evaluation_results/plots/")
    print()

if __name__ == "__main__":
    main()
