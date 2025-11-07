#!/usr/bin/env python3
"""
Quick Causal Critic Accuracy Check
Simplified version for fast testing
"""
import sys
sys.path.append('..')

from test_causal_accuracy import CausalCriticEvaluator
from datasets import load_from_disk

print("""
╔══════════════════════════════════════════════════════════════════════╗
║         CAUSAL CRITIC - QUICK ACCURACY TEST                          ║
║                                                                      ║
║  Tests zero-shot NLI model on D&D fantasy validation data           ║
║  Expected: 75-85% accuracy (zero-shot, no fine-tuning)              ║
╚══════════════════════════════════════════════════════════════════════╝
""")

# Load evaluator
evaluator = CausalCriticEvaluator()

# Load validation data
print("Loading validation dataset...")
val_dataset = load_from_disk('../data/causal_critic_training/val')
print(f"✓ Loaded {len(val_dataset):,} samples\n")

# Run quick test
print("Running quick test on 500 samples...")
results = evaluator.evaluate_dataset(val_dataset, max_samples=500, batch_size=32)

# Show metrics
evaluator.print_detailed_metrics(results)
evaluator.show_sample_predictions(val_dataset, results, num_correct=2, num_incorrect=2)

# Summary
print("\n" + "="*70)
print("QUICK TEST SUMMARY")
print("="*70)
print(f"\n🎯 Accuracy: {results['accuracy']:.4f} ({results['accuracy']*100:.2f}%)")
print(f"📊 Samples tested: {results['num_samples']:,} / {len(val_dataset):,}")

if results['accuracy'] >= 0.80:
    print("✅ EXCELLENT! Model performs well on fantasy data.")
    print("   Ready for PPO integration!")
elif results['accuracy'] >= 0.70:
    print("⚠️  GOOD. Consider fine-tuning for better performance.")
    print("   Usable for PPO but may benefit from training.")
else:
    print("❌ LOW accuracy. Fine-tuning recommended.")
    print("   Consider training with train.py before PPO.")

print("\n💡 Tip: Run test_causal_accuracy.py for full 25K sample evaluation")
print("="*70)
