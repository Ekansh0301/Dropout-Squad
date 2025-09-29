#!/usr/bin/env python
"""
Training script for the Narrative Critic model
This script provides easy execution with predefined parameters
"""

import subprocess
import sys
import os

def train_narrative_critic():
    """Train the narrative critic with optimal parameters"""
    
    # Base directory
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Training arguments
    cmd = [
        sys.executable, "model_critic.py",
        "--model_name_or_path", "microsoft/deberta-v3-base",
        "--train_file", "ROCStories__spring2016 - ROCStories_spring2016.csv",
        "--validation_file", "cloze_test_val__spring2016 - cloze_test_ALL_val.csv",
        "--test_file", "cloze_test_test__spring2016 - cloze_test_ALL_test.csv",
        "--do_train",
        "--do_eval",
        "--do_predict",
        "--max_seq_length", "512",
        "--per_device_train_batch_size", "8",
        "--per_device_eval_batch_size", "16",
        "--learning_rate", "2e-5",
        "--num_train_epochs", "3",
        "--output_dir", "./narrative_critic_model",
        "--overwrite_output_dir",
        "--logging_steps", "100",
        "--eval_steps", "500", 
        "--save_steps", "1000",
        "--eval_strategy", "steps",
        "--save_strategy", "steps",
        "--load_best_model_at_end",
        "--metric_for_best_model", "eval_f1",
        "--greater_is_better", "True",
        "--warmup_steps", "500",
        "--weight_decay", "0.01",
        "--logging_dir", "./logs",
        "--seed", "42",
        "--negative_sampling_ratio", "1.0",
        "--max_train_samples", "10000",  # Remove this for full training
        "--max_eval_samples", "1000",   # Remove this for full evaluation
    ]
    
    print("Starting Narrative Critic training...")
    print("Command:", " ".join(cmd))
    print("-" * 80)
    
    # Run the training
    result = subprocess.run(cmd, cwd=base_dir)
    
    if result.returncode == 0:
        print("\n" + "=" * 80)
        print("Training completed successfully!")
        print("Model saved in: ./narrative_critic_model")
        print("Logs saved in: ./logs")
        print("=" * 80)
    else:
        print(f"\nTraining failed with return code: {result.returncode}")
        return False
    
    return True

def quick_test():
    """Quick test run with minimal data for debugging"""
    
    # Base directory
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Test arguments
    cmd = [
        sys.executable, "model_critic.py",
        "--model_name_or_path", "microsoft/deberta-v3-base",
        "--train_file", "ROCStories__spring2016 - ROCStories_spring2016.csv",
        "--validation_file", "cloze_test_val__spring2016 - cloze_test_ALL_val.csv",
        "--do_train",
        "--do_eval",
        "--max_seq_length", "256",
        "--per_device_train_batch_size", "4",
        "--per_device_eval_batch_size", "8",
        "--learning_rate", "3e-5",
        "--num_train_epochs", "1",
        "--output_dir", "./test_model",
        "--overwrite_output_dir",
        "--logging_steps", "10",
        "--eval_steps", "50",
        "--eval_strategy", "steps",
        "--seed", "42",
        "--max_train_samples", "100",
        "--max_eval_samples", "50",
    ]
    
    print("Starting quick test run...")
    print("Command:", " ".join(cmd))
    print("-" * 80)
    
    # Run the test
    result = subprocess.run(cmd, cwd=base_dir)
    
    if result.returncode == 0:
        print("\n" + "=" * 80)
        print("Quick test completed successfully!")
        print("=" * 80)
    else:
        print(f"\nQuick test failed with return code: {result.returncode}")
        return False
    
    return True

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train Narrative Critic Model")
    parser.add_argument("--quick_test", action="store_true", 
                       help="Run quick test with minimal data")
    
    args = parser.parse_args()
    
    if args.quick_test:
        success = quick_test()
    else:
        success = train_narrative_critic()
    
    if not success:
        sys.exit(1)