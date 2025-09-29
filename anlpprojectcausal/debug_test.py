#!/usr/bin/env python3
"""
Performance and Debugging Tests for Causal Responsiveness Critic

Test performance, error handling, and detailed debugging information
"""

import time
import traceback
from causal_critic import CausalResponsivenessCritic
import torch

def test_performance():
    """Test inference speed and memory usage"""
    
    print("⚡ Performance Testing")
    print("=" * 30)
    
    critic = CausalResponsivenessCritic()
    
    # Test single evaluation speed
    test_cases = [
        ("I attack the dragon", "The dragon roars and strikes back with its claws"),
        ("I pick the lock", "You hear a click as the lock opens"),
        ("I cast a spell", "Magic energy flows from your hands"),
    ]
    
    print("🔄 Single Evaluation Speed:")
    total_time = 0
    for i, (action, response) in enumerate(test_cases, 1):
        start_time = time.time()
        score = critic.evaluate_causality(action, response)
        end_time = time.time()
        elapsed = end_time - start_time
        total_time += elapsed
        
        print(f"Test {i}: {elapsed:.3f}s (Score: {score:.3f})")
    
    avg_time = total_time / len(test_cases)
    print(f"Average time per evaluation: {avg_time:.3f}s")
    
    # Test batch processing speed
    print(f"\n🚀 Batch Evaluation Speed:")
    batch_interactions = [
        {'player_action': action, 'director_response': response} 
        for action, response in test_cases * 10  # 30 evaluations
    ]
    
    start_time = time.time()
    batch_scores = critic.batch_evaluate(batch_interactions, batch_size=8)
    end_time = time.time()
    batch_time = end_time - start_time
    
    print(f"Batch of {len(batch_interactions)} evaluations: {batch_time:.3f}s")
    print(f"Average time per evaluation (batch): {batch_time/len(batch_interactions):.3f}s")
    print(f"Speedup: {avg_time/(batch_time/len(batch_interactions)):.1f}x faster")

def test_error_handling():
    """Test error handling and edge cases"""
    
    print("\n🛠️  Error Handling Tests")
    print("=" * 30)
    
    critic = CausalResponsivenessCritic()
    
    error_test_cases = [
        ("Empty strings", "", ""),
        ("Very long input", "I " + "do something " * 200, "The result is " + "amazing " * 200),
        ("Unicode characters", "I use my 🗡️ sword", "The ⚔️ strikes true! 💥"),
        ("Special characters", "I say: \"Hello @#$%^&*()\"", "They reply: <>&[]{}"),
        ("None context", "I walk forward", "You move ahead", None),
    ]
    
    for test_name, action, response, *context in error_test_cases:
        try:
            ctx = context[0] if context else "Test scenario"
            print(f"Testing {test_name}...")
            
            score = critic.evaluate_causality(action, response, ctx)
            print(f"  ✅ Success: Score {score:.3f}")
            
        except Exception as e:
            print(f"  ❌ Error: {str(e)}")
            print(f"  Details: {traceback.format_exc()}")

def test_detailed_analysis():
    """Get detailed analysis for debugging"""
    
    print("\n🔍 Detailed Analysis")
    print("=" * 30)
    
    critic = CausalResponsivenessCritic()
    
    # Test case with expected strong causality
    strong_case = {
        'player_action': 'I light the torch',
        'director_response': 'The torch flickers to life, casting dancing shadows on the walls',
        'context': 'You are in a dark cave'
    }
    
    # Test case with expected weak causality  
    weak_case = {
        'player_action': 'I ask about the weather',
        'director_response': 'Suddenly a meteor crashes through the roof!',
        'context': 'You are having a casual conversation'
    }
    
    for case_name, case in [("Strong Causality", strong_case), ("Weak Causality", weak_case)]:
        print(f"\n📊 {case_name} Analysis:")
        print(f"Player: {case['player_action']}")
        print(f"Director: {case['director_response']}")
        
        # Get full detailed analysis
        detailed = critic.evaluate_causality(
            case['player_action'], 
            case['director_response'], 
            case['context'],
            return_detailed=True
        )
        
        print(f"\nDetailed Results:")
        print(f"  Causal Score: {detailed.causal_score:.3f}")
        print(f"  Entailment:   {detailed.entailment_prob:.3f}")
        print(f"  Neutral:      {detailed.neutral_prob:.3f}")
        print(f"  Contradiction: {detailed.contradiction_prob:.3f}")
        print(f"  Raw Logits:   {[f'{x:.3f}' for x in detailed.raw_logits]}")
        
        # Get explanation
        explanation_result = critic.evaluate_with_explanations(
            case['player_action'],
            case['director_response'], 
            case['context']
        )
        
        print(f"  Explanation: {explanation_result['explanation']}")
        print(f"  Premise: {explanation_result['premise_hypothesis'][0]}")
        print(f"  Hypothesis: {explanation_result['premise_hypothesis'][1]}")

def test_model_info():
    """Display model and system information"""
    
    print("\n💻 System Information")
    print("=" * 30)
    
    critic = CausalResponsivenessCritic()
    
    print(f"Model: {critic.model_name}")
    print(f"Device: {critic.device}")
    print(f"Model type: {type(critic.model).__name__}")
    print(f"Tokenizer type: {type(critic.tokenizer).__name__}")
    print(f"Label mapping: {critic.label_mapping}")
    
    # Check if CUDA is available
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device count: {torch.cuda.device_count()}")
        print(f"Current CUDA device: {torch.cuda.current_device()}")
    
    # Model parameters info
    total_params = sum(p.numel() for p in critic.model.parameters())
    trainable_params = sum(p.numel() for p in critic.model.parameters() if p.requires_grad)
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

if __name__ == "__main__":
    test_model_info()
    test_performance()
    test_error_handling()
    test_detailed_analysis()
    
    print("\n🎉 Performance and debugging tests completed!")
    print("\n💡 Debugging Tips:")
    print("- Use return_detailed=True for full probability breakdown")
    print("- Check premise/hypothesis formatting for unexpected results")
    print("- Monitor inference time for production deployment")
    print("- Test edge cases specific to your domain")