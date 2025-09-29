#!/usr/bin/env python3
"""
Custom Testing Script for Causal Responsiveness Critic

Create your own test scenarios here to evaluate specific interactions
"""

from causal_critic import CausalResponsivenessCritic
from config import get_config

def test_your_scenarios():
    """Test your own custom scenarios"""
    
    print("🎯 Custom Causal Responsiveness Testing")
    print("=" * 50)
    
    # Initialize critic with your preferred config
    critic = CausalResponsivenessCritic()
    
    # Define your test scenarios
    custom_scenarios = [
        {
            'name': 'Combat Scenario',
            'player_action': 'I swing my sword at the orc',
            'director_response': 'Your blade cuts deep into the orc\'s shoulder, causing it to howl in pain and rage.',
            'context': 'You are fighting an orc warrior in a narrow corridor'
        },
        {
            'name': 'Social Interaction',
            'player_action': 'I try to convince the guard to let us pass',
            'director_response': 'The guard looks skeptical but considers your words carefully before nodding slowly.',
            'context': 'You need to get past a checkpoint guard'
        },
        {
            'name': 'Exploration',
            'player_action': 'I search the ancient library for clues',
            'director_response': 'Among the dusty tomes, you discover a journal with entries about the lost artifact.',
            'context': 'You are in an abandoned wizard\'s tower library'
        }
    ]
    
    total_score = 0
    for i, scenario in enumerate(custom_scenarios, 1):
        print(f"\n🎭 Scenario {i}: {scenario['name']}")
        print(f"Context: {scenario['context']}")
        print(f"Player: {scenario['player_action']}")
        print(f"Director: {scenario['director_response']}")
        
        # Get detailed evaluation
        result = critic.evaluate_with_explanations(
            scenario['player_action'],
            scenario['director_response'],
            scenario['context']
        )
        
        print(f"\n📊 Results:")
        print(f"Causal Score: {result['causal_score']:.3f}")
        print(f"Explanation: {result['explanation']}")
        
        # Quality assessment
        score = result['causal_score']
        if score > 0.7:
            assessment = "🟢 Excellent causality"
        elif score > 0.5:
            assessment = "🟡 Good causality"
        elif score > 0.3:
            assessment = "🟠 Moderate causality"
        else:
            assessment = "🔴 Poor causality"
        
        print(f"Assessment: {assessment}")
        total_score += score
        print("-" * 50)
    
    avg_score = total_score / len(custom_scenarios)
    print(f"\n🏆 Overall Results:")
    print(f"Average Causal Score: {avg_score:.3f}")
    
    if avg_score > 0.7:
        print("✅ Excellent overall causal coherence!")
    elif avg_score > 0.5:
        print("👍 Good overall causal coherence")
    else:
        print("⚠️  Causal coherence needs improvement")

def test_batch_scenarios():
    """Test multiple scenarios in batch for efficiency"""
    
    print("\n🚀 Batch Testing")
    print("=" * 30)
    
    critic = CausalResponsivenessCritic()
    
    # Multiple scenarios for batch processing
    batch_scenarios = [
        {
            'player_action': 'I cast cure light wounds on myself',
            'director_response': 'Warm healing energy flows through your body, closing minor cuts and bruises.',
            'context': 'You are injured after a battle'
        },
        {
            'player_action': 'I pick up the mysterious crystal',
            'director_response': 'The crystal feels warm to the touch and begins to glow brighter in your hand.',
            'context': 'A glowing crystal sits on an ancient pedestal'
        },
        {
            'player_action': 'I ask the merchant about his prices',
            'director_response': 'A meteor crashes through the ceiling, destroying everything!',
            'context': 'You are in a peaceful marketplace'
        },
        {
            'player_action': 'I carefully disarm the trap',
            'director_response': 'Your steady hands work the mechanism until you hear a soft click as the trap becomes safe.',
            'context': 'You\'ve detected a pressure plate trap on the floor'
        }
    ]
    
    # Evaluate all scenarios in batch
    scores = critic.batch_evaluate(batch_scenarios)
    
    print(f"Batch evaluation completed!")
    print(f"Scores: {[f'{score:.3f}' for score in scores]}")
    print(f"Average: {sum(scores)/len(scores):.3f}")

def test_different_configs():
    """Test with different configuration settings"""
    
    print("\n⚙️  Configuration Testing")
    print("=" * 35)
    
    test_scenario = {
        'player_action': 'I negotiate with the dragon',
        'director_response': 'The ancient dragon considers your words, its intelligent eyes studying you carefully.',
        'context': 'You face an ancient red dragon in its lair'
    }
    
    configs = ['default', 'strict', 'lenient', 'balanced']
    
    for config_name in configs:
        config = get_config(config_name)
        critic = CausalResponsivenessCritic(
            model_name=config.model_name,
            device=config.device
        )
        
        score = critic.evaluate_causality(
            test_scenario['player_action'],
            test_scenario['director_response'],
            test_scenario['context']
        )
        
        print(f"{config_name.capitalize()} config: {score:.3f}")

if __name__ == "__main__":
    test_your_scenarios()
    test_batch_scenarios() 
    test_different_configs()
    
    print("\n🎉 Custom testing completed!")
    print("\n💡 Tips for better testing:")
    print("- Test with scenarios similar to your actual use case")
    print("- Include both good and bad examples")
    print("- Test edge cases like empty inputs or very long text")
    print("- Experiment with different model configurations")