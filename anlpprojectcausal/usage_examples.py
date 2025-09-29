"""
Usage Examples and Integration Guide for Causal Responsiveness Critic

This module demonstrates how to use the Causal Responsiveness Critic
in various scenarios and integrate it with the broader Director LLM system.
"""

import sys
import os
from typing import List, Dict, Tuple
import json

# Add current directory to path for imports
sys.path.append(os.path.dirname(__file__))

from causal_critic import CausalResponsivenessCritic, CausalCriticRewardModel
from config import get_config, DOMAIN_CONFIGS


class DirectorLLMIntegration:
    """
    Example integration of Causal Critic with the Director LLM system
    
    This class shows how the causal critic fits into the broader MCRL framework
    described in the paper.
    """
    
    def __init__(self, config_name: str = "default"):
        """Initialize with causal critic and configuration"""
        config = get_config(config_name)
        self.causal_critic = CausalResponsivenessCritic(
            model_name=config.model_name,
            device=config.device
        )
        self.config = config
        
        # Initialize other critics (placeholder - would be implemented similarly)
        self.critics = {
            'causal': CausalCriticRewardModel(self.causal_critic),
            # 'narrative': NarrativeQualityCritic(),  # Would be implemented
            # 'consistency': WorldConsistencyCritic(),  # Would be implemented  
            # 'character': CharacterVoiceCritic(),     # Would be implemented
        }
    
    def evaluate_director_response(self, 
                                 player_input: str,
                                 director_output: str,
                                 context: str = "",
                                 player_intent: str = "DEFAULT") -> Dict:
        """
        Evaluate a Director's response using the causal critic
        
        This simulates how the critic would be used during MCRL training
        
        Args:
            player_input: Player's action or statement
            director_output: Director's generated response
            context: Story/world context
            player_intent: Classified player intent (EXPLORE, ACTION, DIALOGUE)
            
        Returns:
            Dictionary with evaluation results
        """
        # Get detailed causal evaluation
        causal_result = self.causal_critic.evaluate_with_explanations(
            player_input, director_output, context
        )
        
        # Apply dynamic weighting based on intent
        intent_weights = {
            "EXPLORE": 0.6,
            "ACTION": 1.0, 
            "DIALOGUE": 0.8,
            "DEFAULT": 0.7
        }
        
        weighted_score = causal_result['causal_score'] * intent_weights.get(player_intent, 0.7)
        
        return {
            'causal_score': causal_result['causal_score'],
            'weighted_causal_score': weighted_score,
            'entailment_prob': causal_result['entailment_prob'],
            'explanation': causal_result['explanation'],
            'player_intent': player_intent,
            'intent_weight': intent_weights.get(player_intent, 0.7)
        }
    
    def compute_reward_vector(self, 
                            episode_interactions: List[Dict]) -> List[Dict]:
        """
        Compute reward vectors for an episode (simulation of MCRL pipeline)
        
        Args:
            episode_interactions: List of player-director interactions
            
        Returns:
            List of reward dictionaries for each interaction
        """
        rewards = []
        
        for interaction in episode_interactions:
            # Extract components
            player_input = interaction.get('player_input', '')
            director_output = interaction.get('director_output', '')
            context = interaction.get('context', '')
            intent = interaction.get('player_intent', 'DEFAULT')
            
            # Evaluate with causal critic
            causal_reward = self.causal_critic.evaluate_causality(
                player_input, director_output, context
            )
            
            # In full system, would evaluate with all critics
            reward_vector = {
                'causal_responsiveness': causal_reward,
                # 'narrative_quality': narrative_reward,    # Would be computed
                # 'world_consistency': consistency_reward,  # Would be computed
                # 'character_voice': character_reward,      # Would be computed
            }
            
            # Apply dynamic weighting
            intent_weights = {
                'EXPLORE': [0.6, 0.8, 0.9, 0.5],   # [causal, narrative, world, character]
                'ACTION': [1.0, 0.7, 0.8, 0.6], 
                'DIALOGUE': [0.8, 0.9, 0.6, 1.0],
                'DEFAULT': [0.7, 0.7, 0.7, 0.7]
            }
            
            weights = intent_weights.get(intent, intent_weights['DEFAULT'])
            final_reward = causal_reward * weights[0]  # Just causal for now
            
            rewards.append({
                'reward_vector': reward_vector,
                'final_reward': final_reward,
                'weights': weights,
                'intent': intent
            })
        
        return rewards


def example_dnd_session():
    """
    Example D&D session showing causal critic in action
    """
    print("🐉 D&D Session with Causal Responsiveness Critic")
    print("=" * 60)
    
    # Initialize critic
    critic = CausalResponsivenessCritic()
    
    # Simulate a D&D session
    session_interactions = [
        {
            'turn': 1,
            'context': 'You enter a dimly lit tavern. The air is thick with smoke and the smell of ale.',
            'player_input': 'I approach the bartender and ask about rumors of treasure in the nearby ruins',
            'director_output': 'The grizzled bartender looks up from cleaning a mug, his eyes narrowing as he studies you. "Aye, there be talk of ancient gold in the Whispering Ruins," he says in a low voice.'
        },
        {
            'turn': 2,
            'context': 'The bartender has mentioned treasure in the Whispering Ruins.',
            'player_input': 'I buy him a drink to loosen his tongue and ask for more details',
            'director_output': 'The bartender\'s expression softens as you slide a coin across the bar. He pours himself a shot of whiskey and leans in conspiratorially. "The ruins are cursed, they say. But old Malachar\'s treasure vault still lies beneath..."'
        },
        {
            'turn': 3,
            'context': 'You\'ve learned about Malachar\'s treasure vault beneath the cursed ruins.',
            'player_input': 'I thank the bartender and head to the ruins immediately',
            'director_output': 'As you push through the tavern door, a hooded figure at a corner table suddenly stands and follows you outside into the moonlit street.'
        },
        {
            'turn': 4,
            'context': 'A mysterious hooded figure has followed you out of the tavern.',
            'player_input': 'I turn around and confront the figure',
            'director_output': 'The figure throws back their hood, revealing a young elven woman with urgent eyes. "Wait!" she calls out. "I couldn\'t help but overhear - you mustn\'t go to those ruins alone. I know the way past the guardians."'
        }
    ]
    
    total_score = 0
    for interaction in session_interactions:
        print(f"\n🎲 Turn {interaction['turn']}")
        print(f"Context: {interaction['context']}")
        print(f"Player: {interaction['player_input']}")
        print(f"DM: {interaction['director_output']}")
        
        # Evaluate causality
        result = critic.evaluate_with_explanations(
            interaction['player_input'],
            interaction['director_output'],
            interaction['context']
        )
        
        print(f"\n📊 Causal Analysis:")
        print(f"Score: {result['causal_score']:.3f}")
        print(f"Assessment: {result['explanation']}")
        
        total_score += result['causal_score']
        print("-" * 50)
    
    avg_score = total_score / len(session_interactions)
    print(f"\n🏆 Session Summary:")
    print(f"Average Causal Score: {avg_score:.3f}")
    
    if avg_score > 0.7:
        print("✅ Excellent causal coherence throughout the session!")
    elif avg_score > 0.5:
        print("👍 Good causal coherence with room for improvement")
    else:
        print("⚠️  Poor causal coherence - Director needs better training")


def benchmark_different_response_qualities():
    """
    Benchmark the critic against responses of different quality levels
    """
    print("\n🎯 Benchmarking Response Quality Detection")
    print("=" * 60)
    
    critic = CausalResponsivenessCritic()
    
    # Test scenarios with expected quality levels
    test_scenarios = [
        {
            'quality': 'Excellent',
            'player_input': 'I carefully pick the lock on the treasure chest',
            'responses': [
                'You insert your lockpicks into the mechanism and work carefully. After several tense minutes, you hear a satisfying click as the lock opens.',
                'Your skilled fingers manipulate the pins within the lock. The tumblers fall into place one by one until the chest opens with a soft creak.'
            ]
        },
        {
            'quality': 'Good',
            'player_input': 'I cast a fireball at the group of goblins',
            'responses': [
                'The fireball streaks through the air and explodes among the goblins, dealing significant damage.',
                'Your spell erupts in a burst of flame, catching two of the three goblins in the blast.'
            ]
        },
        {
            'quality': 'Poor',
            'player_input': 'I try to sneak past the sleeping guard',
            'responses': [
                'Suddenly, a dragon appears and attacks everyone in the room!',
                'You decide to have a loud conversation with your companion about your plans.'
            ]
        },
        {
            'quality': 'Contradictory',
            'player_input': 'I drink the healing potion',
            'responses': [
                'You throw the potion at the wall, shattering it completely.',
                'Instead of drinking it, you pour the potion on the ground and watch it sizzle.'
            ]
        }
    ]
    
    for scenario in test_scenarios:
        print(f"\n📋 {scenario['quality']} Response Quality")
        print(f"Player Action: {scenario['player_input']}")
        print("-" * 30)
        
        scores = []
        for i, response in enumerate(scenario['responses'], 1):
            score = critic.evaluate_causality(scenario['player_input'], response)
            scores.append(score)
            print(f"Response {i}: {response}")
            print(f"Causal Score: {score:.3f}")
            print()
        
        avg_score = sum(scores) / len(scores)
        print(f"Average Score for {scenario['quality']} responses: {avg_score:.3f}")
        print("=" * 50)


def integration_with_ppo_example():
    """
    Example showing how the causal critic integrates with PPO training
    """
    print("\n🔄 PPO Integration Example")
    print("=" * 60)
    
    # Initialize the integration system
    director_system = DirectorLLMIntegration("balanced")
    
    # Simulate an episode of interactions
    episode = [
        {
            'player_input': 'I examine the mysterious glowing orb',
            'director_output': 'As you approach the orb, it pulses with an otherworldly light. Ancient runes become visible on its surface.',
            'context': 'You are in an ancient wizard\'s tower',
            'player_intent': 'EXPLORE'
        },
        {
            'player_input': 'I try to read the runes',
            'director_output': 'The runes are written in an archaic magical script. You recognize some symbols related to divination magic.',
            'context': 'A glowing orb with ancient runes sits before you',
            'player_intent': 'ACTION'
        },
        {
            'player_input': 'I cast detect magic on the orb',
            'director_output': 'Your spell reveals powerful divination magic emanating from the orb, along with traces of something darker - necromantic energy.',
            'context': 'You\'ve been examining a magical orb with runes',
            'player_intent': 'ACTION'
        }
    ]
    
    # Compute rewards for the episode
    rewards = director_system.compute_reward_vector(episode)
    
    print("Episode Reward Analysis:")
    print("-" * 30)
    
    total_reward = 0
    for i, (interaction, reward) in enumerate(zip(episode, rewards)):
        print(f"\nTurn {i+1}:")
        print(f"Player: {interaction['player_input']}")
        print(f"Director: {interaction['director_output']}")
        print(f"Intent: {interaction['player_intent']}")
        print(f"Causal Score: {reward['reward_vector']['causal_responsiveness']:.3f}")
        print(f"Final Reward: {reward['final_reward']:.3f}")
        
        total_reward += reward['final_reward']
    
    avg_reward = total_reward / len(episode)
    print(f"\n🏆 Episode Summary:")
    print(f"Average Reward: {avg_reward:.3f}")
    print(f"Total Episode Reward: {total_reward:.3f}")
    
    # This reward would be used to update the Director's policy via PPO
    print(f"\n💡 This reward signal would be used to update the Director LLM's policy")
    print(f"   to generate more causally coherent responses in future episodes.")


def main():
    """
    Run all examples and demonstrations
    """
    print("🎭 Director LLM - Causal Responsiveness Critic")
    print("🎯 Comprehensive Usage Examples and Integration Guide")
    print("=" * 80)
    
    try:
        # Run example D&D session
        example_dnd_session()
        
        # Benchmark response quality detection
        benchmark_different_response_qualities()
        
        # Show PPO integration
        integration_with_ppo_example()
        
        print("\n🎉 All examples completed successfully!")
        print("\n📚 Next Steps:")
        print("1. Install requirements: pip install -r requirements.txt")
        print("2. Run tests: python test_causal_critic.py")  
        print("3. Integrate with your Director LLM training pipeline")
        print("4. Experiment with different configurations in config.py")
        
    except Exception as e:
        print(f"❌ Error running examples: {str(e)}")
        print("Make sure you have installed the requirements and have internet access")
        print("for downloading the pre-trained model.")


if __name__ == "__main__":
    main()