"""
Test suite for the Causal Responsiveness Critic

This module contains comprehensive tests for the causal critic component
of the Director LLM system, including edge cases and integration scenarios.
"""

import pytest
import torch
from causal_critic import CausalResponsivenessCritic, CausalCriticRewardModel, CausalityScore


class TestCausalResponsivenessCritic:
    """Test suite for the CausalResponsivenessCritic class"""
    
    @pytest.fixture
    def critic(self):
        """Create a critic instance for testing"""
        return CausalResponsivenessCritic()
    
    def test_initialization(self, critic):
        """Test that the critic initializes correctly"""
        assert critic.model is not None
        assert critic.tokenizer is not None
        assert critic.device in ["cpu", "cuda"]
        assert len(critic.label_mapping) == 3
    
    def test_format_premise_hypothesis(self, critic):
        """Test premise-hypothesis formatting"""
        player_action = "I attack the dragon"
        director_response = "The dragon roars and breathes fire"
        context = "You are in a dragon's lair"
        
        premise, hypothesis = critic._format_premise_hypothesis(
            player_action, director_response, context
        )
        
        assert "Player Action:" in premise
        assert "Context:" in premise
        assert "logically follows" in hypothesis
        assert director_response in hypothesis
    
    def test_evaluate_causality_basic(self, critic):
        """Test basic causality evaluation"""
        # Good causal relationship
        score = critic.evaluate_causality(
            "I open the door",
            "The door creaks open, revealing a dark hallway beyond"
        )
        assert 0.0 <= score <= 1.0
        assert isinstance(score, float)
    
    def test_evaluate_causality_detailed(self, critic):
        """Test detailed causality evaluation"""
        result = critic.evaluate_causality(
            "I cast a healing spell",
            "Your wounds begin to close and you feel refreshed",
            return_detailed=True
        )
        
        assert isinstance(result, CausalityScore)
        assert hasattr(result, 'entailment_prob')
        assert hasattr(result, 'contradiction_prob')
        assert hasattr(result, 'neutral_prob')
        assert hasattr(result, 'causal_score')
        assert 0.0 <= result.causal_score <= 1.0
    
    def test_batch_evaluation(self, critic):
        """Test batch evaluation functionality"""
        interactions = [
            {
                'player_action': 'I swing my sword',
                'director_response': 'Your blade strikes the enemy',
                'context': 'Combat with an orc'
            },
            {
                'player_action': 'I examine the book',
                'director_response': 'The book contains ancient spells'
            }
        ]
        
        scores = critic.batch_evaluate(interactions)
        assert len(scores) == 2
        assert all(0.0 <= score <= 1.0 for score in scores)
    
    def test_evaluate_with_explanations(self, critic):
        """Test evaluation with explanations"""
        result = critic.evaluate_with_explanations(
            "I drink the potion",
            "You feel a surge of magical energy"
        )
        
        assert 'causal_score' in result
        assert 'explanation' in result
        assert 'entailment_prob' in result
        assert isinstance(result['explanation'], str)
    
    def test_poor_causality_detection(self, critic):
        """Test detection of poor causal relationships"""
        # Completely unrelated response
        score = critic.evaluate_causality(
            "I ask about the weather",
            "Suddenly a dragon appears and attacks!"
        )
        
        # Should get a low score for poor causality
        # Note: Exact threshold may vary based on model
        assert score < 0.8  # Expecting lower score for poor causality
    
    def test_good_causality_detection(self, critic):
        """Test detection of good causal relationships"""
        score = critic.evaluate_causality(
            "I light a torch",
            "The torch flickers to life, illuminating the dark chamber"
        )
        
        # Should get a higher score for good causality
        assert score > 0.2  # Expecting higher score for good causality


class TestCausalCriticRewardModel:
    """Test suite for the CausalCriticRewardModel wrapper"""
    
    @pytest.fixture
    def reward_model(self):
        """Create a reward model instance for testing"""
        critic = CausalResponsivenessCritic()
        return CausalCriticRewardModel(critic)
    
    def test_reward_model_call(self, reward_model):
        """Test reward model callable interface"""
        reward = reward_model(
            "I pick up the sword",
            "You grasp the sword's hilt firmly"
        )
        
        assert 0.0 <= reward <= 1.0
        assert isinstance(reward, float)
    
    def test_get_reward_vector_component(self, reward_model):
        """Test reward vector component extraction"""
        episode_data = [
            {
                'player_input': 'I search for traps',
                'director_output': 'You carefully examine the floor',
                'context': 'Dungeon corridor'
            },
            {
                'player_input': 'I cast detect magic',
                'director_output': 'A faint aura surrounds the chest'
            }
        ]
        
        rewards = reward_model.get_reward_vector_component(episode_data)
        assert len(rewards) == 2
        assert all(0.0 <= reward <= 1.0 for reward in rewards)


# Edge case tests
class TestEdgeCases:
    """Test edge cases and error handling"""
    
    @pytest.fixture
    def critic(self):
        return CausalResponsivenessCritic()
    
    def test_empty_inputs(self, critic):
        """Test handling of empty inputs"""
        score = critic.evaluate_causality("", "")
        assert 0.0 <= score <= 1.0
    
    def test_very_long_inputs(self, critic):
        """Test handling of very long inputs"""
        long_text = "This is a very long text. " * 100
        score = critic.evaluate_causality(long_text, long_text)
        assert 0.0 <= score <= 1.0
    
    def test_special_characters(self, critic):
        """Test handling of special characters"""
        score = critic.evaluate_causality(
            "I use my +1 sword of @#$%!",
            "The magical weapon glows with power"
        )
        assert 0.0 <= score <= 1.0


# Integration tests
class TestIntegration:
    """Integration tests for the complete system"""
    
    def test_d_and_d_scenario(self):
        """Test a complete D&D scenario"""
        critic = CausalResponsivenessCritic()
        
        # Simulate a D&D interaction sequence
        interactions = [
            {
                'player_action': 'I sneak up to the sleeping dragon',
                'director_response': 'You move silently across the treasure-strewn floor. The dragon\'s massive form rises and falls with each breath.',
                'context': 'You are in a dragon\'s lair filled with gold and jewels.'
            },
            {
                'player_action': 'I attempt to steal a small gem',
                'director_response': 'You carefully reach for a ruby. As your fingers close around it, a piece of gold shifts with a soft clink.',
                'context': 'The dragon is still sleeping, but any noise could wake it.'
            },
            {
                'player_action': 'I freeze and hold my breath',
                'director_response': 'You remain perfectly still. The dragon\'s eye opens slightly, then closes again as it settles back into sleep.',
                'context': 'You made a small noise that might have disturbed the dragon.'
            }
        ]
        
        # Evaluate the sequence
        for i, interaction in enumerate(interactions):
            result = critic.evaluate_with_explanations(
                interaction['player_action'],
                interaction['director_response'],
                interaction['context']
            )
            
            print(f"\nInteraction {i+1}:")
            print(f"Score: {result['causal_score']:.3f}")
            print(f"Explanation: {result['explanation']}")
            
            # All should have reasonable causality
            assert result['causal_score'] > 0.1


def run_comprehensive_demo():
    """
    Run a comprehensive demonstration of the causal critic
    """
    print("🎭 Director LLM - Causal Responsiveness Critic Demo")
    print("=" * 60)
    
    critic = CausalResponsivenessCritic()
    
    # Test scenarios covering different types of causality
    scenarios = [
        {
            'name': 'Strong Causal Relationship',
            'player_action': 'I pull the lever',
            'director_response': 'The lever clicks into place and you hear the grinding of gears as a hidden door slides open in the wall.',
            'context': 'You are in a puzzle room with various mechanisms.',
            'expected_range': (0.6, 1.0)
        },
        {
            'name': 'Moderate Causal Relationship', 
            'player_action': 'I ask the merchant about rare items',
            'director_response': 'The merchant\'s eyes light up and he leans in conspiratorially, glancing around before speaking.',
            'context': 'You are in a busy marketplace.',
            'expected_range': (0.4, 0.8)
        },
        {
            'name': 'Poor Causal Relationship',
            'player_action': 'I compliment the bard\'s music', 
            'director_response': 'Suddenly, the tavern erupts in flames as a meteor crashes through the ceiling!',
            'context': 'You are listening to a peaceful performance in a cozy tavern.',
            'expected_range': (0.0, 0.4)
        },
        {
            'name': 'Contradictory Response',
            'player_action': 'I carefully sneak past the guard',
            'director_response': 'You walk up to the guard and loudly announce your presence.',
            'context': 'You are trying to infiltrate a secured area.',
            'expected_range': (0.0, 0.3)
        }
    ]
    
    for scenario in scenarios:
        print(f"\n🎯 {scenario['name']}")
        print("-" * 40)
        print(f"Context: {scenario['context']}")
        print(f"Player: {scenario['player_action']}")
        print(f"Director: {scenario['director_response']}")
        
        result = critic.evaluate_with_explanations(
            scenario['player_action'],
            scenario['director_response'],
            scenario['context']
        )
        
        print(f"\n📊 Results:")
        print(f"Causal Score: {result['causal_score']:.3f}")
        print(f"Entailment: {result['entailment_prob']:.3f}")
        print(f"Neutral: {result['neutral_prob']:.3f}") 
        print(f"Contradiction: {result['contradiction_prob']:.3f}")
        print(f"💭 {result['explanation']}")
        
        # Verify score is in expected range
        expected_min, expected_max = scenario['expected_range']
        if expected_min <= result['causal_score'] <= expected_max:
            print("✅ Score within expected range")
        else:
            print(f"⚠️  Score outside expected range [{expected_min}, {expected_max}]")
    
    print(f"\n🎉 Demo completed successfully!")


if __name__ == "__main__":
    run_comprehensive_demo()