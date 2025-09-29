"""
Causal Responsiveness Critic for the Director LLM System

This module implements the Causal Responsiveness Critic that evaluates how well
the Director's narrative responses logically follow from and causally respond to
player actions using a pre-trained NLI model.

Based on the paper: "The Director LLM: A Multi-Critic Reinforcement Learning 
Framework for Domain-Aware Narrative Generation"
"""

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from dataclasses import dataclass

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class CausalityScore:
    """Container for causality evaluation results"""
    entailment_prob: float
    contradiction_prob: float  
    neutral_prob: float
    causal_score: float
    raw_logits: List[float]


class CausalResponsivenessCritic:
    """
    Causal Responsiveness Critic using pre-trained NLI model
    
    This critic evaluates whether the Director's narrative responses
    causally and logically follow from player actions by treating
    this as a natural language inference task.
    
    Model: MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli
    """
    
    def __init__(self, 
                 model_name: str = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli",
                 device: Optional[str] = None):
        """
        Initialize the Causal Responsiveness Critic
        
        Args:
            model_name: Pre-trained NLI model to use
            device: Device to run model on (auto-detected if None)
        """
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        logger.info(f"Initializing Causal Critic with model: {model_name}")
        logger.info(f"Using device: {self.device}")
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        
        # NLI label mapping (standard for most NLI models)
        self.label_mapping = {
            0: "entailment",
            1: "neutral", 
            2: "contradiction"
        }
        
        logger.info("Causal Responsiveness Critic initialized successfully")
    
    def _format_premise_hypothesis(self, 
                                 player_action: str, 
                                 director_response: str,
                                 context: Optional[str] = None) -> Tuple[str, str]:
        """
        Format player action and director response for NLI evaluation
        
        Args:
            player_action: The player's action or statement
            director_response: The Director's narrative response
            context: Optional world/story context
            
        Returns:
            Tuple of (premise, hypothesis) for NLI model
        """
        # Premise: Player action + context
        if context:
            premise = f"Context: {context}\nPlayer Action: {player_action}"
        else:
            premise = f"Player Action: {player_action}"
        
        # Hypothesis: Director's response should logically follow
        hypothesis = f"The following narrative response logically follows: {director_response}"
        
        return premise, hypothesis
    
    def evaluate_causality(self, 
                         player_action: str,
                         director_response: str, 
                         context: Optional[str] = None,
                         return_detailed: bool = False) -> float:
        """
        Evaluate causal responsiveness between player action and director response
        
        Args:
            player_action: The player's action or statement
            director_response: The Director's narrative response
            context: Optional world/story context
            return_detailed: Whether to return detailed CausalityScore object
            
        Returns:
            Causal responsiveness score (0.0 to 1.0) or CausalityScore object
        """
        try:
            # Format inputs for NLI
            premise, hypothesis = self._format_premise_hypothesis(
                player_action, director_response, context
            )
            
            # Tokenize inputs
            inputs = self.tokenizer(
                premise, 
                hypothesis,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            ).to(self.device)
            
            # Get model predictions
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits[0]  # Remove batch dimension
                probabilities = torch.softmax(logits, dim=-1)
            
            # Extract probabilities
            entailment_prob = probabilities[0].item()  # entailment
            neutral_prob = probabilities[1].item()     # neutral
            contradiction_prob = probabilities[2].item()  # contradiction
            
            # Calculate causal score
            # High entailment = good causality
            # High contradiction = poor causality  
            # Neutral is intermediate
            causal_score = entailment_prob + 0.5 * neutral_prob
            
            if return_detailed:
                return CausalityScore(
                    entailment_prob=entailment_prob,
                    contradiction_prob=contradiction_prob,
                    neutral_prob=neutral_prob,
                    causal_score=causal_score,
                    raw_logits=logits.cpu().tolist()
                )
            
            return causal_score
            
        except Exception as e:
            logger.error(f"Error evaluating causality: {str(e)}")
            return 0.0  # Return minimum score on error
    
    def batch_evaluate(self, 
                      interactions: List[Dict[str, str]],
                      batch_size: int = 8) -> List[float]:
        """
        Evaluate causality for multiple interactions in batches
        
        Args:
            interactions: List of dicts with 'player_action', 'director_response', 
                         and optional 'context' keys
            batch_size: Batch size for processing
            
        Returns:
            List of causal responsiveness scores
        """
        scores = []
        
        for i in range(0, len(interactions), batch_size):
            batch = interactions[i:i + batch_size]
            batch_scores = []
            
            for interaction in batch:
                score = self.evaluate_causality(
                    interaction['player_action'],
                    interaction['director_response'], 
                    interaction.get('context')
                )
                batch_scores.append(score)
            
            scores.extend(batch_scores)
            
        return scores
    
    def evaluate_with_explanations(self, 
                                 player_action: str,
                                 director_response: str,
                                 context: Optional[str] = None) -> Dict:
        """
        Evaluate causality and provide human-readable explanations
        
        Args:
            player_action: The player's action or statement
            director_response: The Director's narrative response 
            context: Optional world/story context
            
        Returns:
            Dictionary with score and explanation
        """
        detailed_score = self.evaluate_causality(
            player_action, director_response, context, return_detailed=True
        )
        
        # Generate explanation based on probabilities
        if detailed_score.entailment_prob > 0.7:
            explanation = "Strong causal relationship - the response logically follows from the action"
        elif detailed_score.entailment_prob > 0.4:
            explanation = "Moderate causal relationship - the response somewhat follows from the action"
        elif detailed_score.contradiction_prob > 0.5:
            explanation = "Poor causal relationship - the response contradicts or ignores the action"
        else:
            explanation = "Weak causal relationship - the response is neutral or weakly related to the action"
        
        return {
            'causal_score': detailed_score.causal_score,
            'entailment_prob': detailed_score.entailment_prob,
            'neutral_prob': detailed_score.neutral_prob, 
            'contradiction_prob': detailed_score.contradiction_prob,
            'explanation': explanation,
            'premise_hypothesis': self._format_premise_hypothesis(
                player_action, director_response, context
            )
        }


class CausalCriticRewardModel:
    """
    Wrapper class to integrate Causal Critic into MCRL reward pipeline
    
    This class provides the interface expected by the PPO training loop
    for the multi-critic reinforcement learning framework.
    """
    
    def __init__(self, critic: CausalResponsivenessCritic):
        """
        Initialize reward model wrapper
        
        Args:
            critic: Initialized CausalResponsivenessCritic instance
        """
        self.critic = critic
        self.name = "causal_responsiveness"
    
    def __call__(self, 
                 player_input: str,
                 director_output: str, 
                 context: Optional[str] = None) -> float:
        """
        Compute reward for MCRL training
        
        Args:
            player_input: Player's action/statement
            director_output: Director's generated response
            context: Optional story/world context
            
        Returns:
            Reward score between 0.0 and 1.0
        """
        return self.critic.evaluate_causality(player_input, director_output, context)
    
    def get_reward_vector_component(self, 
                                  episode_data: List[Dict]) -> List[float]:
        """
        Extract causal responsiveness rewards for an entire episode
        
        Args:
            episode_data: List of interaction dictionaries from episode
            
        Returns:
            List of reward values for the episode
        """
        rewards = []
        for interaction in episode_data:
            reward = self(
                interaction.get('player_input', ''),
                interaction.get('director_output', ''),
                interaction.get('context', '')
            )
            rewards.append(reward)
        return rewards


def main():
    """
    Demo function showing how to use the Causal Responsiveness Critic
    """
    print("Initializing Causal Responsiveness Critic...")
    critic = CausalResponsivenessCritic()
    
    # Example interactions for testing
    test_cases = [
        {
            "player_action": "I cast a fireball at the goblin",
            "director_response": "The fireball streaks through the air and explodes against the goblin, dealing 8 points of fire damage. The goblin shrieks and stumbles backward, its clothes singed.",
            "context": "You are in a dark dungeon corridor. A hostile goblin blocks your path, wielding a rusty dagger."
        },
        {
            "player_action": "I try to pick the lock on the chest", 
            "director_response": "You carefully insert your lockpicks into the mechanism. After a few tense moments, you hear a satisfying click as the lock opens.",
            "context": "You've found an ornate treasure chest in the corner of the room."
        },
        {
            "player_action": "I ask the innkeeper about the missing merchant",
            "director_response": "Suddenly, a dragon bursts through the roof and begins attacking everyone in the tavern!",
            "context": "You are in a peaceful tavern in the town of Millbrook. The innkeeper seems nervous when you mention the merchant."
        }
    ]
    
    print("\nEvaluating test cases...")
    print("=" * 60)
    
    for i, case in enumerate(test_cases, 1):
        print(f"\nTest Case {i}:")
        print(f"Player Action: {case['player_action']}")
        print(f"Director Response: {case['director_response']}")
        print(f"Context: {case['context']}")
        
        # Get detailed evaluation
        result = critic.evaluate_with_explanations(
            case['player_action'],
            case['director_response'], 
            case['context']
        )
        
        print(f"\nResults:")
        print(f"Causal Score: {result['causal_score']:.3f}")
        print(f"Entailment: {result['entailment_prob']:.3f}")
        print(f"Neutral: {result['neutral_prob']:.3f}")
        print(f"Contradiction: {result['contradiction_prob']:.3f}")
        print(f"Explanation: {result['explanation']}")
        print("-" * 60)
    
    # Test batch evaluation
    print("\nTesting batch evaluation...")
    batch_scores = critic.batch_evaluate(test_cases)
    print(f"Batch scores: {[f'{score:.3f}' for score in batch_scores]}")
    
    # Test reward model wrapper
    print("\nTesting reward model wrapper...")
    reward_model = CausalCriticRewardModel(critic)
    for case in test_cases:
        reward = reward_model(
            case['player_action'],
            case['director_response'],
            case['context']
        )
        print(f"Reward: {reward:.3f}")


if __name__ == "__main__":
    main()