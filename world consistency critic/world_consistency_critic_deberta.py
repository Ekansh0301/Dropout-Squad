"""
World Consistency Critic - DeBERTa-v3-Large Inference
Standalone trained model for detecting narrative inconsistencies in D&D dialogue.

This is a completely new implementation using DeBERTa-v3-Large.
No connection to previous regex-based implementations.

Usage:
    from world_consistency_critic_deberta import WorldConsistencyCritic
    
    # Load trained model
    critic = WorldConsistencyCritic(
        model_path="/kaggle/input/world-consistency-critic-deberta"
    )
    
    # Score DM response
    history = ["You unlock the door", "The door swings open"]
    dm_response = "The locked door blocks your path"
    score = critic.score(dm_response, history)  # Returns: 0.0 (contradiction)
"""

import json
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import Dict, List, Tuple


class WorldConsistencyCritic:
    """
    World Consistency Critic using fine-tuned DeBERTa-v3-Large.
    
    Detects four types of responses:
    - Contradiction (0.0): Violating established facts
    - Hallucination (0.3): Introducing excessive entities
    - Amnesia (0.5): Forgetting prior information
    - Consistent (1.0): Respecting world state
    """
    
    def __init__(self, model_path: str, device: str = None):
        """
        Initialize the critic with trained DeBERTa model.
        
        Args:
            model_path: Path to trained model directory
                       (e.g., "/kaggle/input/world-consistency-critic-deberta")
            device: Device to run on ('cuda' or 'cpu', auto-detected if None)
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load model and tokenizer
        print(f"Loading World Consistency Critic from {model_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()
        
        # Load configuration
        config_path = f"{model_path}/training_config.json"
        try:
            with open(config_path, 'r') as f:
                self.config = json.load(f)
        except FileNotFoundError:
            # Default configuration
            self.config = {
                'label_mapping': {
                    '0': 'contradiction',
                    '1': 'hallucination',
                    '2': 'amnesia',
                    '3': 'consistent'
                },
                'score_mapping': {
                    'contradiction': 0.0,
                    'hallucination': 0.3,
                    'amnesia': 0.5,
                    'consistent': 1.0
                }
            }
        
        # Convert string keys to int
        self.label_mapping = {int(k): v for k, v in self.config['label_mapping'].items()}
        self.score_mapping = self.config['score_mapping']
        
        print(f"✓ Model loaded on {self.device}")
        print(f"  Accuracy: {self.config.get('test_accuracy', 0)*100:.2f}%")
        print(f"  Macro F1: {self.config.get('test_macro_f1', 0)*100:.2f}%")
    
    def score(self, dm_response: str, history: List[str] = None, return_details: bool = False) -> float:
        """
        Score a DM response for world consistency.
        
        Args:
            dm_response: The DM's generated response
            history: List of previous conversation turns (optional)
            return_details: If True, return dict with details
            
        Returns:
            Consistency score (0.0 to 1.0) or detailed dict
        """
        # Format input text
        if history:
            # Limit history to last 3 turns to avoid exceeding max length
            history_text = " [SEP] ".join(history[-3:])
            text = f"{history_text} [RESPONSE] {dm_response}"
        else:
            text = f"[RESPONSE] {dm_response}"
        
        # Tokenize
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Get prediction
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1)[0]
            predicted_class = torch.argmax(logits, dim=1).item()
        
        # Get label and score
        predicted_label = self.label_mapping[predicted_class]
        consistency_score = self.score_mapping[predicted_label]
        
        if return_details:
            return {
                'score': consistency_score,
                'predicted_class': predicted_label,
                'confidence': probs[predicted_class].item(),
                'probabilities': {
                    self.label_mapping[i]: prob.item()
                    for i, prob in enumerate(probs)
                },
                'explanation': self._get_explanation(predicted_label, probs[predicted_class].item())
            }
        else:
            return consistency_score
    
    def evaluate_with_explanation(self, dm_response: str, history: List[str] = None) -> Dict:
        """
        Evaluate response and return detailed explanation.
        
        Args:
            dm_response: The DM's generated response
            history: List of previous conversation turns (optional)
            
        Returns:
            Dictionary with score, label, confidence, and explanation
        """
        return self.score(dm_response, history, return_details=True)
    
    def _get_explanation(self, label: str, confidence: float) -> str:
        """Generate human-readable explanation"""
        explanations = {
            'contradiction': f"Response contradicts established facts (confidence: {confidence*100:.1f}%)",
            'hallucination': f"Response introduces excessive new entities (confidence: {confidence*100:.1f}%)",
            'amnesia': f"Response forgets prior information (confidence: {confidence*100:.1f}%)",
            'consistent': f"Response is consistent with world state (confidence: {confidence*100:.1f}%)"
        }
        return explanations.get(label, f"Unknown label: {label}")
    
    def batch_score(self, responses: List[str], histories: List[List[str]] = None) -> List[float]:
        """
        Score multiple responses in batch.
        
        Args:
            responses: List of DM responses
            histories: List of conversation histories (one per response)
            
        Returns:
            List of consistency scores
        """
        if histories is None:
            histories = [None] * len(responses)
        
        scores = []
        for response, history in zip(responses, histories):
            scores.append(self.score(response, history))
        
        return scores
    
    def reset(self):
        """Reset state (for compatibility with stateful critics)"""
        pass
    
    def update_world_state(self, text: str):
        """Update world state (for compatibility with stateful critics)"""
        pass


# Convenience function for quick scoring
def score_world_consistency(
    dm_response: str,
    history: List[str] = None,
    critic: WorldConsistencyCritic = None,
    model_path: str = "/kaggle/input/world-consistency-critic-deberta"
) -> float:
    """
    Quick scoring function for world consistency.
    
    Args:
        dm_response: DM response to evaluate
        history: List of previous conversation turns
        critic: Existing critic instance (creates new if None)
        model_path: Path to trained model (default: Kaggle input path)
        
    Returns:
        Consistency score (0.0 to 1.0)
    """
    if critic is None:
        critic = WorldConsistencyCritic(model_path)
    
    return critic.score(dm_response, history)


if __name__ == "__main__":
    # Example usage
    print("World Consistency Critic - Example Usage\n")
    
    # Initialize critic
    critic = WorldConsistencyCritic(model_path="./world_consistency_critic_final")
    
    # Test examples
    test_cases = [
        {
            'history': [
                "You unlock the door with the rusty key",
                "The door swings open, revealing a dark corridor."
            ],
            'response': "The locked door blocks your path.",
            'expected': 'contradiction'
        },
        {
            'history': [
                "You enter the quiet tavern",
                "The room is nearly empty."
            ],
            'response': "The tavern explodes with activity: ten merchants, eight guards, and five bards fill the room.",
            'expected': 'hallucination'
        },
        {
            'history': [
                "The innkeeper says 'Welcome! I am Gregor.'",
                "You chat with the innkeeper."
            ],
            'response': "The innkeeper smiles warmly, though you can't recall his name.",
            'expected': 'amnesia'
        },
        {
            'history': [
                "You carefully pick the lock on the chest",
                "The chest opens with a satisfying click."
            ],
            'response': "Inside the chest, you find a golden amulet and three healing potions.",
            'expected': 'consistent'
        }
    ]
    
    print("Testing examples:\n")
    for i, test in enumerate(test_cases, 1):
        result = critic.evaluate_with_explanation(test['response'], test['history'])
        
        print(f"Example {i} ({test['expected']}):")
        print(f"  Score: {result['score']}")
        print(f"  Predicted: {result['predicted_class']} (confidence: {result['confidence']*100:.1f}%)")
        print(f"  Explanation: {result['explanation']}")
        print()
