"""
Unified reward engine for Multi-Critic Reinforcement Learning pipeline.
Provides efficient, batched interfaces for trained critic models during PPO training.

Design principles:
- Encapsulation: Self-contained critic classes with internal model management
- Efficiency: Models loaded once, evaluation mode, no_grad contexts
- Batching: All methods handle lists of strings for efficient processing
- Clear reward logic: Tailored extraction for each critic's training approach
"""
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from typing import List

class NarrativeCritic:
    """
    Trained narrative quality critic providing continuous reward scores.
    Uses regression model to predict quality scores with sigmoid activation.
    """
    def __init__(self, model_path: str, device: torch.device):
        """
        Initialize narrative critic with trained model.

        Args:
            model_path: Path to trained model directory
            device: Torch device for model inference
        """
        self.device = device
        print(f"Loading Narrative Critic from: {model_path}")
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path).to(self.device).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        print("✓ Narrative Critic loaded and in evaluation mode.")

    def get_reward(self, texts: List[str]) -> torch.Tensor:
        """
        Calculate narrative quality scores for batch of texts.

        Args:
            texts: List of generated DM responses

        Returns:
            Tensor of reward scores (0.0 to 1.0) for each text
        """
        # Tokenize with padding and truncation for batch processing
        inputs = self.tokenizer(
            texts, 
            return_tensors="pt", 
            truncation=True, 
            padding=True,
            max_length=self.tokenizer.model_max_length
        ).to(self.device)
        
        # Efficient inference without gradient calculation
        with torch.no_grad():
            logits = self.model(**inputs).logits
        
        # Apply sigmoid to convert logits to 0-1 reward scores
        scores = torch.sigmoid(logits).squeeze(-1)
        
        return scores

class CausalCritic:
    """
    Trained causal consistency critic using NLI classification.
    Evaluates logical consistency between player actions and DM responses.
    """
    def __init__(self, model_path: str, device: torch.device):
        """
        Initialize causal critic with trained NLI model.

        Args:
            model_path: Path to trained model directory
            device: Torch device for model inference
        """
        self.device = device
        print(f"Loading Causal Critic from: {model_path}")
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path).to(self.device).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        print("✓ Causal Critic loaded and in evaluation mode.")

    def get_reward(self, premises: List[str], hypotheses: List[str]) -> torch.Tensor:
        """
        Calculate causal consistency scores for premise-hypothesis pairs.

        Args:
            premises: List of player prompts/actions
            hypotheses: List of corresponding DM responses

        Returns:
            Tensor of entailment probabilities as reward scores (0.0 to 1.0)
        """
        # Tokenize premise-hypothesis pairs for NLI classification
        inputs = self.tokenizer(
            premises, 
            hypotheses, 
            return_tensors="pt", 
            truncation=True, 
            padding=True,
            max_length=256
        ).to(self.device)
        
        # Efficient inference without gradient calculation
        with torch.no_grad():
            logits = self.model(**inputs).logits
        
        # Apply softmax and extract entailment probabilities
        probs = torch.softmax(logits, dim=-1)
        
        # The reward is the probability of the "entailment" class.
        # Based on your data_prep.py, "entailment" has the label ID 2.
        scores = probs[:, 2]
        
        return scores

# --- Example Usage (for testing this file directly) ---
if __name__ == '__main__':
    # This is a placeholder for testing. Replace with your actual model paths.
    NARRATIVE_CRITIC_PATH = "models/narrative_critic_finetuned" # Use your actual narrative critic path
    CAUSAL_CRITIC_PATH = "models/causal_critic_finetuned"
    
    # Check for GPU
    if torch.cuda.is_available():
        dev = torch.device("cuda:0")
        print("\n--- Testing on GPU ---")
    else:
        dev = torch.device("cpu")
        print("\n--- Testing on CPU ---")

    # --- Test Narrative Critic ---
    print("\n--- Initializing Narrative Critic ---")
    narrative_critic = NarrativeCritic(NARRATIVE_CRITIC_PATH, device=dev)
    
    test_narratives = [
        "You see a room. There are things. You can do stuff.",
        "A profound silence settled over the ancient forest, broken only by the whisper of the wind through the gnarled branches of the elder trees."
    ]
    narrative_scores = narrative_critic.get_reward(test_narratives)
    print("\nNarrative Critic Test:")
    for text, score in zip(test_narratives, narrative_scores):
        print(f"  Score: {score.item():.4f} -> '{text[:50]}...'")

    # --- Test Causal Critic ---
    print("\n--- Initializing Causal Critic ---")
    causal_critic = CausalCritic(CAUSAL_CRITIC_PATH, device=dev)
    
    test_premises = [
        "I use the silver key on the oak door.",
        "I use the silver key on the oak door."
    ]
    test_hypotheses = [
        "The key turns with a satisfying click, and the heavy door swings open.", # Entailment
        "The tavern is filled with the smell of pipe smoke." # Contradiction
    ]
    causal_scores = causal_critic.get_reward(test_premises, test_hypotheses)
    print("\nCausal Critic Test:")
    for p, h, s in zip(test_premises, test_hypotheses, causal_scores):
        print(f"  Score: {s.item():.4f} -> Premise: '{p}' | Hypothesis: '{h[:30]}...'")