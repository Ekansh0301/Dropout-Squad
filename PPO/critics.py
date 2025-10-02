"""
critics.py - The Reward Engine for the MCRL Pipeline

This module loads the final, trained critic models and provides a simple,
unified interface for the PPO trainer to get reward signals.

Key Design Principles:
- Encapsulation: Each critic is a self-contained class, managing its own
  model, tokenizer, and device.
- Efficiency: Models are loaded once, set to evaluation mode (.eval()), and
  all reward calculations are done within a torch.no_grad() block to
  maximize performance.
- Batching: All methods are designed to work on batches (lists of strings)
  for efficient processing during the PPO loop.
- Clear Reward Logic: The reward extraction method is tailored to how each
  critic was trained (Regression for Narrative, Classification for Causal).
"""
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from typing import List

class NarrativeCritic:
    """
    Loads the trained Narrative Critic and provides a reward method.
    This critic was trained as a REGRESSION model to predict a continuous
    quality score. The reward is the sigmoid of the model's single logit output.
    """
    def __init__(self, model_path: str, device: torch.device):
        """
        Initializes the Narrative Critic.

        Args:
            model_path (str): Path to the trained model directory.
            device (torch.device): The device to run the model on (e.g., 'cuda:0').
        """
        self.device = device
        print(f"Loading Narrative Critic from: {model_path}")
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path).to(self.device).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        print("✓ Narrative Critic loaded and in evaluation mode.")

    def get_reward(self, texts: List[str]) -> torch.Tensor:
        """
        Calculates the narrative quality score for a batch of texts.

        Args:
            texts (List[str]): A list of generated DM responses.

        Returns:
            torch.Tensor: A 1D tensor of reward scores (0.0 to 1.0), one for each text.
        """
        # Move to device and tokenize with padding and truncation
        inputs = self.tokenizer(
            texts, 
            return_tensors="pt", 
            truncation=True, 
            padding=True,
            max_length=self.tokenizer.model_max_length
        ).to(self.device)
        
        # Perform inference without calculating gradients for efficiency
        with torch.no_grad():
            logits = self.model(**inputs).logits
        
        # The model outputs a single logit. Apply sigmoid to scale it to a 0-1 reward.
        # .squeeze(-1) changes the shape from [batch_size, 1] to [batch_size]
        scores = torch.sigmoid(logits).squeeze(-1)
        
        return scores

class CausalCritic:
    """
    Loads the trained Causal Critic and provides a reward method.
    This critic was trained as a CLASSIFICATION model to predict the NLI
    relationship (contradiction, neutral, entailment). The reward is the
    softmax probability of the "entailment" class.
    """
    def __init__(self, model_path: str, device: torch.device):
        """
        Initializes the Causal Critic.

        Args:
            model_path (str): Path to the trained model directory.
            device (torch.device): The device to run the model on (e.g., 'cuda:0').
        """
        self.device = device
        print(f"Loading Causal Critic from: {model_path}")
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path).to(self.device).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        print("✓ Causal Critic loaded and in evaluation mode.")

    def get_reward(self, premises: List[str], hypotheses: List[str]) -> torch.Tensor:
        """
        Calculates the causal consistency score for premise-hypothesis pairs.

        Args:
            premises (List[str]): A list of player prompts.
            hypotheses (List[str]): A list of the DM's responses.

        Returns:
            torch.Tensor: A 1D tensor of reward scores (0.0 to 1.0), one for each pair.
        """
        # Tokenize the pairs
        inputs = self.tokenizer(
            premises, 
            hypotheses, 
            return_tensors="pt", 
            truncation=True, 
            padding=True,
            max_length=256 # As defined in your training script
        ).to(self.device)
        
        # Perform inference without calculating gradients
        with torch.no_grad():
            logits = self.model(**inputs).logits
        
        # The model outputs 3 logits. Apply softmax to get probabilities.
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