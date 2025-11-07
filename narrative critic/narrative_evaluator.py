import torch
from transformers import DebertaV2Tokenizer
import sys
import os

# Add parent directory to path to import narrative_critic
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from narrative_critic import NarrativeCritic

class NarrativeEvaluator:
    """Wrapper for narrative quality evaluation during RL training"""
    
    def __init__(self, model_path, device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = DebertaV2Tokenizer.from_pretrained(model_path)
        self.model = NarrativeCritic.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()
        
        print(f"Narrative Evaluator loaded from {model_path}")
    
    def evaluate_batch(self, texts):
        """Evaluate narrative quality for a batch of texts"""
        if not texts:
            return []
        
        with torch.no_grad():
            inputs = self.tokenizer(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=128
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            outputs = self.model(**inputs)
            scores = outputs.logits.cpu().numpy().flatten().tolist()
            
            return scores
    
    def evaluate_single(self, text):
        """Evaluate narrative quality for a single text"""
        return self.evaluate_batch([text])[0]
    
    def __call__(self, texts):
        """Make the evaluator callable for RL integration"""
        if isinstance(texts, str):
            return self.evaluate_single(texts)
        return self.evaluate_batch(texts)

# Example usage
if __name__ == "__main__":
    evaluator = NarrativeEvaluator("../models/narrative_critic")
    
    test_texts = [
        "The dark forest loomed ahead, its twisted branches casting eerie shadows in the moonlight.",
        "You see trees. There are shadows.",
        "The wizard chanted ancient words as blue energy crackled around his fingertips."
    ]
    
    scores = evaluator(test_texts)
    for text, score in zip(test_texts, scores):
        print(f"Score: {score:.3f} | Text: {text}")