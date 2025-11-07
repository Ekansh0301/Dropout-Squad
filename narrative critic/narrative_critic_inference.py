"""
Production-ready Narrative Critic with Calibration
Use this script for inference in your D&D system.
"""

import torch
import pickle
import numpy as np
from pathlib import Path
from typing import Union, List
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class NarrativeCritic:
    """
    Narrative quality assessment with automatic calibration.
    
    Usage:
        critic = NarrativeCritic("models/narrative_critic")
        score = critic.score("You enter a dimly lit tavern...")
        scores = critic.score_batch(["Text 1", "Text 2", "Text 3"])
    """
    
    def __init__(self, model_path: str, use_calibration: bool = True):
        """
        Initialize the narrative critic.
        
        Args:
            model_path: Path to saved model directory
            use_calibration: Whether to apply calibration (default: True)
        """
        self.model_path = Path(model_path)
        self.use_calibration = use_calibration
        
        # Load model and tokenizer
        print(f"Loading model from {model_path}...")
        self.model = AutoModelForSequenceClassification.from_pretrained(str(self.model_path))
        self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_path))
        self.model.eval()
        
        # Setup device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # Load calibrator if available
        self.calibrator = None
        if use_calibration:
            calibrator_path = self.model_path / "calibrator.pkl"
            if calibrator_path.exists():
                with open(calibrator_path, 'rb') as f:
                    self.calibrator = pickle.load(f)
                print(f"✓ Calibrator loaded")
                print(f"  Formula: y = {self.calibrator.coef_[0]:.4f} * pred + {self.calibrator.intercept_:.4f}")
            else:
                print(f"⚠️  No calibrator found at {calibrator_path}")
                print("   Using raw predictions (no calibration)")
        
        print(f"✓ Narrative Critic ready")
        print(f"  Device: {self.device}")
        print(f"  Calibration: {'Enabled' if self.calibrator else 'Disabled'}")
    
    def score(self, text: str) -> float:
        """
        Get quality score for a single text.
        
        Args:
            text: Narrative text to evaluate
            
        Returns:
            Quality score between 0.0 and 1.0
        """
        return self.score_batch([text])[0]
    
    def score_batch(self, texts: List[str]) -> List[float]:
        """
        Get quality scores for multiple texts.
        
        Args:
            texts: List of narrative texts to evaluate
            
        Returns:
            List of quality scores between 0.0 and 1.0
        """
        # Tokenize
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Get predictions
        with torch.no_grad():
            outputs = self.model(**inputs)
            raw_scores = torch.sigmoid(outputs.logits).squeeze()
        
        # Move to CPU and convert to numpy
        raw_scores = raw_scores.cpu().numpy()
        
        # Handle single example
        if raw_scores.ndim == 0:
            raw_scores = raw_scores.reshape(1)
        
        # Apply calibration if available
        if self.calibrator is not None:
            scores = self.calibrator.predict(raw_scores.reshape(-1, 1)).squeeze()
            scores = np.clip(scores, 0.0, 1.0)
        else:
            scores = raw_scores
        
        return scores.tolist()
    
    def __call__(self, text: Union[str, List[str]]) -> Union[float, List[float]]:
        """Make the class callable."""
        if isinstance(text, str):
            return self.score(text)
        else:
            return self.score_batch(text)
    
    def interpret_score(self, score: float) -> str:
        """
        Get human-readable interpretation of quality score.
        
        Args:
            score: Quality score between 0.0 and 1.0
            
        Returns:
            Quality level description
        """
        if score >= 0.8:
            return "Excellent"
        elif score >= 0.6:
            return "Good"
        elif score >= 0.4:
            return "Fair"
        elif score >= 0.2:
            return "Poor"
        else:
            return "Very Poor"


def main():
    """Example usage and testing."""
    import sys
    
    # Check if model path provided
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
    else:
        model_path = "models/narrative_critic"
    
    # Initialize critic
    print("="*70)
    print("NARRATIVE CRITIC - QUALITY ASSESSMENT")
    print("="*70)
    
    critic = NarrativeCritic(model_path)
    
    # Test examples
    print("\n" + "="*70)
    print("TESTING ON EXAMPLE NARRATIVES")
    print("="*70)
    
    test_examples = [
        {
            'text': "The ancient library stretched endlessly before you, its towering shelves groaning under countless leather-bound tomes. Dust motes danced in golden sunlight filtering through stained glass windows, casting rainbow patterns across worn stone floors.",
            'category': 'High Quality D&D Description'
        },
        {
            'text': "You see a room. There is a door. There is a table. There is a chair.",
            'category': 'Low Quality Description'
        },
        {
            'text': "The dragon roars and breathes fire. The dragon roars and breathes fire. The dragon roars and breathes fire again.",
            'category': 'Repetitive Text'
        },
        {
            'text': "Your blade finds its mark with a satisfying thud. The orc's eyes widen in surprise before it crumples to",
            'category': 'Truncated Narrative'
        },
        {
            'text': "The tavern bustles with activity. A bard plays a lute in the corner while patrons laugh and drink. The fire crackles warmly, and the smell of roasted meat fills the air.",
            'category': 'Good D&D Scene'
        }
    ]
    
    for i, example in enumerate(test_examples, 1):
        score = critic.score(example['text'])
        quality = critic.interpret_score(score)
        
        print(f"\nExample {i}: {example['category']}")
        print("-" * 70)
        print(f"Text: {example['text'][:100]}...")
        print(f"Quality Score: {score:.3f} ({quality})")
    
    # Batch processing example
    print("\n" + "="*70)
    print("BATCH PROCESSING EXAMPLE")
    print("="*70)
    
    texts = [ex['text'] for ex in test_examples]
    scores = critic.score_batch(texts)
    
    print(f"\nProcessed {len(texts)} texts in batch:")
    for i, (score, example) in enumerate(zip(scores, test_examples), 1):
        print(f"  {i}. {example['category']}: {score:.3f}")
    
    # Callable interface example
    print("\n" + "="*70)
    print("CALLABLE INTERFACE EXAMPLE")
    print("="*70)
    
    score = critic("The dungeon entrance looms before you, dark and foreboding.")
    print(f"\nDirect call result: {score:.3f}")
    
    print("\n" + "="*70)
    print("✓ TESTING COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()
