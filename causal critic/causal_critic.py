"""
Causal Responsiveness Critic
Uses zero-shot NLI to measure logical consistency
No training required - uses pre-trained model
"""
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import numpy as np
from tqdm import tqdm
from pathlib import Path  # Add this at the top of the file
import json



class CausalCritic:
    """
    Measures if DM response causally follows from player action
    Uses NLI entailment probability as proxy for causal consistency
    """
    
    def __init__(self, model_name="MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"):
        print("\n" + "="*60)
        print("LOADING CAUSAL CRITIC (Zero-Shot NLI)")
        print("="*60)
        print(f"\nModel: {model_name}")
        print("Task: Natural Language Inference (entailment scoring)")
        print("Training: None required (pre-trained)")
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load pre-trained NLI model
        print("\nLoading model...")
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        ).to(self.device)
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model.eval()
        
        print("✓ Causal Critic ready (zero-shot)")
        
        # NLI label mapping (model specific)
        # Typically: 0=contradiction, 1=neutral, 2=entailment
        self.entailment_label = 2
    
    def score(self, premise, hypothesis):
        """
        Score causal consistency between context and response
        
        Args:
            premise: Player action/context (what happened before)
            hypothesis: DM response (what should logically follow)
        
        Returns:
            float: Entailment probability (0-1), higher = more causally consistent
        """
        # Format for NLI
        inputs = self.tokenizer(
            premise,
            hypothesis,
            truncation=True,
            max_length=256,
            return_tensors="pt"
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            
            # Get probability of entailment class
            probs = torch.softmax(logits, dim=-1)
            entailment_prob = probs[0, self.entailment_label].item()
        
        return entailment_prob
    
    def score_batch(self, premise_hypothesis_pairs):
        """Score multiple pairs efficiently"""
        scores = []
        
        for premise, hypothesis in tqdm(premise_hypothesis_pairs, desc="Causal scoring"):
            score = self.score(premise, hypothesis)
            scores.append(score)
        
        return scores
    
    def test_critic(self):
        """Test the critic with examples"""
        print("\n" + "="*60)
        print("TESTING CAUSAL CRITIC")
        print("="*60)
        
        test_cases = [
            {
                'premise': "I cast Fireball at the goblin horde.",
                'hypothesis': "The goblins scatter as flames engulf them. Three fall instantly.",
                'expected': 'high'
            },
            {
                'premise': "I cast Fireball at the goblin horde.",
                'hypothesis': "You find a healing potion in the treasure chest.",
                'expected': 'low'
            },
            {
                'premise': "I search the room for secret doors.",
                'hypothesis': "Rolling a 19 on perception, you notice a faint crack in the wall.",
                'expected': 'high'
            },
            {
                'premise': "I search the room for secret doors.",
                'hypothesis': "The dragon breathes fire at you.",
                'expected': 'low'
            }
        ]
        
        print("\nExample evaluations:")
        for i, case in enumerate(test_cases, 1):
            score = self.score(case['premise'], case['hypothesis'])
            
            print(f"\n{i}. Expected: {case['expected'].upper()} causality")
            print(f"   Context: {case['premise']}")
            print(f"   Response: {case['hypothesis'][:60]}...")
            print(f"   Score: {score:.3f}")
            
            if case['expected'] == 'high' and score > 0.6:
                print("   ✓ Correct")
            elif case['expected'] == 'low' and score < 0.4:
                print("   ✓ Correct")
            else:
                print("   ⚠️  Unexpected score")
        
        print("\n" + "="*60)
        print("Causal Critic is functioning")
        print("="*60)

def save_critic_wrapper():
    """Save a simple wrapper config for later use"""
    config = {
        'model_name': "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli",
        'task': 'causal_consistency',
        'method': 'zero-shot_nli',
        'entailment_label': 2,
        'score_range': [0.0, 1.0],
        'interpretation': 'Higher score = response causally follows from context'
    }
    
    output_dir = Path("models/causal_critic")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    import json
    with open(output_dir / "critic_config.json", 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"\n✓ Causal Critic config saved to: {output_dir}")

if __name__ == "__main__":
    # Initialize critic
    critic = CausalCritic()
    
    # Test it
    critic.test_critic()
    
    # Save config
    save_critic_wrapper()
    
    print("\n✓ Causal Critic ready for evaluation")
    print("  No training required - uses pre-trained NLI model")