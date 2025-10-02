"""
Causal Responsiveness Critic for evaluating DM response consistency.
Uses pre-trained NLI model to score causal relationships between context and response.
"""
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import numpy as np
from tqdm import tqdm
from pathlib import Path
import json



class CausalCritic:
    """
    Evaluates causal consistency between player actions and DM responses.
    Uses NLI entailment probability to measure logical coherence.
    """
    
    def __init__(self, model_name="MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"):
        print("\n" + "="*60)
        print("LOADING CAUSAL CRITIC (Zero-Shot NLI)")
        print("="*60)
        print(f"\nModel: {model_name}")
        print("Task: Natural Language Inference (entailment scoring)")
        print("Training: None required (pre-trained)")
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load pre-trained NLI model for zero-shot inference
        print("\nLoading model...")
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        ).to(self.device)
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model.eval()
        
        print("✓ Causal Critic ready (zero-shot)")
        
        # NLI label mapping: 0=contradiction, 1=neutral, 2=entailment
        self.entailment_label = 2
    
    def score(self, premise, hypothesis):
        """
        Evaluate causal consistency between context and response.
        
        Args:
            premise: Player action/context
            hypothesis: DM response
        
        Returns:
            float: Entailment probability (0-1), higher indicates better consistency
        """
        # Format inputs for NLI model
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
            
            # Extract entailment probability
            probs = torch.softmax(logits, dim=-1)
            entailment_prob = probs[0, self.entailment_label].item()
        
        return entailment_prob
    
    def score_batch(self, premise_hypothesis_pairs):
        """Efficiently score multiple premise-hypothesis pairs."""
        scores = []
        
        for premise, hypothesis in tqdm(premise_hypothesis_pairs, desc="Causal scoring"):
            score = self.score(premise, hypothesis)
            scores.append(score)
        
        return scores
    
    def test_critic(self):
        """Test critic functionality with sample D&D scenarios."""
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
    """Save critic configuration metadata for deployment."""
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
    # Initialize and test critic
    critic = CausalCritic()
    critic.test_critic()
    save_critic_wrapper()
    
    print("\n✓ Causal Critic ready for evaluation")
    print("  No training required - uses pre-trained NLI model")