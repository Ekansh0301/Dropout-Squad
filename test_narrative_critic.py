#!/usr/bin/env python
"""
Inference script for the trained Narrative Critic model
Use this to test your trained model on custom text
"""

import torch
import argparse
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np

class NarrativeCriticInference:
    def __init__(self, model_path: str, device: str = None):
        """Initialize the narrative critic model for inference"""
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Loading model from {model_path}")
        print(f"Using device: {self.device}")
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()
        
        print("Model loaded successfully!")
    
    def score_narrative(self, text: str) -> dict:
        """Score a narrative text and return detailed results"""
        # Tokenize input
        inputs = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=512,
            return_tensors='pt'
        ).to(self.device)
        
        # Get model predictions
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)
        
        # Extract scores
        poor_score = float(probs[0, 0])
        good_score = float(probs[0, 1])
        predicted_label = "Good Narrative" if good_score > poor_score else "Poor Narrative"
        confidence = max(poor_score, good_score)
        
        return {
            'text': text,
            'predicted_label': predicted_label,
            'good_narrative_score': good_score,
            'poor_narrative_score': poor_score,
            'confidence': confidence,
            'quality_score': good_score  # 0-1 scale
        }
    
    def evaluate_story_continuation(self, context: str, ending1: str, ending2: str) -> dict:
        """Evaluate which ending is better for a story context"""
        story1 = f"{context} {ending1}"
        story2 = f"{context} {ending2}"
        
        score1 = self.score_narrative(story1)
        score2 = self.score_narrative(story2)
        
        better_ending = 1 if score1['quality_score'] > score2['quality_score'] else 2
        
        return {
            'context': context,
            'ending1': ending1,
            'ending2': ending2,
            'ending1_score': score1['quality_score'],
            'ending2_score': score2['quality_score'],
            'better_ending': better_ending,
            'score_difference': abs(score1['quality_score'] - score2['quality_score'])
        }
    
    def batch_score(self, texts: list) -> list:
        """Score multiple texts at once"""
        results = []
        for text in texts:
            results.append(self.score_narrative(text))
        return results

def main():
    parser = argparse.ArgumentParser(description="Test Narrative Critic Model")
    parser.add_argument("--model_path", type=str, default="./narrative_critic_model",
                       help="Path to the trained model")
    parser.add_argument("--text", type=str, 
                       help="Single text to evaluate")
    parser.add_argument("--context", type=str, 
                       help="Story context for ending comparison")
    parser.add_argument("--ending1", type=str, 
                       help="First ending option")
    parser.add_argument("--ending2", type=str, 
                       help="Second ending option")
    parser.add_argument("--interactive", action="store_true",
                       help="Start interactive mode")
    
    args = parser.parse_args()
    
    # Initialize model
    try:
        critic = NarrativeCriticInference(args.model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Make sure you have trained the model first!")
        return
    
    if args.interactive:
        # Interactive mode
        print("\n" + "="*80)
        print("NARRATIVE CRITIC - INTERACTIVE MODE")
        print("="*80)
        print("Commands:")
        print("  1. Type a story to evaluate its quality")
        print("  2. Type 'compare' to compare two story endings")
        print("  3. Type 'quit' to exit")
        print("-"*80)
        
        while True:
            try:
                user_input = input("\nEnter text or command: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    break
                elif user_input.lower() == 'compare':
                    context = input("Enter story context: ").strip()
                    ending1 = input("Enter first ending: ").strip()
                    ending2 = input("Enter second ending: ").strip()
                    
                    result = critic.evaluate_story_continuation(context, ending1, ending2)
                    
                    print(f"\nComparison Results:")
                    print(f"Ending 1 Score: {result['ending1_score']:.3f}")
                    print(f"Ending 2 Score: {result['ending2_score']:.3f}")
                    print(f"Better Ending: {result['better_ending']}")
                    print(f"Score Difference: {result['score_difference']:.3f}")
                    
                elif user_input:
                    result = critic.score_narrative(user_input)
                    
                    print(f"\nNarrative Quality Assessment:")
                    print(f"Prediction: {result['predicted_label']}")
                    print(f"Quality Score: {result['quality_score']:.3f}")
                    print(f"Confidence: {result['confidence']:.3f}")
                    
            except KeyboardInterrupt:
                break
    
    elif args.text:
        # Single text evaluation
        result = critic.score_narrative(args.text)
        print(f"\nNarrative Quality Assessment:")
        print(f"Text: {result['text']}")
        print(f"Prediction: {result['predicted_label']}")
        print(f"Quality Score: {result['quality_score']:.3f}")
        print(f"Confidence: {result['confidence']:.3f}")
    
    elif args.context and args.ending1 and args.ending2:
        # Story continuation comparison
        result = critic.evaluate_story_continuation(args.context, args.ending1, args.ending2)
        
        print(f"\nStory Continuation Comparison:")
        print(f"Context: {result['context']}")
        print(f"Ending 1: {result['ending1']}")
        print(f"Ending 1 Score: {result['ending1_score']:.3f}")
        print(f"Ending 2: {result['ending2']}")
        print(f"Ending 2 Score: {result['ending2_score']:.3f}")
        print(f"Better Ending: {result['better_ending']}")
    
    else:
        # Demo examples
        print("\nRunning demo examples...")
        
        # Good story example
        good_story = "Sarah had always dreamed of becoming a doctor. She studied hard through medical school and finally graduated. When she got her first job at the hospital, she was nervous but excited. On her first day, she helped save a patient's life. Sarah knew she had found her calling."
        
        # Poor story example (incoherent)
        poor_story = "The cat was purple and flew to Mars. Then it rained chocolate milk on Tuesday. The elephant decided to become a piano. Music played backwards while the sun was cold. Everyone laughed at the serious joke."
        
        print("\nGood Story Example:")
        result1 = critic.score_narrative(good_story)
        print(f"Score: {result1['quality_score']:.3f} - {result1['predicted_label']}")
        
        print("\nPoor Story Example:")
        result2 = critic.score_narrative(poor_story)
        print(f"Score: {result2['quality_score']:.3f} - {result2['predicted_label']}")
        
        # Story continuation example
        print("\nStory Continuation Example:")
        context = "John had been preparing for the big race for months. He trained every day and ate healthy foods. On the day of the race, he felt ready."
        ending1 = "John ran his best time ever and won first place."
        ending2 = "John suddenly decided to quit running and became a chef instead."
        
        result3 = critic.evaluate_story_continuation(context, ending1, ending2)
        print(f"Better ending: {result3['better_ending']} (score difference: {result3['score_difference']:.3f})")

if __name__ == "__main__":
    main()