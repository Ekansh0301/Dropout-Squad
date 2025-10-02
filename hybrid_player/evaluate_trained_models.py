"""
Evaluation script for trained hybrid player models.
Tests both language model and intent classifier on holdout data.
"""
import os
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import classification_report, accuracy_score
import numpy as np

def evaluate_trained_models():
    """Evaluate trained models on test data."""
    print("Evaluating Trained Models on Test Data\n")
    
    # Load test data
    test_data_path = "../data/processed/hybrid_player_data.csv"
    if not os.path.exists(test_data_path):
        print(f"Test data not found at: {test_data_path}")
        print("Looking for alternative paths...")
        
        # Try different possible locations
        possible_paths = [
            "data/processed/hybrid_player_data.csv",
            "../data/processed/hybrid_player_data.csv", 
            "../../data/processed/hybrid_player_data.csv"
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                test_data_path = path
                print(f"Found data at: {path}")
                break
        else:
            print(" Could not find test data!")
            return
    
    test_data = pd.read_csv(test_data_path)
    print(f" Loaded test data: {len(test_data)} samples")
    
    # 1. Evaluate Intent Classifier
    print("\n" + "="*50)
    print("INTENT CLASSIFIER EVALUATION")
    print("="*50)
    
    classifier_path = "models/intent_classifier/final"
    if os.path.exists(classifier_path):
        try:
            classifier_tokenizer = AutoTokenizer.from_pretrained(classifier_path)
            classifier_model = AutoModelForSequenceClassification.from_pretrained(classifier_path)
            
            # Use a subset for quick evaluation (500 samples)
            eval_data = test_data.sample(min(500, len(test_data)), random_state=42)
            
            true_labels = []
            predicted_labels = []
            confidence_scores = []
            
            id2label = {0: "EXPLORE", 1: "ACTION", 2: "DIALOGUE"}
            label2id = {"EXPLORE": 0, "ACTION": 1, "DIALOGUE": 2}
            
            print("Evaluating on 500 samples...")
            for _, row in eval_data.iterrows():
                text = row['text']
                true_intent = row['intent']
                
                inputs = classifier_tokenizer(
                    text, 
                    return_tensors="pt", 
                    truncation=True, 
                    padding=True,
                    max_length=128
                )
                
                with torch.no_grad():
                    outputs = classifier_model(**inputs)
                    predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
                    predicted_class = torch.argmax(predictions, dim=1).item()
                    confidence = predictions[0][predicted_class].item()
                
                true_labels.append(label2id[true_intent])
                predicted_labels.append(predicted_class)
                confidence_scores.append(confidence)
            
            # Calculate metrics
            accuracy = accuracy_score(true_labels, predicted_labels)
            print(f" Accuracy: {accuracy:.3f}")
            print(f" Average Confidence: {np.mean(confidence_scores):.3f}")
            
            print("\n Detailed Classification Report:")
            print(classification_report(
                true_labels, 
                predicted_labels, 
                target_names=["EXPLORE", "ACTION", "DIALOGUE"]
            ))
            
            # Show some examples
            print("\n Example Predictions:")
            sample_indices = np.random.choice(len(eval_data), 5, replace=False)
            for idx in sample_indices:
                row = eval_data.iloc[idx]
                pred_idx = list(eval_data.index).index(row.name)
                print(f"  Text: '{row['text'][:50]}...'")
                print(f"  True: {row['intent']}, Pred: {id2label[predicted_labels[pred_idx]]}")
                print(f"  Confidence: {confidence_scores[pred_idx]:.3f}")
                print()
            
        except Exception as e:
            print(f" Error evaluating intent classifier: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(" Intent classifier not found!")
    
    # 2. Evaluate Language Model Perplexity
    print("\n" + "="*50)
    print("LANGUAGE MODEL QUALITY EVALUATION")
    print("="*50)
    
    lm_path = "models/language_model/final"
    if os.path.exists(lm_path):
        try:
            lm_tokenizer = AutoTokenizer.from_pretrained(lm_path)
            lm_model = AutoModelForCausalLM.from_pretrained(lm_path)
            
            # Calculate perplexity on test samples
            print("Calculating perplexity on 100 samples...")
            test_texts = test_data.sample(100)['text'].tolist()
            
            total_loss = 0
            total_tokens = 0
            
            for text in test_texts:
                inputs = lm_tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
                with torch.no_grad():
                    outputs = lm_model(**inputs, labels=inputs["input_ids"])
                    loss = outputs.loss
                    total_loss += loss.item() * inputs["input_ids"].size(1)
                    total_tokens += inputs["input_ids"].size(1)
            
            avg_loss = total_loss / total_tokens
            perplexity = torch.exp(torch.tensor(avg_loss)).item()
            
            print(f" Perplexity: {perplexity:.2f}")
            print(f" Average Loss: {avg_loss:.4f}")
            
            # Test generation diversity
            print("\n Generation Diversity Test:")
            test_scenarios = [
                "You see a dragon. What do you do?",
                "The door is locked.",
                "An old man approaches you."
            ]
            
            for scenario in test_scenarios:
                print(f"\nScenario: {scenario}")
                
                # Generate multiple responses with different temperatures
                for temp in [0.5, 0.7, 0.9]:
                    print(f"  Temperature {temp}:")
                    for i in range(2):
                        inputs = lm_tokenizer.encode(scenario, return_tensors="pt")
                        outputs = lm_model.generate(
                            inputs,
                            max_length=len(inputs[0]) + 20,
                            num_return_sequences=1,
                            temperature=temp,
                            do_sample=True,
                            pad_token_id=lm_tokenizer.eos_token_id
                        )
                        response = lm_tokenizer.decode(outputs[0], skip_special_tokens=True)
                        new_text = response[len(scenario):].strip()
                        print(f"    {i+1}. {new_text}")
                    
        except Exception as e:
            print(f" Error evaluating language model: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(" Language model not found!")

if __name__ == "__main__":
    evaluate_trained_models()