# test_trained_models.py
import os
import sys
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSequenceClassification
import torch

def load_and_test_models():
    print(" Loading and Testing Trained Models\n")
    
    # 1. Test Language Model
    print("1. Testing Language Model...")
    lm_path = "models/language_model/final"
    
    if os.path.exists(lm_path):
        try:
            lm_tokenizer = AutoTokenizer.from_pretrained(lm_path)
            lm_model = AutoModelForCausalLM.from_pretrained(lm_path)
            print(" Language model loaded successfully!")
            
            # Test generation
            test_prompts = [
                "I look around and",
                "I attack the",
                "I say to the NPC:"
            ]
            
            for prompt in test_prompts:
                inputs = lm_tokenizer.encode(prompt, return_tensors="pt")
                outputs = lm_model.generate(
                    inputs, 
                    max_length=20, 
                    num_return_sequences=1,
                    pad_token_id=lm_tokenizer.eos_token_id
                )
                generated = lm_tokenizer.decode(outputs[0], skip_special_tokens=True)
                print(f"   Prompt: '{prompt}'")
                print(f"   Generated: '{generated}'")
                print()
                
        except Exception as e:
            print(f" Error loading language model: {e}")
    else:
        print(" Language model path not found!")
    
    # 2. Test Intent Classifier
    print("\n2. Testing Intent Classifier...")
    classifier_path = "models/intent_classifier/final"
    
    if os.path.exists(classifier_path):
        try:
            classifier_tokenizer = AutoTokenizer.from_pretrained(classifier_path)
            classifier_model = AutoModelForSequenceClassification.from_pretrained(classifier_path)
            print(" Intent classifier loaded successfully!")
            
            # Test classification
            test_utterances = [
                "I look around the room carefully",
                "I attack the monster with my sword",
                "Hello, my name is John"
            ]
            
            id2label = {0: "EXPLORE", 1: "ACTION", 2: "DIALOGUE"}
            
            for utterance in test_utterances:
                inputs = classifier_tokenizer(
                    utterance, 
                    return_tensors="pt", 
                    truncation=True, 
                    padding=True,
                    max_length=128
                )
                
                with torch.no_grad():
                    outputs = classifier_model(**inputs)
                    predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
                    predicted_class = torch.argmax(predictions, dim=1).item()
                    
                print(f"   Utterance: '{utterance}'")
                print(f"   Predicted: {id2label[predicted_class]} (conf: {predictions[0][predicted_class]:.3f})")
                print()
                
        except Exception as e:
            print(f" Error loading intent classifier: {e}")
    else:
        print(" Intent classifier path not found!")

if __name__ == "__main__":
    load_and_test_models()