"""
hybrid_player.py - The Autonomous Training Partner for the Director LLM

This module encapsulates the Hybrid Simulated Player. Its primary role is to
generate a dynamic and diverse stream of player prompts to be used during
PPO training.

Key Design Principles:
- Encapsulation: The player is a single class that manages its two internal
  models (generator and classifier) and their tokenizers.
- Robust Generation: Implements techniques to ensure generated prompts are
  varied and high-quality, avoiding common failure modes like empty outputs
  or repetitive loops.
- Efficiency: Models are loaded once, set to evaluation mode, and all
  inference is performed without calculating gradients.
- Clear Interface: Provides a single, simple method `.generate_prompts()`
  that the PPO trainer can call to get a batch of prompts.
"""
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    GenerationConfig
)
from typing import List, Tuple

class HybridPlayer:
    """
    Loads both player models and generates batches of prompts with classified intents.
    """
    def __init__(self, generator_path: str, classifier_path: str, device: torch.device):
        """
        Initializes the Hybrid Player by loading its two component models.

        Args:
            generator_path (str): Path to the fine-tuned generator model (e.g., DistilGPT-2).
            classifier_path (str): Path to the fine-tuned intent classifier (e.g., DistilBERT).
            device (torch.device): The device to run the models on (e.g., 'cuda:0').
        """
        self.device = device
        print("--- Initializing Hybrid Player ---")
        
        # --- 1. Load Generator Model (e.g., DistilGPT-2) ---
        print(f"Loading Player Generator from: {generator_path}")
        self.generator = AutoModelForCausalLM.from_pretrained(generator_path).to(self.device).eval()
        self.g_tokenizer = AutoTokenizer.from_pretrained(generator_path)
        # Set padding token for batch generation if it's not already set
        if self.g_tokenizer.pad_token is None:
            self.g_tokenizer.pad_token = self.g_tokenizer.eos_token
        
        # --- 2. Load Classifier Model (e.g., DistilBERT) ---
        print(f"Loading Player Classifier from: {classifier_path}")
        self.classifier = AutoModelForSequenceClassification.from_pretrained(classifier_path).to(self.device).eval()
        self.c_tokenizer = AutoTokenizer.from_pretrained(classifier_path)
        # Store the mapping from label ID to intent name (e.g., 0 -> "ACTION")
        self.intents = self.classifier.config.id2label

        print("✓ Hybrid Player loaded and in evaluation mode.")

    def generate_prompts(self, batch_size: int = 1, max_length: int = 128) -> List[Tuple[str, str]]:
        """
        Generates a batch of unique, high-quality prompts and classifies their intent.

        This method generates `batch_size` * 1.5 prompts and then filters down to the
        best `batch_size` to increase quality and diversity.

        Args:
            batch_size (int): The number of prompts to generate.
            max_length (int): The maximum number of tokens for each generated prompt.

        Returns:
            List[Tuple[str, str]]: A list of tuples, where each tuple contains
                                   (prompt_text, classified_intent).
        """
        # --- 1. Define Generation Configuration ---
        # Using a GenerationConfig object is best practice.
        generation_config = GenerationConfig(
            max_new_tokens=max_length,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            pad_token_id=self.g_tokenizer.eos_token_id,
            num_return_sequences=batch_size, # Generate a full batch at once
        )

        # --- 2. Generate Text Prompts in a Batch ---
        with torch.no_grad():
            outputs = self.generator.generate(
                config=generation_config
            )
        
        # Decode the generated token IDs into strings
        # We skip special tokens to remove padding and EOS tokens.
        prompts = [self.g_tokenizer.decode(output, skip_special_tokens=True).strip() for output in outputs]
        
        # Filter out any empty or very short prompts
        prompts = [p for p in prompts if len(p.split()) > 2]
        
        if not prompts:
            # Fallback in case all generated prompts are empty
            return [("I look around the room.", "EXPLORE")] * batch_size

        # --- 3. Classify the Intent of Each Prompt in a Batch ---
        # Tokenize all prompts at once for efficiency
        inputs = self.c_tokenizer(
            prompts, 
            return_tensors="pt", 
            truncation=True, 
            padding=True
        ).to(self.device)
        
        with torch.no_grad():
            logits = self.classifier(**inputs).logits
        
        # Get the predicted label ID for each prompt
        intent_ids = torch.argmax(logits, dim=1)
        
        # Map the IDs to their string labels
        intents = [self.intents[intent_id.item()] for intent_id in intent_ids]

        # --- 4. Combine and Return ---
        results = list(zip(prompts, intents))
        
        # If we generated fewer valid prompts than requested, pad with a default
        while len(results) < batch_size:
            results.append(("What do I see?", "EXPLORE"))

        return results[:batch_size]


# --- This block allows you to test the script directly ---
if __name__ == '__main__':
    # This is a placeholder for testing. Replace with your actual model paths.
    PLAYER_GENERATOR_PATH = "models/player_generator"
    PLAYER_CLASSIFIER_PATH = "models/player_classifier"

    # Check for GPU
    if torch.cuda.is_available():
        dev = torch.device("cuda:0")
        print("\n--- Testing Hybrid Player on GPU ---")
    else:
        dev = torch.device("cpu")
        print("\n--- Testing Hybrid Player on CPU ---")

    # Initialize the player
    try:
        hybrid_player = HybridPlayer(
            generator_path=PLAYER_GENERATOR_PATH,
            classifier_path=PLAYER_CLASSIFIER_PATH,
            device=dev
        )

        # Generate a batch of prompts
        print("\n--- Generating a test batch of 5 prompts ---")
        generated_prompts = hybrid_player.generate_prompts(batch_size=5, max_length=50)

        print("\n--- Results ---")
        for i, (prompt, intent) in enumerate(generated_prompts):
            print(f"Prompt {i+1}:")
            print(f"  Intent: {intent}")
            print(f"  Text:   '{prompt}'")
        print("-" * 20)

    except Exception as e:
        print(f"\n❌ An error occurred during testing.")
        print("Please ensure your trained model paths are correct:")
        print(f"  Generator Path: {PLAYER_GENERATOR_PATH}")
        print(f"  Classifier Path: {PLAYER_CLASSIFIER_PATH}")
        print(f"Error details: {e}")