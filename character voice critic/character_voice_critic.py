"""
Character Voice Critic Implementation
Uses DeBERTa-v3-base to evaluate NPC dialogue consistency with character voice.
"""

import json
import os
import random
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from collections import defaultdict

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import (
    AutoTokenizer,
    AutoModel,
    get_linear_schedule_with_warmup
)
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


@dataclass
class CharacterProfile:
    """Stores character information and dialogue history."""
    name: str
    dialogue_history: List[Dict[str, str]] = field(default_factory=list)
    traits: List[str] = field(default_factory=list)
    speech_patterns: List[str] = field(default_factory=list)
    
    def add_dialogue(self, text: str, context: str):
        self.dialogue_history.append({"text": text, "context": context})
    
    def add_trait(self, trait: str):
        if trait not in self.traits:
            self.traits.append(trait)
    
    def add_speech_pattern(self, pattern: str):
        if pattern not in self.speech_patterns:
            self.speech_patterns.append(pattern)
    
    def get_dialogue_count(self) -> int:
        return len(self.dialogue_history)


class CharacterVoiceDataset(Dataset):
    """Dataset for character voice matching."""
    
    def __init__(self, examples: List[Dict], tokenizer, max_length: int = 256):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        example = self.examples[idx]
        
        # Format: [Character: {name}] [Context: {context}] [Dialogue: {text}]
        text = (
            f"[Character: {example['character']}] "
            f"[Context: {example['context']}] "
            f"[Dialogue: {example['text']}]"
        )
        
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'character_id': example['character_id'],
            'label': torch.tensor(example['label'], dtype=torch.float)
        }


class CharacterVoiceModel(nn.Module):
    """DeBERTa-based model with character embeddings."""
    
    def __init__(self, model_name: str, num_characters: int, embedding_dim: int = 128):
        super().__init__()
        self.deberta = AutoModel.from_pretrained(model_name)
        self.hidden_size = self.deberta.config.hidden_size
        
        # Character embedding layer
        self.character_embeddings = nn.Embedding(num_characters, embedding_dim)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_size + embedding_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    
    def forward(self, input_ids, attention_mask, character_ids):
        # Get DeBERTa contextual embeddings
        outputs = self.deberta(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0, :]  # [CLS] token
        
        # Get character embeddings
        char_embeds = self.character_embeddings(character_ids)
        
        # Concatenate and classify
        combined = torch.cat([pooled_output, char_embeds], dim=1)
        logits = self.classifier(combined)
        
        return logits.squeeze()


class CharacterVoiceCritic:
    """Main critic class for character voice evaluation."""
    
    def __init__(
        self,
        model_name: str = "microsoft/deberta-v3-base",
        num_characters: int = 100,
        embedding_dim: int = 128,
        device: str = None
    ):
        self.model_name = model_name
        self.num_characters = num_characters
        self.embedding_dim = embedding_dim
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = None
        self.characters: Dict[str, CharacterProfile] = {}
        self.character_to_id: Dict[str, int] = {}
        self.id_to_character: Dict[int, str] = {}
        
    def build_training_data_from_crd3(
        self,
        crd3_dialogue_file: str,
        output_file: str = None,
        min_dialogues: int = 10,
        max_characters: int = None
    ) -> List[Dict]:
        """
        Build training data from CRD3 NPC dialogues.
        Creates positive and negative pairs for character voice matching.
        """
        print("Loading CRD3 NPC dialogues...")
        with open(crd3_dialogue_file, 'r', encoding='utf-8') as f:
            dialogues = json.load(f)
        
        # Group dialogues by character
        character_dialogues = defaultdict(list)
        for item in dialogues:
            character = item['character']
            character_dialogues[character].append({
                'text': item['text'],
                'context': item['context'],
                'episode': item.get('episode', ''),
                'turn': item.get('turn_number', 0)
            })
        
        # Filter characters with sufficient dialogue
        print(f"Filtering characters with at least {min_dialogues} dialogues...")
        valid_characters = {
            char: dlgs for char, dlgs in character_dialogues.items()
            if len(dlgs) >= min_dialogues
        }
        
        print(f"Found {len(valid_characters)} valid characters")
        
        # Limit number of characters if specified
        if max_characters and len(valid_characters) > max_characters:
            # Sort by dialogue count and take top characters
            sorted_chars = sorted(
                valid_characters.items(),
                key=lambda x: len(x[1]),
                reverse=True
            )[:max_characters]
            valid_characters = dict(sorted_chars)
            print(f"Limited to top {max_characters} characters by dialogue count")
        
        # Create character profiles and mappings
        for idx, (char_name, dlgs) in enumerate(sorted(valid_characters.items())):
            profile = CharacterProfile(name=char_name)
            for dlg in dlgs:
                profile.add_dialogue(dlg['text'], dlg['context'])
            self.characters[char_name] = profile
            self.character_to_id[char_name] = idx
            self.id_to_character[idx] = char_name
        
        print(f"Created {len(self.characters)} character profiles")
        
        # Update num_characters
        self.num_characters = len(self.characters)
        
        # Build training examples
        print("Building training examples...")
        training_examples = []
        character_names = list(valid_characters.keys())
        
        for char_name in tqdm(character_names, desc="Creating pairs"):
            char_dialogues = valid_characters[char_name]
            char_id = self.character_to_id[char_name]
            
            # Create positive examples (character matches their own dialogue)
            for dlg in char_dialogues:
                training_examples.append({
                    'character': char_name,
                    'character_id': char_id,
                    'text': dlg['text'],
                    'context': dlg['context'],
                    'label': 1.0  # Match
                })
            
            # Create negative examples (character with other characters' dialogue)
            num_negatives = len(char_dialogues)  # Balance positive/negative
            for _ in range(num_negatives):
                # Pick random different character
                other_char = random.choice([c for c in character_names if c != char_name])
                other_dlg = random.choice(valid_characters[other_char])
                
                training_examples.append({
                    'character': char_name,  # Current character
                    'character_id': char_id,
                    'text': other_dlg['text'],  # Other character's dialogue
                    'context': other_dlg['context'],
                    'label': 0.0  # Mismatch
                })
        
        # Shuffle
        random.shuffle(training_examples)
        
        print(f"Created {len(training_examples)} training examples")
        print(f"  Positive: {sum(1 for ex in training_examples if ex['label'] == 1.0)}")
        print(f"  Negative: {sum(1 for ex in training_examples if ex['label'] == 0.0)}")
        
        # Save if output file specified
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(training_examples, f, indent=2)
            print(f"Saved training data to {output_file}")
        
        return training_examples
    
    def train(
        self,
        training_data: List[Dict],
        output_dir: str,
        validation_split: float = 0.1,
        num_epochs: int = 3,
        batch_size: int = 16,
        learning_rate: float = 2e-5,
        warmup_steps: int = 500,
        max_length: int = 256,
        save_steps: int = 500
    ):
        """Train the character voice critic model."""
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Split data
        random.shuffle(training_data)
        split_idx = int(len(training_data) * (1 - validation_split))
        train_data = training_data[:split_idx]
        val_data = training_data[split_idx:]
        
        print(f"\nTraining samples: {len(train_data)}")
        print(f"Validation samples: {len(val_data)}")
        
        # Create datasets
        train_dataset = CharacterVoiceDataset(train_data, self.tokenizer, max_length)
        val_dataset = CharacterVoiceDataset(val_data, self.tokenizer, max_length)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        
        # Initialize model
        print(f"\nInitializing model with {self.num_characters} characters...")
        self.model = CharacterVoiceModel(
            self.model_name,
            self.num_characters,
            self.embedding_dim
        ).to(self.device)
        
        # Optimizer and scheduler
        optimizer = AdamW(self.model.parameters(), lr=learning_rate)
        total_steps = len(train_loader) * num_epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
        
        # Loss function
        criterion = nn.BCELoss()
        
        # Training loop
        print(f"\nStarting training for {num_epochs} epochs...")
        print(f"Device: {self.device}")
        print("=" * 70)
        
        best_val_acc = 0.0
        training_history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
        
        for epoch in range(num_epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            train_preds = []
            train_labels = []
            
            # Single progress bar for training - update every 50 batches
            with tqdm(total=len(train_loader), desc=f"Epoch {epoch + 1}/{num_epochs} [Train]", 
                     leave=True, ncols=120, miniters=50) as pbar:
                for batch_idx, batch in enumerate(train_loader):
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    character_ids = batch['character_id'].to(self.device)
                    labels = batch['label'].to(self.device)
                    
                    optimizer.zero_grad()
                    
                    outputs = self.model(input_ids, attention_mask, character_ids)
                    loss = criterion(outputs, labels)
                    
                    loss.backward()
                    optimizer.step()
                    scheduler.step()
                    
                    train_loss += loss.item()
                    
                    # Collect predictions
                    preds = (outputs > 0.5).float()
                    train_preds.extend(preds.cpu().numpy())
                    train_labels.extend(labels.cpu().numpy())
                    
                    # Update progress bar every 50 batches
                    if (batch_idx + 1) % 50 == 0 or (batch_idx + 1) == len(train_loader):
                        pbar.set_postfix({
                            'loss': f'{loss.item():.4f}',
                            'avg_loss': f'{train_loss / (batch_idx + 1):.4f}'
                        })
                        pbar.update(min(50, len(train_loader) - pbar.n))
            
            # Calculate training metrics
            train_loss /= len(train_loader)
            train_acc = accuracy_score(train_labels, train_preds)
            
            # Validation phase
            self.model.eval()
            val_loss = 0.0
            val_preds = []
            val_labels = []
            
            # Single progress bar for validation - update every 50 batches
            with tqdm(total=len(val_loader), desc=f"Epoch {epoch + 1}/{num_epochs} [Val]  ", 
                     leave=True, ncols=120, miniters=50) as pbar:
                with torch.no_grad():
                    for batch_idx, batch in enumerate(val_loader):
                        input_ids = batch['input_ids'].to(self.device)
                        attention_mask = batch['attention_mask'].to(self.device)
                        character_ids = batch['character_id'].to(self.device)
                        labels = batch['label'].to(self.device)
                        
                        outputs = self.model(input_ids, attention_mask, character_ids)
                        loss = criterion(outputs, labels)
                        
                        val_loss += loss.item()
                        
                        preds = (outputs > 0.5).float()
                        val_preds.extend(preds.cpu().numpy())
                        val_labels.extend(labels.cpu().numpy())
                        
                        # Update progress bar every 50 batches
                        if (batch_idx + 1) % 50 == 0 or (batch_idx + 1) == len(val_loader):
                            pbar.set_postfix({'loss': f'{val_loss / (batch_idx + 1):.4f}'})
                            pbar.update(min(50, len(val_loader) - pbar.n))
            
            val_loss /= len(val_loader)
            val_acc = accuracy_score(val_labels, val_preds)
            precision, recall, f1, _ = precision_recall_fscore_support(
                val_labels, val_preds, average='binary'
            )
            
            # Store metrics
            training_history['train_loss'].append(train_loss)
            training_history['train_acc'].append(train_acc)
            training_history['val_loss'].append(val_loss)
            training_history['val_acc'].append(val_acc)
            
            # Print epoch summary
            print(f"\n{'=' * 70}")
            print(f"Epoch {epoch + 1}/{num_epochs} Summary:")
            print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
            print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.4f}")
            print(f"  Val Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f}")
            print(f"{'=' * 70}\n")
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                self.save_model(output_dir)
                print(f"✓ New best model saved! (Val Acc: {val_acc:.4f})")
        
        # Save training history
        history_path = os.path.join(output_dir, 'training_history.json')
        with open(history_path, 'w') as f:
            json.dump(training_history, f, indent=2)
        
        print(f"\n{'=' * 70}")
        print(f"Training completed!")
        print(f"Best validation accuracy: {best_val_acc:.4f}")
        print(f"Model saved to: {output_dir}")
        print(f"{'=' * 70}")
        
        return training_history
    
    def save_model(self, output_dir: str):
        """Save model, tokenizer, and character mappings."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save model
        torch.save(self.model.state_dict(), os.path.join(output_dir, 'model.pt'))
        
        # Save tokenizer
        self.tokenizer.save_pretrained(output_dir)
        
        # Save character information
        char_info = {
            'character_to_id': self.character_to_id,
            'id_to_character': self.id_to_character,
            'num_characters': self.num_characters,
            'embedding_dim': self.embedding_dim,
            'model_name': self.model_name,
            'characters': {
                name: {
                    'dialogue_count': profile.get_dialogue_count(),
                    'traits': profile.traits,
                    'speech_patterns': profile.speech_patterns
                }
                for name, profile in self.characters.items()
            }
        }
        
        with open(os.path.join(output_dir, 'character_info.json'), 'w') as f:
            json.dump(char_info, f, indent=2)
    
    def load_model(self, model_dir: str):
        """Load saved model and character mappings."""
        # Load character info
        with open(os.path.join(model_dir, 'character_info.json'), 'r') as f:
            char_info = json.load(f)
        
        self.character_to_id = char_info['character_to_id']
        self.id_to_character = {int(k): v for k, v in char_info['id_to_character'].items()}
        self.num_characters = char_info['num_characters']
        self.embedding_dim = char_info['embedding_dim']
        self.model_name = char_info['model_name']
        
        # Recreate character profiles
        for name, info in char_info['characters'].items():
            profile = CharacterProfile(name=name)
            profile.traits = info.get('traits', [])
            profile.speech_patterns = info.get('speech_patterns', [])
            self.characters[name] = profile
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        
        # Load model
        self.model = CharacterVoiceModel(
            self.model_name,
            self.num_characters,
            self.embedding_dim
        ).to(self.device)
        
        self.model.load_state_dict(
            torch.load(os.path.join(model_dir, 'model.pt'), map_location=self.device)
        )
        self.model.eval()
        
        print(f"Model loaded from {model_dir}")
        print(f"Characters: {len(self.characters)}")
    
    def score(
        self,
        character_name: str,
        dialogue: str,
        context: str = "",
        return_probability: bool = True
    ) -> float:
        """
        Score how well dialogue matches character voice.
        
        Returns:
            Float between 0.0 (poor match) and 1.0 (strong match)
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() or train() first.")
        
        # Check if character exists
        if character_name not in self.character_to_id:
            print(f"Warning: Unknown character '{character_name}'. Returning neutral score.")
            return 0.5
        
        char_id = self.character_to_id[character_name]
        
        # Format input
        text = (
            f"[Character: {character_name}] "
            f"[Context: {context}] "
            f"[Dialogue: {dialogue}]"
        )
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            max_length=256,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Move to device
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        character_ids = torch.tensor([char_id]).to(self.device)
        
        # Get prediction
        self.model.eval()
        with torch.no_grad():
            score = self.model(input_ids, attention_mask, character_ids)
        
        return score.item()
    
    def evaluate_with_explanation(
        self,
        character_name: str,
        dialogue: str,
        context: str = ""
    ) -> Dict:
        """Evaluate with detailed explanation."""
        score = self.score(character_name, dialogue, context)
        
        # Interpretation
        if score >= 0.8:
            interpretation = "Strong character voice match"
        elif score >= 0.6:
            interpretation = "Moderate character voice match"
        elif score >= 0.4:
            interpretation = "Weak character voice match"
        else:
            interpretation = "Poor character voice match"
        
        # Character info
        char_info = None
        if character_name in self.characters:
            profile = self.characters[character_name]
            char_info = {
                'name': character_name,
                'dialogue_count': profile.get_dialogue_count(),
                'traits': profile.traits,
                'patterns': profile.speech_patterns
            }
        
        return {
            'score': score,
            'interpretation': interpretation,
            'character_info': char_info
        }
    
    def get_character_embedding(self, character_name: str) -> Optional[np.ndarray]:
        """Get learned embedding for a character."""
        if self.model is None or character_name not in self.character_to_id:
            return None
        
        char_id = self.character_to_id[character_name]
        embedding = self.model.character_embeddings.weight[char_id]
        return embedding.detach().cpu().numpy()
    
    def compare_characters(self, char1: str, char2: str) -> float:
        """Compare similarity between two characters based on embeddings."""
        emb1 = self.get_character_embedding(char1)
        emb2 = self.get_character_embedding(char2)
        
        if emb1 is None or emb2 is None:
            return 0.0
        
        # Cosine similarity
        similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
        return float(similarity)
    
    def add_character_profile(self, character_name: str) -> CharacterProfile:
        """Add a new character profile manually."""
        if character_name in self.characters:
            return self.characters[character_name]
        
        profile = CharacterProfile(name=character_name)
        self.characters[character_name] = profile
        
        # Assign new ID
        new_id = len(self.character_to_id)
        self.character_to_id[character_name] = new_id
        self.id_to_character[new_id] = character_name
        
        return profile


def main():
    """Main function for standalone training."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train Character Voice Critic')
    parser.add_argument('--data', type=str, required=True,
                       help='Path to CRD3 NPC dialogues JSON file')
    parser.add_argument('--output', type=str, default='./character_voice_model',
                       help='Output directory for trained model')
    parser.add_argument('--epochs', type=int, default=3,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=2e-5,
                       help='Learning rate')
    parser.add_argument('--min_dialogues', type=int, default=10,
                       help='Minimum dialogues per character')
    parser.add_argument('--max_characters', type=int, default=None,
                       help='Maximum number of characters to include')
    
    args = parser.parse_args()
    
    # Initialize critic
    print("Initializing Character Voice Critic...")
    critic = CharacterVoiceCritic()
    
    # Build training data
    training_data = critic.build_training_data_from_crd3(
        crd3_dialogue_file=args.data,
        output_file=os.path.join(args.output, 'training_data.json'),
        min_dialogues=args.min_dialogues,
        max_characters=args.max_characters
    )
    
    # Train model
    critic.train(
        training_data=training_data,
        output_dir=args.output,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate
    )
    
    print("\nTraining complete!")


if __name__ == '__main__':
    main()
