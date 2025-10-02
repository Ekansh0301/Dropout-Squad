"""
Data loading utilities for hybrid player training.
Extracts player utterances from CRD3 and LIGHT datasets.
"""
import os
import json
import pickle
from typing import List, Dict, Any, Tuple
import pandas as pd
from tqdm import tqdm
from .config import DataConfig
from .utils import load_json_files, save_pickle, load_pickle, ensure_dir
import logging

logger = logging.getLogger(__name__)

class CRD3DataLoader:
    """Loads player utterances from Critical Role Dungeons & Dragons 3 dataset."""
    
    def __init__(self, config: DataConfig):
        self.config = config
        self.dm_names = [name.upper() for name in config.dm_names]
    
    def is_player_turn(self, names: List[str]) -> bool:
        """Check if turn is from a player (not DM)."""
        if not names:
            return False
        return not any(name.upper() in self.dm_names for name in names)
    
    def extract_player_utterances(self) -> List[str]:
        """Extract all player utterances from CRD3 dataset."""
        all_utterances = []
        
        for chunk_size in self.config.crd3_chunk_sizes:
            chunk_dir = os.path.join(
                self.config.crd3_base_path, 
                self.config.crd3_aligned_data_dir,
                f"c={chunk_size}"
            )
            if not os.path.exists(chunk_dir):
                logger.warning(f"Directory not found: {chunk_dir}")
                continue
                
            json_files = load_json_files(chunk_dir)
            
            for file_data in tqdm(json_files, desc=f"Processing c={chunk_size}"):
                if not isinstance(file_data, list):
                    continue
                    
                for chunk in file_data:
                    if not isinstance(chunk, dict):
                        continue
                        
                    turns = chunk.get("TURNS", [])
                    for turn in turns:
                        if self.is_player_turn(turn.get("NAMES", [])):
                            utterances = turn.get("UTTERANCES", [])
                            full_utterance = " ".join(utterances).strip()
                            if full_utterance:
                                all_utterances.append(full_utterance)
        
        logger.info(f"Extracted {len(all_utterances)} player utterances from CRD3")
        return all_utterances

class LIGHTDataLoader:
    """Loads player utterances from LIGHT dialogue dataset."""
    
    def __init__(self, config: DataConfig):
        self.config = config
    
    def extract_player_utterances(self) -> List[str]:
        """Extract player utterances from LIGHT dataset."""
        all_utterances = []
        
        # Load main LIGHT data file
        light_data_path = os.path.join(self.config.light_base_path, "light_data.pkl")
        if os.path.exists(light_data_path):
            try:
                light_data = load_pickle(light_data_path)
                logger.info(f"Loaded LIGHT data from {light_data_path}")
                all_utterances.extend(self._extract_from_light_list(light_data))
            except Exception as e:
                logger.error(f"Error loading LIGHT data from {light_data_path}: {e}")
        
        # Load unseen data file
        light_unseen_path = os.path.join(self.config.light_base_path, "light_unseen_data.pkl")
        if os.path.exists(light_unseen_path):
            try:
                light_unseen_data = load_pickle(light_unseen_path)
                logger.info(f"Loaded LIGHT unseen data from {light_unseen_path}")
                all_utterances.extend(self._extract_from_light_list(light_unseen_data))
            except Exception as e:
                logger.error(f"Error loading LIGHT unseen data from {light_unseen_path}: {e}")
        
        # Remove duplicates
        all_utterances = list(set(all_utterances))
        logger.info(f"Extracted {len(all_utterances)} player utterances from LIGHT")
        return all_utterances
    
    def _extract_from_light_list(self, light_data: List) -> List[str]:
        """Extract utterances from LIGHT list structure."""
        utterances = []
        if not isinstance(light_data, list):
            return utterances
            
        for game_instance in light_data:
            if isinstance(game_instance, dict):
                # Extract from speech, action, emote fields
                speech = game_instance.get('speech', '')
                if speech and isinstance(speech, str) and speech.strip():
                    utterances.append(speech.strip())
                
                action = game_instance.get('action', '')
                if action and isinstance(action, str) and action.strip():
                    utterances.append(action.strip())
                
                emote = game_instance.get('emote', '')
                if emote and isinstance(emote, str) and emote.strip():
                    utterances.append(emote.strip())
        
        return utterances

class IntentLabeler:
    def __init__(self):
        self.intent_keywords = {
            'EXPLORE': ['look', 'examine', 'search', 'go to', 'enter', 'open', 'close', 
                       'north', 'south', 'east', 'west', 'up', 'down', 'room', 'area',
                       'explore', 'investigate', 'check', 'inspect', 'walk', 'move',
                       'approach', 'navigate', 'traverse', 'investigate', 'scan'],
            'ACTION': ['attack', 'hit', 'cast', 'use', 'take', 'get', 'drop', 'give',
                      'equip', 'wear', 'remove', 'drink', 'eat', 'read', 'push', 'pull',
                      'fight', 'kill', 'destroy', 'create', 'build', 'craft', 'throw',
                      'shoot', 'stab', 'slash', 'block', 'defend', 'heal', 'brew'],
            'DIALOGUE': ['say', 'tell', 'ask', 'speak', 'talk', 'hello', 'hi', 'thank',
                        'please', 'sorry', 'yes', 'no', 'maybe', 'why', 'what', 'how',
                        'greet', 'question', 'answer', 'respond', 'reply', 'whisper',
                        'shout', 'yell', 'converse', 'discuss', 'negotiate']
        }
    
    def label_utterance(self, text: str) -> str:
        """Label utterance with intent based on keywords"""
        text_lower = text.lower()
        
        # Count keyword matches for each intent
        scores = {}
        for intent, keywords in self.intent_keywords.items():
            scores[intent] = sum(1 for keyword in keywords if keyword in text_lower)
        
        # Get intent with highest score
        if sum(scores.values()) == 0:
            return 'DIALOGUE'  # Default to dialogue if no keywords found
            
        max_intent = max(scores.items(), key=lambda x: x[1])
        return max_intent[0]

class HybridPlayerDataProcessor:
    def __init__(self, config: DataConfig):
        self.config = config
        self.crd3_loader = CRD3DataLoader(config)
        self.light_loader = LIGHTDataLoader(config)
        self.labeler = IntentLabeler()
    
    def process_all_data(self) -> pd.DataFrame:
        """Process both datasets and return labeled data by reading actual files"""
        logger.info("Starting data processing from actual files...")
        
        # Check if data directories exist
        if not os.path.exists(self.config.crd3_base_path):
            logger.warning(f"CRD3 base path not found: {self.config.crd3_base_path}")
        
        if not os.path.exists(self.config.light_base_path):
            logger.warning(f"LIGHT base path not found: {self.config.light_base_path}")
        
        # Extract utterances from both datasets by reading actual files
        logger.info("Extracting from CRD3 dataset...")
        crd3_utterances = self.crd3_loader.extract_player_utterances()
        
        logger.info("Extracting from LIGHT dataset...")
        light_utterances = self.light_loader.extract_player_utterances()
        
        # Combine and deduplicate
        all_utterances = list(set(crd3_utterances + light_utterances))
        logger.info(f"Total unique utterances extracted: {len(all_utterances)}")
        
        if len(all_utterances) == 0:
            logger.error("No utterances found in the dataset files!")
            logger.error("Please check that:")
            logger.error(f"1. CRD3 data is in: {self.config.crd3_base_path}")
            logger.error(f"2. LIGHT data is in: {self.config.light_base_path}")
            logger.error("3. The file structures match the expected formats")
            raise ValueError("No data found in the specified paths")
        
        # Label intents
        logger.info("Labeling utterances with intents...")
        labeled_data = []
        for utterance in tqdm(all_utterances, desc="Labeling intents"):
            intent = self.labeler.label_utterance(utterance)
            labeled_data.append({
                'text': utterance,
                'intent': intent,
                'intent_id': ['EXPLORE', 'ACTION', 'DIALOGUE'].index(intent)
            })
        
        df = pd.DataFrame(labeled_data)
        
        # Save processed data
        ensure_dir(self.config.output_path)
        output_file = os.path.join(self.config.output_path, "hybrid_player_data.csv")
        df.to_csv(output_file, index=False)
        logger.info(f"Saved processed data to {output_file}")
        
        # Print class distribution
        self._print_class_distribution(df)
        
        return df
    
    def _print_class_distribution(self, df: pd.DataFrame) -> None:
        """Print distribution of intent classes"""
        distribution = df['intent'].value_counts()
        logger.info("Intent class distribution:")
        for intent, count in distribution.items():
            percentage = count / len(df) * 100
            logger.info(f"  {intent}: {count} ({percentage:.1f}%)")
    
    def train_val_test_split(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Split data into train/val/test sets"""
        from sklearn.model_selection import train_test_split
        
        # Ensure we have enough samples for each class
        if len(df) < 10:
            logger.warning("Very small dataset, using simple split")
            train_df = df
            val_df = df.head(0)  # Empty
            test_df = df.head(0)  # Empty
        else:
            try:
                train_df, temp_df = train_test_split(
                    df, test_size=(self.config.val_split + self.config.test_split), 
                    random_state=42, stratify=df['intent']
                )
                
                val_df, test_df = train_test_split(
                    temp_df, 
                    test_size=self.config.test_split/(self.config.val_split + self.config.test_split),
                    random_state=42, stratify=temp_df['intent']
                )
            except ValueError:
                # Fallback if stratification fails
                logger.warning("Stratified split failed, using random split")
                train_df, temp_df = train_test_split(
                    df, test_size=(self.config.val_split + self.config.test_split), 
                    random_state=42
                )
                val_df, test_df = train_test_split(
                    temp_df, 
                    test_size=self.config.test_split/(self.config.val_split + self.config.test_split),
                    random_state=42
                )
        
        logger.info(f"Data split - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
        return train_df, val_df, test_df