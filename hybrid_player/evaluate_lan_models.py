# unified_lm_evaluation.py
import os
import pandas as pd
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.model_selection import train_test_split
import logging
from typing import Dict, List, Tuple, Any
import json
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class UnifiedLMEvaluator:
    def __init__(self, model_path: str = "models/language_model/final"):
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        self.test_data = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.load_models()
        self.load_test_data()
    
    def load_models(self):
        """Load model and tokenizer"""
        try:
            logger.info(f"Loading model from {self.model_path}...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.model = AutoModelForCausalLM.from_pretrained(self.model_path).to(self.device)
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                
            logger.info(f"✅ Model loaded successfully on {self.device}")
            
        except Exception as e:
            logger.error(f"❌ Failed to load model: {e}")
            raise
    
    def load_test_data(self):
        """Load and prepare test data"""
        data_paths = [
            "data/processed/hybrid_player_data.csv",
            "./data/processed/hybrid_player_data.csv",
            "../data/processed/hybrid_player_data.csv"
        ]
        
        for path in data_paths:
            if os.path.exists(path):
                full_data = pd.read_csv(path)
                logger.info(f"✅ Loaded dataset from {path}: {len(full_data)} samples")
                
                # Create consistent test split
                if os.path.exists("data/processed/test.csv"):
                    self.test_data = pd.read_csv("data/processed/test.csv")
                    logger.info(f"📂 Using existing test split: {len(self.test_data)} samples")
                else:
                    _, self.test_data = train_test_split(full_data, test_size=0.1, random_state=42)
                    logger.info(f"🧪 Created new test split: {len(self.test_data)} samples")
                break
        else:
            logger.warning("❌ No test data found, using built-in examples")
            self.test_data = None
    
    def calculate_robust_perplexity(self, num_samples: int = 100) -> Dict[str, float]:
        """Calculate perplexity using multiple methods for comparison"""
        
        if self.test_data is not None:
            test_texts = self.test_data["text"].dropna().tolist()
            test_texts = test_texts[:min(num_samples, len(test_texts))]
            data_source = "test_split"
        else:
            # Use comprehensive built-in examples
            test_texts = self._get_comprehensive_examples()[:num_samples]
            data_source = "built_in"
        
        logger.info(f"Calculating perplexity on {len(test_texts)} samples from {data_source}...")
        
        results = {}
        
        # Method 1: Standard perplexity (your second code's approach)
        total_loss, total_tokens = 0, 0
        for text in tqdm(test_texts, desc="Method 1 - Standard"):
            try:
                inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=128).to(self.device)
                with torch.no_grad():
                    outputs = self.model(**inputs, labels=inputs["input_ids"])
                loss = outputs.loss
                total_loss += loss.item() * inputs["input_ids"].size(1)
                total_tokens += inputs["input_ids"].size(1)
            except Exception as e:
                logger.warning(f"Error processing text: {e}")
                continue
        
        if total_tokens > 0:
            avg_loss = total_loss / total_tokens
            perplexity = torch.exp(torch.tensor(avg_loss)).item()
            results['standard_perplexity'] = perplexity
            results['standard_loss'] = avg_loss
        else:
            results['standard_perplexity'] = float('inf')
            results['standard_loss'] = float('inf')
        
        # Method 2: Batched processing for efficiency
        if len(test_texts) > 10:
            try:
                batched_perplexity = self._calculate_batched_perplexity(test_texts)
                results['batched_perplexity'] = batched_perplexity
            except Exception as e:
                logger.warning(f"Batched perplexity failed: {e}")
                results['batched_perplexity'] = 'Error'
        
        # Method 3: Filtered perplexity (remove very short/long texts)
        filtered_texts = [t for t in test_texts if 10 <= len(t.split()) <= 100]
        if len(filtered_texts) >= 10:
            filtered_perplexity = self._calculate_simple_perplexity(filtered_texts[:50])
            results['filtered_perplexity'] = filtered_perplexity
            results['filtered_sample_size'] = len(filtered_texts[:50])
        
        results['total_samples'] = len(test_texts)
        results['data_source'] = data_source
        
        return results
    
    def _calculate_batched_perplexity(self, texts: List[str]) -> float:
        """Calculate perplexity using batching for efficiency"""
        batch_size = 8
        total_loss, total_tokens = 0, 0
        
        for i in tqdm(range(0, len(texts), batch_size), desc="Method 2 - Batched"):
            batch_texts = texts[i:i+batch_size]
            try:
                inputs = self.tokenizer(
                    batch_texts, 
                    return_tensors="pt", 
                    truncation=True, 
                    padding=True,
                    max_length=128
                ).to(self.device)
                
                with torch.no_grad():
                    outputs = self.model(**inputs, labels=inputs["input_ids"])
                    loss = outputs.loss
                    total_loss += loss.item() * inputs["input_ids"].numel()
                    total_tokens += inputs["input_ids"].numel()
                    
            except Exception as e:
                logger.warning(f"Batch {i} failed: {e}")
                continue
        
        if total_tokens == 0:
            return float('inf')
        
        avg_loss = total_loss / total_tokens
        return torch.exp(torch.tensor(avg_loss)).item()
    
    def _calculate_simple_perplexity(self, texts: List[str]) -> float:
        """Simple perplexity calculation"""
        total_loss, total_tokens = 0, 0
        
        for text in texts:
            try:
                inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=128).to(self.device)
                with torch.no_grad():
                    outputs = self.model(**inputs, labels=inputs["input_ids"])
                    loss = outputs.loss
                    total_loss += loss.item() * inputs["input_ids"].size(1)
                    total_tokens += inputs["input_ids"].size(1)
            except:
                continue
        
        if total_tokens == 0:
            return float('inf')
        
        avg_loss = total_loss / total_tokens
        return torch.exp(torch.tensor(avg_loss)).item()
    
    def _get_comprehensive_examples(self) -> List[str]:
        """Get comprehensive built-in examples covering different game scenarios"""
        
        exploration = [
            "I look around the room carefully for any hidden passages",
            "Searching the area thoroughly, I check for traps",
            "I examine the ancient runes on the wall closely",
            "Scanning the forest, I look for any movement or danger",
            "I inspect the mysterious artifact for any markings or clues"
        ]
        
        combat = [
            "I attack the dragon with my enchanted sword",
            "Casting a fireball spell at the goblin horde",
            "I draw my bow and aim carefully at the distant target",
            "Using my shield to block the ogre's massive club",
            "I strike the enemy with a powerful two-handed blow"
        ]
        
        social = [
            "I greet the merchant warmly and ask about his wares",
            "Hello there, my name is Valerius the brave",
            "I politely ask the king for assistance in our quest",
            "Thank you for your help, we are in your debt",
            "I explain the situation to the village elder carefully"
        ]
        
        item_interaction = [
            "I carefully pick up the glowing crystal and examine it",
            "Using the health potion to heal my wounds quickly",
            "I read the ancient scroll to learn the magic spell",
            "Equipping the magical armor for better protection",
            "I unlock the chest with the silver key we found earlier"
        ]
        
        # Combine all categories
        all_examples = exploration + combat + social + item_interaction
        return all_examples * 3  # Repeat to get more samples
    
    def evaluate_generation_quality(self, num_examples: int = 20) -> Dict[str, Any]:
        """Comprehensive generation quality evaluation"""
        
        test_scenarios = [
            {"prompt": "I look around and", "category": "exploration"},
            {"prompt": "I attack the", "category": "combat"},
            {"prompt": "I say to the", "category": "social"},
            {"prompt": "The magic spell", "category": "magic"},
            {"prompt": "My sword", "category": "combat"},
            {"prompt": "The treasure chest", "category": "exploration"},
            {"prompt": "I cast a", "category": "magic"},
            {"prompt": "The dragon", "category": "combat"},
            {"prompt": "I ask the", "category": "social"},
            {"prompt": "In the dungeon", "category": "exploration"}
        ]
        
        results = {}
        
        for scenario in test_scenarios[:num_examples]:
            prompt = scenario["prompt"]
            category = scenario["category"]
            
            try:
                # Generate with different temperatures
                responses = {}
                for temp in [0.5, 0.7, 0.9]:
                    response = self._generate_response(prompt, temperature=temp)
                    responses[f"temp_{temp}"] = response
                
                results[prompt] = {
                    "category": category,
                    "responses": responses,
                    "response_lengths": {k: len(v.split()) for k, v in responses.items()}
                }
                
            except Exception as e:
                results[prompt] = {"error": str(e)}
        
        return results
    
    def _generate_response(self, prompt: str, temperature: float = 0.7, max_new_tokens: int = 20) -> str:
        """Generate a single response"""
        try:
            inputs = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
            
            outputs = self.model.generate(
                inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                no_repeat_ngram_size=2
            )
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return response[len(prompt):].strip()
            
        except Exception as e:
            return f"[Generation error: {e}]"
    
    def analyze_response_quality(self, generation_results: Dict[str, Any]) -> Dict[str, float]:
        """Analyze the quality of generated responses"""
        
        all_responses = []
        for prompt_data in generation_results.values():
            if "responses" in prompt_data:
                for temp, response in prompt_data["responses"].items():
                    if not response.startswith("[Generation error"):
                        all_responses.append(response)
        
        if not all_responses:
            return {"error": "No valid responses generated"}
        
        # Calculate basic statistics
        response_lengths = [len(resp.split()) for resp in all_responses]
        unique_responses = len(set(all_responses)) / len(all_responses)
        
        # Simple quality heuristics
        meaningful_responses = 0
        for resp in all_responses:
            words = resp.split()
            if (len(words) >= 3 and 
                not any(word in resp.lower() for word in ["error", "sorry", "cannot"]) and
                not resp.endswith(('...', '--', '???'))):
                meaningful_responses += 1
        
        meaningful_ratio = meaningful_responses / len(all_responses)
        
        return {
            "avg_response_length": np.mean(response_lengths),
            "response_length_std": np.std(response_lengths),
            "unique_response_ratio": unique_responses,
            "meaningful_response_ratio": meaningful_ratio,
            "total_responses_analyzed": len(all_responses)
        }
    
    def run_complete_evaluation(self) -> Dict[str, Any]:
        """Run complete evaluation"""
        
        logger.info("🚀 Starting Unified Language Model Evaluation")
        
        results = {
            "model_info": {
                "model_path": self.model_path,
                "device": str(self.device),
                "test_data_size": len(self.test_data) if self.test_data is not None else 0
            },
            "perplexity_analysis": {},
            "generation_quality": {},
            "response_analysis": {}
        }
        
        # 1. Perplexity Analysis
        logger.info("📊 Calculating robust perplexity metrics...")
        results["perplexity_analysis"] = self.calculate_robust_perplexity(100)
        
        # 2. Generation Quality
        logger.info("🎨 Evaluating generation quality...")
        generation_results = self.evaluate_generation_quality(15)
        results["generation_quality"] = generation_results
        
        # 3. Response Analysis
        logger.info("🔍 Analyzing response quality...")
        results["response_analysis"] = self.analyze_response_quality(generation_results)
        
        return results
    
    def generate_report(self, results: Dict[str, Any]):
        """Generate comprehensive evaluation report"""
        
        print("\n" + "="*80)
        print("🤖 UNIFIED LANGUAGE MODEL EVALUATION REPORT")
        print("="*80)
        
        # Model Info
        model_info = results["model_info"]
        print(f"\n📋 MODEL INFORMATION:")
        print(f"   • Model Path: {model_info['model_path']}")
        print(f"   • Device: {model_info['device']}")
        print(f"   • Test Data Size: {model_info['test_data_size']}")
        
        # Perplexity Results
        perplexity = results["perplexity_analysis"]
        print(f"\n📊 PERPLEXITY ANALYSIS:")
        print(f"   • Standard Perplexity: {perplexity.get('standard_perplexity', 'N/A'):.2f}")
        print(f"   • Standard Loss: {perplexity.get('standard_loss', 'N/A'):.4f}")
        
        if 'batched_perplexity' in perplexity:
            print(f"   • Batched Perplexity: {perplexity['batched_perplexity']:.2f}")
        
        if 'filtered_perplexity' in perplexity:
            print(f"   • Filtered Perplexity: {perplexity['filtered_perplexity']:.2f}")
            print(f"   • Filtered Samples: {perplexity.get('filtered_sample_size', 'N/A')}")
        
        print(f"   • Total Samples: {perplexity.get('total_samples', 'N/A')}")
        print(f"   • Data Source: {perplexity.get('data_source', 'N/A')}")
        
        # Response Analysis
        response_analysis = results["response_analysis"]
        if "error" not in response_analysis:
            print(f"\n💬 RESPONSE QUALITY ANALYSIS:")
            print(f"   • Avg Response Length: {response_analysis.get('avg_response_length', 'N/A'):.1f} words")
            print(f"   • Response Length Std: {response_analysis.get('response_length_std', 'N/A'):.1f}")
            print(f"   • Unique Responses: {response_analysis.get('unique_response_ratio', 'N/A'):.1%}")
            print(f"   • Meaningful Responses: {response_analysis.get('meaningful_response_ratio', 'N/A'):.1%}")
            print(f"   • Total Analyzed: {response_analysis.get('total_responses_analyzed', 'N/A')}")
        
        # Generation Examples
        generation_quality = results["generation_quality"]
        print(f"\n🎭 GENERATION EXAMPLES (Temperature Comparison):")
        
        example_count = 0
        for prompt, data in generation_quality.items():
            if example_count >= 5:  # Show only 5 examples
                break
            if "responses" in data:
                print(f"\n   Prompt: '{prompt}'")
                for temp, response in data["responses"].items():
                    print(f"     {temp}: '{response}'")
                example_count += 1
        
        # Recommendations
        self._print_recommendations(results)
        
        print("\n" + "="*80)
        print("EVALUATION COMPLETE")
        print("="*80)
    
    def _print_recommendations(self, results: Dict[str, Any]):
        """Print recommendations based on evaluation results"""
        
        perplexity = results["perplexity_analysis"].get("standard_perplexity", float('inf'))
        response_analysis = results["response_analysis"]
        
        print(f"\n💡 RECOMMENDATIONS:")
        
        if perplexity < 50:
            print("   ✅ Excellent perplexity - model is well-trained")
        elif perplexity < 100:
            print("   ✅ Good perplexity - model performs reasonably well")
        elif perplexity < 200:
            print("   ⚠️  Moderate perplexity - consider more training data")
        else:
            print("   ❌ High perplexity - model needs significant improvement")
        
        if "unique_response_ratio" in response_analysis:
            uniqueness = response_analysis["unique_response_ratio"]
            if uniqueness < 0.5:
                print("   ⚠️  Low response diversity - try higher temperatures (0.8-1.0)")
            else:
                print("   ✅ Good response diversity")
        
        if "meaningful_response_ratio" in response_analysis:
            meaningful = response_analysis["meaningful_response_ratio"]
            if meaningful < 0.6:
                print("   ⚠️  Many nonsensical responses - consider better training data filtering")
            else:
                print("   ✅ Good response quality")
        
        print(f"\n   🎯 Suggested Generation Parameters:")
        print(f"      • Temperature: 0.7-0.9 for balanced responses")
        print(f"      • max_new_tokens: 15-25 for concise responses")
        print(f"      • no_repeat_ngram_size: 2 to reduce repetition")

def main():
    """Main evaluation function"""
    
    print("🤖 Unified Language Model Evaluation")
    print("=" * 50)
    
    try:
        evaluator = UnifiedLMEvaluator()
        results = evaluator.run_complete_evaluation()
        evaluator.generate_report(results)
        
        # Save results
        output_file = "unified_evaluation_results.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            # Convert numpy types for JSON serialization
            def convert_types(obj):
                if isinstance(obj, (np.float32, np.float64, np.float16)):
                    return float(obj)
                elif isinstance(obj, (np.int32, np.int64, np.int16)):
                    return int(obj)
                elif isinstance(obj, np.bool_):
                    return bool(obj)
                elif isinstance(obj, dict):
                    return {k: convert_types(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_types(item) for item in obj]
                else:
                    return obj
            
            json.dump(convert_types(results), f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 Results saved to: {output_file}")
        
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        raise

if __name__ == "__main__":
    main()