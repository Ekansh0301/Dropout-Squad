"""
Multi-Critic Reinforcement Learning training pipeline for Director LLM.
Implements PPO with dynamic reward weighting, comprehensive logging, and robust error handling.

Features:
- Dynamic reward weighting based on player intent
- Comprehensive logging and visualization
- Automatic checkpointing and recovery
- Multi-objective reward tracking
"""

import torch
import yaml
import wandb
import json
import numpy as np
from pathlib import Path
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
from tqdm import tqdm
from collections import defaultdict, deque
import warnings
import sys
import signal

# Import project components
from critics import NarrativeCritic, CausalCritic
from hybrid_player import HybridPlayer

warnings.filterwarnings('ignore')

class PPOTrainingOrchestrator:
    """
    Main orchestrator for Multi-Critic RL training with comprehensive
    error handling, checkpointing, and evaluation capabilities.
    """
    
    def __init__(self, config_path="configs/ppo_config.yaml"):
        """Initialize training orchestrator with configuration."""
        print("=" * 80)
        print("DIRECTOR LLM - MULTI-CRITIC PPO TRAINING PIPELINE")
        print("=" * 80)
        
        # Load configuration
        self.config_path = config_path
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Setup paths and directories
        self.output_dir = Path(self.config['training_settings']['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / 'checkpoints').mkdir(exist_ok=True)
        (self.output_dir / 'logs').mkdir(exist_ok=True)
        (self.output_dir / 'plots').mkdir(exist_ok=True)
        
        # Setup device
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"\n✓ Device: {self.device}")
        if torch.cuda.is_available():
            print(f"✓ GPU: {torch.cuda.get_device_name(0)}")
            print(f"✓ VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        
        # Set random seeds
        self.set_seeds(self.config['training_settings']['seed'])
        
        # Initialize tracking variables
        self.global_step = 0
        self.best_reward = -float('inf')
        self.training_history = defaultdict(list)
        self.recent_rewards = deque(maxlen=100)  # For moving average
        
        # Setup graceful shutdown handler
        signal.signal(signal.SIGINT, self._signal_handler)
        self.interrupted = False
        
        print(f"✓ Output directory: {self.output_dir}")
        print(f"✓ Configuration loaded from: {config_path}")

    def set_seeds(self, seed):
        """Set all random seeds for reproducibility."""
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        print(f"✓ Random seed set to: {seed}")

    def _signal_handler(self, signum, frame):
        """Handle Ctrl+C gracefully."""
        print("\n\n⚠️  Training interrupted by user. Saving checkpoint...")
        self.interrupted = True

    def setup_wandb(self):
        """Initialize Weights & Biases logging."""
        if self.config['training_settings'].get('log_with') == 'wandb':
            try:
                run_name = self.config.get('run_name', 'ppo-run').format(
                    date=datetime.now().strftime('%Y%m%d-%H%M%S')
                )
                
                wandb.init(
                    project=self.config.get('project_name', 'Director-LLM-MCRL'),
                    name=run_name,
                    config=self.config,
                    resume="allow",
                    dir=str(self.output_dir / 'logs')
                )
                print("✓ W&B tracking initialized")
                return True
            except ImportError:
                print("⚠️  W&B not installed. Install with: pip install wandb")
            except Exception as e:
                print(f"⚠️  W&B initialization failed: {e}")
        return False

    def load_models(self):
        """Load all models and components."""
        print("\n[1/5] Loading Models and Components")
        paths = self.config['model_paths']
        
        # Load tokenizer first
        print("  → Loading tokenizer")
        self.tokenizer = AutoTokenizer.from_pretrained(paths['policy_model_path'])
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        print("✓ Tokenizer loaded")
        
        # Load policy model with value head for PPO
        print("  → Loading policy model with value head")
        try:
            # Try loading with value head directly
            self.model = AutoModelForCausalLMWithValueHead.from_pretrained(
                paths['policy_model_path'],
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
                is_trainable=True,
            )
        except Exception as e:
            print(f"  → Direct loading failed, loading base model first: {e}")
            # Load base model, then add value head
            base_model = AutoModelForCausalLM.from_pretrained(
                "microsoft/phi-2",
                torch_dtype=torch.bfloat16,
                trust_remote_code=True
            )
            base_model = PeftModel.from_pretrained(base_model, paths['policy_model_path'])
            base_model = base_model.merge_and_unload()
            self.model = AutoModelForCausalLMWithValueHead.from_pretrained(
                base_model,
                is_trainable=True
            )
        
        self.model = self.model.to(self.device)
        print(f"✓ Policy model loaded: {paths['policy_model_path']}")
        
        # Load reference model (frozen copy for KL divergence)
        print("  → Loading reference model")
        try:
            self.ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(
                paths['policy_model_path'],
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
                is_trainable=False,
            )
        except Exception as e:
            print(f"  → Using base model approach for reference")
            base_model = AutoModelForCausalLM.from_pretrained(
                "microsoft/phi-2",
                torch_dtype=torch.bfloat16,
                trust_remote_code=True
            )
            base_model = PeftModel.from_pretrained(base_model, paths['policy_model_path'])
            base_model = base_model.merge_and_unload()
            self.ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(
                base_model,
                is_trainable=False
            )
        
        self.ref_model = self.ref_model.to(self.device).eval()
        print("✓ Reference model loaded")
        
        # Load critics
        print("  → Loading critic models")
        self.narrative_critic = NarrativeCritic(paths['narrative_critic_path'], self.device)
        self.causal_critic = CausalCritic(paths['causal_critic_path'], self.device)
        print("✓ Critics loaded")
        
        # Load hybrid player
        print("  → Loading hybrid player")
        self.hybrid_player = HybridPlayer(
            paths['player_generator_path'],
            paths['player_classifier_path'],
            self.device
        )
        print("✓ Hybrid player loaded")
        
        # Print model sizes
        policy_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"\n✓ All components loaded successfully")
        print(f"  • Trainable parameters: {policy_params:,}")

    def get_dynamic_weights(self, intents):
        """
        Dynamically adjust reward weights based on player intents.
        
        This is a key innovation of the MCRL framework: different intents
        require different reward priorities.
        """
        dynamic_weights = self.config.get('dynamic_reward_weights', {
            'EXPLORE': {'narrative': 0.8, 'causal': 0.2},
            'ACTION': {'narrative': 0.3, 'causal': 0.7},
            'DIALOGUE': {'narrative': 0.6, 'causal': 0.4},
            'default': {'narrative': 0.5, 'causal': 0.5}
        })
        
        # Get weights for each intent in the batch
        batch_weights = []
        for intent in intents:
            weights = dynamic_weights.get(intent, dynamic_weights['default'])
            batch_weights.append([weights['narrative'], weights['causal']])
        
        return torch.tensor(batch_weights, device=self.device)

    def compute_rewards(self, prompts, responses, intents):
        """
        Compute multi-objective rewards with dynamic weighting.
        
        Returns:
            rewards: Final scalar rewards for each sample
            reward_components: Dict with individual critic scores for logging
        """
        # Get individual critic scores
        narrative_rewards = self.narrative_critic.get_reward(responses)
        causal_rewards = self.causal_critic.get_reward(prompts, responses)
        
        # Normalize rewards to prevent scale issues
        narrative_rewards = self._normalize_rewards(narrative_rewards)
        causal_rewards = self._normalize_rewards(causal_rewards)
        
        # Stack rewards into matrix [batch_size, 2]
        reward_matrix = torch.stack([narrative_rewards, causal_rewards], dim=1)
        
        # Get dynamic weights based on intents
        if self.config['training_settings'].get('use_dynamic_weighting', True):
            weights = self.get_dynamic_weights(intents)
        else:
            # Use static weights
            static_weights = self.config['reward_weights']
            weights = torch.tensor(
                [[static_weights['narrative'], static_weights['causal']]] * len(intents),
                device=self.device
            )
        
        # Compute weighted combination: sum(rewards * weights) for each sample
        final_rewards = (reward_matrix * weights).sum(dim=1)
        
        # Apply reward clipping to prevent extreme values
        max_reward = self.config['training_settings'].get('max_reward_clip', 10.0)
        final_rewards = torch.clamp(final_rewards, -max_reward, max_reward)
        
        reward_components = {
            'narrative': narrative_rewards,
            'causal': causal_rewards,
            'final': final_rewards,
            'weights': weights
        }
        
        return final_rewards, reward_components

    def _normalize_rewards(self, rewards):
        """Z-score normalization of rewards."""
        mean = rewards.mean()
        std = rewards.std()
        if std > 0:
            return (rewards - mean) / (std + 1e-8)
        return rewards - mean

    def initialize_ppo_trainer(self):
        """Initialize the PPO trainer with config."""
        print("\n[2/5] Initializing PPO Trainer")
        
        ppo_config = PPOConfig(
            **self.config['ppo_hyperparameters'],
            log_with=self.config['training_settings'].get('log_with'),
            tracker_project_name=self.config.get('project_name', 'Director-LLM-MCRL'),
        )
        
        self.ppo_trainer = PPOTrainer(
            config=ppo_config,
            model=self.model,
            ref_model=self.ref_model,
            tokenizer=self.tokenizer,
        )
        
        print("✓ PPOTrainer initialized with config:")
        print(f"  • Learning rate: {ppo_config.learning_rate}")
        print(f"  • Batch size: {ppo_config.batch_size}")
        print(f"  • Mini batch size: {ppo_config.mini_batch_size}")
        print(f"  • PPO epochs: {ppo_config.ppo_epochs}")
        print(f"  • KL coefficient: {ppo_config.init_kl_coef}")

    def save_checkpoint(self, step, is_best=False):
        """Save training checkpoint."""
        checkpoint_dir = self.output_dir / 'checkpoints' / f'step_{step}'
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model
        self.ppo_trainer.save_pretrained(checkpoint_dir)
        self.tokenizer.save_pretrained(checkpoint_dir)
        
        # Save training state
        state = {
            'global_step': step,
            'best_reward': self.best_reward,
            'training_history': dict(self.training_history),
            'config': self.config
        }
        
        with open(checkpoint_dir / 'training_state.json', 'w') as f:
            json.dump(state, f, indent=2)
        
        # If best model, also save to 'best' directory
        if is_best:
            best_dir = self.output_dir / 'best_model'
            best_dir.mkdir(parents=True, exist_ok=True)
            self.ppo_trainer.save_pretrained(best_dir)
            self.tokenizer.save_pretrained(best_dir)
            print(f"  ⭐ New best model saved (reward: {self.best_reward:.4f})")
        
        print(f"  💾 Checkpoint saved: {checkpoint_dir}")

    def evaluate(self, num_samples=50):
        """
        Perform evaluation on a held-out set.
        
        This generates responses to fixed prompts and computes average rewards.
        """
        print("\n  🔍 Running evaluation...")
        self.model.eval()
        
        # Generate evaluation prompts
        eval_prompts_data = self.hybrid_player.generate_prompts(
            batch_size=num_samples,
            max_length=self.config['generation_settings']['max_prompt_length']
        )
        eval_prompts, eval_intents = zip(*eval_prompts_data)
        
        all_rewards = []
        narrative_scores = []
        causal_scores = []
        
        # Generate in smaller batches to avoid OOM
        eval_batch_size = 8
        for i in range(0, len(eval_prompts), eval_batch_size):
            batch_prompts = eval_prompts[i:i+eval_batch_size]
            batch_intents = eval_intents[i:i+eval_batch_size]
            
            # Tokenize
            prompt_tensors = [
                self.tokenizer.encode(p, return_tensors="pt").to(self.device) 
                for p in batch_prompts
            ]
            
            # Generate
            with torch.no_grad():
                response_tensors = self.ppo_trainer.generate(
                    prompt_tensors,
                    **self.config['generation_settings']
                )
            
            # Decode
            responses = [
                self.tokenizer.decode(r.squeeze(), skip_special_tokens=True) 
                for r in response_tensors
            ]
            
            # Compute rewards
            rewards, components = self.compute_rewards(
                list(batch_prompts), 
                responses, 
                list(batch_intents)
            )
            
            all_rewards.extend(rewards.cpu().numpy())
            narrative_scores.extend(components['narrative'].cpu().numpy())
            causal_scores.extend(components['causal'].cpu().numpy())
        
        self.model.train()
        
        eval_metrics = {
            'eval/mean_reward': np.mean(all_rewards),
            'eval/std_reward': np.std(all_rewards),
            'eval/mean_narrative': np.mean(narrative_scores),
            'eval/mean_causal': np.mean(causal_scores),
        }
        
        print(f"  ✓ Evaluation complete:")
        print(f"    • Mean reward: {eval_metrics['eval/mean_reward']:.4f}")
        print(f"    • Mean narrative: {eval_metrics['eval/mean_narrative']:.4f}")
        print(f"    • Mean causal: {eval_metrics['eval/mean_causal']:.4f}")
        
        return eval_metrics

    def log_metrics(self, metrics, step):
        """Log metrics to wandb and local file."""
        # Add to history
        for key, value in metrics.items():
            self.training_history[key].append(value)
        
        # Log to wandb
        if self.use_wandb:
            wandb.log(metrics, step=step)
        
        # Log to local file
        log_file = self.output_dir / 'logs' / 'training_log.jsonl'
        with open(log_file, 'a') as f:
            log_entry = {'step': step, **metrics}
            f.write(json.dumps(log_entry) + '\n')

    def plot_training_curves(self):
        """Generate training curve plots."""
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        sns.set_style("whitegrid")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot 1: Combined reward
        if 'ppo/mean_reward' in self.training_history:
            axes[0, 0].plot(self.training_history['ppo/mean_reward'], label='Mean Reward')
            if 'eval/mean_reward' in self.training_history:
                eval_steps = np.arange(0, len(self.training_history['ppo/mean_reward']), 
                                      self.config['training_settings'].get('eval_interval', 100))
                axes[0, 0].plot(eval_steps[:len(self.training_history['eval/mean_reward'])], 
                              self.training_history['eval/mean_reward'], 
                              'ro-', label='Eval Reward')
            axes[0, 0].set_xlabel('Step')
            axes[0, 0].set_ylabel('Reward')
            axes[0, 0].set_title('Combined Reward Over Training')
            axes[0, 0].legend()
            axes[0, 0].grid(alpha=0.3)
        
        # Plot 2: Individual critic rewards
        if 'ppo/narrative_reward' in self.training_history:
            axes[0, 1].plot(self.training_history['ppo/narrative_reward'], label='Narrative')
            axes[0, 1].plot(self.training_history['ppo/causal_reward'], label='Causal')
            axes[0, 1].set_xlabel('Step')
            axes[0, 1].set_ylabel('Reward')
            axes[0, 1].set_title('Individual Critic Rewards')
            axes[0, 1].legend()
            axes[0, 1].grid(alpha=0.3)
        
        # Plot 3: KL divergence
        if 'ppo/kl' in self.training_history:
            axes[1, 0].plot(self.training_history['ppo/kl'])
            axes[1, 0].axhline(y=self.config['ppo_hyperparameters']['target_kl'], 
                             color='r', linestyle='--', label='Target KL')
            axes[1, 0].set_xlabel('Step')
            axes[1, 0].set_ylabel('KL Divergence')
            axes[1, 0].set_title('KL Divergence from Reference Model')
            axes[1, 0].legend()
            axes[1, 0].grid(alpha=0.3)
        
        # Plot 4: Policy loss
        if 'ppo/policy_loss' in self.training_history:
            axes[1, 1].plot(self.training_history['ppo/policy_loss'])
            axes[1, 1].set_xlabel('Step')
            axes[1, 1].set_ylabel('Loss')
            axes[1, 1].set_title('Policy Loss')
            axes[1, 1].grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'plots' / 'training_curves.png', dpi=300)
        plt.close()
        
        print(f"  📊 Training curves saved to: {self.output_dir / 'plots' / 'training_curves.png'}")

    def train(self):
        """Main PPO training loop."""
        print("\n[3/5] Starting PPO Training Loop")
        print(f"  • Total steps: {self.config['training_settings']['total_ppo_steps']}")
        print(f"  • Batch size: {self.config['ppo_hyperparameters']['batch_size']}")
        print(f"  • Checkpoint interval: {self.config['training_settings'].get('checkpoint_interval', 100)}")
        print(f"  • Evaluation interval: {self.config['training_settings'].get('eval_interval', 100)}")
        
        generation_kwargs = self.config['generation_settings'].copy()
        # Remove prompt length as it's not a generation parameter
        generation_kwargs.pop('max_prompt_length', None)
        
        checkpoint_interval = self.config['training_settings'].get('checkpoint_interval', 100)
        eval_interval = self.config['training_settings'].get('eval_interval', 100)
        total_steps = self.config['training_settings']['total_ppo_steps']
        
        # Main training loop
        pbar = tqdm(range(total_steps), desc="PPO Training")
        
        for step in pbar:
            if self.interrupted:
                self.save_checkpoint(step, is_best=False)
                break
            
            try:
                # 1. Generate batch of prompts from hybrid player
                generated_batch = self.hybrid_player.generate_prompts(
                    batch_size=self.config['ppo_hyperparameters']['batch_size'],
                    max_length=self.config['generation_settings']['max_prompt_length']
                )
                prompts, intents = zip(*generated_batch)
                prompts = list(prompts)
                intents = list(intents)
                
                # 2. Tokenize prompts
                prompt_tensors = [
                    self.tokenizer.encode(p, return_tensors="pt").to(self.device)
                    for p in prompts
                ]
                
                # 3. Generate responses
                response_tensors = self.ppo_trainer.generate(
                    prompt_tensors,
                    **generation_kwargs
                )
                
                # 4. Decode responses
                responses = [
                    self.tokenizer.decode(r.squeeze(), skip_special_tokens=True)
                    for r in response_tensors
                ]
                
                # 5. Compute rewards
                rewards, reward_components = self.compute_rewards(prompts, responses, intents)
                
                # Track recent rewards for moving average
                self.recent_rewards.extend(rewards.cpu().numpy())
                
                # 6. Perform PPO update
                stats = self.ppo_trainer.step(
                    prompt_tensors,
                    response_tensors,
                    rewards.tolist()
                )
                
                # 7. Log metrics
                metrics = {
                    'ppo/mean_reward': rewards.mean().item(),
                    'ppo/std_reward': rewards.std().item(),
                    'ppo/narrative_reward': reward_components['narrative'].mean().item(),
                    'ppo/causal_reward': reward_components['causal'].mean().item(),
                    'ppo/moving_avg_reward': np.mean(list(self.recent_rewards)),
                    'ppo/kl': stats.get('ppo/policy/kl', stats.get('objective/kl', 0)),
                    'ppo/policy_loss': stats.get('ppo/loss/policy', 0),
                    'ppo/value_loss': stats.get('ppo/loss/value', 0),
                }
                
                # Add intent distribution to metrics
                intent_dist = {f'intent/{intent}': intents.count(intent) / len(intents) 
                             for intent in set(intents)}
                metrics.update(intent_dist)
                
                self.log_metrics(metrics, step)
                
                # Update progress bar
                pbar.set_postfix({
                    'reward': f"{metrics['ppo/mean_reward']:.3f}",
                    'narrative': f"{metrics['ppo/narrative_reward']:.3f}",
                    'causal': f"{metrics['ppo/causal_reward']:.3f}"
                })
                
                # 8. Periodic evaluation
                if (step + 1) % eval_interval == 0:
                    eval_metrics = self.evaluate()
                    self.log_metrics(eval_metrics, step)
                    
                    # Check if best model
                    if eval_metrics['eval/mean_reward'] > self.best_reward:
                        self.best_reward = eval_metrics['eval/mean_reward']
                        self.save_checkpoint(step, is_best=True)
                
                # 9. Periodic checkpointing
                if (step + 1) % checkpoint_interval == 0:
                    self.save_checkpoint(step, is_best=False)
                
                self.global_step = step
                
            except Exception as e:
                print(f"\n❌ Error at step {step}: {e}")
                import traceback
                traceback.print_exc()
                print("Saving emergency checkpoint...")
                self.save_checkpoint(step, is_best=False)
                raise

        print("\n✓ Training loop completed successfully")

    def finalize_training(self):
        """Save final model and generate reports."""
        print("\n[4/5] Finalizing Training")
        
        # Save final model
        print("  → Saving final model")
        final_dir = self.output_dir / 'final_model'
        self.ppo_trainer.save_pretrained(final_dir)
        self.tokenizer.save_pretrained(final_dir)
        
        # Save final configuration
        with open(final_dir / 'final_config.yaml', 'w') as f:
            yaml.dump(self.config, f)
        
        # Generate training curves
        print("  → Generating training curves")
        self.plot_training_curves()
        
        # Save training history
        history_df = pd.DataFrame(self.training_history)
        history_df.to_csv(self.output_dir / 'logs' / 'training_history.csv', index=False)
        
        # Generate final report
        self.generate_training_report()
        
        print(f"\n✓ All artifacts saved to: {self.output_dir}")

    def generate_training_report(self):
        """Generate a comprehensive training report."""
        report = f"""# PPO Training Report - Director LLM

**Training Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Total Steps:** {self.global_step + 1}
**Best Reward:** {self.best_reward:.4f}

## Configuration

**Model:**
- Base: microsoft/phi-2
- SFT Checkpoint: {self.config['model_paths']['policy_model_path']}

**PPO Hyperparameters:**
- Learning Rate: {self.config['ppo_hyperparameters']['learning_rate']}
- Batch Size: {self.config['ppo_hyperparameters']['batch_size']}
- PPO Epochs: {self.config['ppo_hyperparameters']['ppo_epochs']}
- KL Target: {self.config['ppo_hyperparameters']['target_kl']}

**Reward Configuration:**
- Dynamic Weighting: {self.config['training_settings'].get('use_dynamic_weighting', True)}
- Static Weights: Narrative={self.config['reward_weights']['narrative']}, Causal={self.config['reward_weights']['causal']}

## Training Summary

**Final Metrics:**
"""
        
        if self.training_history:
            last_10_avg = np.mean(list(self.training_history.get('ppo/mean_reward', []))[-10:])
            report += f"- Mean Reward (last 10 steps): {last_10_avg:.4f}\n"
            report += f"- Mean Narrative (last 10): {np.mean(list(self.training_history.get('ppo/narrative_reward', []))[-10:]):.4f}\n"
            report += f"- Mean Causal (last 10): {np.mean(list(self.training_history.get('ppo/causal_reward', []))[-10:]):.4f}\n"
        
        report += f"""

## Files Generated

- `final_model/`: Final PPO-trained model
- `best_model/`: Best model by evaluation reward
- `checkpoints/`: Periodic training checkpoints
- `logs/training_history.csv`: Complete training metrics
- `plots/training_curves.png`: Visualization of training progress

## Next Steps

1. Run comprehensive evaluation: `python generate_research_report.py`
2. Compare against baselines (SFT, single-critic, fixed-weights)
3. Perform zero-shot transfer tests (Jericho games, Story Cloze)
4. Conduct human evaluation study

---
*Generated by the Director LLM PPO Training Pipeline*
"""
        
        with open(self.output_dir / 'training_report.md', 'w') as f:
            f.write(report)
        
        print(f"  📄 Training report saved: {self.output_dir / 'training_report.md'}")

    def run(self):
        """Execute the complete PPO training pipeline."""
        try:
            self.use_wandb = self.setup_wandb()
            self.load_models()
            self.initialize_ppo_trainer()
            self.train()
            self.finalize_training()
            
            print("\n" + "=" * 80)
            print("✅ PPO TRAINING COMPLETED SUCCESSFULLY!")
            print("=" * 80)
            print(f"\n📁 All outputs saved to: {self.output_dir}")
            print(f"⭐ Best model reward: {self.best_reward:.4f}")
            
            if self.use_wandb:
                wandb.finish()
            
        except KeyboardInterrupt:
            print("\n\n⚠️  Training was interrupted by the user.")
            if self.global_step > 0:
                self.finalize_training()
            else:
                print("No training steps were completed. No final artifacts to save.")

        # --- THIS COMPLETES THE PROVIDED CODE ---
        except Exception as e:
            print(f"\n\n❌ AN UNEXPECTED ERROR OCCURRED DURING THE PIPELINE.")
            print(f"Error Type: {type(e).__name__}")
            print(f"Error Details: {e}")
            traceback.print_exc()
            
            if self.global_step > 0:
                print("\nAttempting to save a final report and plots with data up to the point of failure...")
                try:
                    self.finalize_training()
                except Exception as final_e:
                    print(f"❌ Could not finalize training artifacts. Error: {final_e}")
            else:
                print("Error occurred before any training steps completed.")

# --- This is the main execution block that runs the orchestrator ---
if __name__ == "__main__":
    orchestrator = PPOTrainingOrchestrator()
    orchestrator.run()