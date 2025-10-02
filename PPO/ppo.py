"""
train_ppo.py - The Main Orchestrator for MCRL Training

This script is the final stage of the project, integrating all previously
trained components to fine-tune the SFT Director LLM using Reinforcement
Learning (PPO).

The pipeline is as follows:
1.  Load all configurations from `configs/ppo_config.yaml`.
2.  Initialize all models:
    - The SFT Agent (as the PPO Policy Model).
    - A frozen reference copy of the SFT Agent (for KL-divergence).
    - The Narrative and Causal Critic models.
    - The autonomous Hybrid Player.
3.  Initialize the TRL `PPOTrainer`.
4.  Begin the main training loop:
    a. The Hybrid Player generates a batch of prompts.
    b. The PPO Policy Model generates responses to these prompts.
    c. Both Critics score the prompt-response pairs.
    d. The scores are combined into a single reward signal.
    e. The `PPOTrainer` performs an update step using this data.
5.  Save the final, PPO-trained model adapters.
"""

import torch
import yaml
import wandb
from pathlib import Path
from transformers import AutoTokenizer
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
from tqdm import tqdm

# Import our custom, modular components
from critics import NarrativeCritic, CausalCritic
from hybrid_player import HybridPlayer

def setup_wandb(config):
    """Initializes Weights & Biases if configured."""
    if config['training_settings'].get('log_with') == 'wandb':
        try:
            wandb.init(
                project=config.get('project_name', 'Director-LLM-MCRL'),
                name=config.get('run_name', 'ppo-run').format(date=Path.cwd().name),
                config=config,
                resume="allow",
            )
            print("✓ W&B tracking initialized.")
            return True
        except ImportError:
            print("⚠️ W&B not installed. Skipping. Install with: pip install wandb")
        except Exception as e:
            print(f"⚠️ W&B initialization failed: {e}")
    return False

def main():
    """The main PPO training pipeline."""
    
    # --- 1. Load Configuration from YAML ---
    print("--- [1/5] Loading Configuration ---")
    with open("configs/ppo_config.yaml", 'r') as f:
        config = yaml.safe_load(f)
    print("✓ Config loaded successfully.")
    
    # --- 2. Setup Environment and Logging ---
    print("\n--- [2/5] Setting up Environment ---")
    ppo_config = PPOConfig(**config['ppo_hyperparameters'])
    torch.manual_seed(config['training_settings']['seed'])
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"✓ Using device: {device}")
    
    use_wandb = setup_wandb(config)

    # --- 3. Load All Models and Tokenizers ---
    print("\n--- [3/5] Loading All Models ---")
    paths = config['model_paths']

    # Load the SFT Agent as the main policy model.
    # TRL's `AutoModelForCausalLMWithValueHead` automatically adds a value head
    # on top of the language model, which is essential for PPO.
    model = AutoModelForCausalLMWithValueHead.from_pretrained(
        paths['policy_model_path'],
        torch_dtype=torch.bfloat16,
        is_trainable=True,
    ).to(device)
    print(f"✓ Policy Model loaded: {paths['policy_model_path']}")

    # Load a second, frozen copy of the SFT agent. This 'reference model'
    # is used to calculate the KL-divergence penalty, preventing the PPO
    # model from straying too far from its original language capabilities.
    ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(
        paths['policy_model_path'],
        torch_dtype=torch.bfloat16,
        is_trainable=False,
    ).to(device)
    print(f"✓ Reference Model loaded: {paths['policy_model_path']}")
    
    tokenizer = AutoTokenizer.from_pretrained(paths['policy_model_path'])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print("✓ Tokenizer loaded.")
    
    # Load our custom, modular components
    narrative_critic = NarrativeCritic(paths['narrative_critic_path'], device)
    causal_critic = CausalCritic(paths['causal_critic_path'], device)
    hybrid_player = HybridPlayer(
        paths['player_generator_path'],
        paths['player_classifier_path'],
        device
    )

    # --- 4. Initialize the PPOTrainer ---
    print("\n--- [4/5] Initializing PPOTrainer ---")
    ppo_trainer = PPOTrainer(
        config=ppo_config,
        model=model,
        ref_model=ref_model,
        tokenizer=tokenizer
    )
    print("✓ PPOTrainer initialized.")

    # --- 5. The Main PPO Training Loop ---
    print("\n--- [5/5] Starting PPO Training Loop ---")
    generation_kwargs = config['generation_settings']
    reward_weights = config['reward_weights']
    
    # Use tqdm for a nice progress bar
    for step in tqdm(range(config['training_settings']['total_ppo_steps']), "PPO Step"):
        # 5.1 Get a batch of prompts from our autonomous Hybrid Player
        # The player returns a list of (prompt_text, intent) tuples
        generated_batch = hybrid_player.generate_prompts(
            batch_size=ppo_config.batch_size, 
            max_length=generation_kwargs['max_prompt_length']
        )
        prompts, intents = zip(*generated_batch)
        prompts = list(prompts) # Convert from tuple to list
        prompt_tensors = [tokenizer.encode(p, return_tensors="pt").to(device) for p in prompts]

        # 5.2 Generate responses from the main policy model
        response_tensors = ppo_trainer.generate(prompt_tensors, **generation_kwargs)
        responses = [tokenizer.decode(r.squeeze(), skip_special_tokens=True) for r in response_tensors]

        # 5.3 Get reward scores from both critics
        narrative_rewards = narrative_critic.get_reward(responses)
        causal_rewards = causal_critic.get_reward(prompts, responses)

        # 5.4 Combine rewards into a single signal
        # This is where the core logic of our MCRL framework is implemented.
        rewards = (reward_weights['narrative'] * narrative_rewards) + \
                  (reward_weights['causal'] * causal_rewards)
        
        # 5.5 Perform the PPO update step
        # The trainer handles all the complex math: GAE, value loss, policy loss, KL penalty, etc.
        # It requires lists of tensors and a list of python floats for the rewards.
        stats = ppo_trainer.step(prompt_tensors, response_tensors, rewards.tolist())
        
        # Log statistics to W&B or terminal
        if use_wandb:
            log_data = {
                "ppo/mean_reward": rewards.mean().item(),
                "ppo/narrative_reward": narrative_rewards.mean().item(),
                "ppo/causal_reward": causal_rewards.mean().item(),
                **stats,
            }
            ppo_trainer.log_stats(stats, {"prompts": prompts, "responses": responses}, rewards.tolist())
            wandb.log(log_data)

    # --- Post-Training: Save the final model ---
    print("\n--- PPO training complete. Saving final model... ---")
    output_dir = config['training_settings']['output_dir']
    ppo_trainer.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    # Save the final config used for this run
    final_config_path = Path(output_dir) / "final_ppo_config.yaml"
    with open(final_config_path, 'w') as f:
        yaml.dump(config, f)

    print(f"✓ Final model adapters and config saved to: {output_dir}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user. No model was saved.")
    except Exception as e:
        print(f"\n\n❌ An unexpected error occurred during training.")
        # It's very helpful to print the full traceback for debugging
        import traceback
        traceback.print_exc()