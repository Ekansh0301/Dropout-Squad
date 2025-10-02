"""
Standalone evaluation for trained critic models.
Tests narrative and causal critics against curated examples to verify performance.
Generates markdown report for project documentation.
"""
import torch
from pathlib import Path

# Import custom critic implementations
from critics import NarrativeCritic, CausalCritic

def test_narrative_critic(critic, output_dir):
    """Evaluate narrative critic on curated test examples."""
    print("\n--- Evaluating Narrative Critic ---")
    
    test_cases = {
        "High Quality (Expected Score: High)": [
            "The ancient library was a labyrinth of forgotten knowledge, its shelves groaning under the weight of countless dusty tomes.",
            "Sunlight streamed through the stained-glass windows, painting the stone floor in a mosaic of vibrant colors as a lone knight knelt before the altar."
        ],
        "Low Quality (Expected Score: Low)": [
            "You see a room. There is a table. There is a chair.",
            "go north get key open door go east"
        ],
        "Repetitive (Expected Score: Low)": [
            "The dragon roars. The dragon is big. The dragon roars again because the dragon is angry. The dragon roars.",
            "I hit the goblin. I hit the goblin. I hit the goblin."
        ]
    }

    report = "## Narrative Critic Evaluation\n\n"
    report += "| Category | Expected Score | Actual Score | Text Snippet |\n"
    report += "|---|---|---|---|\n"

    for category, examples in test_cases.items():
        scores = critic.get_reward(examples).cpu().numpy()
        for i, text in enumerate(examples):
            score = scores[i]
            report += f"| {category} | High/Low | **{score:.3f}** | `{text[:50]}...` |\n"
    
    print("✓ Narrative Critic evaluation complete.")
    return report

def test_causal_critic(critic, output_dir):
    """Evaluate causal critic on premise-hypothesis pairs."""
    print("\n--- Evaluating Causal Critic ---")
    
    test_cases = [
        ("I cast Fireball at the goblin horde.", "The goblins scatter as flames engulf them.", "High"),
        ("I search the room for secret doors.", "Rolling a 19 on perception, you notice a faint crack in the wall.", "High"),
        ("I cast Fireball at the goblin horde.", "You find a healing potion in the treasure chest.", "Low"),
        ("I search the room for secret doors.", "The dragon breathes fire at you.", "Low"),
    ]
    
    report = "\n## Causal Critic Evaluation\n\n"
    report += "| Expected | Actual Score | Premise (Player) | Hypothesis (DM) |\n"
    report += "|---|---|---|---|\n"

    for premise, hypothesis, expected in test_cases:
        score = critic.get_reward([premise], [hypothesis]).cpu().numpy()[0]
        report += f"| {expected} | **{score:.3f}** | `{premise}` | `{hypothesis[:50]}...` |\n"

    print("✓ Causal Critic evaluation complete.")
    return report


def main():
    """Execute critic evaluation and generate report."""
    # Configuration paths matching ppo_config.yaml
    NARRATIVE_CRITIC_PATH = "critic/models/narrative_critic"
    CAUSAL_CRITIC_MODEL_NAME = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
    OUTPUT_DIR = Path("evaluation_results")
    
    # Setup output directory and device
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"--- Running Critic Evaluation on {device} ---")

    # Load and test critics
    try:
        narrative_critic = NarrativeCritic(NARRATIVE_CRITIC_PATH, device)
        narrative_report = test_narrative_critic(narrative_critic, OUTPUT_DIR)
    except Exception as e:
        narrative_report = f"## Narrative Critic Evaluation FAILED\n\nCould not load model from `{NARRATIVE_CRITIC_PATH}`.\nError: {e}"
        print(f"❌ {narrative_report}")

    try:
        causal_critic = CausalCritic(CAUSAL_CRITIC_MODEL_NAME, device)
        causal_report = test_causal_critic(causal_critic, OUTPUT_DIR)
    except Exception as e:
        causal_report = f"## Causal Critic Evaluation FAILED\n\nCould not load model from Hugging Face Hub: `{CAUSAL_CRITIC_MODEL_NAME}`.\nError: {e}"
        print(f"❌ {causal_report}")

    # Generate and save evaluation report
    final_report = f"# Critic Evaluation Report\n\nThis document verifies the behavior of the individual critic models.\n\n{narrative_report}{causal_report}"
    report_path = OUTPUT_DIR / "critic_evaluation_report.md"
    with open(report_path, 'w') as f:
        f.write(final_report)
    
    print(f"\n--- ✅ Critic Evaluation Complete! ---")
    print(f"Full report saved to: {report_path}")

if __name__ == "__main__":
    main()