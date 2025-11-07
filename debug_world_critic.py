"""
Debug script for World Consistency Critic

Run this to diagnose why you're getting score 1.0 on all examples.
"""

import sys
sys.path.append('./world consistency critic/')

from world_consistency_critic import WorldConsistencyCritic

print("="*60)
print("World Consistency Critic Debug Test")
print("="*60)

# Initialize critic
print("\n1. Initializing critic...")
critic = WorldConsistencyCritic(model_name="google/flan-t5-large")
print("✓ Critic initialized")

# Test 1: Consistent response (should score 1.0)
print("\n" + "="*60)
print("TEST 1: Consistent World State (Expected: 1.0)")
print("="*60)

player_1 = "I use the rusty key to unlock the ancient door"
critic.update_world_state(player_1)

dm_1 = "The key turns with a satisfying click. The ancient door swings open, revealing a dark corridor beyond."
result_1 = critic.evaluate_with_explanation(dm_1, player_1, debug=True)

print(f"\n>>> Final Score: {result_1['score']:.2f}")
print(f">>> Reason: {result_1['reason']}")

critic.update_world_state(dm_1)

# Test 2: Contradiction (should score 0.0)
print("\n" + "="*60)
print("TEST 2: Contradiction Detection (Expected: 0.0)")
print("="*60)
print("Previous state: Door was unlocked and opened")

player_2 = "I walk through the open doorway"
critic.update_world_state(player_2)

dm_2 = "You reach for the handle, but the door is locked tight."
result_2 = critic.evaluate_with_explanation(dm_2, player_2, debug=True)

print(f"\n>>> Final Score: {result_2['score']:.2f}")
print(f">>> Reason: {result_2['reason']}")

# Test 3: Hallucination (should score 0.3)
print("\n" + "="*60)
print("TEST 3: Hallucination Detection (Expected: 0.3)")
print("="*60)

critic.reset()  # Start fresh

player_3 = "I enter the empty tavern"
critic.update_world_state(player_3)
print(f"Player said tavern is 'empty'")

dm_3 = "The tavern is bustling with activity. The innkeeper greets you warmly, while the bard plays a lively tune. Three merchants argue in the corner, a mysterious hooded figure watches from the shadows, and two guards play dice."
result_3 = critic.evaluate_with_explanation(dm_3, player_3, debug=True)

print(f"\n>>> Final Score: {result_3['score']:.2f}")
print(f">>> Reason: {result_3['reason']}")

# Summary
print("\n" + "="*60)
print("DIAGNOSIS")
print("="*60)

if result_1['score'] == 1.0 and result_2['score'] == 1.0 and result_3['score'] == 1.0:
    print("\n❌ ISSUE CONFIRMED: All scores are 1.0")
    print("\nPossible causes:")
    print("1. Flan-T5 extraction is not working properly")
    print("2. Model not loading correctly")
    print("3. Parsing logic failing")
    print("\nCheck the [DEBUG] output above to see if entities/objects are being extracted.")
    
elif result_1['score'] != 1.0:
    print("\n❌ ISSUE: Test 1 should be consistent (1.0)")
    print(f"   Got: {result_1['score']}")
    
elif result_2['score'] == 1.0:
    print("\n❌ ISSUE: Test 2 should detect contradiction (0.0)")
    print("   The door was opened but is now locked without explanation")
    
elif result_3['score'] == 1.0:
    print("\n❌ ISSUE: Test 3 should detect hallucination (0.3)")
    print("   6+ NPCs introduced when tavern was described as 'empty'")
    
else:
    print("\n✓ All tests passed correctly!")
    print(f"   Test 1 (consistent): {result_1['score']}")
    print(f"   Test 2 (contradiction): {result_2['score']}")
    print(f"   Test 3 (hallucination): {result_3['score']}")

print("\n" + "="*60)
