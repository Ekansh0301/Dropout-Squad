"""
Quick test to verify World Consistency Critic fixes work correctly.
Run this to see if the contradiction/hallucination/amnesia detection is working.
"""

import sys
sys.path.insert(0, 'world consistency critic')

from world_consistency_critic import WorldConsistencyCritic

def test_critic():
    print("=" * 100)
    print("TESTING WORLD CONSISTENCY CRITIC FIXES")
    print("=" * 100)
    
    critic = WorldConsistencyCritic()
    
    # Test 0: AttributeError fix (None state/location handling)
    print("\n" + "─" * 100)
    print("TEST 0: AttributeError Fix (None State Handling)")
    print("─" * 100)
    try:
        critic.reset()
        critic.update_world_state("I use the rusty key to unlock the ancient door")
        score = critic.score("The key turns with a satisfying click. The ancient door swings open, revealing a dark corridor beyond.")
        print(f"  Score: {score:.2f}")
        print(f"  Result: ✓ PASS (No AttributeError)")
    except AttributeError as e:
        print(f"  Result: ✗ FAIL (AttributeError: {e})")
    
    # Test 1: Torch contradiction (lit in hand → unlit on ground)
    print("\n" + "─" * 100)
    print("TEST 1: Torch Contradiction")
    print("─" * 100)
    critic.reset()
    critic.update_world_state("I light the torch")
    critic.update_world_state("You ignite the torch. It burns brightly, illuminating the dark corridor.")
    critic.update_world_state("I examine the torch in my hand")
    score = critic.score("The torch lies on the ground, unlit and covered in dust.", debug=True)
    print(f"\n  Score: {score:.2f}")
    print(f"  Expected: ≤0.3 (contradiction)")
    print(f"  Result: {'✓ PASS' if score <= 0.3 else '✗ FAIL'}")
    
    # Test 2: Chalice contradiction (in bag → on pedestal)
    print("\n" + "─" * 100)
    print("TEST 2: Chalice Location Contradiction")
    print("─" * 100)
    critic.reset()
    critic.update_world_state("I take the golden chalice")
    critic.update_world_state("You pick up the ornate golden chalice and place it in your bag.")
    critic.update_world_state("I look around")
    score = critic.score("The golden chalice remains on the pedestal, gleaming in the torchlight.", debug=True)
    print(f"\n  Score: {score:.2f}")
    print(f"  Expected: ≤0.3 (contradiction)")
    print(f"  Result: {'✓ PASS' if score <= 0.3 else '✗ FAIL'}")
    
    # Test 3: Guard name contradiction (Aldric → Brennan)
    print("\n" + "─" * 100)
    print("TEST 3: NPC Name Contradiction")
    print("─" * 100)
    critic.reset()
    critic.update_world_state("I ask the guard his name")
    critic.update_world_state("The guard responds, 'I am Sir Aldric, captain of the watch.'")
    critic.update_world_state("I greet the guard")
    score = critic.score("The guard, Sir Brennan, nods in acknowledgment.", debug=True)
    print(f"\n  Score: {score:.2f}")
    print(f"  Expected: ≤0.3 (contradiction)")
    print(f"  Result: {'✓ PASS' if score <= 0.3 else '✗ FAIL'}")
    
    # Test 4: Hallucination (empty chamber with 10+ entities)
    print("\n" + "─" * 100)
    print("TEST 4: Hallucination in Empty Chamber")
    print("─" * 100)
    critic.reset()
    critic.update_world_state("I enter the empty chamber")
    score = critic.score("The chamber bustles with activity: five merchants haggle loudly, three guards patrol, a bard plays music, two servants clean, and a mysterious hooded figure lurks in the corner.", debug=True)
    print(f"\n  Score: {score:.2f}")
    print(f"  Expected: ≤0.4 (hallucination)")
    print(f"  Result: {'✓ PASS' if score <= 0.4 else '✗ FAIL'}")
    
    # Test 5: Amnesia (forgotten amulet)
    print("\n" + "─" * 100)
    print("TEST 5: Inventory Amnesia")
    print("─" * 100)
    critic.reset()
    critic.update_world_state("I pick up the ruby amulet")
    critic.update_world_state("You take the glowing ruby amulet and wear it around your neck.")
    critic.update_world_state("I check my equipment")
    score = critic.score("You have: a sword, a shield, and a backpack. Nothing else.", debug=True)
    print(f"\n  Score: {score:.2f}")
    print(f"  Expected: ≤0.5 (amnesia)")
    print(f"  Result: {'✓ PASS' if score <= 0.5 else '✗ FAIL'}")
    
    # Test 6: Consistent (should score high)
    print("\n" + "─" * 100)
    print("TEST 6: Consistent State (Control)")
    print("─" * 100)
    critic.reset()
    critic.update_world_state("I draw my sword")
    critic.update_world_state("You unsheathe your blade. It gleams in the light.")
    critic.update_world_state("I ready my weapon")
    score = critic.score("You hold your drawn sword at the ready, prepared for combat.", debug=True)
    print(f"\n  Score: {score:.2f}")
    print(f"  Expected: ≥0.8 (consistent)")
    print(f"  Result: {'✓ PASS' if score >= 0.8 else '✗ FAIL'}")
    
    print("\n" + "=" * 100)
    print("TESTING COMPLETE")
    print("=" * 100)

if __name__ == "__main__":
    test_critic()
