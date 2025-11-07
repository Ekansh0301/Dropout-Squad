"""
Test script to verify character_voice_critic.py implementation.
Run this to check if the implementation works correctly.
"""

import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test that all imports work."""
    print("Testing imports...")
    try:
        from character_voice_critic import (
            CharacterVoiceCritic,
            CharacterProfile,
            CharacterVoiceDataset,
            CharacterVoiceModel
        )
        print("✓ All imports successful")
        return True
    except Exception as e:
        print(f"✗ Import failed: {e}")
        return False


def test_character_profile():
    """Test CharacterProfile class."""
    print("\nTesting CharacterProfile...")
    try:
        from character_voice_critic import CharacterProfile
        
        profile = CharacterProfile(name="Test Character")
        profile.add_dialogue("Hello!", "in tavern")
        profile.add_trait("friendly")
        profile.add_speech_pattern("casual")
        
        assert profile.name == "Test Character"
        assert profile.get_dialogue_count() == 1
        assert "friendly" in profile.traits
        assert "casual" in profile.speech_patterns
        
        print("✓ CharacterProfile works correctly")
        return True
    except Exception as e:
        print(f"✗ CharacterProfile failed: {e}")
        return False


def test_critic_initialization():
    """Test CharacterVoiceCritic initialization."""
    print("\nTesting CharacterVoiceCritic initialization...")
    try:
        from character_voice_critic import CharacterVoiceCritic
        
        critic = CharacterVoiceCritic(
            model_name="microsoft/deberta-v3-base",
            num_characters=10,
            embedding_dim=128
        )
        
        assert critic.num_characters == 10
        assert critic.embedding_dim == 128
        assert len(critic.characters) == 0  # No characters yet
        
        print("✓ CharacterVoiceCritic initializes correctly")
        return True
    except Exception as e:
        print(f"✗ CharacterVoiceCritic initialization failed: {e}")
        return False


def test_data_loading():
    """Test loading CRD3 data."""
    print("\nTesting data loading...")
    try:
        import json
        
        # Check if crd3_npc_dialogues.json exists
        data_path = "../crd3_npc_dialogues.json"
        if not os.path.exists(data_path):
            print(f"⚠ Dataset not found at {data_path}")
            print("  This is OK - dataset should be provided during training")
            return True
        
        # Try loading
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"✓ Found dataset with {len(data)} dialogues")
        
        # Check format
        sample = data[0]
        required_fields = ['character', 'text', 'context']
        for field in required_fields:
            assert field in sample, f"Missing field: {field}"
        
        print("✓ Dataset format is correct")
        return True
        
    except FileNotFoundError:
        print("⚠ Dataset not found (this is OK for testing)")
        return True
    except Exception as e:
        print(f"✗ Data loading failed: {e}")
        return False


def test_training_data_creation():
    """Test training data creation (mock)."""
    print("\nTesting training data creation...")
    try:
        from character_voice_critic import CharacterVoiceCritic
        
        # Create mock data
        mock_dialogues = [
            {"character": "Char1", "text": "Hello!", "context": "tavern", "episode": "E1", "turn_number": 1},
            {"character": "Char1", "text": "How are you?", "context": "tavern", "episode": "E1", "turn_number": 2},
            {"character": "Char1", "text": "Nice day!", "context": "street", "episode": "E1", "turn_number": 3},
            {"character": "Char2", "text": "Greetings!", "context": "tavern", "episode": "E1", "turn_number": 4},
            {"character": "Char2", "text": "Good to see you!", "context": "street", "episode": "E1", "turn_number": 5},
        ] * 5  # Repeat to meet min_dialogues requirement
        
        # Save mock data
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            import json
            json.dump(mock_dialogues, f)
            temp_path = f.name
        
        try:
            critic = CharacterVoiceCritic()
            training_data = critic.build_training_data_from_crd3(
                crd3_dialogue_file=temp_path,
                min_dialogues=5,
                max_characters=2
            )
            
            assert len(training_data) > 0, "No training data created"
            assert critic.num_characters == 2, "Wrong number of characters"
            
            # Check data format
            sample = training_data[0]
            assert 'character' in sample
            assert 'text' in sample
            assert 'context' in sample
            assert 'label' in sample
            assert 'character_id' in sample
            
            # Check labels
            labels = [ex['label'] for ex in training_data]
            assert 1.0 in labels, "No positive examples"
            assert 0.0 in labels, "No negative examples"
            
            print(f"✓ Created {len(training_data)} training examples")
            print(f"  Characters: {critic.num_characters}")
            print(f"  Positive: {sum(1 for l in labels if l == 1.0)}")
            print(f"  Negative: {sum(1 for l in labels if l == 0.0)}")
            
            return True
            
        finally:
            os.unlink(temp_path)
            
    except Exception as e:
        print(f"✗ Training data creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_character_management():
    """Test character profile management."""
    print("\nTesting character management...")
    try:
        from character_voice_critic import CharacterVoiceCritic
        
        critic = CharacterVoiceCritic()
        
        # Add character
        profile = critic.add_character_profile("Test Character")
        assert "Test Character" in critic.characters
        assert "Test Character" in critic.character_to_id
        
        # Get embedding (should return None before model is initialized)
        embedding = critic.get_character_embedding("Test Character")
        assert embedding is None, "Should return None before model init"
        
        print("✓ Character management works correctly")
        return True
        
    except Exception as e:
        print(f"✗ Character management failed: {e}")
        return False


def main():
    """Run all tests."""
    print("=" * 70)
    print("CHARACTER VOICE CRITIC - IMPLEMENTATION TEST")
    print("=" * 70)
    
    tests = [
        ("Imports", test_imports),
        ("CharacterProfile", test_character_profile),
        ("Critic Initialization", test_critic_initialization),
        ("Data Loading", test_data_loading),
        ("Training Data Creation", test_training_data_creation),
        ("Character Management", test_character_management),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n✗ Test '{name}' crashed: {e}")
            results.append((name, False))
    
    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status:8s} - {name}")
    
    passed = sum(1 for _, r in results if r)
    total = len(results)
    
    print("=" * 70)
    print(f"RESULT: {passed}/{total} tests passed")
    print("=" * 70)
    
    if passed == total:
        print("\n🎉 All tests passed! Implementation is ready to use.")
        print("\nNext steps:")
        print("  1. Run training: python character_voice_critic.py --data ../crd3_npc_dialogues.json --output ./model")
        print("  2. Or use the Kaggle notebook for interactive training")
        return 0
    else:
        print("\n⚠ Some tests failed. Please check the errors above.")
        return 1


if __name__ == '__main__':
    exit_code = main()
    exit(exit_code)
