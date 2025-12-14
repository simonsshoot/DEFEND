"""
Test script to verify R-Judge data loading and basic functionality
"""

import json
from pathlib import Path

def test_load_rjudge():
    """Test loading R-Judge dataset"""
    dataset_root = Path("../../data/R-Judge")
    
    subdirs = ["Application", "Finance", "IoT", "Program", "Web"]
    
    total_samples = 0
    for subdir in subdirs:
        subdir_path = dataset_root / subdir
        
        harmful_path = subdir_path / "harmful.json"
        benign_path = subdir_path / "benign.json"
        
        harmful_count = 0
        benign_count = 0
        
        if harmful_path.exists():
            with open(harmful_path, "r", encoding="utf-8") as f:
                harmful_data = json.load(f)
                harmful_count = len(harmful_data)
        
        if benign_path.exists():
            with open(benign_path, "r", encoding="utf-8") as f:
                benign_data = json.load(f)
                benign_count = len(benign_data)
        
        print(f"{subdir}: {harmful_count} harmful, {benign_count} benign")
        total_samples += harmful_count + benign_count
    
    print(f"\nTotal samples: {total_samples}")
    
    # Test loading one sample
    print("\n" + "="*80)
    print("Sample from Application/harmful.json:")
    with open(dataset_root / "Application" / "harmful.json", "r", encoding="utf-8") as f:
        data = json.load(f)
        sample = data[0]
        print(f"ID: {sample['id']}")
        print(f"Scenario: {sample['scenario']}")
        print(f"Label: {sample['label']}")
        print(f"Goal: {sample['goal']}")
        print(f"Number of dialogue turns: {len(sample['contents'])}")
        print(f"\nFirst turn:")
        for turn in sample['contents'][0]:
            print(f"  Role: {turn['role']}")
            if turn.get('content'):
                print(f"  Content: {turn['content'][:100]}...")

if __name__ == "__main__":
    test_load_rjudge()
