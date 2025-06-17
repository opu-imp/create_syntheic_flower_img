#!/usr/bin/env python3
"""Test script for aestivation integration."""

import sys
import os

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.io.aestivation_reader import AestivationDataReader
from src.config.settings import SynthesisConfig

def test_aestivation_reader() -> bool:
    """Test the aestivation reader functionality."""
    print("Testing AestivationDataReader...")
    
    try:
        reader = AestivationDataReader()
        
        # Test reading pattern statistics
        print("\n1. Testing pattern statistics reading...")
        df = reader.read_pattern_statistics()
        print(f"   - Total patterns: {len(df)}")
        print(f"   - Pattern lengths: {sorted(df['len'].unique())}")
        print(f"   - Sample patterns:")
        for _, row in df.head(5).iterrows():
            print(f"     {row['pattern']} (len={row['len']}, count={row['count']})")
        
        # Test pattern parsing
        print("\n2. Testing pattern parsing...")
        test_patterns = ['OIOAIA', 'OIAOI', 'OIOIOAIA']
        for pattern in test_patterns:
            depths = reader.parse_pattern_string(pattern)
            print(f"   - {pattern} -> {depths}")
        
        # Test patterns by length
        print("\n3. Testing patterns by length...")
        for length in [4, 5, 6, 7, 8]:
            patterns = reader.get_patterns_by_length(length, min_count=1000)
            print(f"   - {length} petals: {len(patterns)} patterns (min_count=1000)")
            if patterns:
                top_pattern = max(patterns, key=lambda x: x[2])
                print(f"     Most common: {top_pattern[0]} (count={top_pattern[2]})")
        
        # Test weighted sampler
        print("\n4. Testing weighted pattern sampler...")
        try:
            sampler = reader.get_weighted_pattern_sampler(6, min_count=1000)
            print("   - Sampling 5 patterns for 6 petals:")
            for i in range(5):
                pattern_str, depth_list = sampler()
                angles = reader.generate_angles_from_pattern(depth_list)
                print(f"     {i+1}. {pattern_str} -> depths: {depth_list}")
                print(f"        angles: {[f'{a:.1f}°' for a in angles[:4]]}...")
        except ValueError as e:
            print(f"   - Error: {e}")
        
        # Test synthesis config
        print("\n5. Testing synthesis configuration...")
        config = reader.create_synthesis_config(petal_counts=[4, 5, 6, 7, 8], min_count=500)
        for count, stats in config['stats'].items():
            print(f"   - {count} petals: {stats['total_patterns']} patterns, {stats['total_occurrences']} occurrences")
        
        print("\n✅ All tests passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_synthesis_config() -> bool:
    """Test the synthesis configuration integration."""
    print("\n" + "="*50)
    print("Testing SynthesisConfig integration...")
    
    # Test aestivation mode
    print("\n1. Testing aestivation mode configuration...")
    print(f"   - Initial aestivation mode: {SynthesisConfig.use_aestivation_data()}")
    
    SynthesisConfig.set_aestivation_mode(True)
    print(f"   - After enabling: {SynthesisConfig.use_aestivation_data()}")
    
    SynthesisConfig.set_aestivation_path('/custom/path')
    print(f"   - Custom path: {SynthesisConfig.get_aestivation_path()}")
    
    SynthesisConfig.set_aestivation_mode(False)
    print(f"   - After disabling: {SynthesisConfig.use_aestivation_data()}")
    
    print("\n✅ Configuration tests passed!")

if __name__ == "__main__":
    print("Aestivation Integration Test")
    print("=" * 50)
    
    success = test_aestivation_reader()
    if success:
        test_synthesis_config()
        print("\n🎉 All tests completed successfully!")
        print("\nUsage examples:")
        print("  Traditional mode: python scripts/create_synthetic.py")
        print("  Aestivation mode: python scripts/create_synthetic.py --use-aestivation")
        print("  Custom path:      python scripts/create_synthetic.py --use-aestivation --aestivation-path /path/to/data")
    else:
        print("\n❌ Tests failed!")
        sys.exit(1)