"""Reader for aestivation simulation output data."""

import os
import csv
from typing import List, Dict, Tuple, Optional
import pandas as pd
import numpy as np


class AestivationDataReader:
    """Reader for aestivation simulation output files."""
    
    def __init__(self, base_path: str = "aestivation"):
        """Initialize the reader with base path to aestivation directory.
        
        Args:
            base_path: Path to aestivation directory
        """
        self.base_path = base_path
        
    def read_pattern_statistics(self, filename: str = "tmp.csv") -> pd.DataFrame:
        """Read pattern statistics from CSV file.
        
        Args:
            filename: Name of the CSV file with pattern statistics
            
        Returns:
            DataFrame with pattern statistics
        """
        filepath = os.path.join(self.base_path, filename)
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Pattern statistics file not found: {filepath}")
            
        return pd.read_csv(filepath)
    
    def parse_pattern_string(self, pattern: str) -> List[int]:
        """Parse pattern string (e.g., 'OIOAIA') to depth levels.
        
        Pattern characters:
        - O: outer (depth 0)
        - I: inner (depth 2) 
        - A: adjacent/middle (depth 1)
        
        Args:
            pattern: Pattern string like 'OIOAIA'
            
        Returns:
            List of depth levels [0, 2, 0, 1, 2, 1]
        """
        depth_map = {'O': 0, 'A': 1, 'I': 2}
        return [depth_map[char] for char in pattern]
    
    def get_patterns_by_length(self, length: int, min_count: int = 100) -> List[Tuple[str, List[int], int]]:
        """Get patterns of specific length with minimum occurrence count.
        
        Args:
            length: Number of petals
            min_count: Minimum occurrence count to include pattern
            
        Returns:
            List of (pattern_string, depth_list, count) tuples
        """
        df = self.read_pattern_statistics()
        filtered = df[(df['len'] == length) & (df['count'] >= min_count)]
        
        patterns = []
        for _, row in filtered.iterrows():
            pattern_str = row['pattern']
            depth_list = self.parse_pattern_string(pattern_str)
            count = row['count']
            patterns.append((pattern_str, depth_list, count))
            
        return patterns
    
    def get_weighted_pattern_sampler(self, length: int, min_count: int = 100) -> callable:
        """Create a weighted random pattern sampler for given length.
        
        Args:
            length: Number of petals
            min_count: Minimum occurrence count to include pattern
            
        Returns:
            Function that samples patterns based on their occurrence frequency
        """
        patterns = self.get_patterns_by_length(length, min_count)
        
        if not patterns:
            raise ValueError(f"No patterns found for length {length} with min_count {min_count}")
        
        pattern_strings = [p[0] for p in patterns]
        depth_lists = [p[1] for p in patterns]
        weights = np.array([p[2] for p in patterns], dtype=float)
        weights = weights / weights.sum()  # Normalize to probabilities
        
        def sampler() -> Tuple[str, List[int]]:
            """Sample a pattern based on occurrence frequency.
            
            Returns:
                Tuple of (pattern_string, depth_list)
            """
            idx = np.random.choice(len(patterns), p=weights)
            return pattern_strings[idx], depth_lists[idx]
        
        return sampler
    
    def generate_angles_from_pattern(self, depth_list: List[int], 
                                   base_angle: float = 137.5,
                                   noise_sigma: float = 10.0) -> List[float]:
        """Generate petal angles from depth pattern using golden angle spiral.
        
        Args:
            depth_list: List of depth levels for each petal
            base_angle: Base divergence angle (golden angle ≈ 137.5°)
            noise_sigma: Standard deviation for angle noise
            
        Returns:
            List of angles in degrees
        """
        n_petals = len(depth_list)
        angles = []
        
        for i in range(n_petals):
            # Golden angle spiral
            angle = (i * base_angle) % 360
            
            # Add noise
            if noise_sigma > 0:
                angle += np.random.normal(0, noise_sigma)
            
            angles.append(angle)
        
        return angles
    
    def create_synthesis_config(self, 
                              petal_counts: List[int] = [4, 5, 6, 7, 8, 9, 10],
                              min_count: int = 100,
                              base_angle: float = 137.5) -> Dict[str, List]:
        """Create synthesis configuration from aestivation data.
        
        Args:
            petal_counts: List of petal counts to include
            min_count: Minimum occurrence count for patterns
            base_angle: Base divergence angle
            
        Returns:
            Configuration dictionary with arrangements and angles
        """
        config = {
            'arrangements': {},
            'samplers': {},
            'stats': {}
        }
        
        for count in petal_counts:
            patterns = self.get_patterns_by_length(count, min_count)
            
            if patterns:
                # Store arrangements in MatchConfig format
                config['arrangements'][f'dynamic_{count}'] = [p[1] for p in patterns]
                
                # Create weighted sampler
                config['samplers'][count] = self.get_weighted_pattern_sampler(count, min_count)
                
                # Store statistics
                config['stats'][count] = {
                    'total_patterns': len(patterns),
                    'total_occurrences': sum(p[2] for p in patterns),
                    'patterns': patterns
                }
        
        return config


def demo_aestivation_reader() -> None:
    """Demonstrate usage of AestivationDataReader."""
    reader = AestivationDataReader()
    
    # Read pattern statistics
    print("Reading pattern statistics...")
    df = reader.read_pattern_statistics()
    print(f"Total patterns: {len(df)}")
    print(f"Pattern lengths: {sorted(df['len'].unique())}")
    
    # Get patterns for 6 petals
    print("\nPatterns for 6 petals (min 1000 occurrences):")
    patterns_6 = reader.get_patterns_by_length(6, min_count=1000)
    for pattern_str, depth_list, count in patterns_6:
        print(f"  {pattern_str}: {depth_list} (count: {count})")
    
    # Create weighted sampler
    print("\nSampling patterns for 6 petals:")
    sampler = reader.get_weighted_pattern_sampler(6, min_count=1000)
    for i in range(5):
        pattern_str, depth_list = sampler()
        angles = reader.generate_angles_from_pattern(depth_list)
        print(f"  Sample {i+1}: {pattern_str} -> depths: {depth_list} -> angles: {[f'{a:.1f}' for a in angles]}")
    
    # Create full synthesis config
    print("\nCreating synthesis configuration...")
    config = reader.create_synthesis_config(petal_counts=[4, 5, 6, 7, 8], min_count=500)
    for count, stats in config['stats'].items():
        print(f"  {count} petals: {stats['total_patterns']} patterns, {stats['total_occurrences']} total occurrences")


if __name__ == "__main__":
    demo_aestivation_reader()