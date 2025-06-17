#!/usr/bin/env python3
"""Main script for generating synthetic flower images."""

import os
import multiprocessing
from concurrent.futures.process import ProcessPoolExecutor
from typing import Tuple, Optional

import cv2

from src.config.settings import SynthesisConfig
from src.core.geometry import SynthesisParameterConfig
from src.core.synthesis import synthesize_single_flower
from src.io.multi_channel import write_img
from src.io.aestivation_reader import AestivationDataReader


def setup_directories(color: str, batch_idx: int) -> Tuple[str, str]:
    """Setup output directories for synthetic images.
    
    Args:
        color: Color name
        batch_idx: Batch index
        
    Returns:
        Tuple of (flower_dir, mask_dir) paths
    """
    synthe_flw_dir = SynthesisConfig.get_output_flw_dir(color)
    synthe_mask_dir = SynthesisConfig.get_output_mask_dir(color)
    
    if batch_idx == 0:
        os.makedirs(synthe_flw_dir, exist_ok=True)
        os.makedirs(synthe_mask_dir, exist_ok=True)
        os.makedirs(f'{synthe_mask_dir}_10c', exist_ok=True)
    
    return synthe_flw_dir, synthe_mask_dir


def main(args: Tuple[int, int, int]) -> None:
    """Main function for generating synthetic flower images.
    
    Args:
        args: Tuple of (color_index, batch_index, num_create)
    """
    color_idx, batch_idx, NUM_CREATE = args
    
    # Get color and setup
    color = SynthesisConfig.get_color_by_index(color_idx)
    
    # Initialize configuration
    config = SynthesisParameterConfig(
        SynthesisConfig.get_petals_path(color),
        SynthesisConfig.get_crown_path(color),
        SynthesisConfig.PETAL_ARRANGEMENTS
    )
    
    # Initialize aestivation reader if enabled
    aestivation_reader = None
    if SynthesisConfig.use_aestivation_data():
        try:
            aestivation_reader = AestivationDataReader(SynthesisConfig.get_aestivation_path())
            if batch_idx == 0:
                print(f"Using aestivation data from {SynthesisConfig.get_aestivation_path()}")
        except Exception as e:
            print(f"Warning: Failed to initialize aestivation reader: {e}")
            print("Falling back to traditional synthesis")
    
    # Setup parameters
    max_len = config.get_max_len()
    side = max_len * SynthesisConfig.IMAGE_SIZE_MULTIPLIER
    
    # Setup directories
    synthe_flw_dir, synthe_mask_dir = setup_directories(color, batch_idx)
    
    print(f'{color_idx=}, {batch_idx=}, {NUM_CREATE=}')
    
    # Generate images
    for j in range(NUM_CREATE):
        # Synthesize single flower
        img_synthetic, agg_synthetic_mask, mask_10c = synthesize_single_flower(
            config, max_len, side, aestivation_reader
        )
        
        # Save images
        idx = str(j + batch_idx * NUM_CREATE).zfill(6)
        print(idx)
        
        try:
            success = cv2.imwrite(f'{synthe_flw_dir}/{idx}.png', img_synthetic)
            if not success:
                print(f"Warning: Failed to save flower image {idx}")
                
            success = cv2.imwrite(f'{synthe_mask_dir}/{idx}.png', agg_synthetic_mask)
            if not success:
                print(f"Warning: Failed to save mask image {idx}")
                
            write_img(mask_10c, f'{synthe_mask_dir}_10c/{idx}.h5')
        except Exception as e:
            print(f"Error saving images for {idx}: {e}")
            continue


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate synthetic flower images')
    parser.add_argument('--use-aestivation', action='store_true', 
                       help='Use aestivation data for petal arrangements')
    parser.add_argument('--aestivation-path', type=str, default='aestivation',
                       help='Path to aestivation directory')
    
    parsed_args = parser.parse_args()
    
    # Configure aestivation mode
    if parsed_args.use_aestivation:
        SynthesisConfig.set_aestivation_mode(True)
        SynthesisConfig.set_aestivation_path(parsed_args.aestivation_path)
        print(f"Aestivation mode enabled with path: {parsed_args.aestivation_path}")
    else:
        print("Using traditional synthesis configuration")
    
    assert multiprocessing.cpu_count() > SynthesisConfig.MIN_CPU_COUNT, \
        f'num_cpu is less than {SynthesisConfig.MIN_CPU_COUNT}'
    
    args = []
    for color_idx in range(SynthesisConfig.NUM_COLORS):
        for batch_idx in range(SynthesisConfig.NUM_BATCHES):
            args.append((color_idx, batch_idx, SynthesisConfig.NUM_CREATE_PER_BATCH))
    
    with ProcessPoolExecutor() as executor:
        executor.map(main, args)