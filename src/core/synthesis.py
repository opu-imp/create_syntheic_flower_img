"""Core synthesis functionality for creating synthetic flower images."""

import random
import math
from typing import List, Tuple, Optional

import cv2
import numpy as np

from ..config.settings import SynthesisConfig
from ..io.multi_channel import write_img
from ..io.aestivation_reader import AestivationDataReader
from .image_processing import crop_foreground, refine_edge
from .geometry import rotate_image, put_crown


def noise(mu: float = SynthesisConfig.NOISE_MU, sigma: float = SynthesisConfig.NOISE_SIGMA) -> float:
    """Generate Gaussian noise.
    
    Args:
        mu: Mean value
        sigma: Standard deviation
        
    Returns:
        Random noise value
    """
    return random.gauss(mu, sigma)


def apply_petal_augmentation(petal: np.ndarray, sigma: float = SynthesisConfig.AUGMENTATION_SIGMA) -> np.ndarray:
    """Apply augmentation to a petal image.
    
    Args:
        petal: Input petal image
        sigma: Standard deviation for scale augmentation
        
    Returns:
        Augmented petal image
    """
    petal_aug = petal.copy()
    
    # Random horizontal flip
    if np.random.randint(2):
        petal_aug = cv2.flip(petal_aug, 1)
    
    # Random scale
    fx = random.gauss(1, sigma)
    fy = random.gauss(1, sigma)
    petal_aug = cv2.resize(petal_aug, dsize=None, fx=fx, fy=fy)
    
    return petal_aug


def place_petal_on_synthetic(img_synthetic: np.ndarray, 
                           synthetic_mask: np.ndarray,
                           agg_synthetic_mask: np.ndarray,
                           petal_result: np.ndarray,
                           petal_index: int,
                           side: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Place a single petal on the synthetic image and update masks.
    
    Args:
        img_synthetic: Main synthetic image
        synthetic_mask: Individual petal mask
        agg_synthetic_mask: Aggregated mask with petal indices
        petal_result: Processed petal image to place
        petal_index: Index of the petal (0-based)
        side: Side length of the synthetic image
        
    Returns:
        Updated (img_synthetic, synthetic_mask, agg_synthetic_mask)
    """
    # Define mask for foreground
    mask_new = np.any(petal_result != [0, 0, 0], axis=-1)
    
    # Calculate placement position
    synthetic_center = [side // 2, side // 2]
    square_center = [petal_result.shape[0] // 2, petal_result.shape[1] // 2]
    top_left = [synthetic_center[0] - square_center[0], synthetic_center[1] - square_center[1]]
    
    # Update synthetic image
    y_end = top_left[0] + petal_result.shape[0]
    x_end = top_left[1] + petal_result.shape[1]
    img_synthetic[top_left[0]:y_end, top_left[1]:x_end][mask_new] = petal_result[mask_new]
    
    # Update masks
    synthetic_mask[top_left[0]:y_end, top_left[1]:x_end][mask_new] = 1
    agg_synthetic_mask[top_left[0]:y_end, top_left[1]:x_end][mask_new] = petal_index + 1
    
    return img_synthetic, synthetic_mask, agg_synthetic_mask


def get_petal_arrangement_aestivation(aestivation_reader: AestivationDataReader, 
                                    petal_counts: List[int] = [4, 5, 6, 7, 8, 9, 10]) -> Tuple[List[float], List[int]]:
    """Get petal arrangement from aestivation data.
    
    Args:
        aestivation_reader: Reader for aestivation data
        petal_counts: Preferred petal counts
        
    Returns:
        Tuple of (angles, depth_levels)
    """
    # Select number of petals randomly from available counts
    available_counts = []
    for count in petal_counts:
        try:
            patterns = aestivation_reader.get_patterns_by_length(count, min_count=100)
            if patterns:
                available_counts.append(count)
        except:
            continue
    
    if not available_counts:
        # Fallback to default arrangement
        return [0, 90, 180, 270], [0, 1, 2, 0]
    
    # Select petal count
    petal_count = random.choice(available_counts)
    
    # Get weighted sampler and sample pattern
    sampler = aestivation_reader.get_weighted_pattern_sampler(petal_count, min_count=100)
    pattern_str, depth_list = sampler()
    
    # Generate angles from pattern
    angles = aestivation_reader.generate_angles_from_pattern(
        depth_list, 
        base_angle=137.5,  # Golden angle
        noise_sigma=SynthesisConfig.ANGLE_NOISE_SIGMA
    )
    
    return angles, depth_list


def synthesize_single_flower(config: SynthesisConfig, max_len: int, side: int, 
                           aestivation_reader: Optional[AestivationDataReader] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Synthesize a single flower image with masks.
    
    Args:
        config: Synthesis parameter configuration
        max_len: Maximum dimension of petals
        side: Side length of output image
        aestivation_reader: Optional reader for aestivation data
        
    Returns:
        Tuple of (synthetic_image, aggregated_mask, multi_channel_mask)
    """
    # Initialize images
    img_synthetic = np.zeros((side, side, 3), dtype=np.uint8)
    agg_synthetic_mask = np.zeros((side, side), dtype=np.uint8)
    synthetic_masks = []
    
    # Select crown and resize
    img_crown = random.choice(config.crowns)
    crown_size = int(max_len * SynthesisConfig.CROWN_SIZE_RATIO)
    img_crown = cv2.resize(img_crown, (crown_size, crown_size))
    
    # Select angles and depth levels
    if SynthesisConfig.use_aestivation_data() and aestivation_reader:
        angles, depth_levels = get_petal_arrangement_aestivation(aestivation_reader)
    else:
        # Use traditional configuration
        angles = random.choice(config.angles_list)
        depth_levels = None
    
    # Select petals
    petals = random.choices(config.petals, k=SynthesisConfig.N_SAMPLE_PETAL)
    
    # Sort petals by depth if using aestivation data
    if depth_levels:
        # Create list of (angle, depth, index) and sort by depth (back to front: 0, 1, 2)
        petal_info = [(angles[i], depth_levels[i], i) for i in range(len(angles))]
        petal_info.sort(key=lambda x: x[1])  # Sort by depth: 0=back, 1=middle, 2=front
        angles = [info[0] for info in petal_info]
        processing_order = [info[2] for info in petal_info]
    else:
        processing_order = list(range(len(angles)))
    
    # Process each petal in depth order
    for idx, angle in enumerate(angles):
        synthetic_mask = np.zeros((side, side), dtype=np.uint8)
        
        # Select and augment petal
        petal = random.choice(petals)
        petal_aug = apply_petal_augmentation(petal)
        
        # Calculate petal center
        h, w = petal.shape[:2]
        petal_img_center = [
            int(h * SynthesisConfig.PETAL_CENTER_Y_RATIO), 
            int(w * SynthesisConfig.PETAL_CENTER_X_RATIO)
        ]
        
        # Add angle noise and rotate
        if not (SynthesisConfig.use_aestivation_data() and aestivation_reader):
            # Only add noise if not using aestivation (which already includes noise)
            angle += noise(mu=0, sigma=SynthesisConfig.ANGLE_NOISE_SIGMA)
        angle_radian = math.radians(angle)
        result = rotate_image(petal, petal_img_center, angle_radian, SynthesisConfig.PADDING_SIZE)
        
        # Use original index for mask labeling
        original_idx = processing_order[idx] if depth_levels else idx
        
        # Place petal on synthetic image
        img_synthetic, synthetic_mask, agg_synthetic_mask = place_petal_on_synthetic(
            img_synthetic, synthetic_mask, agg_synthetic_mask, result, original_idx, side
        )
        
        synthetic_masks.append(synthetic_mask)
    
    # Create multi-channel mask
    while len(synthetic_masks) < SynthesisConfig.MAX_MASK_CHANNELS:
        synthetic_masks.append(np.zeros((side, side), dtype=np.uint8))
    mask_10c = np.stack(synthetic_masks[:SynthesisConfig.MAX_MASK_CHANNELS], -1)
    
    # Add crown
    img_synthetic = put_crown(img_synthetic, img_crown)
    
    return img_synthetic, agg_synthetic_mask, mask_10c