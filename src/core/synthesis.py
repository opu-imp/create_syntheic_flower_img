"""Core synthesis functionality for creating synthetic flower images."""

import random
import math
from typing import List, Tuple

import cv2
import numpy as np

from ..config.settings import SynthesisConfig
from ..io.multi_channel import write_img
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


def synthesize_single_flower(config, max_len: int, side: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Synthesize a single flower image with masks.
    
    Args:
        config: Synthesis parameter configuration
        max_len: Maximum dimension of petals
        side: Side length of output image
        
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
    
    # Select angles and petals
    angles = random.choice(config.angles_list)
    petals = random.choices(config.petals, k=SynthesisConfig.N_SAMPLE_PETAL)
    
    # Process each petal
    for i, angle in enumerate(angles):
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
        angle += noise(mu=0, sigma=SynthesisConfig.ANGLE_NOISE_SIGMA)
        angle_radian = math.radians(angle)
        result = rotate_image(petal, petal_img_center, angle_radian, SynthesisConfig.PADDING_SIZE)
        
        # Place petal on synthetic image
        img_synthetic, synthetic_mask, agg_synthetic_mask = place_petal_on_synthetic(
            img_synthetic, synthetic_mask, agg_synthetic_mask, result, i, side
        )
        
        synthetic_masks.append(synthetic_mask)
    
    # Create multi-channel mask
    while len(synthetic_masks) < SynthesisConfig.MAX_MASK_CHANNELS:
        synthetic_masks.append(np.zeros((side, side), dtype=np.uint8))
    mask_10c = np.stack(synthetic_masks[:SynthesisConfig.MAX_MASK_CHANNELS], -1)
    
    # Add crown
    img_synthetic = put_crown(img_synthetic, img_crown)
    
    return img_synthetic, agg_synthetic_mask, mask_10c