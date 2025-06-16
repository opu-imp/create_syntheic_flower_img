"""Image processing functions for synthetic flower generation."""

from typing import List

import cv2
import numpy as np

from ..config.settings import SynthesisConfig


def crop_foreground(image: np.ndarray, pad: int = SynthesisConfig.FOREGROUND_CROP_PAD) -> np.ndarray:
    """Crop foreground object from image with padding.
    
    Args:
        image: Input image
        pad: Padding around the bounding box
        
    Returns:
        Cropped image
        
    Raises:
        ValueError: If no contours found or image is invalid
    """
    if image is None or image.size == 0:
        raise ValueError("Invalid input image")
    
    try:
        grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(grayscale, SynthesisConfig.BINARY_THRESHOLD, 255, cv2.THRESH_BINARY)

        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            raise ValueError("No contours found in image")

        x, y, w, h = cv2.boundingRect(contours[0])

        # Add padding to the bounding box
        x = max(0, x - pad)
        y = max(0, y - pad)
        w = min(image.shape[1] - x, w + 2*pad)
        h = min(image.shape[0] - y, h + 2*pad)

        # Crop the image using the bounding box
        cropped = image[y:y+h, x:x+w]
        
        if cropped.size == 0:
            raise ValueError("Cropped image is empty")

        return cropped
    except Exception as e:
        raise RuntimeError(f"Error cropping foreground: {e}")


def refine_edge(img_rgb: np.ndarray) -> np.ndarray:
    """Refine edges of the image using morphological operations.
    
    Args:
        img_rgb: Input RGB image
        
    Returns:
        Image with refined edges
        
    Raises:
        ValueError: If input image is invalid
    """
    if img_rgb is None or img_rgb.size == 0:
        raise ValueError("Invalid input image")
    
    try:
        kernel = np.ones(SynthesisConfig.EDGE_REFINE_KERNEL_SIZE, np.uint8)
        squared_distances = np.sum(np.square(img_rgb), axis=2)

        mask = np.where(squared_distances == 0, 0, 255)
        dilation = cv2.dilate(
            mask.astype(np.uint8), 
            kernel, 
            iterations=SynthesisConfig.EDGE_REFINE_DILATION_ITERATIONS
        )
        erosion = cv2.erode(
            dilation.astype(np.uint8), 
            kernel, 
            iterations=SynthesisConfig.EDGE_REFINE_EROSION_ITERATIONS
        )
        refined_img = cv2.bitwise_and(img_rgb, img_rgb, mask=erosion)
        
        return refined_img
    except Exception as e:
        raise RuntimeError(f"Error refining edges: {e}")