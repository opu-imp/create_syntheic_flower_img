"""Geometric transformation functions for petal arrangement."""

from typing import List, Tuple
import glob

import cv2
import numpy as np

from ..config.settings import SynthesisConfig
from .image_processing import crop_foreground, refine_edge


def get_raw_angles(num_petals: int, base_angle: int = 137) -> List[int]:
    """花弁の発生角を模した螺旋構造になるような角度配列を取得
    
    Args:
        num_petals: Number of petals
        base_angle: Base angle for spiral structure
        
    Returns:
        List of raw angles
    """
    return [base_angle * i for i in range(num_petals)]


def _sorted_divider_pairs(raw_angles: List[int]) -> List[Tuple[int, int]]:
    """実際の発生ロジックを満たす螺旋構造で得るための関数
    
    Args:
        raw_angles: List of raw angles
        
    Returns:
        Sorted list of (angle_mod_360, rotation_count) pairs
    """
    return sorted([(angle % 360, angle // 360) for angle in raw_angles])


def get_ideal_angles(num_petals: int, base_angle: int = 137) -> List[int]:
    """花弁が等間隔に並ぶような理想的な角度配列を取得
    
    花弁の発生角を模した螺旋構造になるように計算する。
    
    Args:
        num_petals: Number of petals
        base_angle: Base angle for spiral structure (default: 137 - golden angle)
        
    Returns:
        Sorted list of ideal angles for petal placement
    """
    raw_angles = get_raw_angles(num_petals, base_angle)
    sorted_angle_rank_pairs = _sorted_divider_pairs(raw_angles)
    remainder = [360 * divider for _, divider in sorted_angle_rank_pairs]
    unit_angle = 360 / num_petals
    
    return sorted([int(unit_angle * i + r) for i, r in enumerate(remainder)])


def rotate_image(image: np.ndarray, marked_center: List[int], angle: float, pad: int) -> np.ndarray:
    """Rotate image around a specific center point.
    
    Args:
        image: Input image
        marked_center: Center point for rotation [y, x]
        angle: Rotation angle in radians
        pad: Padding to apply
        
    Returns:
        Rotated image
    """
    h, w, c = image.shape

    # Create a black square image with the same dimensions as the original image
    side = max(h, w) * 4
    square_image = np.zeros((side, side, c), dtype=np.uint8)

    # Calculate the offset for pasting the original image onto the square image
    offset_x = (side - w) // 2
    offset_y = (side - h) // 2

    # Paste the original image onto the square image
    square_image[offset_y:offset_y + h, offset_x:offset_x + w] = image

    # Calculate the position of marked_center in square_image
    square_marked_center = [marked_center[0] + offset_y, marked_center[1] + offset_x]

    # Calculate the position of the center of square_image
    square_center = [side // 2, side // 2]

    # Define the translation matrix for shifting the image
    T = np.float32([[1, 0, square_center[1] - square_marked_center[1]], 
                    [0, 1, square_center[0] - square_marked_center[0] - pad]])

    # Shift the image so that marked_center is now at the center of the image
    shifted_image = cv2.warpAffine(square_image, T, (side, side))

    # Define the rotation matrix
    # Here we adjust the center of rotation to be the center of shifted_image
    M = cv2.getRotationMatrix2D((square_center[1], square_center[0]), np.degrees(angle), 1.0)

    # Apply the rotation to the shifted image
    rotated_image = cv2.warpAffine(shifted_image, M, (side, side))
    
    return refine_edge(rotated_image)


def put_crown(img_synthetic: np.ndarray, img_crown: np.ndarray) -> np.ndarray:
    """Place crown image at the center of synthetic image.
    
    Args:
        img_synthetic: Synthetic flower image
        img_crown: Crown image to place
        
    Returns:
        Synthetic image with crown
    """
    # Calculate the shape of img_add and img_synthetic
    h_crown, w_crown, c_crown = img_crown.shape
    h_syn, w_syn, c_syn = img_synthetic.shape
    
    # Calculate the offset for pasting img_add onto img_synthetic
    offset_y = (h_syn - h_crown) // 2
    offset_x = (w_syn - w_crown) // 2
    
    # Create a mask for the foreground of img_add
    mask_crown = np.any(img_crown != [0, 0, 0], axis=-1)
    
    # Paste img_add onto img_synthetic using the mask
    img_synthetic[offset_y:offset_y + h_crown, offset_x:offset_x + w_crown][mask_crown] = img_crown[mask_crown]
    return img_synthetic


class SynthesisParameterConfig:
    """
    Synthesis Parameter Config class for generating synthetic data.

    Attributes:
        petals (list): List of images from the directory specified in img_petals.
        crowns (list): List of images from the directory specified in img_crowns.
        angles_list (list): List of angle_list.
    """
    def __init__(self, path_petals: str, path_crowns: str, dict_pairs: dict):
        self.petals = self._get_imgs(path_petals)
        self.crowns = self._get_imgs(path_crowns)
        self.angles_list = self._get_angles_list(dict_pairs)

        self._standardize_petals()

    def _get_imgs(self, path_dir: str) -> List[np.ndarray]:
        """Load and process images from directory.
        
        Args:
            path_dir: Path pattern for images
            
        Returns:
            List of processed images
            
        Raises:
            FileNotFoundError: If no images found or image loading fails
        """
        paths = glob.glob(path_dir)
        if not paths:
            raise FileNotFoundError(f"No images found at {path_dir}")
        
        imgs = []
        for path in paths:
            img = cv2.imread(path)
            if img is None:
                print(f"Warning: Could not load image {path}")
                continue
            imgs.append(img)
        
        if not imgs:
            raise FileNotFoundError(f"No valid images could be loaded from {path_dir}")
        
        try:
            cropped_imgs = [crop_foreground(img) for img in imgs]
            refined_imgs = [refine_edge(img) for img in cropped_imgs]
            return refined_imgs
        except Exception as e:
            raise RuntimeError(f"Error processing images from {path_dir}: {e}")

    def get_max_len(self):
        # Get dimensions of all images
        dimensions = [img.shape for img in self.petals]

        # Separate widths and heights
        widths = [dim[1] for dim in dimensions]
        heights = [dim[0] for dim in dimensions]

        # Find maximum width and height
        max_width = max(widths)
        max_height = max(heights)

        return max(max_width, max_height)

    def _standardize_petals(self) -> None:
        """Standardize petal images to consistent size."""
        max_len = self.get_max_len()
        resized_imgs = []
        for img in self.petals:
            h, w = img.shape[:2]
            if h > w:
                new_size = (int(w * max_len / h), max_len)
            else:
                new_size = (max_len, int(h * max_len / w))
    
            resized_img = cv2.resize(img, new_size)
            resized_imgs.append(resized_img)
    
        self.petals = resized_imgs

    def _get_angles_list(self, dict_pairs: dict) -> List[List[int]]:
        """Generate list of angle arrangements from petal configurations.
        
        Args:
            dict_pairs: Dictionary of petal arrangement configurations
            
        Returns:
            List of angle arrangements
        """
        all_pairs = [pair for pairs in dict_pairs.values() for pair in pairs]
        angles = [get_ideal_angles(*pair) for pair in all_pairs]
        return angles