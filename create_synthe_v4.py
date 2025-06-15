import os
import glob
import math
import random
from typing import List, Tuple

import cv2
import numpy as np
import multiprocessing
from concurrent.futures.process import ProcessPoolExecutor

from multi_channel_img_io import write_img
from config import SynthesisConfig


def noise(mu: float = SynthesisConfig.NOISE_MU, sigma: float = SynthesisConfig.NOISE_SIGMA) -> float:
    """Generate Gaussian noise.
    
    Args:
        mu: Mean value
        sigma: Standard deviation
        
    Returns:
        Random noise value
    """
    return random.gauss(mu, sigma)

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


def rotate_image(image, marked_center, angle, pad):
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
    T = np.float32([[1, 0, square_center[1] - square_marked_center[1]], [0, 1, square_center[0] - square_marked_center[0] - pad]])

    # Shift the image so that marked_center is now at the center of the image
    shifted_image = cv2.warpAffine(square_image, T, (side, side))

    # Define the rotation matrix
    # Here we adjust the center of rotation to be the center of shifted_image
    M = cv2.getRotationMatrix2D((square_center[1], square_center[0]), np.degrees(angle), 1.0)

    # Apply the rotation to the shifted image
    rotated_image = cv2.warpAffine(shifted_image, M, (side, side))
    
    return refine_edge(rotated_image)

def put_crown(img_synthetic, img_crown):
    
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
    
<<<<<<< Updated upstream
    config = SynthesisParameterConfig(
        f'./work/data/petals/{color}/*.png',
        f'./work/data/petals/{color}/crown/*.png',
        dic_pairs
    )

    max_len = config.get_max_len()
    side = max_len * 4
    padding_size = 6
    n_sample_petal = 3
    NUM_CREATE = 2
    sigma = 0.15

    synthe_flw_dir = f'./work/data/synthetic_flw/flw/{color}'
    synthe_mask_dir = f'./work/data/synthetic_flw/mask/{color}'
=======
>>>>>>> Stashed changes
    if batch_idx == 0:
        os.makedirs(synthe_flw_dir, exist_ok=True)
        os.makedirs(synthe_mask_dir, exist_ok=True)
        os.makedirs(f'{synthe_mask_dir}_10c', exist_ok=True)
    
    return synthe_flw_dir, synthe_mask_dir


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


def synthesize_single_flower(config: SynthesisParameterConfig, 
                           max_len: int, 
                           side: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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
            config, max_len, side
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
    assert multiprocessing.cpu_count() > SynthesisConfig.MIN_CPU_COUNT, f'num_cpu is less than {SynthesisConfig.MIN_CPU_COUNT}'
    
    args = []
    for color_idx in range(SynthesisConfig.NUM_COLORS):
        for batch_idx in range(SynthesisConfig.NUM_BATCHES):
            args.append((color_idx, batch_idx, SynthesisConfig.NUM_CREATE_PER_BATCH))
    
    with ProcessPoolExecutor() as executor:
        executor.map(main, args)
