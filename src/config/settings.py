"""Configuration file for synthetic flower image generation."""

from typing import Dict, List, Tuple, Optional
import os

class SynthesisConfig:
    """Configuration class for synthetic flower image generation."""
    
    # Image processing constants
    IMAGE_SIZE_MULTIPLIER = 4  # side = max_len * 4
    PADDING_SIZE = 6
    N_SAMPLE_PETAL = 3
    AUGMENTATION_SIGMA = 0.15
    CROWN_SIZE_RATIO = 0.5  # Crown size relative to max_len
    PETAL_CENTER_Y_RATIO = 0.95  # Petal center Y position relative to height
    PETAL_CENTER_X_RATIO = 0.5   # Petal center X position relative to width
    
    # Noise parameters
    ANGLE_NOISE_SIGMA = 10  # degrees
    NOISE_MU = 0
    NOISE_SIGMA = 5
    
    # Processing parameters
    MIN_CPU_COUNT = 40
    NUM_CREATE_PER_BATCH = 12500
    NUM_BATCHES = 8
    NUM_COLORS = 5
    
    # Color definitions
    COLOR_LIST = ['黄色丸', '紫', '白紫', '薄い白緑', '薄黄色']
    
    # Petal arrangement configurations
    # Format: [num_petals, base_angle]
    PETAL_ARRANGEMENTS: Dict[str, List[List[int]]] = {
        'A1': [[4, 144]],
        'A3': [[4, 100]],
        'B1': [[5, 100], [5, 137]],
        'C2': [[6, 137]],
        'D1': [[7, 100], [7, 137]],
        'E2': [[8, 100]],
        'E3': [[8, 137]],
        'F1': [[9, 100], [9, 137]],
        'G1': [[10, 137]],
    }
    
    # Directory paths
    BASE_DATA_DIR = '../data'
    PETALS_DIR_TEMPLATE = f'{BASE_DATA_DIR}/petals/{{color}}/*.png'
    CROWN_DIR_TEMPLATE = f'{BASE_DATA_DIR}/petals/{{color}}/crown/*.png'
    OUTPUT_FLW_DIR_TEMPLATE = f'{BASE_DATA_DIR}/synthetic_flw/flw/{{color}}'
    OUTPUT_MASK_DIR_TEMPLATE = f'{BASE_DATA_DIR}/synthetic_flw/mask/{{color}}'
    
    # Image processing parameters
    FOREGROUND_CROP_PAD = 5
    EDGE_REFINE_KERNEL_SIZE = (3, 3)
    EDGE_REFINE_DILATION_ITERATIONS = 2
    EDGE_REFINE_EROSION_ITERATIONS = 4
    BINARY_THRESHOLD = 10
    MAX_MASK_CHANNELS = 10
    
    @classmethod
    def get_color_by_index(cls, index: int) -> str:
        """Get color name by index."""
        if 0 <= index < len(cls.COLOR_LIST):
            return cls.COLOR_LIST[index]
        raise IndexError(f"Color index {index} out of range")
    
    @classmethod
    def get_petals_path(cls, color: str) -> str:
        """Get petals directory path for given color."""
        return cls.PETALS_DIR_TEMPLATE.format(color=color)
    
    @classmethod
    def get_crown_path(cls, color: str) -> str:
        """Get crown directory path for given color."""
        return cls.CROWN_DIR_TEMPLATE.format(color=color)
    
    @classmethod
    def get_output_flw_dir(cls, color: str) -> str:
        """Get output flower directory for given color."""
        return cls.OUTPUT_FLW_DIR_TEMPLATE.format(color=color)
    
    @classmethod
    def get_output_mask_dir(cls, color: str) -> str:
        """Get output mask directory for given color."""
        return cls.OUTPUT_MASK_DIR_TEMPLATE.format(color=color)
    
    @classmethod
    def use_aestivation_data(cls) -> bool:
        """Check if aestivation data should be used for synthesis."""
        return hasattr(cls, '_use_aestivation') and cls._use_aestivation
    
    @classmethod
    def set_aestivation_mode(cls, enabled: bool) -> None:
        """Enable or disable aestivation data mode."""
        cls._use_aestivation = enabled
    
    @classmethod
    def get_aestivation_path(cls) -> str:
        """Get path to aestivation directory."""
        return getattr(cls, '_aestivation_path', 'aestivation')
    
    @classmethod
    def set_aestivation_path(cls, path: str) -> None:
        """Set path to aestivation directory."""
        cls._aestivation_path = path


class MatchConfig:
    """Configuration for matching algorithms."""
    
    # Flower arrangement patterns (depth levels: 0=back, 1=middle, 2=front)
    ARRANGEMENTS = {
        'a1': [2, 0, 2, 0],
        'a2': [1, 2, 1, 0],
        'a3': [1, 1, 2, 0],
        'b1': [1, 0, 2, 0, 2],
        'c1': [1, 0, 1, 2, 0, 2],
        'c2': [0, 2, 0, 2, 0, 2],
        'c3': [2, 1, 0, 2, 0, 1],
        'd1': [2, 1, 0, 2, 1, 0, 1],
        'd2': [0, 2, 0, 1, 2, 0, 2],
        'e1': [2, 1, 0, 1, 2, 0, 2, 0],
        'e2': [0, 2, 0, 2, 0, 2, 0, 2],
        'e3': [0, 2, 0, 2, 1, 0, 2, 1],
        'f1': [0, 2, 0, 2, 0, 2, 0, 1, 2],
        'g1': [2, 0, 2, 0, 2, 0, 2, 0, 2, 0],
    }
    
    # Score thresholds
    SCORE_THRESHOLD = 0.5
    
    @classmethod
    def get_all_arrangements(cls) -> List[List[int]]:
        """Get all arrangement patterns."""
        return list(cls.ARRANGEMENTS.values())
    
    @classmethod
    def get_concatenated_arrangements(cls) -> List[List[int]]:
        """Get arrangements concatenated with themselves for circular matching."""
        arrangements = cls.get_all_arrangements()
        return [arrange + arrange for arrange in arrangements]


# Utility functions for configuration
def validate_config() -> bool:
    """Validate configuration parameters."""
    config = SynthesisConfig()
    
    # Check if all required directories exist
    import os
    if not os.path.exists(config.BASE_DATA_DIR):
        raise FileNotFoundError(f"Base data directory not found: {config.BASE_DATA_DIR}")
    
    # Validate color list
    if not config.COLOR_LIST:
        raise ValueError("COLOR_LIST cannot be empty")
    
    # Validate arrangement configurations
    if not config.PETAL_ARRANGEMENTS:
        raise ValueError("PETAL_ARRANGEMENTS cannot be empty")
    
    return True