"""Tests for synthesis functionality."""

import pytest
import numpy as np
from unittest.mock import Mock, patch

from src.core.synthesis import noise, apply_petal_augmentation
from src.config.settings import SynthesisConfig


class TestSynthesis:
    """Test synthesis functions."""
    
    def test_noise_default_parameters(self):
        """Test noise generation with default parameters."""
        result = noise()
        assert isinstance(result, float)
        
    def test_noise_custom_parameters(self):
        """Test noise generation with custom parameters."""
        result = noise(mu=10, sigma=2)
        assert isinstance(result, float)
        
    def test_apply_petal_augmentation(self):
        """Test petal augmentation."""
        # Create a simple test image
        test_image = np.ones((100, 100, 3), dtype=np.uint8) * 128
        
        with patch('random.gauss', return_value=1.1), \
             patch('np.random.randint', return_value=1):
            result = apply_petal_augmentation(test_image)
            
        assert result.shape[2] == 3  # Should maintain 3 channels
        assert result.dtype == np.uint8
        
    def test_apply_petal_augmentation_no_flip(self):
        """Test petal augmentation without flip."""
        test_image = np.ones((100, 100, 3), dtype=np.uint8) * 128
        
        with patch('random.gauss', return_value=1.0), \
             patch('np.random.randint', return_value=0):
            result = apply_petal_augmentation(test_image)
            
        assert result.shape[2] == 3  # Should maintain 3 channels
        assert result.dtype == np.uint8


if __name__ == "__main__":
    pytest.main([__file__])