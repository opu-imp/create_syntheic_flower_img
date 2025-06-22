"""Tests for utility functions."""

import pytest
import numpy as np

from src.utils.common import iou, calculate_center, nms


class TestUtils:
    """Test utility functions."""
    
    def test_iou_no_overlap(self):
        """Test IoU calculation with no overlap."""
        bbox1 = (0, 0, 10, 10)
        bbox2 = (20, 20, 30, 30)
        result = iou(bbox1, bbox2)
        assert result == 0.0
        
    def test_iou_complete_overlap(self):
        """Test IoU calculation with complete overlap."""
        bbox1 = (0, 0, 10, 10)
        bbox2 = (0, 0, 10, 10)
        result = iou(bbox1, bbox2)
        assert result == 1.0
        
    def test_iou_partial_overlap(self):
        """Test IoU calculation with partial overlap."""
        bbox1 = (0, 0, 10, 10)
        bbox2 = (5, 5, 15, 15)
        result = iou(bbox1, bbox2)
        # Intersection: 5x5 = 25, Union: 100 + 100 - 25 = 175
        expected = 25 / 175
        assert abs(result - expected) < 1e-6
        
    def test_calculate_center(self):
        """Test bounding box center calculation."""
        bbox = (0, 0, 10, 10)
        center = calculate_center(bbox)
        assert center == (5.0, 5.0)
        
    def test_calculate_center_asymmetric(self):
        """Test center calculation for asymmetric bbox."""
        bbox = (2, 3, 8, 9)
        center = calculate_center(bbox)
        assert center == (5.0, 6.0)
        
    def test_nms_empty_input(self):
        """Test NMS with empty input."""
        bboxes, scores, classes = nms([], [], [])
        assert bboxes == []
        assert scores == []
        assert classes == []
        
    def test_nms_single_bbox(self):
        """Test NMS with single bounding box."""
        bboxes = [(0, 0, 10, 10)]
        scores = [0.9]
        classes = [1]
        
        result_bboxes, result_scores, result_classes = nms(bboxes, scores, classes)
        
        assert result_bboxes == [(0, 0, 10, 10)]
        assert result_scores == [0.9]
        assert result_classes == [1]


if __name__ == "__main__":
    pytest.main([__file__])