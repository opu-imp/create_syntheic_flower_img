"""Arrangement and sorting functions for bounding boxes."""

import numpy as np
from typing import List, Tuple, Any

from ..utils.common import calculate_center, _clockwise_sort_angle, sort_by_bboxes
from ..config.settings import MatchConfig


def calculate_circular_fit(cycle1: List[int], cycle2: List[int], isLog: bool = False) -> int:
    """Calculate circular fit between two cycles.
    
    Args:
        cycle1: First cycle
        cycle2: Second cycle
        isLog: Whether to enable logging
        
    Returns:
        Maximum fit score
    """
    # 円順列の長さ
    len_cycle1 = len(cycle1)
    len_cycle2 = len(cycle2)
    
    # cycle2がcycle1より長い場合、適合度は0
    if len_cycle2 != len_cycle1:
        if isLog:
            print("not same length")
        return 0  

    # 連結したcycle1を作成
    doubled_cycle1 = cycle1 + cycle1

    # 適合度をカウントする変数
    max_fit = 0

    # cycle1の各回転での適合度を計算
    for start in range(len_cycle1):
        # 現在の回転に対応する部分を取得
        current_fit = sum(1 for i in range(len_cycle2) if doubled_cycle1[start + i] == cycle2[i])
        max_fit = max(max_fit, current_fit)

    if isLog:
        print(f"配列1: {cycle1},配列2: {cycle2},適合度: {max_fit}")

    return max_fit


def calculate_circular_fit_with_arranges(cycle1: List[int], isLog: bool = False) -> Tuple[int, List[int]]:
    """Calculate the best circular fit with all available arrangements.
    
    Args:
        cycle1: Input cycle to match
        isLog: Whether to enable logging
        
    Returns:
        Tuple of (max_fit_score, best_matching_arrangement)
    """
    max_fit = 0
    fit_arrange = []
    arranges = MatchConfig.get_all_arrangements()
    
    for cycle2 in arranges:
        # Test normal arrangement
        fit = calculate_circular_fit(cycle1, cycle2, isLog)
        if max_fit < fit:
            max_fit = fit
            fit_arrange = cycle2
            
        # Test reversed arrangement
        reverse_arrange = reverse_order(cycle2)
        reverse_fit = calculate_circular_fit(cycle1, reverse_arrange, isLog)
        if max_fit < reverse_fit:
            max_fit = reverse_fit
            fit_arrange = reverse_arrange
    
    return max_fit, fit_arrange


def reverse_order(arr: List[int]) -> List[int]:
    """Reverse the order of an array."""
    return arr[::-1]


def sort_bbox_clockwise(bboxes: List[Tuple], scores: List[float] = None, labels: List[Any] = None):
    """Sort bounding boxes in clockwise order.
    
    Args:
        bboxes: List of bounding boxes
        scores: Optional list of scores
        labels: Optional list of labels
        
    Returns:
        Sorted bboxes (and scores, labels if provided)
    """
    if scores is None and labels is None:
        # Simple sorting for bboxes only
        centers = [calculate_center(bbox) for bbox in bboxes]
        centroid = np.mean(centers, axis=0)
        return sorted(bboxes, key=lambda bbox: _clockwise_sort_angle(calculate_center(bbox), centroid))
    else:
        # Use the full sorting function from utils
        return sort_by_bboxes(bboxes, scores or [], labels or [])


# Legacy function for backward compatibility
def sort_by_bbox(bboxes, scores, labels):
    """Sort bounding boxes, scores, and labels in clockwise order."""
    return sort_by_bboxes(bboxes, scores, labels)


# Example usage and test data
if __name__ == "__main__":
    # Test data from original files
    bounding_boxes = [
        (0, 0, 100, 100),
        (100, 0, 200, 100),
        (200, 0, 300, 100),
        (300, 0, 400, 100),
    ]
    
    sorted_bboxes = sort_bbox_clockwise(bounding_boxes)
    print("Sorted Bounding Boxes (clockwise):")
    for bbox in sorted_bboxes:
        print(bbox)
        
    # Test with scores and labels
    bboxes = [(1, 1, 2, 2), (2, 1, 3, 3), (0, 0, 1, 1), (1, 2, 2, 3)]
    scores = [0.9, 0.8, 0.7, 0.6]   
    labels = [0, 1, 2, 3]
    
    sorted_bboxes, sorted_scores, sorted_labels = sort_by_bbox(bboxes, scores, labels)
    print("\nSorted with scores and labels:")
    for bbox, score, label in zip(sorted_bboxes, sorted_scores, sorted_labels):
        print(f"BBox: {bbox}, Score: {score}, Label: {label}")
        
    # Test reverse order
    arrange = [1, 2, 3, 4, 5]
    reversed_arrange = reverse_order(arrange)
    print(f"\nReversed: {reversed_arrange}")