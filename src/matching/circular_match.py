"""Calculate circular fit for flower petal arrangements."""
from typing import List, Tuple
from ..config.settings import MatchConfig

# Use arrangements from config
arranges = MatchConfig.get_all_arrangements()

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
        if(isLog):
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

    if (isLog):
        print(f"配列1: {cycle1},配列2: {cycle2},適合度: {max_fit}")

    return max_fit

def reverse_order(arr: List[int]) -> List[int]:
    """Reverse the order of an array."""
    return arr[::-1]

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

# Example usage
if __name__ == "__main__":
    cycle1 = [1, 2, 3, 4]
    cycle2 = [4, 3, 2, 1]  # Rotated version of circular permutation
    fit = calculate_circular_fit(cycle1, cycle2, True)
    print(f"Fit score: {fit}")