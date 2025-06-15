import numpy as np
from utils import calculate_center, _clockwise_sort_angle
from config import MatchConfig

# Use arrangements from config
arranges = MatchConfig.get_all_arrangements()
concat_arranges = MatchConfig.get_concatenated_arrangements()
# 0,1,2からなる配列とマッチング

scores = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0]

# Use threshold from config
keep = [1 if score >= MatchConfig.SCORE_THRESHOLD else 0 for score in scores]

# keepが1のscoreのみ取り出す
selected_scores = [score for score, k in zip(scores, keep) if k]

def calculate_circular_fit(cycle1, cycle2, isLog=False):
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

    if isLog:
        print(f"配列1: {cycle1},配列2: {cycle2},適合度: {max_fit}")

    return max_fit

def calculate_circular_fit_with_arranges(cycle1, isLog=False):
    max_fit = 0
    fit_arrange = []
    for cycle2 in concat_arranges:
        max_fit = calculate_circular_fit(cycle1, cycle2, isLog)
        fit_arrange = cycle2
    
    return max_fit, fit_arrange

bounding_boxes = [
    (0, 0, 100, 100),
    (100, 0, 200, 100),
    (200, 0, 300, 100),
    (300, 0, 400, 100),
    (400, 0, 500, 100),
    (500, 0, 600, 100),
    (600, 0, 700, 100),
    (700, 0, 800, 100),
    (800, 0, 900, 100),
    (900, 0, 1000, 100),
]


# バウンディングボックスの中心点を計算
centers = [calculate_center(bbox) for bbox in bounding_boxes]

# 中心点の重心を計算
centroid = np.mean(centers, axis=0)


# 中心点を時計回りに並び替え
sorted_bboxes = sorted(bounding_boxes, key=lambda bbox: _clockwise_sort_angle(calculate_center(bbox), centroid))

if __name__ == "__main__":
    print("Sorted Bounding Boxes (clockwise):")
    for bbox in sorted_bboxes:
        print(bbox)
