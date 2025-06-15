# bboxを時計回りに並び替える関数
import numpy as np
from utils import calculate_center, _clockwise_sort_angle

# バウンディングボックスのリスト (x_min, y_min, x_max, y_max)
bounding_boxes = [
    (1, 1, 2, 2),
    (2, 1, 3, 3),
    (0, 0, 1, 1),
    (1, 2, 2, 3)
]


# バウンディングボックスの中心点を計算
centers = [calculate_center(bbox) for bbox in bounding_boxes]

# 中心点の重心を計算
centroid = np.mean(centers, axis=0)


# 中心点を時計回りに並び替え
sorted_bboxes = sorted(bounding_boxes, key=lambda bbox: _clockwise_sort_angle(calculate_center(bbox), centroid))

def sort_by_bbox(bboxes, scores, labels):
    """Sort bounding boxes, scores, and labels in clockwise order."""
    from utils import sort_by_bboxes
    return sort_by_bboxes(bboxes, scores, labels)

bboxes = [
    (1, 1, 2, 2),
    (2, 1, 3, 3),
    (0, 0, 1, 1),
    (1, 2, 2, 3)
]

scores = [0.9, 0.8, 0.7, 0.6]   

labels = [0, 1, 2, 3]

sorted_bboxes, sorted_scores, sorted_labels = sort_by_bbox(bboxes, scores, labels)


# 配列を逆順に並び替える関数
def reverse_order(arr):
    return arr[::-1]

if __name__ == "__main__":
    print("Sorted Bounding Boxes (clockwise):")
    for bbox in sorted_bboxes:
        print(bbox)
        
    arrange = [1, 2, 3, 4, 5]
    reversed_arrange = reverse_order(arrange)
    print(reversed_arrange)
