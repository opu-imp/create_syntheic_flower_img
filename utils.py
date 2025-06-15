import numpy as np
from typing import List, Tuple, Any

def iou(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
    """Calculate Intersection over Union (IoU) of two bounding boxes.
    
    Args:
        a: Bounding box (xmin, ymin, xmax, ymax)
        b: Bounding box (xmin, ymin, xmax, ymax)
        
    Returns:
        IoU value between 0 and 1
    """
    ax_mn, ay_mn, ax_mx, ay_mx = a[0], a[1], a[2], a[3]
    bx_mn, by_mn, bx_mx, by_mx = b[0], b[1], b[2], b[3]

    a_area = (ax_mx - ax_mn + 1) * (ay_mx - ay_mn + 1)
    b_area = (bx_mx - bx_mn + 1) * (by_mx - by_mn + 1)

    abx_mn = max(ax_mn, bx_mn)
    aby_mn = max(ay_mn, by_mn)
    abx_mx = min(ax_mx, bx_mx)
    aby_mx = min(ay_mx, by_mx)
    w = max(0, abx_mx - abx_mn + 1)
    h = max(0, aby_mx - aby_mn + 1)
    intersect = w*h

    iou = intersect / (a_area + b_area - intersect)
    return iou

def nms(bboxes, scores, classes, iou_threshold=0.5):
    new_bboxes = [] # NMS適用後の矩形リスト
    new_scores = [] # NMS適用後の信頼度(スコア値)リスト
    new_classes = [] # NMS適用後のクラスのリスト

    while len(bboxes) > 0:
        # スコア最大の矩形のインデックスを取得
        argmax = scores.index(max(scores))

        # スコア最大の矩形、スコア値、クラスをそれぞれのリストから消去
        bbox = bboxes.pop(argmax)
        score = scores.pop(argmax)
        clss = classes.pop(argmax)        

        # スコア最大の矩形と、対応するスコア値、クラスをNMS適用後のリストに格納
        new_bboxes.append(bbox)
        new_scores.append(score)
        new_classes.append(clss)

        pop_i = []
        for i, bbox_tmp in enumerate(bboxes):
            # スコア最大の矩形bboxとのIoUがiou_threshold以上のインデックスを取得
            if iou(bbox, bbox_tmp) >= iou_threshold:
                pop_i.append(i)

        # 取得したインデックス(pop_i)の矩形、スコア値、クラスをそれぞれのリストから消去
        for i in pop_i[::-1]:
            bboxes.pop(i)
            scores.pop(i)
            classes.pop(i)

    return new_bboxes, new_scores, new_classes

def probas_to_scores_and_classes(probas):
    scores = []
    classes = []
    for p in probas:
        cl = p.argmax()
        score = p[cl]
        scores.append(score)
        classes.append(cl)
    return scores, classes

def calculate_center(bbox: Tuple[float, float, float, float]) -> Tuple[float, float]:
    """Calculate the center point of a bounding box.
    
    Args:
        bbox: Bounding box (xmin, ymin, xmax, ymax)
        
    Returns:
        Center point (center_x, center_y)
    """
    x_min, y_min, x_max, y_max = bbox
    center_x = (x_min + x_max) / 2
    center_y = (y_min + y_max) / 2
    return (center_x, center_y)

def _clockwise_sort_angle(center: Tuple[float, float], centroid: Tuple[float, float]) -> float:
    """Calculate angle for clockwise sorting.
    
    Args:
        center: Point (x, y)
        centroid: Reference point (x, y)
        
    Returns:
        Angle in radians
    """
    return np.arctan2(center[1] - centroid[1], center[0] - centroid[0])


def sort_by_bboxes(bboxes: List[Tuple[float, float, float, float]], 
                   scores: List[float], 
                   labels: List[Any]) -> Tuple[List, List, List]:
    """Sort bounding boxes, scores, and labels in clockwise order.
    
    Args:
        bboxes: List of bounding boxes
        scores: List of confidence scores
        labels: List of labels
        
    Returns:
        Tuple of sorted (bboxes, scores, labels)
    """
    centers = [calculate_center(bbox) for bbox in bboxes]
    centroid = np.mean(centers, axis=0)

    def sort_key(item):
        bbox, _, _ = item
        center = calculate_center(bbox)
        return _clockwise_sort_angle(center, centroid)

    sorted_items = sorted(zip(bboxes, scores, labels), key=sort_key)
    return list(zip(*sorted_items))