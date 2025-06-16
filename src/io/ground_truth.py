import os
from datetime import datetime
import numpy as np
import json


n_classes = 10
img_size = 256
coord_filter = np.array([[(i, j) for j in range(img_size)] for i in range(img_size)])

images_info = []
image_paths = image_paths[:10]

for i, image_path in enumerate(image_paths):
    info = {}
    fname = os.path.basename(image_path)
    info['path'] = image_path.replace('../../', '')

    mask_path = image_path.replace('/flw/', '/mask/')
    mask = Image.open(mask_path)
    one_hot = get_one_hot(mask, n_classes=n_classes, img_size=img_size)
    centers = get_center_array(one_hot, coord_filter)
    cls_dict = get_cls_dict(centers, img_size=img_size)
    center_dict = dict([[i + 1,  [center[0] , center[1] , 10 , 10 ]] for i, center in enumerate(centers)])
    # print(cls_dict)
    # print(center_dict)
    info['bboxes'] = [[key, {'cls': cls_dict[key], 'bb': bb}] for key, bb in center_dict.items()]
    # print(info)
    images_info.append(info)
    if i % 1000 == 0:
        print(f'just processed {i}th image ! ({datetime.now()})')

"""
ここからはjsonファイルの作成
"""
data = {}

# 画像とバウンディングボックスの情報を処理
for image_id, img_info in enumerate(images_info, start=1):
    img_path = img_info["path"]
    bboxes = img_info["bboxes"]
    
    # 画像サイズを取得
    with Image.open('../../' + img_path) as img:
        width, height = img.size

    # フォルダ名とファイル名を取得
    folder_name = os.path.basename(os.path.dirname(img_path))
    file_name = os.path.basename(img_path)
    
    # フォルダごとのデータ構造を作成
    if folder_name not in data:
        data[folder_name] = {}
        
    annotations = []
    
    for bbox in bboxes:
        annotations.append({
            "category_id": bbox[1]['cls'],
            "bbox": bbox[1]['bb'],  # [x_min, y_min, width, height]
            "area": 100,  # 面積 = width * height
            "iscrowd": 0  # 複数のオブジェクトを1bbで表現しているか
        })

    data[folder_name][file_name] = {
        "width": img_size,
        "height": img_size,
        "file_path": img_path,
        "annotations": annotations
    }

with open(f'./output/ground_truth.json', 'w') as json_file:
    json.dump(data, json_file)
