# 合成花画像の生成

## pythonファイル説明
* create_synthe_v4.ipynb: 合成用のnotebook
* create_synthe_v4.py: スクリプトver.
* multi_channel_img_io.py: マルチチャネルのマスクを作るときに必要
* rotate_petals: 切り出した花弁の向きを揃えるための画像回転用notebook


## 事前準備
* 花弁の画像を種類(color)ごとに以下のパスに配置
  * crownは、花弁中心の画像
```py
    f'../data/petals/{color}/*.png',
    f'../data/petals/{color}/crown/*.png',
```