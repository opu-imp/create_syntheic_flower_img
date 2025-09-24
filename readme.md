# 合成花画像生成

合成花画像を生成するためのコード

## 必要要件

- Python 3.8以上
- OpenCV、NumPy、H5Py

## インストール

```bash
pip install -e .
```

## データ構成
//TODO: 画像フォルダへのリンクとどういうデータなのかを追加
花弁画像を以下のように配置してください：

```
../data/
└── petals/
    ├── 黄色丸/
    │   ├── *.png
    │   └── crown/
    │       └── *.png
    ├── 紫/
    ├── 白紫/
    ├── 薄い白緑/
    └── 薄黄色/
```

## 使用方法

### 従来方式（ルールベース）
```bash
python scripts/create_synthetic.py
```

### aestivation統合方式（葉序の発生モデルのシミュレーションベース）

**事前準備:**
[aestivation](https://github.com/opu-imp/aestivation)リポジトリをクローンし、コンパイルして実行し、出力ファイル`tmp.csv`を以下のパスに配置：

```
create_syntheic_flower_img/
└── aestivation/
    └── tmp.csv    # パターン統計ファイル
```

**合成画像を生成:**
```bash
python scripts/create_synthetic.py --use-aestivation
```
