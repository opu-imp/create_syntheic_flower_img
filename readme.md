# 合成花画像生成

合成花画像を生成するためのコード

## 必要要件

- Python 3.8 以上
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

- パラメータ
  - NUMCREATE: 生成する合成画像の枚数
  - color: 花弁の色（ディレクトリ名）
- 出力先
  - 合成画像: f'work/data/synthetic_flw/flw/{color}'
  - 重なり情報: f'work/data/synthetic_flw/mask/{color}'
- 出力

## 実行

### 従来方式（ルールベース）

```bash
python scripts/create_synthetic.py
```

### aestivation 統合方式（葉序の発生モデルのシミュレーションベース）

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

---

```
@InCollection{信田2025,
  author =	{信田 浩希 and 内海 ゆづ子 and 藤本 仰一 and 岩村 雅一},
  title =	{花弁配置推定システムのための合成画像を用いた分類手法の評価},
  booktitle =	{情報処理学会 研究報告コンピュータビジョンとイメージメディア（CVIM）},
  year =	2025,
  month =	may,
  volume =	{2025-CVIM-242},
  presenID =	{32},
  pages =	{1--8},
  numpages =	{8},
  URL =		{https://ipsj.ixsq.nii.ac.jp/records/2001863},
  publisher =	{情報処理学会},
  location =	{奈良女子大学}
}
```
