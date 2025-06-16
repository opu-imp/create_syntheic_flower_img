# 合成花画像生成システム

葉序理論に基づく花弁配置アルゴリズムを使用して、合成花画像を生成するコンピュータビジョンパッケージです。

## 🌸 概要

本プロジェクトは、自然な花の成長パターンを模倣した螺旋パターンで個々の花弁画像を配置することで、リアルな合成花画像を生成します。黄金角（137°）やその他の葉序パターンを使用して、多様な花の構成を作成します。

## ✨ 特徴

- **葉序理論ベースの配置**: 自然な花の成長パターンの数学的モデルを使用
- **マルチチャンネルマスク生成**: 各花弁の詳細なセグメンテーションマスクを作成
- **高性能処理**: マルチプロセシングによる並列処理をサポート
- **設定可能なパラメータ**: 柔軟な花弁数、配置、拡張オプション
- **HDF5サポート**: マルチチャンネルマスクの効率的な保存

## 📁 プロジェクト構成

```
create_syntheic_flower_img/
├── src/                          # ソースコードモジュール
│   ├── core/                     # コア合成機能
│   │   ├── synthesis.py          # メイン合成ロジック
│   │   ├── image_processing.py   # 画像処理ユーティリティ
│   │   └── geometry.py           # 幾何学的変換
│   ├── matching/                 # パターンマッチングアルゴリズム
│   │   ├── circular_match.py     # 円形パターンマッチング
│   │   ├── arrange.py            # バウンディングボックス配置
│   │   └── pattern_match.py      # パターンマッチングユーティリティ
│   ├── io/                       # 入出力操作
│   │   ├── multi_channel.py      # マルチチャンネルマスクI/O
│   │   └── ground_truth.py       # グラウンドトゥルース生成
│   ├── utils/                    # ユーティリティ関数
│   │   └── common.py             # 共通ユーティリティ（IoU、NMSなど）
│   └── config/                   # 設定
│       └── settings.py           # 設定クラス
├── scripts/                      # 実行スクリプト
│   ├── create_synthetic.py       # メイン合成スクリプト
│   └── run_synthesis.sh          # バッチ実行スクリプト
├── notebooks/                    # Jupyterノートブック
├── docker/                       # Docker設定
├── docs/                         # ドキュメント
└── tests/                        # ユニットテスト
```

## 🚀 インストール

### 必要要件

- Python 3.8以上
- 40コア以上のCPU（フルスケール生成時の推奨）
- OpenCV、NumPy、H5Py

### パッケージインストール

```bash
# リポジトリをクローン
git clone https://github.com/your-org/synthetic-flower-generator.git
cd synthetic-flower-generator

# 開発モードでインストール
pip install -e .

# 開発用依存関係も含めてインストール
pip install -e ".[dev]"

# Jupyter サポート付きでインストール
pip install -e ".[jupyter]"
```

### Docker セットアップ

```bash
# Docker イメージをビルド
docker build -f docker/Dockerfile -t synthetic-flowers .

# コンテナを実行
docker run -v $(pwd):/work -p 8899:8899 synthetic-flowers

# コンテナ内でJupyter Labを開始
docker exec -it <container_id> /work/docker/start-jupyter.sh
```

## 📊 データ要件

花弁画像は以下のように整理されている必要があります：

```
../data/
└── petals/
    ├── 黄色丸/              # 黄色丸花弁
    │   ├── *.png           # 個別の花弁画像
    │   └── crown/          # 中心部/花冠画像
    │       └── *.png
    ├── 紫/                  # 紫花弁
    ├── 白紫/                # 白紫花弁
    ├── 薄い白緑/             # 薄い白緑花弁
    └── 薄黄色/               # 薄黄色花弁
```

サポートされている色：
- `黄色丸` （黄色丸）
- `紫` （紫）
- `白紫` （白紫）
- `薄い白緑` （薄い白緑）
- `薄黄色` （薄黄色）

## 💻 使用方法

### 基本的な使用方法

```bash
# メイン合成スクリプトを実行
python scripts/create_synthetic.py

# またはバッチスクリプトを使用
./scripts/run_synthesis.sh
```

### Python API

```python
from src.core.synthesis import synthesize_single_flower
from src.core.geometry import SynthesisParameterConfig
from src.config.settings import SynthesisConfig

# 設定を初期化
config = SynthesisParameterConfig(
    path_petals="../data/petals/黄色丸/*.png",
    path_crowns="../data/petals/黄色丸/crown/*.png",
    dict_pairs=SynthesisConfig.PETAL_ARRANGEMENTS
)

# 単一の合成花を生成
img, mask, multi_channel_mask = synthesize_single_flower(
    config, 
    max_len=256, 
    side=1024
)
```

### 設定

`src/config/settings.py`の主要パラメータ：

```python
# 画像処理
IMAGE_SIZE_MULTIPLIER = 4
PADDING_SIZE = 6
N_SAMPLE_PETAL = 3
AUGMENTATION_SIGMA = 0.15

# 花弁配置 (花弁数, 基準角度)
PETAL_ARRANGEMENTS = {
    'A1': [[4, 144]],    # 4花弁、144°角度
    'B1': [[5, 100], [5, 137]],  # 5花弁、様々な角度
    'C2': [[6, 137]],    # 6花弁、黄金角
    # ... その他の設定
}
```

## 🔬 アルゴリズムの詳細

### 葉序理論ベースの螺旋配置

コアアルゴリズムは葉序理論（植物の葉/花弁の配置）の原理を使用します：

1. **黄金角**: デフォルトの基準角137°が最適なパッキングを作成
2. **螺旋生成**: 花弁が螺旋経路に沿って配置される
3. **ノイズ注入**: 自然な外観のためのランダムな変動
4. **拡張**: 各花弁のスケールと反転バリエーション

### 出力形式

各生成は以下を生成します：
- **合成花画像**: 配置された花弁を持つRGB画像
- **集約マスク**: 花弁インデックスを持つ単一チャンネルマスク
- **マルチチャンネルマスク**: 個別の花弁マスクを持つ10チャンネルHDF5ファイル

## 🧪 テスト

```bash
# すべてのテストを実行
pytest tests/

# カバレッジ付きで実行
pytest --cov=src tests/

# 特定のテストファイルを実行
pytest tests/test_synthesis.py
```

## 🛠️ 開発

### コードスタイル

```bash
# Blackでコードをフォーマット
black src/ scripts/ tests/

# flake8でチェック
flake8 src/ scripts/

# mypyで型チェック
mypy src/
```

### 貢献

1. リポジトリをフォーク
2. フィーチャーブランチを作成 (`git checkout -b feature/amazing-feature`)
3. 変更をコミット (`git commit -m 'Add amazing feature'`)
4. ブランチにプッシュ (`git push origin feature/amazing-feature`)
5. プルリクエストを開く

## 📈 パフォーマンス

- **処理能力**: 色/バッチの組み合わせごとに12,500枚の画像を生成
- **並列化**: 5色 × 8バッチでProcessPoolExecutorを使用
- **要件**: フルスケール生成には40コア以上のCPUを推奨
- **出力**: 合計約500,000枚の合成画像
