# 動作確認手順

このドキュメントでは、合成花画像生成システムの動作確認手順を説明します。

## 目次

- [前提条件](#前提条件)
- [環境構築の確認](#環境構築の確認)
- [データディレクトリの確認](#データディレクトリの確認)
- [基本機能テスト](#基本機能テスト)
- [Aestivation機能のテスト](#aestivation機能のテスト)
- [実行テスト](#実行テスト)
- [出力確認](#出力確認)
- [トラブルシューティング](#トラブルシューティング)

## 前提条件

### システム要件
- Python 3.8以上
- 十分なメモリ（最低8GB推奨）
- 40コア以上のCPU（フルスケール生成時）

### 必要なパッケージ
```bash
# Python環境の確認
python --version  # 3.8以上必要

# 必要パッケージの確認
pip list | grep -E "(opencv|numpy|h5py|pandas)"
```

## 環境構築の確認

### 1. プロジェクトルートへの移動
```bash
cd /Users/nobu/research/create_syntheic_flower_img
```

### 2. 依存関係の確認
```bash
# 必要なパッケージがインポートできるか確認
python -c "
import cv2
import numpy as np
import h5py
import pandas as pd
print('✅ All required packages imported successfully')
"
```

### 3. パッケージインストール（必要な場合）
```bash
# 開発モードでインストール
pip install -e .

# または必要なパッケージを個別にインストール
pip install opencv-python numpy h5py pandas
```

## データディレクトリの確認

### 1. データディレクトリ構造の確認
```bash
# データディレクトリの存在確認
ls -la ../data/petals/
```

### 2. 必要なディレクトリ
以下のディレクトリが必要です：
- `黄色丸/` - 黄色丸花弁画像
- `紫/` - 紫花弁画像
- `白紫/` - 白紫花弁画像
- `薄い白緑/` - 薄い白緑花弁画像
- `薄黄色/` - 薄黄色花弁画像

### 3. 花弁画像の確認
```bash
# 各ディレクトリに画像ファイルが存在するか確認
find ../data/petals -name "*.png" | head -10
```

## 基本機能テスト

### 1. 設定ファイルの検証
```bash
python -c "
from src.config.settings import validate_config
try:
    validate_config()
    print('✅ Configuration validation passed')
except Exception as e:
    print(f'❌ Configuration error: {e}')
"
```

### 2. コアモジュールのインポートテスト
```bash
python -c "
from src.core.synthesis import synthesize_single_flower
from src.core.geometry import SynthesisParameterConfig
from src.config.settings import SynthesisConfig
from src.io.multi_channel import read_img, write_img
print('✅ Core modules imported successfully')
"
```

## Aestivation機能のテスト

### 1. Aestivationデータの確認
```bash
# Aestivationディレクトリの確認
ls -la aestivation/

# 必要なファイルの存在確認
if [ -f "aestivation/tmp.csv" ]; then
    echo "✅ Aestivation data file found"
else
    echo "❌ Aestivation data file not found"
fi
```

### 2. Aestivation機能テストの実行
```bash
# テストスクリプトの実行
python scripts/test_aestivation.py
```

## 実行テスト

### 1. 小規模テスト（従来方式）
```bash
# Pythonスクリプトで小規模テストを実行
python -c "
from src.core.synthesis import synthesize_single_flower
from src.core.geometry import SynthesisParameterConfig
from src.config.settings import SynthesisConfig
import numpy as np

# 基本設定でテスト
config = SynthesisParameterConfig(
    path_petals='../data/petals/黄色丸/*.png',
    path_crowns='../data/petals/黄色丸/crown/*.png',
    dict_pairs=SynthesisConfig.PETAL_ARRANGEMENTS
)

try:
    img, mask, multi_mask = synthesize_single_flower(config, 128, 512)
    print(f'✅ Success! Generated image shape: {img.shape}')
    print(f'   Mask shape: {mask.shape}')
    print(f'   Multi-channel mask shape: {multi_mask.shape}')
except Exception as e:
    print(f'❌ Error: {e}')
"
```

### 2. Aestivationモードテスト
```bash
# Aestivationモードでの小規模テスト
python -c "
from src.core.synthesis import synthesize_single_flower
from src.core.geometry import SynthesisParameterConfig
from src.config.settings import SynthesisConfig
from src.io.aestivation_reader import AestivationDataReader

# Aestivationモード有効化
SynthesisConfig.set_aestivation_mode(True)
reader = AestivationDataReader('aestivation')

config = SynthesisParameterConfig(
    path_petals='../data/petals/黄色丸/*.png',
    path_crowns='../data/petals/黄色丸/crown/*.png',
    dict_pairs=SynthesisConfig.PETAL_ARRANGEMENTS
)

try:
    img, mask, multi_mask = synthesize_single_flower(
        config, 128, 512, aestivation_reader=reader
    )
    print(f'✅ Aestivation mode success! Image shape: {img.shape}')
except Exception as e:
    print(f'❌ Aestivation mode error: {e}')
"
```

### 3. メインスクリプトの実行

#### デバッグモード（小規模）
```bash
# デバッグモードで実行（少数の画像のみ生成）
python scripts/create_synthetic.py --debug

# または環境変数で制御
DEBUG=1 python scripts/create_synthetic.py
```

#### Aestivationモード
```bash
# Aestivationモードでの実行
python scripts/create_synthetic.py --use-aestivation --debug

# カスタムパス指定
python scripts/create_synthetic.py --use-aestivation --aestivation-path /path/to/aestivation
```

#### フルスケール実行（注意：時間がかかります）
```bash
# 全データ生成（40コアCPU推奨）
python scripts/create_synthetic.py

# バッチスクリプトを使用
./scripts/run_synthesis.sh
```

## 出力確認

### 1. 出力ディレクトリの確認
```bash
# 出力ディレクトリの存在確認
ls -la output/
ls -la masks/
ls -la multi_channel_masks/
```

### 2. 生成されたファイルの確認
```bash
# 生成された画像数の確認
echo "Generated images: $(find output -name "*.png" | wc -l)"
echo "Generated masks: $(find masks -name "*.png" | wc -l)"
echo "Generated HDF5 masks: $(find multi_channel_masks -name "*.h5" | wc -l)"
```

### 3. サンプル画像の確認
```bash
# 最初の数枚を確認
ls output/*.png | head -5
```

## トラブルシューティング

### よくある問題と解決方法

#### 1. データディレクトリが見つからない
```bash
# エラー: FileNotFoundError: Base data directory not found
# 解決方法:
mkdir -p ../data/petals
# または settings.py で BASE_DATA_DIR を適切なパスに変更
```

#### 2. メモリ不足エラー
```bash
# エラー: MemoryError または システムが応答しない
# 解決方法: パラメータを小さくして実行
python scripts/create_synthetic.py --max-len 128 --side 512
```

#### 3. Aestivationファイルが見つからない
```bash
# エラー: FileNotFoundError: aestivation/tmp.csv not found
# 解決方法:
mkdir -p aestivation
# または --aestivation-path で正しいパスを指定
```

#### 4. インポートエラー
```bash
# エラー: ModuleNotFoundError
# 解決方法:
pip install -e .
# または必要なパッケージを個別インストール
```

### ログとデバッグ

#### ログファイルの確認
```bash
# ログファイルが存在する場合
tail -f synthesis.log
```

#### 詳細なエラー情報の取得
```bash
# Python警告をエラーとして扱う
python -W error scripts/create_synthetic.py --debug

# 詳細なトレースバックを表示
python -X dev scripts/create_synthetic.py --debug
```

### パフォーマンスの確認
```bash
# 処理時間の測定
time python scripts/create_synthetic.py --debug

# システムリソースの監視
# 別のターミナルで実行
top -p $(pgrep -f create_synthetic.py)
```

## 次のステップ

動作確認が完了したら：

1. **小規模テストから開始**: `--debug` フラグで動作を確認
2. **段階的にスケールアップ**: パラメータを徐々に大きくして実行
3. **本番実行**: フルスケールでの生成を実行

詳細な使用方法は [README.md](../README.md) を参照してください。