# Aestivation Data Integration for Synthetic Flower Generation

このドキュメントでは、aestivationシミュレーションデータを使用した合成花画像生成機能について説明します。

## 概要

従来の固定パターンによる花弁配置に加えて、植物の原基発生シミュレーション（aestivation）から得られた統計データを使用して、より自然な花弁配置パターンで合成画像を生成できるようになりました。

## 主な特徴

- **統計的に妥当な配置**: 実際の植物原基発生パターンに基づく花弁配置
- **重み付きサンプリング**: 出現頻度に基づいた確率的パターン選択
- **深度順配置**: 花弁の前後関係（深度）を考慮した自然な重なり
- **黄金角スパイラル**: 137.5°の黄金角を基本とした角度生成
- **互換性**: 従来の固定パターンとの切り替え可能

## aestivationデータ形式

### パターン統計ファイル (tmp.csv)
```csv
len,pattern,count
4,OAAI,21000
5,OIOAI,21000
6,OIOAIA,12280
```

- `len`: 花弁数
- `pattern`: パターン文字列（O=外側、I=内側、A=中間）
- `count`: 出現回数

### パターン文字列の意味
- `O` (Outer): 深度0 - 最も外側（奥）
- `A` (Adjacent): 深度1 - 中間
- `I` (Inner): 深度2 - 最も内側（手前）

例: `OIOAIA` → 深度レベル `[0, 2, 0, 1, 2, 1]`

## インストールと設定

### 必要な依存関係
```bash
pip install pandas numpy opencv-python
```

### ディレクトリ構造
```
create_syntheic_flower_img/
├── aestivation/           # aestivationシミュレーションデータ
│   └── tmp.csv           # パターン統計ファイル
├── src/
│   ├── io/
│   │   └── aestivation_reader.py  # データリーダー
│   ├── config/
│   │   └── settings.py   # 設定ファイル
│   └── core/
│       └── synthesis.py  # 合成処理
└── scripts/
    ├── create_synthetic.py        # メインスクリプト
    └── test_aestivation.py       # テストスクリプト
```

## 使用方法

### 基本的な使用方法

#### 1. 従来方式（固定パターン）
```bash
python scripts/create_synthetic.py
```

#### 2. aestivationデータ使用
```bash
python scripts/create_synthetic.py --use-aestivation
```

#### 3. カスタムパス指定
```bash
python scripts/create_synthetic.py --use-aestivation --aestivation-path /path/to/aestivation
```

### コマンドライン引数

| 引数 | 説明 | デフォルト |
|------|------|-----------|
| `--use-aestivation` | aestivationデータを使用 | False |
| `--aestivation-path` | aestivationディレクトリのパス | `aestivation` |

### 設定パラメータ

#### AestivationDataReader設定
```python
from src.io.aestivation_reader import AestivationDataReader

reader = AestivationDataReader('aestivation')

# パターン取得（6花弁、最小出現回数1000）
patterns = reader.get_patterns_by_length(6, min_count=1000)

# 重み付きサンプラー作成
sampler = reader.get_weighted_pattern_sampler(6, min_count=1000)
pattern_str, depth_list = sampler()
```

#### SynthesisConfig設定
```python
from src.config.settings import SynthesisConfig

# aestivationモード有効化
SynthesisConfig.set_aestivation_mode(True)
SynthesisConfig.set_aestivation_path('/custom/path')

# 状態確認
is_enabled = SynthesisConfig.use_aestivation_data()
path = SynthesisConfig.get_aestivation_path()
```

## APIリファレンス

### AestivationDataReader

#### 主要メソッド

```python
class AestivationDataReader:
    def __init__(self, base_path: str = "aestivation")
    
    def read_pattern_statistics(self, filename: str = "tmp.csv") -> pd.DataFrame
    
    def parse_pattern_string(self, pattern: str) -> List[int]
    
    def get_patterns_by_length(self, length: int, min_count: int = 100) -> List[Tuple[str, List[int], int]]
    
    def get_weighted_pattern_sampler(self, length: int, min_count: int = 100) -> callable
    
    def generate_angles_from_pattern(self, depth_list: List[int], 
                                   base_angle: float = 137.5,
                                   noise_sigma: float = 10.0) -> List[float]
    
    def create_synthesis_config(self, 
                              petal_counts: List[int] = [4, 5, 6, 7, 8, 9, 10],
                              min_count: int = 100,
                              base_angle: float = 137.5) -> Dict[str, List]
```

#### 使用例

```python
from src.io.aestivation_reader import AestivationDataReader

# 初期化
reader = AestivationDataReader('aestivation')

# パターン統計読み込み
df = reader.read_pattern_statistics()
print(f"Total patterns: {len(df)}")

# 6花弁のパターン取得
patterns_6 = reader.get_patterns_by_length(6, min_count=1000)
for pattern_str, depth_list, count in patterns_6:
    print(f"{pattern_str}: {depth_list} (count: {count})")

# 重み付きサンプリング
sampler = reader.get_weighted_pattern_sampler(6, min_count=1000)
pattern_str, depth_list = sampler()
angles = reader.generate_angles_from_pattern(depth_list)
print(f"Pattern: {pattern_str}")
print(f"Depths: {depth_list}")
print(f"Angles: {angles}")
```

### SynthesisConfig拡張

#### 新しいメソッド

```python
@classmethod
def use_aestivation_data(cls) -> bool
    """aestivationデータ使用フラグを取得"""

@classmethod
def set_aestivation_mode(cls, enabled: bool)
    """aestivationモードを設定"""

@classmethod
def get_aestivation_path(cls) -> str
    """aestivationディレクトリパスを取得"""

@classmethod
def set_aestivation_path(cls, path: str)
    """aestivationディレクトリパスを設定"""
```

## テスト

### テストスクリプト実行
```bash
python scripts/test_aestivation.py
```

### テスト内容
1. パターン統計読み込みテスト
2. パターン解析テスト
3. 花弁数別パターン取得テスト
4. 重み付きサンプラーテスト
5. 設定機能テスト

### 期待される出力例
```
Aestivation Integration Test
==================================================
Testing AestivationDataReader...

1. Testing pattern statistics reading...
   - Total patterns: 797
   - Pattern lengths: [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
   - Sample patterns:
     OI (len=2, count=21000)
     OAI (len=3, count=21000)
     ...

2. Testing pattern parsing...
   - OIOAIA -> [0, 2, 0, 1, 2, 1]
   - OIAOI -> [0, 2, 1, 0, 2]
   ...

✅ All tests passed!
```

## パフォーマンス

### 従来方式との比較

| 項目 | 従来方式 | aestivation方式 |
|------|----------|----------------|
| パターン数 | 固定（設定ファイル） | 動的（統計データ）|
| 角度生成 | 固定角度 + ノイズ | 黄金角 + ノイズ |
| 深度考慮 | 固定配置順 | 統計的配置順 |
| 計算オーバーヘッド | 最小 | 軽微（初期化時のみ）|

### メモリ使用量
- パターン統計データ: ~50KB
- サンプラー初期化: ~1MB
- 実行時オーバーヘッド: 無視できるレベル

## トラブルシューティング

### よくある問題

#### 1. FileNotFoundError: Pattern statistics file not found
```
解決方法: aestivationディレクトリにtmp.csvが存在することを確認
```

#### 2. ValueError: No patterns found for length X
```
解決方法: min_countを下げるか、異なる花弁数を指定
```

#### 3. ImportError: No module named 'pandas'
```
解決方法: pip install pandas
```

### デバッグ方法

#### 1. ログ出力の有効化
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

#### 2. パターン統計の確認
```python
reader = AestivationDataReader()
df = reader.read_pattern_statistics()
print(df.groupby('len')['count'].sum())
```

#### 3. サンプリング結果の検証
```python
sampler = reader.get_weighted_pattern_sampler(6)
patterns = [sampler()[0] for _ in range(100)]
from collections import Counter
print(Counter(patterns))
```

## 拡張機能

### カスタムパターンフィルタリング
```python
def custom_filter(pattern_str: str, depth_list: List[int], count: int) -> bool:
    # 特定の条件でフィルタリング
    return count > 5000 and len(set(depth_list)) > 1

# カスタムフィルタの適用
patterns = reader.get_patterns_by_length(6, min_count=100)
filtered = [p for p in patterns if custom_filter(*p)]
```

### 角度生成のカスタマイズ
```python
def custom_angle_generator(depth_list: List[int]) -> List[float]:
    # カスタム角度生成ロジック
    n = len(depth_list)
    return [(i * 360 / n) for i in range(n)]

# カスタム生成器の使用
angles = custom_angle_generator(depth_list)
```

## 今後の改善予定

1. **リアルタイム統計更新**: 新しいシミュレーション結果の自動取り込み
2. **パターン可視化**: 花弁配置パターンの3D可視化
3. **バッチ処理最適化**: 大規模データセット処理の高速化
4. **パラメータ調整UI**: GUI による設定変更インターフェース

## 参考資料

- [aestivation シミュレーション詳細](aestivation/README.md)
- [植物原基発生理論](https://example.com/phyllotaxis)
- [黄金角と植物配置](https://example.com/golden-angle)

## ライセンス

このプロジェクトは元のライセンスに従います。aestivation統合機能も同様のライセンス条件が適用されます。