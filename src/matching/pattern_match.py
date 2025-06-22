import torch
from models import build_model
import os
import torch
from torchvision import models, transforms
from PIL import Image
from glob import glob

device = 'cuda:0'

# 学習済みモデルをロード
checkpoint = torch.load(f'../output/5kind50000img500epoch/partialV2_resize1/checkpoint.pth', map_location='cpu')
args = checkpoint['args']
model, criterion, postprocessors = build_model(args)
model.to(device)

model_without_ddp = model
model_without_ddp.load_state_dict(checkpoint['model'])
model_without_ddp
model.eval() # 評価モードに設定
criterion.eval()
args


# 画像の前処理
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),  # 事前学習済みモデル用の正規化
])

# 画像が保存されているフォルダのパス
flw_glob_path = '../../create_syntheic_flower_img/data/synthetic_flw/5kind_each10000img/flw/*/*'
image_paths = sorted(glob(flw_glob_path))
image_paths = image_paths[:2]

# 有効な拡張子
valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp']

# 結果を格納するリスト
results: list = []

# フォルダ内の画像を読み込み、識別する
for image_path in image_paths:
    if any(image_path.endswith(ext) for ext in valid_extensions):  # 有効な画像ファイルの拡張子
        # 画像を読み込み、前処理を行う
        print('image_path: {}'.format(image_path))
        img = Image.open(image_path).convert("RGB")  # RGBモードに変換
        img_tensor = preprocess(img).unsqueeze(0)  # バッチサイズの次元を追加

        # GPUが使える場合はGPUにデータとモデルを移す
        if torch.cuda.is_available():
            img_tensor = img_tensor.to('cuda')
            model.to('cuda')

        # モデルを使って予測を行う
        with torch.no_grad():  # 勾配計算を行わない
            outputs = model(img_tensor)
        print(outputs)

        results.append(outputs)

        # 結果を処理する
        # _, predicted_class = torch.max(outputs, 1)  # 最も高い確率のクラスを取得
        # confidence = torch.nn.functional.softmax(outputs, dim=1)[0][predicted_class].item()  # 信頼度を取得

        # 結果を保存
        # results.append((filename, predicted_class.item(), confidence))