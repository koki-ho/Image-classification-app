# 画像分類アプリ

このアプリは、ユーザーがアップロードした画像を、学習済みのディープラーニングモデルを用いて分類するWEBアプリケーションです。

---

## 機能

- **画像アップロード**: 保存した画像のアップロード可能。
- **分類**: アップロードした画像から、分類確率を表示します。
- **UI**: Flaskを利用したシンプルなインターフェース
- **モデル**: 学習済みディープラーニングモデル

---

## 使用方法

1. **アプリを起動**
   ```bash
   git clone https://github.com/koki-ho/Image-classification-app.git
   cd Image-classification-app
   python3 -m venv venv
   source venv/bin/activate  # Windowsの場合: venv\Scripts\activate
   pip install -r requirements.txt
   python app.py
   ```
2. ブラウザで `http://127.0.0.1:5000` を開く。
3. 画像をアップロードし、分類確率を確認。

---

## ディレクトリ構成

```plaintext
Image-classification-app/
│
├── app.py             　# メインアプリ
├── model/             　# 事前学習モデル
│   ├── model_vgg.pth  　# 学習済パラメーター
│   └── predict_net.py 　# モデル定義
├── templates/         　# HTML テンプレート
│   ├── layout.html    　# 全体のレイアウト
│   ├── index.html     　# 画像アップロードフォーム
│   └── result.html    　# 分類結果の表示
└── requirements.txt   　# アプリに必要な依存関係
```

---

## 技術スタック

- **Flask**: PythonのWebフレームワーク。アプリケーションのサーバーサイドを担当。
- **PyTorch**: ニューラルネットワークフレームワーク。
- **HTML/CSS**: ユーザーインターフェースの構築。

---
