from flask import Flask, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms

import os
import shutil
from PIL import Image

#定義したクラスの呼び出し
from model.predict_net import VGGNet

#フォルダ保存先
UPLOAD_FOLDER = "./static/images/"
#指定ファイル
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "gif"}

labels = ["飛行機", "自動車", "鳥", "猫", "鹿", "犬", "カエル", "馬", "船", "トラック"]
n_class = len(labels)
img_size = 32
n_result = 3

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/", methods=["GET", "POST"])
def index():
    return render_template("index.html")

@app.route("/result", methods=["GET", "POST"])
def result():
    if request.method == "POST":
        #ファイルの存在と形式を確認
        if "file" not in request.files:
            print("File doesn't exist")
            return redirect(url_for("index"))
        file = request.files["file"]
        if not allowed_file(file.filename):
            print(f'{file.filename}: File not allowed')
            return redirect(url_for("index"))
        
        # ファイルの保存
        if os.path.isdir(UPLOAD_FOLDER):
            shutil.rmtree(UPLOAD_FOLDER)
        os.makedirs(UPLOAD_FOLDER, exist_ok=True)
        filename = secure_filename(file.filename) #ファイル名を安全なものに
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)

        # 画像の読み込み
        image = Image.open(filepath)
        image = image.convert("RGB")
        image = image.resize((img_size, img_size))

        normalize = transforms.Normalize((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)) #平均値を0、標準偏差1
        to_tensor = transforms.ToTensor()
        transform = transforms.Compose([to_tensor, normalize])

        x = transform(image)
        x = x.reshape(1, 3, img_size, img_size) # バッチサイズ、入力チャンネル、高さ、幅

        # 予測
        net = VGGNet()
        net.load_state_dict(torch.load(
            "./model/model_vgg.pth", map_location=torch.device("cpu")))
        net.eval() # 評価モード

        y = net(x)
        y = F.softmax(y, dim=1)[0]
        sorted_idx = torch.argsort(-y) # 降順
        result = ""
        for i in range(n_result):
            idx = sorted_idx[i].item()
            ratio = y[idx].item()
            label = labels[idx]
            result += "<p>" + str(round(ratio*100, 1)) + \
                "%の確率で" + label + "です。</p>"
        return render_template("result.html", result=result, filepath=filepath)
    else:
        return redirect(url_for("index"))
    
if __name__ == "__main__":
    app.run(debug=True)
