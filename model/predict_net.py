import torch.nn as nn
import torch.nn.functional as F

#VGGNet
class VGGNet(nn.Module):
    def __init__(self):
        super().__init__()
        #第1層,畳み込み層1
        self.conv1 = nn.Conv2d(3, 32, 3, padding=(1, 1))
        #第1層,畳み込み層2
        self.conv2 = nn.Conv2d(32, 32, 3, padding=(1, 1))
        #第2層,畳み込み層3
        self.conv3 = nn.Conv2d(32, 64, 3, padding=(1, 1))
        #第2層,畳み込み層4
        self.conv4 = nn.Conv2d(64, 64, 3, padding=(1, 1))
        #第2層,畳み込み層3
        self.conv5 = nn.Conv2d(64, 128, 3, padding=(1, 1))
        #第2層,畳み込み層4
        self.conv6 = nn.Conv2d(128, 128, 3, padding=(1, 1))

        #Batch Normalization
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(32)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(64)
        self.bn5 = nn.BatchNorm2d(128)
        self.bn6 = nn.BatchNorm2d(128)

        #全結合層
        self.fc1 = nn.Linear(128*4*4, 256)
        self.fc2 = nn.Linear(256, 10)
        #プーリング層:(領域サイズ, ストライド)
        self.pool = nn.MaxPool2d(2, 2)
        #ドロップアウト
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.dropout(x)
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        x = self.dropout(x)
        x = F.relu(self.bn5(self.conv5(x)))
        x = self.pool(F.relu(self.bn6(self.conv6(x))))
        x = self.dropout(x)
        x = x.view(-1, 128*4*4)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x