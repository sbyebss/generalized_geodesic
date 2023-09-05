# pylint: skip-file
import math

import torch
from torch import nn
from torch.nn import functional as F

Half_width = 512
layer_width = 128

# Best model from the leadboard:
# https://paperswithcode.com/sota/image-classification-on-kuzushiji-mnist


class SpinalNet(nn.Module):
    """
    Based on - https://github.com/kkweon/mnist-competition
    from: https://github.com/ranihorev/Kuzushiji_MNIST/blob/master/KujuMNIST.ipynb
    """

    def two_conv_pool(self, in_channels, f1, f2):
        s = nn.Sequential(
            nn.Conv2d(in_channels, f1, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(f1),
            nn.ReLU(inplace=True),
            nn.Conv2d(f1, f2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(f2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        for m in s.children():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        return s

    def three_conv_pool(self, in_channels, f1, f2, f3):
        s = nn.Sequential(
            nn.Conv2d(in_channels, f1, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(f1),
            nn.ReLU(inplace=True),
            nn.Conv2d(f1, f2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(f2),
            nn.ReLU(inplace=True),
            nn.Conv2d(f2, f3, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(f3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        for m in s.children():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        return s

    def __init__(self, num_class=26):
        super().__init__()
        self.l1 = self.two_conv_pool(3, 64, 64)
        self.l2 = self.two_conv_pool(64, 128, 128)
        self.l3 = self.three_conv_pool(128, 256, 256, 256)
        self.l4 = self.three_conv_pool(256, 256, 256, 256)

        self.fc_spinal_layer1 = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(Half_width, layer_width),
            nn.BatchNorm1d(layer_width),
            nn.ReLU(inplace=True),
        )
        self.fc_spinal_layer2 = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(Half_width + layer_width, layer_width),
            nn.BatchNorm1d(layer_width),
            nn.ReLU(inplace=True),
        )
        self.fc_spinal_layer3 = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(Half_width + layer_width, layer_width),
            nn.BatchNorm1d(layer_width),
            nn.ReLU(inplace=True),
        )
        self.fc_spinal_layer4 = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(Half_width + layer_width, layer_width),
            nn.BatchNorm1d(layer_width),
            nn.ReLU(inplace=True),
        )
        self.fc_out = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(layer_width * 4, num_class),
        )
        # self.softmax = nn.Softmax(dim=1)

    def forward(self, x, label):
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        x = x.view(x.size(0), -1)

        x1 = self.fc_spinal_layer1(x[:, 0:Half_width])
        x2 = self.fc_spinal_layer2(
            torch.cat([x[:, Half_width : 2 * Half_width], x1], dim=1)
        )
        x3 = self.fc_spinal_layer3(torch.cat([x[:, 0:Half_width], x2], dim=1))
        x4 = self.fc_spinal_layer4(
            torch.cat([x[:, Half_width : 2 * Half_width], x3], dim=1)
        )

        x = torch.cat([x1, x2], dim=1)
        x = torch.cat([x, x3], dim=1)
        x = torch.cat([x, x4], dim=1)

        x = self.fc_out(x)
        return x
        # return self.softmax(x)


class LeNet(nn.Module):
    def __init__(self, num_class=10):
        # only applicable to 32*32
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(400, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_class)
        self.prelu = nn.PReLU()

    def forward(self, x, label):
        del label
        # label_embedding = self.emb(label.long())
        x = F.max_pool2d(self.prelu(self.conv1(x)), 2)
        x = F.max_pool2d(self.prelu(self.conv2(x)), 2)
        x = x.view(x.size()[0], -1)
        x = self.prelu(self.fc1(x))
        top_hidden = self.prelu(self.fc2(x))
        x = self.fc3(top_hidden)
        return x


class SimpleMLP(nn.Module):
    def __init__(self, input_dim, num_class=10, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 500),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(500, 200),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(200, 100),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(100, num_class),
        )

    def forward(self, X, *args):
        return self.net(X)


if __name__ == "__main__":
    from torchinfo import summary

    snet = SpinalNet()
    summary(snet, input_size=[(7, 3, 32, 32), (7,)])

    # lenet = LeNet()
    # summary(lenet, input_size=[(7, 3, 32, 32), (7,)])
