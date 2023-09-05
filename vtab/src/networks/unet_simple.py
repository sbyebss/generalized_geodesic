# pylint: skip-file
import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaIN(nn.Module):
    def __init__(self, in_channel, num_classes, eps=1e-5):
        super().__init__()
        self.num_classes = num_classes
        self.eps = eps
        self.l1 = nn.Linear(num_classes, in_channel * 4, bias=True)  # bias is good :)

    def c_norm(self, x, bs, ch, eps=1e-7):
        assert isinstance(x, torch.cuda.FloatTensor)
        x_var = x.var(dim=-1) + eps
        x_std = x_var.sqrt().view(bs, ch, 1, 1)
        x_mean = x.mean(dim=-1).view(bs, ch, 1, 1)
        return x_std, x_mean

    def forward(self, x, y):
        assert x.size(0) == y.size(0)
        size = x.size()
        bs, ch = size[:2]
        x_ = x.view(bs, ch, -1)
        y_ = self.l1(y).view(bs, ch, -1)
        x_std, x_mean = self.c_norm(x_, bs, ch, eps=self.eps)
        y_std, y_mean = self.c_norm(y_, bs, ch, eps=self.eps)
        out = ((x - x_mean.expand(size)) / x_std.expand(size)) * y_std.expand(
            size
        ) + y_mean.expand(size)
        return out


def r_double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True),
    )


class Conditional_UNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes
        self.dconv_down1 = r_double_conv(3, 64)
        self.dconv_down2 = r_double_conv(64, 128)
        self.dconv_down3 = r_double_conv(128, 256)
        self.dconv_down4 = r_double_conv(256, 512)

        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.maxpool = nn.MaxPool2d(2)
        self.dropout = nn.Dropout(p=0.3)
        # self.dropout_half = HalfDropout(p=0.3)

        self.adain3 = AdaIN(512, num_classes=num_classes)
        self.adain2 = AdaIN(256, num_classes=num_classes)
        self.adain1 = AdaIN(128, num_classes=num_classes)

        self.dconv_up3 = r_double_conv(256 + 512, 256)
        self.dconv_up2 = r_double_conv(128 + 256, 128)
        self.dconv_up1 = r_double_conv(64 + 128, 64)

        self.conv_last = nn.Conv2d(64, 3, 1)
        self.activation = nn.Tanh()
        # self.init_weight()

    def forward(self, x, c):
        # input c is hard label
        c = F.one_hot(c, num_classes=self.num_classes).float()
        conv1 = self.dconv_down1(x)
        x = self.maxpool(conv1)

        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)

        conv3 = self.dconv_down3(x)
        x = self.maxpool(conv3)

        x = self.dconv_down4(x)

        # dropout
        # x = self.dropout_half(x)

        x = self.adain3(x, c)
        x = self.upsample(x)
        x = self.dropout(x)
        x = torch.cat([x, conv3], dim=1)

        x = self.dconv_up3(x)

        x = self.adain2(x, c)
        x = self.upsample(x)
        x = self.dropout(x)
        x = torch.cat([x, conv2], dim=1)

        x = self.dconv_up2(x)

        x = self.adain1(x, c)
        x = self.upsample(x)
        x = self.dropout(x)
        x = torch.cat([x, conv1], dim=1)

        x = self.dconv_up1(x)

        out = self.conv_last(x)

        return self.activation(out)


if __name__ == "__main__":
    from torchinfo import summary

    unet = Conditional_UNet(num_classes=11)
    summary(unet, input_size=[(7, 3, 32, 32), (7,)])
