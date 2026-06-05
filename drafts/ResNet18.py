"""
ResNet-18 Binary Classification Model
=====================================
纯净版 ResNet-18 二分类模型, 从零实现。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Basic Residual Block
# ============================================================
class BasicBlock(nn.Module):
    """两个 3x3 卷积 + 残差连接。"""
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=3, stride=stride, padding=1, bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels,
            kernel_size=3, stride=1, padding=1, bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out, inplace=True)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out + identity
        out = F.relu(out, inplace=True)
        return out


# ============================================================
# ResNet-18
# ============================================================
class ResNet18(nn.Module):
    """
    ResNet-18 Binary Classifier.

    Args:
        num_classes: 类别数。
            - 2 -> 配合 nn.CrossEntropyLoss
            - 1 -> 配合 nn.BCEWithLogitsLoss
    """

    def __init__(self, num_classes=2):
        super().__init__()
        self.in_channels_running = 64

        # Stem
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # 4 个 stage, 每个 stage 包含 2 个 BasicBlock
        self.layer1 = self._make_layer(out_channels=64,  blocks=2, stride=1)
        self.layer2 = self._make_layer(out_channels=128, blocks=2, stride=2)
        self.layer3 = self._make_layer(out_channels=256, blocks=2, stride=2)
        self.layer4 = self._make_layer(out_channels=512, blocks=2, stride=2)

        # 分类头
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

        # 参数初始化
        self._init_weights()

    def _make_layer(self, out_channels, blocks, stride):
        downsample = None
        if stride != 1 or self.in_channels_running != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.in_channels_running, out_channels,
                    kernel_size=1, stride=stride, bias=False,
                ),
                nn.BatchNorm2d(out_channels),
            )

        layers = [BasicBlock(self.in_channels_running, out_channels, stride, downsample)]
        self.in_channels_running = out_channels
        for _ in range(1, blocks):
            layers.append(BasicBlock(self.in_channels_running, out_channels))

        return nn.Sequential(*layers)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x, inplace=True)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


# ============================================================
# 结构自检
# ============================================================
if __name__ == "__main__":
    model = ResNet18(num_classes=2)
    x = torch.randn(4, 3, 224, 224)
    logits = model(x)

    print(f"Input  shape : {tuple(x.shape)}")      # (4, 3, 224, 224)
    print(f"Logits shape : {tuple(logits.shape)}") # (4, 2)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Total params : {n_params:,}")          # 约 11.17M