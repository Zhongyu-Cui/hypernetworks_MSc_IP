"""
ResNet-18 for MIMIC-CXR Binary Classification (Image-only)
===========================================================
迁移自 drafts/ResNet18.py 的从零实现版 ResNet-18, 网络结构保持不变 (4 个 stage,
每个 stage 2 个 BasicBlock), 仅将默认 num_classes 改为 1 以配合二分类设置
(U-Zeros 标签策略, forward 输出单个 logit, 训练时配合 nn.BCEWithLogitsLoss 使用,
与 drafts/train_UTKface_preactive.py 的稳定性配置保持一致)。

输入约定: [B, 3, 224, 224]。CXR 原图为单通道灰度图, 经
transforms.Grayscale(num_output_channels=3) 复制为 3 通道, 以匹配标准
ResNet-18 stem (3 通道输入) 并保留接入 ImageNet 预训练权重的可能性。

模型 forward 仅接收图像, 不接收 sex / race; 这两个敏感属性只在训练完成后用于
分组公平性评估 (见 src/utils/mimic_fairness.py)。
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

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        downsample: nn.Module | None = None,
    ) -> None:
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
    ResNet-18 图像分类器, 用于 MIMIC-CXR 二分类。
    Image-only: forward 只接收图像张量, 不接收 sex / race 等敏感属性
    (敏感属性仅在评估阶段用于分组公平性统计)。

    Args:
        num_classes: 输出 logit 维度。
            - 1 -> 配合 nn.BCEWithLogitsLoss (默认; 对应 U-Zeros 二分类设置)
            - 2 -> 配合 nn.CrossEntropyLoss
    """

    def __init__(self, num_classes: int = 1) -> None:
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

    def _make_layer(self, out_channels: int, blocks: int, stride: int) -> nn.Sequential:
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

    def _init_weights(self) -> None:
        # 只对 Conv2d / BatchNorm2d 做 Kaiming / 常数初始化, fc 保持 PyTorch 默认
        # (nn.Linear 默认的 kaiming_uniform_ + fan_in) —— 根目录 CLAUDE.md 记录过
        # PreactivResNet18 把单输出 fc 误用 fan_out 模式 Kaiming 初始化, 导致初始
        # logit 量级 ~10+、训练第一步就坍缩到多数类的教训, 这里特意不重复该错误。
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
    model = ResNet18(num_classes=1)
    x = torch.randn(4, 3, 224, 224)
    logits = model(x)

    print(f"Input  shape : {tuple(x.shape)}")       # (4, 3, 224, 224)
    print(f"Logits shape : {tuple(logits.shape)}")  # (4, 1)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Total params : {n_params:,}")           # 约 11.17M
