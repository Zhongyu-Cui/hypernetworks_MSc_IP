"""
ImageNet 预训练 ResNet-18 for MIMIC-CXR Binary Classification (Image-only)
===========================================================================
与 src/models/resnet18.py 中从零实现的 ResNet18 接口对齐 (forward 仅接收图像,
输出单个 logit, 配合 nn.BCEWithLogitsLoss 使用), 但 backbone 改为
torchvision.models.resnet18 + ImageNet 预训练权重, 仅替换最后的全连接层。

输入约定: [B, 3, 224, 224]。CXR 原图为单通道灰度图, 经
transforms.Grayscale(num_output_channels=3) 复制为 3 通道, 与 ImageNet
预训练 backbone 的 3 通道输入约定兼容。
"""

import torch
import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18


class ResNet18Pretrained(nn.Module):
    """
    ImageNet 预训练 ResNet-18 图像分类器, 用于 MIMIC-CXR 二分类。

    Args:
        num_classes: 输出 logit 维度。
            - 1 -> 配合 nn.BCEWithLogitsLoss (默认; 对应 U-Zeros 二分类设置)
            - 2 -> 配合 nn.CrossEntropyLoss
    """

    def __init__(
        self,
        num_classes: int = 1,
        pretrained: bool = True,
        freeze_backbone: bool = False,
    ) -> None:
        super().__init__()
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = resnet18(weights=weights)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)
        if freeze_backbone:
            self._freeze_backbone()

    def _freeze_backbone(self) -> None:
        """
        冻结特征提取 backbone 的可学习参数 (requires_grad=False)，只保留任务头 fc 可训练——
        即「linear probe on frozen ImageNet features」，作为冻结 regime 下 HN 的 ERM 对照。

        与 HN 变体的 _freeze_backbone 语义严格一致：只冻结卷积/BN 仿射参数 (θ)；BN 的
        running_mean/var 是 buffer (非梯度更新)，仍随训练分布自适应——把实验变量干净隔离
        为「backbone 卷积/线性权重能否被梯度更新」，避免 ImageNet BN 统计硬套拖垮整体 AUC。
        """
        for name, param in self.backbone.named_parameters():
            if not name.startswith("fc."):
                param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


# ============================================================
# 结构自检
# ============================================================
if __name__ == "__main__":
    model = ResNet18Pretrained(num_classes=1)
    x = torch.randn(4, 3, 224, 224)
    logits = model(x)

    print(f"Input  shape : {tuple(x.shape)}")       # (4, 3, 224, 224)
    print(f"Logits shape : {tuple(logits.shape)}")  # (4, 1)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Total params : {n_params:,}")
