"""
Pre-activation ResNet-18 (2D & 3D)
===================================
基于 HyperFusion 仓库 (daniel4725/HyperFusion, base_models.py) 中
`PreactivResBlock_bn` 与 `PreactivResNet` 的风格, 把深度扩展到标准 ResNet-18:
每 stage 包含 2 个 pre-activation 残差块, 共 4 stage = 8 个 block.

更新 (vs. 上一版):
    修复 _init_weights: 不再对 nn.Linear 做 fan_out 模式的 Kaiming-normal,
    恢复 PyTorch 默认 (kaiming_uniform_(a=sqrt(5))).  上一版对 fc2 (out=1)
    用 fan_out 会得到 std≈1.414 的巨大权重, 导致初始 logit 数量级在 10 以上,
    第一个 batch 的 BCE loss 直接飙到 6+, 训练随后坍缩到全预测多数类.
"""

import torch
import torch.nn as nn


# ============================================================
# Dim-aware module 工厂
# ============================================================
def _nd_modules(dim):
    """根据 dim ∈ {2, 3} 返回对应的 (Conv, BN, Dropout, MaxPool, GAP) 类。"""
    if dim == 2:
        return nn.Conv2d, nn.BatchNorm2d, nn.Dropout2d, nn.MaxPool2d, nn.AdaptiveAvgPool2d
    if dim == 3:
        return nn.Conv3d, nn.BatchNorm3d, nn.Dropout3d, nn.MaxPool3d, nn.AdaptiveAvgPool3d
    raise ValueError(f"dim must be 2 or 3, got {dim}")


def conv_bn_relu(in_channels, out_channels, dim=3,
                 kernel_size=3, stride=1, padding=1,
                 bn_momentum=0.05, conv_bias=True):
    """对应仓库 conv3d_bn3d_relu, 同时支持 2D / 3D。"""
    Conv, BN, *_ = _nd_modules(dim)
    return nn.Sequential(
        Conv(in_channels, out_channels, kernel_size,
             stride=stride, padding=padding, bias=conv_bias),
        BN(out_channels, momentum=bn_momentum),
        nn.ReLU(inplace=True),
    )


# ============================================================
# Pre-activation Res Block
# ============================================================
class PreactivResBlock(nn.Module):
    """
    Pre-activation 残差块 (He et al., 2016) 的 dim-aware 版本。
    与 base_models.py / PreactivResBlock_bn 严格对齐:

        BN1 -> ReLU -> Dropout -> Conv1 ->
        BN2 -> ReLU -> Dropout -> Conv2  ( + residual )

    Downsample 分支: BN -> 1x1 Conv (有 BN, 与仓库一致)。
    """

    def __init__(self, in_channels, out_channels, dim=3,
                 stride=1, dropout=0.0,
                 bn_momentum=0.05, conv_bias=True):
        super().__init__()
        Conv, BN, Dropout_nd, *_ = _nd_modules(dim)

        self.bn1 = BN(in_channels, momentum=bn_momentum)
        self.conv1 = Conv(in_channels, out_channels,
                          kernel_size=3, stride=stride, padding=1, bias=conv_bias)
        self.bn2 = BN(out_channels, momentum=bn_momentum)
        self.conv2 = Conv(out_channels, out_channels,
                          kernel_size=3, stride=1, padding=1, bias=conv_bias)

        # 共享一份 ReLU / Dropout (与原仓库写法一致, 注意 ReLU 非 inplace)
        self.relu = nn.ReLU()
        self.dropout = Dropout_nd(p=dropout)

        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                BN(in_channels, momentum=bn_momentum),
                Conv(in_channels, out_channels,
                     kernel_size=1, stride=stride, padding=0, bias=conv_bias),
            )
        else:
            self.downsample = None

    def forward(self, x):
        identity = self.downsample(x) if self.downsample is not None else x

        out = self.bn1(x)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.conv2(out)

        out = out + identity   # pre-activation 末端不再接 ReLU
        return out


# ============================================================
# Pre-activation ResNet-18
# ============================================================
class PreactivResNet18(nn.Module):
    """
    Pre-activation ResNet-18 (2D / 3D 共用).

    架构 (init_features = f, 默认 16):
        stem    :    in_ch ->  f         (3x3 Conv + BN + ReLU)
        maxpool : /2
        layer1  :    f     ->  2f        (2 blocks, 第一个 stride=1)
        layer2  :    2f    ->  4f        (2 blocks, 第一个 stride=2)
        layer3  :    4f    ->  8f        (2 blocks, 第一个 stride=2)
        layer4  :    8f    -> 16f        (2 blocks, 第一个 stride=2)
        head    : GAP -> Drop(0.6) -> Linear -> ReLU -> Drop(0.5) -> Linear

    Args:
        dim:            2 (2D image) 或 3 (3D volume).
        in_channels:    输入通道数.
        n_outputs:      类别数 (>=2 配 nn.CrossEntropyLoss; =1 配 BCEWithLogitsLoss).
        init_features:  起始通道数, 默认 16 (与 HyperFusion AD baseline 一致).
        bn_momentum:    BatchNorm momentum.
    """

    # 每个 stage 的 dropout 率, 与原 PreactivResNet 一致
    _STAGE_DROPOUTS = (0.1, 0.2, 0.2, 0.3)

    def __init__(self, dim=3, in_channels=1, n_outputs=3,
                 init_features=64, bn_momentum=0.1):
        super().__init__()
        assert dim in (2, 3), f"dim must be 2 or 3, got {dim}"
        self.dim = dim
        self.bn_momentum = bn_momentum
        self.in_channels_running = init_features
        f = init_features

        _, _, _, MaxPool, GAP = _nd_modules(dim)

        # Stem
        self.stem = conv_bn_relu(in_channels, f, dim=dim, bn_momentum=bn_momentum)
        self.maxpool = MaxPool(kernel_size=2, stride=2)

        # 4 stages, 每个 stage 2 个 PreactivResBlock
        self.layer1 = self._make_layer(out_channels=2 * f,  blocks=2, stride=1,
                                       dropout=self._STAGE_DROPOUTS[0])
        self.layer2 = self._make_layer(out_channels=4 * f,  blocks=2, stride=2,
                                       dropout=self._STAGE_DROPOUTS[1])
        self.layer3 = self._make_layer(out_channels=8 * f,  blocks=2, stride=2,
                                       dropout=self._STAGE_DROPOUTS[2])
        self.layer4 = self._make_layer(out_channels=16 * f, blocks=2, stride=2,
                                       dropout=self._STAGE_DROPOUTS[3])

        # 分类头 (HyperFusion 双 fc 风格)
        self.gap = GAP(1)
        self.linear_drop1 = nn.Dropout(0.6)
        self.fc1 = nn.Linear(16 * f, 4 * f)
        self.relu = nn.ReLU()
        self.linear_drop2 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(4 * f, n_outputs)

        self._init_weights()

    def _make_layer(self, out_channels, blocks, stride, dropout):
        """同一 stage 内的多个 block 共享 dropout 率; 仅第一个 block 负责降采样/升通道."""
        layers = [PreactivResBlock(
            self.in_channels_running, out_channels,
            dim=self.dim, stride=stride, dropout=dropout,
            bn_momentum=self.bn_momentum,
        )]
        self.in_channels_running = out_channels
        for _ in range(1, blocks):
            layers.append(PreactivResBlock(
                out_channels, out_channels,
                dim=self.dim, stride=1, dropout=dropout,
                bn_momentum=self.bn_momentum,
            ))
        return nn.Sequential(*layers)

    def _init_weights(self):
        """
        仅对 Conv (fan_out, ReLU 增益) 和 BN (gamma=1, beta=0) 做显式初始化.
        nn.Linear 保留 PyTorch 默认 (kaiming_uniform_(a=sqrt(5))), 这相当于
        fan_in 模式 — 对 fc2(out=1) 这种 fan_out=1 的情形, 显式 fan_out 初始化
        会得到 std≈1.4 的巨大权重并导致初始 logit 爆炸.
        """
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Conv3d)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm3d)):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            # Linear: 保留 PyTorch 默认初始化, 不显式覆盖

    def forward(self, x):
        x = self.stem(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.gap(x)
        x = torch.flatten(x, 1)

        x = self.linear_drop1(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.linear_drop2(x)
        x = self.fc2(x)
        return x


# ============================================================
# Convenience constructors
# ============================================================
def PreactivResNet18_2D(in_channels=3, n_outputs=2, init_features=16, **kwargs):
    """2D 版本: 默认 in_channels=3 (RGB), n_outputs=2 (binary)."""
    return PreactivResNet18(dim=2, in_channels=in_channels, n_outputs=n_outputs,
                            init_features=init_features, **kwargs)


def PreactivResNet18_3D(in_channels=1, n_outputs=3, init_features=16, **kwargs):
    """3D 版本: 默认 in_channels=1 (灰度 MRI), n_outputs=3 (CN/MCI/AD)."""
    return PreactivResNet18(dim=3, in_channels=in_channels, n_outputs=n_outputs,
                            init_features=init_features, **kwargs)


# ============================================================
# 结构自检
# ============================================================
if __name__ == "__main__":
    # ----- 2D 版本 -----
    print("=== PreactivResNet18_2D (RGB, binary) ===")
    m2d = PreactivResNet18_2D(in_channels=3, n_outputs=1, init_features=16)
    x2d = torch.randn(4, 3, 224, 224)
    y2d = m2d(x2d)
    print(f"  Input  shape : {tuple(x2d.shape)}")
    print(f"  Output shape : {tuple(y2d.shape)}")
    print(f"  Total params : {sum(p.numel() for p in m2d.parameters()):,}")
    print(f"  Output stats : mean={y2d.mean().item():+.3f}  std={y2d.std().item():.3f}")
    print(f"                  (健康初始化下 std 应在个位数内, 不应 ~10+)")

    # ----- 3D 版本 -----
    print("\n=== PreactivResNet18_3D (gray MRI, 3-class) ===")
    m3d = PreactivResNet18_3D(in_channels=1, n_outputs=3, init_features=16)
    x3d = torch.randn(2, 1, 64, 96, 64)
    y3d = m3d(x3d)
    print(f"  Input  shape : {tuple(x3d.shape)}")
    print(f"  Output shape : {tuple(y3d.shape)}")
    print(f"  Total params : {sum(p.numel() for p in m3d.parameters()):,}")