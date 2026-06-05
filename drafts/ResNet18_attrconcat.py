"""
ResNet-18 with Attribute Concatenation at Input
===============================================
不使用 HyperNetwork。把 sex 和 race 做 embedding 后 broadcast 成空间维度,
与图像沿通道维度 concat, 作为 ResNet-18 的输入。

输入通道数 = 3 (RGB) + sex_embed + race_embed
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Basic Residual Block (和之前一样)
# ============================================================
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels,
                               kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels,
                               kernel_size=3, stride=1, padding=1, bias=False)
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
# ResNet-18 with Attribute Concat
# ============================================================
class ResNet18AttrConcat(nn.Module):
    """
    把 sex/race 的 embedding 当作额外通道拼接到输入图像上。

    Args:
        out_dim    : 输出维度 (BCE 二分类 = 1, 多分类 = k)
        num_sex    : sex 的类别数 (UTKFace = 2)
        num_race   : race 的类别数 (UTKFace = 5)
        sex_embed  : sex embedding 维度 (默认 2)
        race_embed : race embedding 维度 (默认 4)
        img_channels: 输入图像通道数 (默认 3)
    """

    def __init__(self, out_dim=1, num_sex=2, num_race=5,
                 sex_embed=2, race_embed=4, img_channels=3):
        super().__init__()
        self.in_channels_running = 64
        self.img_channels = img_channels
        self.sex_embed_dim = sex_embed
        self.race_embed_dim = race_embed

        # 敏感属性 embedding
        self.sex_emb = nn.Embedding(num_sex, sex_embed)
        self.race_emb = nn.Embedding(num_race, race_embed)

        # conv1 输入通道数 = 图像通道 + 属性通道
        total_in = img_channels + sex_embed + race_embed   # 3 + 2 + 4 = 9

        # --- Stem ---
        self.conv1 = nn.Conv2d(total_in, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # --- 4 个 stage ---
        self.layer1 = self._make_layer(64,  2, stride=1)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)

        # --- 分类头 ---
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, out_dim)

        self._init_weights()

    def _make_layer(self, out_channels, blocks, stride):
        downsample = None
        if stride != 1 or self.in_channels_running != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels_running, out_channels,
                          kernel_size=1, stride=stride, bias=False),
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

    def _build_attr_map(self, sex, race, H, W):
        """
        把 sex/race 的 embedding 广播成 [B, sex_embed+race_embed, H, W].
        同一张图的每个空间位置都是相同的属性向量。
        """
        s = self.sex_emb(sex)     # [B, sex_embed]
        r = self.race_emb(race)   # [B, race_embed]
        attr = torch.cat([s, r], dim=1)               # [B, sex_embed+race_embed]
        # 扩展到 [B, C, H, W]: 在空间维上复制同一个向量
        attr = attr.unsqueeze(-1).unsqueeze(-1)       # [B, C, 1, 1]
        attr = attr.expand(-1, -1, H, W)              # [B, C, H, W]
        return attr

    def forward(self, image, sex, race):
        """
        image: [B, 3, H, W]
        sex  : [B] long
        race : [B] long
        returns: [B, out_dim] logits
        """
        B, _, H, W = image.shape

        # 1) 构造属性 map 并和图像沿通道维 concat
        attr_map = self._build_attr_map(sex, race, H, W)   # [B, C_attr, H, W]
        x = torch.cat([image, attr_map], dim=1)            # [B, 3+C_attr, H, W]

        # 2) 标准 ResNet-18 前向
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
    model = ResNet18AttrConcat(out_dim=1, num_sex=2, num_race=5)

    B = 4
    image = torch.randn(B, 3, 224, 224)
    sex = torch.randint(0, 2, (B,))
    race = torch.randint(0, 5, (B,))

    pred = model(image, sex, race)

    print(f"Image: {tuple(image.shape)}")
    print(f"Sex  : {tuple(sex.shape)}  values={sex.tolist()}")
    print(f"Race : {tuple(race.shape)}  values={race.tolist()}")
    print(f"Pred : {tuple(pred.shape)}")  # [4, 1]

    # 参数量
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal params: {n_params:,}")

    # 验证反向传播
    loss = pred.sum()
    loss.backward()
    print("✅ Backward pass OK.")
    