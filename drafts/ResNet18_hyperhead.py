"""
ResNet-18 + HyperNetwork for UTKFace Age Regression
====================================================
- Backbone (ResNet-18): 人脸图像 -> 512 维特征
- HyperNet: (sex, race) -> fc 层的 weight 和 bias
- 最终预测: 年龄 (分类)
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
# HyperNetwork: 生成 fc 层的权重
# ============================================================
class HyperNet(nn.Module):
    """
    输入: sex [B] (long, 取值 0/1), race [B] (long, 取值 0~4)
    输出: weight [B, out_dim, in_dim], bias [B, out_dim]

    Args:
        in_dim     : 主网络分类头的输入维度 (ResNet-18 = 512)
        out_dim    : 主网络分类头的输出维度 (年龄回归 = 1)
        num_sex    : sex 的类别数 (UTKFace = 2)
        num_race   : race 的类别数 (UTKFace = 5)
        sex_embed  : sex 的 embedding 维度
        race_embed : race 的 embedding 维度
        hidden_dim : HyperNet 内部 MLP 的隐藏层维度
    """

    def __init__(self, in_dim=512, out_dim=1,
                 num_sex=2, num_race=5,
                 sex_embed=4, race_embed=8,
                 hidden_dim=64):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        # 各自 embedding
        self.sex_emb = nn.Embedding(num_sex, sex_embed)
        self.race_emb = nn.Embedding(num_race, race_embed)

        # MLP 把拼接后的属性 embedding 映射成 fc 的 weight + bias
        cond_dim = sex_embed + race_embed
        weight_numel = in_dim * out_dim   # 512 * 1 = 512
        bias_numel = out_dim              # 1

        self.mlp = nn.Sequential(
            nn.Linear(cond_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, weight_numel + bias_numel),
        )

    def forward(self, sex, race):
        """
        sex, race: LongTensor of shape [B]
        returns:
            weight: [B, out_dim, in_dim]
            bias  : [B, out_dim]
        """
        s = self.sex_emb(sex)      # [B, sex_embed]
        r = self.race_emb(race)    # [B, race_embed]
        cond = torch.cat([s, r], dim=1)  # [B, cond_dim]

        out = self.mlp(cond)       # [B, 512 + 1]
        weight = out[:, :self.in_dim * self.out_dim]  # [B, 512]
        bias = out[:, self.in_dim * self.out_dim:]    # [B, 1]

        # reshape 成 fc 层参数的标准形状
        weight = weight.view(-1, self.out_dim, self.in_dim)  # [B, 1, 512]
        return weight, bias


# ============================================================
# ResNet-18 with HyperHead
# ============================================================
class ResNet18HyperHead(nn.Module):
    """
    ResNet-18 (backbone) + HyperNet (生成分类头权重).

    Args:
        out_dim: 输出维度
            - 1 -> 二分类, 配合 nn.BCEWithLogitsLoss
            - k -> k 分类, 配合 nn.CrossEntropyLoss
        num_sex, num_race: 敏感属性的类别数
    """

    def __init__(self, out_dim=1, num_sex=2, num_race=5,
                 sex_embed=4, race_embed=8, hyper_hidden=64):
        super().__init__()
        self.in_channels_running = 64
        self.feat_dim = 512
        self.out_dim = out_dim

        # --- Backbone (和之前的 ResNet18 一样, 只是去掉 fc) ---
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(64,  2, stride=1)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # --- HyperNet 替代原来的 self.fc ---
        self.hyper = HyperNet(
            in_dim=self.feat_dim, out_dim=out_dim,
            num_sex=num_sex, num_race=num_race,
            sex_embed=sex_embed, race_embed=race_embed,
            hidden_dim=hyper_hidden,
        )

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

    def extract_features(self, x):
        """只跑 backbone, 返回 [B, 512]."""
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x, inplace=True)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)   # [B, 512]
        return x

    def forward(self, image, sex, race):
        """
        image: [B, 3, H, W]
        sex  : [B] long
        race : [B] long
        returns: [B, out_dim]  (logits)
        """
        # 1) backbone 抽特征
        feat = self.extract_features(image)            # [B, 512]

        # 2) hyper 生成分类头的权重和偏置 (每个样本不同)
        weight, bias = self.hyper(sex, race)           # [B, 1, 512], [B, 1]

        # 3) 逐样本做线性变换: y_i = W_i x_i + b_i
        #    用 bmm: [B, 1, 512] @ [B, 512, 1] -> [B, 1, 1]
        y = torch.bmm(weight, feat.unsqueeze(-1))      # [B, 1, 1]
        y = y.squeeze(-1) + bias                       # [B, 1]
        return y


# ============================================================
# 结构自检
# ============================================================
if __name__ == "__main__":
    model = ResNet18HyperHead(out_dim=1, num_sex=2, num_race=5)

    B = 4
    image = torch.randn(B, 3, 224, 224)
    sex = torch.randint(0, 2, (B,))
    race = torch.randint(0, 5, (B,))

    pred = model(image, sex, race)

    print(f"Image: {tuple(image.shape)}")
    print(f"Sex  : {tuple(sex.shape)}  values={sex.tolist()}")
    print(f"Race : {tuple(race.shape)}  values={race.tolist()}")
    print(f"Pred : {tuple(pred.shape)}")  # [4, 1]

    # 参数量分解
    backbone_params = sum(p.numel() for n, p in model.named_parameters()
                          if not n.startswith("hyper"))
    hyper_params = sum(p.numel() for n, p in model.named_parameters()
                       if n.startswith("hyper"))
    print(f"\nBackbone params: {backbone_params:,}")  # 约 11.17M - 513 = ~11.17M
    print(f"HyperNet params: {hyper_params:,}")       # 约 33K

    # 验证反向传播
    loss = pred.sum()
    loss.backward()
    print("\n✅ Backward pass OK.")
