"""
ImageNet 预训练 ResNet-18 + HyperHead for MIMIC-CXR Binary Classification
=========================================================================
在 image-only baseline (src/models/resnet18_pretrained.py) 的基础上，把最后的
全连接分类头 (nn.Linear) 替换为一个由「敏感属性」驱动的 HyperNetwork：

  - Backbone : 与 baseline **完全一致**——torchvision.models.resnet18 +
               ImageNet 预训练权重，输出 512 维特征 (fc 之前的表征)。backbone
               参数照常参与微调。
  - HyperNet : 输入 MIMIC-CXR 预处理后包含的全部离散敏感属性
               (sex / race / age_group)，输出 **逐样本** 的分类头权重与偏置。
  - 预测     : 用 HyperNet 生成的逐样本线性层对 backbone 特征做分类。

设计参考 drafts/ResNet18_hyperhead.py（UTKFace 原型），但做了两点适配：
  1. backbone 换成与 MIMIC baseline 对齐的 ImageNet 预训练 torchvision ResNet-18，
     而非 drafts 里从零实现的版本，确保与 baseline 的图像通路严格可比。
  2. 条件属性从 (sex, race) 扩展为 (sex, race, age_group)，覆盖
     mimic_cxr_dataset.py 返回的全部离散敏感属性；这也对应 baseline 报告里
     EqOdds gap 最大的 Age 维度与最弱交叉组 Male & Non-White & >=60。

forward 签名: (image, sex, race, age_group)，与 MIMICCXRDataset.__getitem__
返回的 (image, label, sex, race, age_group) 中的属性顺序一致，训练循环可直接解包。
"""

import torch
import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18


# ============================================================
# HyperNetwork: 由离散敏感属性生成分类头 (fc) 的权重与偏置
# ============================================================
class HyperHeadNet(nn.Module):
    """
    输入若干离散敏感属性 (sex / race / age_group)，输出主干分类头的逐样本参数。

    每个属性先各自过一个 nn.Embedding，再拼接成条件向量 cond，经一个小 MLP
    映射成「展平后的分类头权重 + 偏置」。因此同一张图像在不同敏感属性下会得到
    不同的分类边界，这正是用超网络注入属性、改善子群公平性的机制。

    Args:
        in_dim     : 主干分类头的输入维度 (ResNet-18 = 512)。
        out_dim    : 主干分类头的输出维度 (No Finding 二分类 = 1)。
        num_sex    : sex 的类别数 (MIMIC: Male/Female = 2)。
        num_race   : race 的类别数 (MIMIC: White/Non-White = 2)。
        num_age    : age_group 的类别数 (MIMIC: <thr / >=thr = 2)。
        sex_embed  : sex embedding 维度。
        race_embed : race embedding 维度。
        age_embed  : age_group embedding 维度。
        hidden_dim : HyperNet 内部 MLP 的隐藏层维度。
    """

    def __init__(
        self,
        in_dim: int = 512,
        out_dim: int = 1,
        num_sex: int = 2,
        num_race: int = 2,
        num_age: int = 2,
        sex_embed: int = 4,
        race_embed: int = 4,
        age_embed: int = 4,
        hidden_dim: int = 64,
    ) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        # 各离散属性各自的 embedding 表（离散变量 -> 稠密向量）
        self.sex_emb = nn.Embedding(num_sex, sex_embed)
        self.race_emb = nn.Embedding(num_race, race_embed)
        self.age_emb = nn.Embedding(num_age, age_embed)

        cond_dim = sex_embed + race_embed + age_embed
        weight_numel = in_dim * out_dim   # 展平后的 fc 权重元素数 (512*1)
        bias_numel = out_dim              # fc 偏置元素数 (1)

        # 把条件向量映射成「展平的分类头权重 + 偏置」
        self.mlp = nn.Sequential(
            nn.Linear(cond_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, weight_numel + bias_numel),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        """
        参数初始化。

        关键：把 MLP 最后一层 (生成头参数的那层) 初始化为小权重 + 零偏置，使得
        训练初期生成的分类头权重接近 0、logit 接近 0、各样本几乎一致。这样做有两个
        好处：(1) 起步稳定，避免重蹈 PreactivResNet18 里 fc(out=1) 用 fan_out 初始化
        导致 logit 爆炸、训练坍缩到多数类的覆辙；(2) 让超网络从「与属性无关的统一头」
        出发，逐步学到属性相关的差异，而不是一上来就引入高方差扰动。
        """
        # 第一层 (cond -> hidden) 用默认 Linear 初始化即可
        nn.init.normal_(self.mlp[-1].weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(
        self, sex: torch.Tensor, race: torch.Tensor, age_group: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            sex       : LongTensor [B]，取值 0/1。
            race      : LongTensor [B]，取值 0/1。
            age_group : LongTensor [B]，取值 0/1。

        Returns:
            weight: [B, out_dim, in_dim] 逐样本分类头权重。
            bias  : [B, out_dim]         逐样本分类头偏置。
        """
        s = self.sex_emb(sex)         # [B, sex_embed]
        r = self.race_emb(race)       # [B, race_embed]
        a = self.age_emb(age_group)   # [B, age_embed]
        cond = torch.cat([s, r, a], dim=1)   # [B, cond_dim]

        out = self.mlp(cond)          # [B, in_dim*out_dim + out_dim]
        weight = out[:, : self.in_dim * self.out_dim]   # [B, in_dim*out_dim]
        bias = out[:, self.in_dim * self.out_dim :]     # [B, out_dim]

        # reshape 成标准 fc 权重形状 [B, out_dim, in_dim]
        weight = weight.view(-1, self.out_dim, self.in_dim)
        return weight, bias


# ============================================================
# ResNet-18 (ImageNet 预训练) + HyperHead
# ============================================================
class ResNet18HyperHead(nn.Module):
    """
    与 MIMIC baseline 对齐的 ImageNet 预训练 ResNet-18 backbone，
    其分类头由 HyperHeadNet 依据离散敏感属性逐样本生成。

    Args:
        num_classes: 分类头输出维度。
            - 1 -> 配合 nn.BCEWithLogitsLoss (默认; No Finding U-Zeros 二分类)
            - k -> 配合 nn.CrossEntropyLoss
        num_sex, num_race, num_age: 三个离散敏感属性的类别数 (MIMIC 均为 2)。
        sex_embed, race_embed, age_embed: 各属性 embedding 维度。
        hyper_hidden: HyperNet 内部 MLP 隐藏层维度。
    """

    def __init__(
        self,
        num_classes: int = 1,
        num_sex: int = 2,
        num_race: int = 2,
        num_age: int = 2,
        sex_embed: int = 4,
        race_embed: int = 4,
        age_embed: int = 4,
        hyper_hidden: int = 64,
    ) -> None:
        super().__init__()

        # --- Backbone: 与 baseline 完全一致的 ImageNet 预训练 ResNet-18 ---
        # 把 fc 换成 Identity，使 backbone(x) 直接输出 512 维特征 (fc 之前的表征)。
        self.backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.feat_dim = self.backbone.fc.in_features   # 512
        self.backbone.fc = nn.Identity()

        # --- HyperNet 取代原 fc，按敏感属性生成逐样本分类头 ---
        self.hyper = HyperHeadNet(
            in_dim=self.feat_dim,
            out_dim=num_classes,
            num_sex=num_sex,
            num_race=num_race,
            num_age=num_age,
            sex_embed=sex_embed,
            race_embed=race_embed,
            age_embed=age_embed,
            hidden_dim=hyper_hidden,
        )

    def extract_features(self, image: torch.Tensor) -> torch.Tensor:
        """只跑 backbone，返回 [B, feat_dim] 特征 (fc 之前)。"""
        return self.backbone(image)

    def forward(
        self,
        image: torch.Tensor,
        sex: torch.Tensor,
        race: torch.Tensor,
        age_group: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            image     : [B, 3, H, W] 图像 (CXR 经 Grayscale(3) 复制为 3 通道)。
            sex       : [B] long，0=Male, 1=Female。
            race      : [B] long，0=White, 1=Non-White。
            age_group : [B] long，0=<thr, 1=>=thr。

        Returns:
            logits: [B, num_classes]。
        """
        # 1) backbone 抽取图像特征 (与 baseline 同一条通路)
        feat = self.extract_features(image)            # [B, 512]

        # 2) HyperNet 依据敏感属性生成逐样本分类头参数
        weight, bias = self.hyper(sex, race, age_group)  # [B, C, 512], [B, C]

        # 3) 逐样本线性变换 y_i = W_i x_i + b_i (batched matmul)
        #    [B, C, 512] @ [B, 512, 1] -> [B, C, 1]
        logits = torch.bmm(weight, feat.unsqueeze(-1)).squeeze(-1) + bias  # [B, C]
        return logits


# ============================================================
# 结构自检
# ============================================================
if __name__ == "__main__":
    model = ResNet18HyperHead(num_classes=1)

    B = 4
    image = torch.randn(B, 3, 224, 224)
    sex = torch.randint(0, 2, (B,))
    race = torch.randint(0, 2, (B,))
    age = torch.randint(0, 2, (B,))

    logits = model(image, sex, race, age)

    print(f"Image      : {tuple(image.shape)}")
    print(f"Sex/Race/Age values: {sex.tolist()} / {race.tolist()} / {age.tolist()}")
    print(f"Logits     : {tuple(logits.shape)}")   # (4, 1)

    backbone_params = sum(p.numel() for p in model.backbone.parameters())
    hyper_params = sum(p.numel() for p in model.hyper.parameters())
    print(f"\nBackbone params: {backbone_params:,}")
    print(f"HyperNet params: {hyper_params:,}")
    print(f"Total params   : {backbone_params + hyper_params:,}")

    # 验证反向传播
    loss = logits.sum()
    loss.backward()
    print("\nBackward pass OK.")
