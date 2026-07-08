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
        init_std: float = 1e-3,
    ) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.init_std = init_std

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

        init_std 控制末层权重初始尺度：默认 1e-3（近零起步）；诊断「属性通路是否被
        过小初始化人为压制」时可调大（见 src/training/diagnose_hyperhead_pathway.py）。
        """
        # 第一层 (cond -> hidden) 用默认 Linear 初始化即可
        nn.init.normal_(self.mlp[-1].weight, mean=0.0, std=self.init_std)
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
# HyperNetwork (单属性版): 由单一 age 生成分类头 (fc) 的权重与偏置
# ============================================================
class AgeHyperHeadNet(nn.Module):
    """
    HyperHeadNet 的单属性版：只用 age_group (HAM10000: 4 有效组) 一个敏感属性生成主干分类头
    的逐样本参数。结构与 HyperHeadNet 对齐 (embedding → 小 MLP → 展平 fc 权重+偏置、末层近零
    初始化)，仅条件向量由「sex+race+age 三 embedding 拼接」缩为「单一 age embedding」。

    为什么只用 age：HAM 条件互信息诊断 (作业 68297) 显示 sex 轴 I(Y;sex|X)≈0、age 轴读出可
    复现 I(Y;age|X)>0。而 HyperHead「在共享 φ(X) 上按属性生成逐样本线性头」恰好等价于诊断 E1
    的 int 模型 (逐组独立线性头)——是与该正向信号最对口的架构。

    ⚠️ age_group==-1 (0-20 排除组) 不是合法 embedding 索引，训练/评估前须在数据侧过滤 age_group>=0。

    Args:
        in_dim    : 主干分类头输入维度 (ResNet-18 = 512)。
        out_dim   : 主干分类头输出维度 (malignant 二分类 = 1)。
        num_age   : age_group 有效类别数 (HAM: 4)。
        age_embed : age embedding 维度。
        hidden_dim: HyperNet 内部 MLP 隐藏层维度。
        init_std  : 末层权重初始尺度 (近零起步，使初始 logit≈0、各样本几乎一致)。
    """

    def __init__(
        self,
        in_dim: int = 512,
        out_dim: int = 1,
        num_age: int = 4,
        age_embed: int = 4,
        hidden_dim: int = 64,
        init_std: float = 1e-3,
    ) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.init_std = init_std

        self.age_emb = nn.Embedding(num_age, age_embed)

        weight_numel = in_dim * out_dim   # 展平后的 fc 权重元素数 (512*1)
        bias_numel = out_dim              # fc 偏置元素数 (1)
        self.mlp = nn.Sequential(
            nn.Linear(age_embed, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, weight_numel + bias_numel),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        """末层近零 + 零偏置：初期生成头≈0、logit≈0、各样本几乎一致 (与 HyperHeadNet 同理)。"""
        nn.init.normal_(self.mlp[-1].weight, mean=0.0, std=self.init_std)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, age_group: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """age_group: [B] long (0–3) → (weight [B, out_dim, in_dim], bias [B, out_dim])。"""
        a = self.age_emb(age_group)            # [B, age_embed]
        out = self.mlp(a)                      # [B, in_dim*out_dim + out_dim]
        weight = out[:, : self.in_dim * self.out_dim].view(-1, self.out_dim, self.in_dim)
        bias = out[:, self.in_dim * self.out_dim :]
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
        init_std: float = 1e-3,
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
            init_std=init_std,
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
# ResNet-18 + HyperHead for HAM10000 (单一 age 属性)
# ============================================================
class ResNet18HyperHeadAge(nn.Module):
    """
    HAM10000 版 HyperHead: backbone 与 baseline 完全一致 (ImageNet 预训练 torchvision
    ResNet-18, fc→Identity, 输出 512 维特征)，分类头由 AgeHyperHeadNet 依据单一 age_group
    逐样本生成。**只条件化 age** (sex 轴诊断读弱/null，先不接)。

    forward 签名 (image, age_group)，与 HAM10000Dataset 的 (image, label, sex, age_group)
    中 age 对齐 (sex 仍解包但不进模型)。⚠️ 仅接受 age_group∈{0..3}; -1 须在数据侧先过滤。

    Args:
        num_classes : 分类头输出维度 (malignant 二分类 → 1)。
        num_age     : age_group 有效类别数 (HAM: 4)。
        age_embed   : age embedding 维度。
        hyper_hidden: HyperNet 内部 MLP 隐藏层维度。
        init_std    : 末层近零初始尺度 (与 baseline 起步稳定一致)。
    """

    def __init__(
        self,
        num_classes: int = 1,
        num_age: int = 4,
        age_embed: int = 4,
        hyper_hidden: int = 64,
        init_std: float = 1e-3,
    ) -> None:
        super().__init__()
        # backbone 与 baseline 完全一致 (与 ResNet18HyperHead 同构)
        self.backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.feat_dim = self.backbone.fc.in_features   # 512
        self.backbone.fc = nn.Identity()

        # 单属性 age HyperNet 取代原 fc
        self.hyper = AgeHyperHeadNet(
            in_dim=self.feat_dim,
            out_dim=num_classes,
            num_age=num_age,
            age_embed=age_embed,
            hidden_dim=hyper_hidden,
            init_std=init_std,
        )

    def extract_features(self, image: torch.Tensor) -> torch.Tensor:
        """只跑 backbone，返回 [B, feat_dim] 特征 (fc 之前)。"""
        return self.backbone(image)

    def forward(self, image: torch.Tensor, age_group: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image    : [B, 3, 224, 224] 皮肤镜 RGB 图像。
            age_group: [B] long，0–3 (4 有效年龄组；-1 须已过滤)。

        Returns:
            logits: [B, num_classes]。
        """
        feat = self.extract_features(image)            # [B, 512]
        weight, bias = self.hyper(age_group)           # [B, C, 512], [B, C]
        logits = torch.bmm(weight, feat.unsqueeze(-1)).squeeze(-1) + bias  # [B, C]
        return logits


class SkinHyperHeadNet(nn.Module):
    """
    HyperHeadNet 的单属性版：只用 skin (Fitzpatrick I–VI → 0–5) 一个敏感属性生成主干分类头
    的逐样本参数。结构与 AgeHyperHeadNet 完全对齐 (单 embedding → 小 MLP → 展平 fc 权重+偏置、
    末层近零初始化)，仅把 age embedding 换成 6 类 skin embedding。

    为什么用 HyperHead 做 Fitzpatrick：HyperHead「在共享 φ(X) 上按属性生成逐样本线性头」是
    最浅的属性介入 HN，与 HyperFusion (单点深层)、HyperAdapt (每层低秩) 构成注入深度对照轴。
    注：Fitzpatrick 条件互信息 I(Y;skin|X)≈0 (作业 67913)，据核心论点 HN 在此注定无收益，
    本脚本产出的是「信号缺失处的 null result」对照，与 HAM-age (信号处) 并列解释。

    Args:
        in_dim    : 主干分类头输入维度 (ResNet-18 = 512)。
        out_dim   : 主干分类头输出维度 (malignant 二分类 = 1)。
        num_skin  : skin 类别数 (Fitzpatrick I–VI = 6)。
        skin_embed: skin embedding 维度。
        hidden_dim: HyperNet 内部 MLP 隐藏层维度。
        init_std  : 末层权重初始尺度 (近零起步，使初始 logit≈0、各样本几乎一致)。
    """

    def __init__(
        self,
        in_dim: int = 512,
        out_dim: int = 1,
        num_skin: int = 6,
        skin_embed: int = 4,
        hidden_dim: int = 64,
        init_std: float = 1e-3,
    ) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.init_std = init_std

        self.skin_emb = nn.Embedding(num_skin, skin_embed)

        weight_numel = in_dim * out_dim   # 展平后的 fc 权重元素数 (512*1)
        bias_numel = out_dim              # fc 偏置元素数 (1)
        self.mlp = nn.Sequential(
            nn.Linear(skin_embed, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, weight_numel + bias_numel),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        """末层近零 + 零偏置：初期生成头≈0、logit≈0、各样本几乎一致 (与 AgeHyperHeadNet 同理)。"""
        nn.init.normal_(self.mlp[-1].weight, mean=0.0, std=self.init_std)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, skin: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """skin: [B] long (0–5) → (weight [B, out_dim, in_dim], bias [B, out_dim])。"""
        a = self.skin_emb(skin)                # [B, skin_embed]
        out = self.mlp(a)                      # [B, in_dim*out_dim + out_dim]
        weight = out[:, : self.in_dim * self.out_dim].view(-1, self.out_dim, self.in_dim)
        bias = out[:, self.in_dim * self.out_dim :]
        return weight, bias


# ============================================================
# ResNet-18 + HyperHead for Fitzpatrick17k (单一 skin 属性)
# ============================================================
class ResNet18HyperHeadSkin(nn.Module):
    """
    Fitzpatrick17k 版 HyperHead: backbone 与 baseline 完全一致 (ImageNet 预训练 torchvision
    ResNet-18, fc→Identity, 输出 512 维特征)，分类头由 SkinHyperHeadNet 依据单一肤色 skin
    逐样本生成。与 ResNet18HyperHeadAge 同构，仅条件属性由 age 换成 skin。

    forward 签名 (image, skin)，与 FitzpatrickDataset 的 (image, label, skin) 对齐。

    Args:
        num_classes : 分类头输出维度 (malignant 二分类 → 1)。
        num_skin    : skin 有效类别数 (Fitzpatrick I–VI = 6)。
        skin_embed  : skin embedding 维度。
        hyper_hidden: HyperNet 内部 MLP 隐藏层维度。
        init_std    : 末层近零初始尺度 (与 baseline 起步稳定一致)。
    """

    def __init__(
        self,
        num_classes: int = 1,
        num_skin: int = 6,
        skin_embed: int = 4,
        hyper_hidden: int = 64,
        init_std: float = 1e-3,
    ) -> None:
        super().__init__()
        # backbone 与 baseline 完全一致 (与 ResNet18HyperHeadAge 同构)
        self.backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.feat_dim = self.backbone.fc.in_features   # 512
        self.backbone.fc = nn.Identity()

        # 单属性 skin HyperNet 取代原 fc
        self.hyper = SkinHyperHeadNet(
            in_dim=self.feat_dim,
            out_dim=num_classes,
            num_skin=num_skin,
            skin_embed=skin_embed,
            hidden_dim=hyper_hidden,
            init_std=init_std,
        )

    def extract_features(self, image: torch.Tensor) -> torch.Tensor:
        """只跑 backbone，返回 [B, feat_dim] 特征 (fc 之前)。"""
        return self.backbone(image)

    def forward(self, image: torch.Tensor, skin: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image: [B, 3, 224, 224] 皮肤镜 RGB 图像。
            skin : [B] long，0–5 (6 肤色型)。

        Returns:
            logits: [B, num_classes]。
        """
        feat = self.extract_features(image)            # [B, 512]
        weight, bias = self.hyper(skin)                # [B, C, 512], [B, C]
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

    # ---- HAM10000 单属性 (age) 变体自检 ----
    age_model = ResNet18HyperHeadAge(num_classes=1, num_age=4)
    age = torch.tensor([0, 1, 2, 3], dtype=torch.long)   # 4 个有效 age 组 (无 -1)
    age_logits = age_model(image, age)
    print(f"\n[HAM age 1-attr] Logits : {tuple(age_logits.shape)}")  # (4, 1)
    n_bb = sum(p.numel() for p in age_model.backbone.parameters())
    n_hy = sum(p.numel() for p in age_model.hyper.parameters())
    print(f"Backbone {n_bb:,} + HyperNet {n_hy:,} = {n_bb + n_hy:,}")
    age_logits.sum().backward()
    print("Age variant backward pass OK.")
