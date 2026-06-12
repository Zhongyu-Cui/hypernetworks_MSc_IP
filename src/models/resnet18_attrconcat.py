"""
ImageNet 预训练 ResNet-18 + Attribute Concatenation for MIMIC-CXR Binary Classification
=======================================================================================
在 image-only baseline (src/models/resnet18_pretrained.py) 的基础上, **不使用超网络**,
而是把离散敏感属性 (sex / race / age_group) 各自 embedding 后, broadcast 成空间特征图,
沿通道维与输入图像 concat, 作为加宽后的 ResNet-18 stem (conv1) 的输入。这是属性融合里
最朴素的「输入端拼接」基线 (input-level conditioning), 用以与超网络系列变体
(HyperHead / HyperAdapt / HyperFusion) 在相同预处理/早停下对比公平性。

设计参考 drafts/ResNet18_attrconcat.py (UTKFace 原型), 但做了三点适配以与 MIMIC baseline 对齐:

  1. Backbone 换成与 baseline 完全一致的 torchvision.models.resnet18 + ImageNet 预训练
     权重 (drafts 里是从零实现的 ResNet18)。本模块直接持有一个完整的 torchvision
     resnet18 作为 self.backbone, 仅把 stem 的 conv1 加宽以容纳额外属性通道, 其余层
     (bn1、maxpool、layer1~4、avgpool、fc) 与 baseline 严格一致。
  2. 条件属性从 (sex, race) 扩展为 (sex, race, age_group), 覆盖 MIMICCXRDataset 返回的
     全部离散敏感属性; 这也对应 baseline 报告里 EqOdds gap 最大的 Age 维度与最弱交叉组
     Male & Non-White & >=60。
  3. forward 签名统一为 (image, sex, race, age_group), 与 ResNet18HyperHead /
     ResNet18HyperFusion 一致, 训练循环可直接复用 (与 MIMICCXRDataset.__getitem__ 返回
     顺序对齐: image, label, sex, race, age_group)。

「初始等价于 baseline」的处理 (与超网络变体的 Δθ≈0 / MIP 思想一致):
  - 加宽 conv1 时, 把预训练 conv1 的 [64, 3, 7, 7] 权重原样拷入新 conv1 的前 3 个输入
    通道, 其余「属性通道」对应的卷积权重 **零初始化**。于是第 0 步前向时属性通道贡献为 0,
    stem 输出严格等于纯 ImageNet 预训练 ResNet-18, 训练从 baseline 起步, 再逐步学到把
    属性信息注入 stem 的权重。这样既保留了预训练权重, 又使图像通路在起点上与 baseline
    可比, 公平性对比不被「随机重置 stem」的噪声干扰。
"""

import torch
import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18


# ============================================================
# ResNet-18 (ImageNet 预训练) + Attribute Concat
# ============================================================
class ResNet18AttrConcat(nn.Module):
    """
    与 MIMIC baseline 对齐的 ImageNet 预训练 ResNet-18, 在输入端把离散敏感属性
    (sex / race / age_group) 的 embedding broadcast 成空间图后, 与图像沿通道维 concat,
    再送入加宽 stem 的 backbone。属性以「额外输入通道」的形式参与所有后续卷积。

    Args:
        num_classes: 分类头输出维度。
            - 1 -> 配合 nn.BCEWithLogitsLoss (默认; No Finding U-Zeros 二分类)
            - k -> 配合 nn.CrossEntropyLoss
        num_sex, num_race, num_age: 三个离散敏感属性的类别数 (MIMIC 均为 2)。
        sex_embed, race_embed, age_embed: 各属性 embedding 维度 (= 额外输入通道数之分量)。
            默认各 2, 共 6 个属性通道, conv1 输入通道 = 3 + 6 = 9。
        pretrained: 是否载入 torchvision ImageNet 预训练 backbone 权重
            (与 baseline 一致, 默认 True)。
    """

    IMG_CHANNELS = 3   # CXR 经 Grayscale(3) 复制为 3 通道, 与 ImageNet backbone 约定一致

    def __init__(
        self,
        num_classes: int = 1,
        num_sex: int = 2,
        num_race: int = 2,
        num_age: int = 2,
        sex_embed: int = 2,
        race_embed: int = 2,
        age_embed: int = 2,
        pretrained: bool = True,
    ) -> None:
        super().__init__()
        self.sex_embed_dim = sex_embed
        self.race_embed_dim = race_embed
        self.age_embed_dim = age_embed
        self.attr_channels = sex_embed + race_embed + age_embed   # 额外输入通道总数

        # --- Backbone: 与 baseline 完全一致的 ImageNet 预训练 ResNet-18 ---
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = resnet18(weights=weights)
        # 替换分类头: 1000 类 -> num_classes (与 baseline ResNet18Pretrained 一致)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)

        # --- 离散属性各自的 embedding 表 (离散变量 -> 稠密向量) ---
        self.sex_emb = nn.Embedding(num_sex, sex_embed)
        self.race_emb = nn.Embedding(num_race, race_embed)
        self.age_emb = nn.Embedding(num_age, age_embed)

        # --- 加宽 stem: conv1 从 3 通道扩成 3 + attr_channels 通道 ---
        # 预训练权重原样拷入前 3 通道, 属性通道零初始化 ⇒ 初始前向严格等价于 baseline
        self._expand_conv1()

    def _expand_conv1(self) -> None:
        """
        把 backbone.conv1 由 Conv2d(3, 64, 7, s2, p3) 替换为
        Conv2d(3 + attr_channels, 64, 7, s2, p3), 并:
          - 前 3 个输入通道的卷积权重 = 原预训练权重 (保留 ImageNet 图像通路);
          - 属性通道对应的卷积权重 = 0 (初始时属性不影响 stem 输出 ⇒ 等价 baseline)。
        torchvision resnet18 的 conv1 无 bias, 故无需处理 bias。
        """
        old_conv: nn.Conv2d = self.backbone.conv1
        total_in = self.IMG_CHANNELS + self.attr_channels

        new_conv = nn.Conv2d(
            in_channels=total_in,
            out_channels=old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=(old_conv.bias is not None),
        )
        with torch.no_grad():
            new_conv.weight.zero_()                                # 先全零
            # 前 3 通道拷入预训练权重; 其余属性通道保持 0
            new_conv.weight[:, : self.IMG_CHANNELS] = old_conv.weight
            if old_conv.bias is not None:
                new_conv.bias.copy_(old_conv.bias)

        self.backbone.conv1 = new_conv

    def _build_attr_map(
        self,
        sex: torch.Tensor,
        race: torch.Tensor,
        age_group: torch.Tensor,
        height: int,
        width: int,
    ) -> torch.Tensor:
        """
        把 sex / race / age_group 的 embedding 拼接并 broadcast 成
        [B, attr_channels, H, W]: 同一张图的每个空间位置都填同一个属性向量。

        Args:
            sex / race / age_group: [B] long。
            height, width: 目标空间尺寸 (与输入图像一致)。

        Returns:
            attr_map: [B, attr_channels, H, W]。
        """
        s = self.sex_emb(sex)          # [B, sex_embed]
        r = self.race_emb(race)        # [B, race_embed]
        a = self.age_emb(age_group)    # [B, age_embed]
        attr = torch.cat([s, r, a], dim=1)            # [B, attr_channels]
        # 在空间维上复制同一向量: [B, C] -> [B, C, 1, 1] -> [B, C, H, W]
        attr = attr.unsqueeze(-1).unsqueeze(-1)
        attr = attr.expand(-1, -1, height, width)
        return attr

    def forward(
        self,
        image: torch.Tensor,
        sex: torch.Tensor,
        race: torch.Tensor,
        age_group: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            image     : [B, 3, 224, 224] 图像 (CXR 经 Grayscale(3) 复制为 3 通道)。
            sex       : [B] long, 0=Male, 1=Female。
            race      : [B] long, 0=White, 1=Non-White。
            age_group : [B] long, 0=<thr, 1=>=thr。

        Returns:
            logits: [B, num_classes]。
        """
        _, _, height, width = image.shape

        # 1) 构造属性 map 并与图像沿通道维 concat -> [B, 3 + attr_channels, H, W]
        attr_map = self._build_attr_map(sex, race, age_group, height, width)
        x = torch.cat([image, attr_map], dim=1)

        # 2) 标准 ResNet-18 前向 (stem 已加宽, 其余与 baseline 完全一致)
        return self.backbone(x)

    # ------------------------------------------------------------
    # 工具: 分别统计 backbone / 属性融合 (embedding) 的参数量
    # ------------------------------------------------------------
    def param_breakdown(self) -> tuple[int, int, int]:
        """返回 (总参数量, 属性 embedding 参数量, backbone 参数量)。"""
        attr_modules = list(self.sex_emb.parameters())
        attr_modules += list(self.race_emb.parameters())
        attr_modules += list(self.age_emb.parameters())
        n_attr = sum(p.numel() for p in attr_modules)
        n_total = sum(p.numel() for p in self.parameters())
        return n_total, n_attr, n_total - n_attr


# ============================================================
# 结构自检
# ============================================================
if __name__ == "__main__":
    torch.manual_seed(0)
    # 自检不下载预训练权重 (pretrained=False), 仅验证结构与前向/反向
    model = ResNet18AttrConcat(num_classes=1, pretrained=False)

    batch_size = 4
    image = torch.randn(batch_size, 3, 224, 224)
    sex = torch.tensor([0, 1, 0, 1], dtype=torch.long)
    race = torch.tensor([0, 1, 0, 1], dtype=torch.long)
    age = torch.tensor([0, 0, 1, 1], dtype=torch.long)

    model.eval()                                          # eval 避免 BN 在小 batch 上不稳
    logits = model(image, sex, race, age)
    print(f"Input  shape : {tuple(image.shape)}")          # (4, 3, 224, 224)
    print(f"conv1  in_ch : {model.backbone.conv1.in_channels}")  # 3 + 6 = 9
    print(f"Logits shape : {tuple(logits.shape)}")         # (4, 1)
    print(f"Logit  stats : mean={logits.mean().item():+.3f}  std={logits.std().item():.3f}")

    n_total, n_attr, n_backbone = model.param_breakdown()
    print(f"Total    params : {n_total:>12,}")
    print(f"  Backbone      : {n_backbone:>12,}")
    print(f"  Attr embed    : {n_attr:>12,}")

    # --- 检查 1: 初始输出对属性严格不变 (属性通道零初始化 ⇒ 等价于纯 baseline) ---
    # 同一图像换不同 sex/race/age, 初始 logit 应完全一致。
    alt = model(image, 1 - sex, 1 - race, 1 - age)
    max_diff = (logits - alt).abs().max().item()
    print(f"\n[init] 属性翻转后 logit 最大差异: {max_diff:.2e} "
          f"({'✅ 等价 baseline' if max_diff < 1e-5 else '❌ 应为 0'})")

    # --- 检查 2: 零初始化门控的自举行为 ---
    # 第 0 步: conv1 属性通道权重为 0, 故属性 embedding 梯度恰为 0 (符合预期);
    # 但 conv1 属性通道权重本身能拿到梯度 ⇒ 一步优化后即非零, embedding 随即恢复梯度。
    model.train()
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    for step in range(2):
        opt.zero_grad()
        loss = model(image, sex, race, age).sum()
        loss.backward()
        attr_grad = sum(
            p.grad.abs().sum().item()
            for p in (*model.sex_emb.parameters(),
                      *model.race_emb.parameters(),
                      *model.age_emb.parameters())
            if p.grad is not None
        )
        conv1_grad = model.backbone.conv1.weight.grad.abs().sum().item()
        print(f"[step {step}] conv1 grad={conv1_grad:.2f}  attr-embed grad={attr_grad:.4f}")
        opt.step()
    print("Backward / bootstrap OK." if attr_grad > 0 and conv1_grad > 0 else
          "自举失败, 检查 forward 链路!")
