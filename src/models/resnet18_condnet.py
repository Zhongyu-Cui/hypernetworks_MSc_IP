"""
ResNet18CondNet — 条件化范围消融的统一可配置模型
==================================================
条件化范围消融方案（docs/conditioning_ablation_plan.md）的**唯一模型**：一次实现、四个
cell 共用，只暴露 `location` 一个开关，其余全部锁死，以得到「只改条件化范围、其余不变」的
严格嵌套阶梯。相较 src/models/resnet18_hyperadapt.py（HyperAdapt，恒定条件化除 stem 外每一层）
的关键差异是：**哪些卷积层挂 adapter 可配置**，从而把「条件化范围」隔离成唯一变量。

四个 cell（严格嵌套 C-Head ⊂ C-Deep ⊂ C-Full；fc 条件化在三个 conditioning cell 间恒定）：

    | cell   | location | 条件化范围                     |
    |--------|----------|------------------------------|
    | ERM    | "none"   | 无条件化（= 纯预训练 ResNet-18）|
    | C-Head | "head"   | 仅 fc                         |
    | C-Deep | "deep"   | fc ＋ layer4 全部 conv         |
    | C-Full | "full"   | fc ＋ layer1–4 全部 conv       |

因 fc 在三个 conditioning cell 恒定、conv 集合严格嵌套（∅ ⊂ {layer4} ⊂ {layer1..4}），
增量干净：`C-Deep − C-Head` = 加 layer4 conv，`C-Full − C-Deep` = 加 layer1–3 conv。

锁死的公共设定（不作变量，对齐 docs/conditioning_ablation_plan.md §1）：
  · 统一条件编码器（cat_embed=16 → out_dim=128），所有 cell 相同；
  · 参数共享固定 shared-A（A 按 stage 跨层共享、B 每层独立，HyperAdapt-style）；
  · conv rank=4，conv 乘性低秩 (1+M)、fc 加性低秩 ΔW；
  · 初始化令 Δθ≈0（shared-A/fc_A 置零 ⇒ 严格等价 base；B 用 Kaiming 保梯度非零）；
  · 训练 regime 对齐现有 HAM HN（BCE、AdamW、bs=128、≤30 epoch）；主结果用 best_overall。

**canonical base state（硬约束，§1）**：每 (dataset, fold, seed) 冻结同一份 base backbone
（ImageNet 预训练，天然跨 cell 相同）＋ base fc（用**独立、确定性**的生成器初始化，与模块注册
顺序无关），使所有 cell（含比较用 ERM）在同一 (fold,seed) 下 base fc 逐元素相等——否则
「Δθ=0 等价 ERM」只对 cell 自身成立、不对用于比较的 ERM 成立。见 build_base_fc_seed /
apply_canonical_base_state 与本文件末尾自检的逐元素相等断言。

复用 resnet18_hyperadapt 的 adapted_conv2d / adapted_linear 与三种条件编码器
（PatientEmbedding / SkinEmbedding / AgeEmbedding），避免重复实现。
"""

from __future__ import annotations

import hashlib

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import ResNet18_Weights, resnet18

from src.models.resnet18_hyperadapt import (
    AgeEmbedding,
    PatientEmbedding,
    SkinEmbedding,
    adapted_conv2d,
    adapted_linear,
)

# ============================================================
# location 范围表：(条件化的 stage 集合, 是否条件化 fc)
# ============================================================
# 键 = plan 的 cell 开关；值 = (conditioned_stages, condition_fc)。
# conditioned_stages ⊆ {"layer1","layer2","layer3","layer4"}；空集 = 不改 backbone。
LOCATION_SCOPES: dict[str, tuple[frozenset[str], bool]] = {
    "none": (frozenset(), False),                                              # ERM
    "head": (frozenset(), True),                                              # C-Head：仅 fc
    "deep": (frozenset({"layer4"}), True),                                    # C-Deep：fc + layer4
    "full": (frozenset({"layer1", "layer2", "layer3", "layer4"}), True),      # C-Full：fc + 全 conv
    # ---- E1 敏感性实验③（fc-off）：关掉 fc 条件化、只留 conv，检验超网络是否被逼着激活 conv ----
    # 与 deep/full 唯一差别 = condition_fc=False（fc 退化为属性无关共享 base 头）；conv 范围不变。
    "deep_nofc": (frozenset({"layer4"}), False),                              # C-Deep 去 fc 条件化
    "full_nofc": (frozenset({"layer1", "layer2", "layer3", "layer4"}), False),  # C-Full 去 fc 条件化
}

# ResNet-18 各 stage 的输出通道数（shared-A 按此分组）
STAGE_CHANNELS: dict[str, int] = {"layer1": 64, "layer2": 128, "layer3": 256, "layer4": 512}


# ============================================================
# canonical base state：确定性 base fc 种子
# ============================================================
def build_base_fc_seed(dataset: str, fold: int, seed: int) -> int:
    """
    由 (dataset, fold, seed) 派生一个**确定性**的 base fc 初始化种子。

    用途见模块 docstring 的 canonical base state：所有 cell（含比较用 ERM）在同一
    (dataset, fold, seed) 下必须共享逐元素相等的 base fc，才能保证「Δθ=0 等价 ERM」对
    *比较用* ERM 也成立。用独立种子初始化 base fc（而非依赖模块注册顺序消耗的全局 RNG），
    从根上消除「不同 cell 因构造的 adapter 生成器数量不同、导致 fc 初始化前 RNG 状态不同」的错配。

    用 blake2b 对三元组做稳定哈希（跨进程/跨机器一致，不依赖 Python 的 hash 随机化），
    取低 63 位作为 torch.Generator 的合法种子。

    Args:
        dataset: 数据集名（如 "ham10000"）。
        fold   : 折号（单次划分记为 0）。
        seed   : 训练 seed。

    Returns:
        一个可复现的非负整数种子（< 2**63）。
    """
    key = f"{dataset}|fold{fold}|seed{seed}".encode("utf-8")
    digest = hashlib.blake2b(key, digest_size=8).digest()
    return int.from_bytes(digest, "big") & ((1 << 63) - 1)


# ============================================================
# CondBasicBlock（可条件化 / 可退化为普通 BasicBlock）
# ============================================================
class CondBasicBlock(nn.Module):
    """
    与 torchvision.models.resnet.BasicBlock **命名严格对齐**（conv1/bn1/conv2/bn2 +
    可选 downsample[conv,bn]），故能直接 load ImageNet 预训练权重。按 `conditioned` 开关
    在两种前向之间切换：

      · conditioned=True ：每个卷积层挂 channel-wise 乘性低秩调制 Θ_ℓ → Θ_ℓ*(1+A·B_ℓ)，
        A 为本 stage 跨层共享（外部传入）、B_ℓ 层独有（HyperAdapt-style）。
      · conditioned=False：退化为普通 BasicBlock（不建 B 生成器、前向不带任何调制、FLOPs
        与显存与原 torchvision block 相同）——这是「范围外的层保持基线」的实现基础。

    Args:
        in_channels      : 输入通道数。
        out_channels     : 输出通道数。
        stride           : 第一个卷积 / downsample 的步幅。
        downsample       : 维度对齐用的 1x1 conv+bn (nn.Sequential) 或 None。
        patient_embed_dim: 条件向量维度。
        rank             : 低秩分解的秩 k。
        conditioned      : 本 block 是否挂 adapter（由所属 stage 是否在 location 范围内决定）。
    """

    expansion = 1

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        downsample: nn.Sequential | None = None,
        patient_embed_dim: int = 128,
        rank: int = 4,
        conditioned: bool = True,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.rank = rank
        self.conditioned = conditioned

        # ----- backbone 卷积层（命名与 torchvision BasicBlock 一致）-----
        self.conv1 = nn.Conv2d(in_channels, out_channels,
                               kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels,
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

        # ----- 层独有的 B 生成器：仅在 conditioned 时建立 -----
        if conditioned:
            self.B_gen_conv1 = nn.Linear(patient_embed_dim, rank * in_channels)
            self.B_gen_conv2 = nn.Linear(patient_embed_dim, rank * out_channels)
            self.B_gen_down = (nn.Linear(patient_embed_dim, rank * in_channels)
                               if downsample is not None else None)
        else:
            self.B_gen_conv1 = None
            self.B_gen_conv2 = None
            self.B_gen_down = None

    def _make_modulation(
        self, a_shared: torch.Tensor, b_gen: nn.Linear,
        embedding: torch.Tensor, c_in: int,
    ) -> torch.Tensor:
        """生成某层的 channel-wise 调制 M = A·B（[B, C_out, c_in]）。"""
        batch_size = embedding.shape[0]
        b_mat = b_gen(embedding).view(batch_size, self.rank, c_in)
        return torch.bmm(a_shared, b_mat)

    def _forward_plain(self, x: torch.Tensor) -> torch.Tensor:
        """普通 BasicBlock 前向（conditioned=False）：不带任何 adapter 调制。"""
        identity = x
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        return F.relu(out + identity, inplace=True)

    def _forward_cond(
        self, x: torch.Tensor, embedding: torch.Tensor, a_shared: torch.Tensor
    ) -> torch.Tensor:
        """条件化前向（conditioned=True）：每卷积层 channel-wise 乘性低秩调制。"""
        identity = x
        # conv1
        m1 = self._make_modulation(a_shared, self.B_gen_conv1, embedding, self.in_channels)
        out = adapted_conv2d(x, self.conv1.weight, m1, stride=self.stride, padding=1)
        out = F.relu(self.bn1(out), inplace=True)
        # conv2
        m2 = self._make_modulation(a_shared, self.B_gen_conv2, embedding, self.out_channels)
        out = adapted_conv2d(out, self.conv2.weight, m2, stride=1, padding=1)
        out = self.bn2(out)
        # downsample（也带 adapter）
        if self.downsample is not None:
            ds_conv: nn.Conv2d = self.downsample[0]
            ds_bn: nn.BatchNorm2d = self.downsample[1]
            md = self._make_modulation(a_shared, self.B_gen_down, embedding, self.in_channels)
            ds_stride = ds_conv.stride[0] if isinstance(ds_conv.stride, tuple) else ds_conv.stride
            identity = adapted_conv2d(x, ds_conv.weight, md, stride=ds_stride, padding=0)
            identity = ds_bn(identity)
        return F.relu(out + identity, inplace=True)

    def forward(
        self,
        x: torch.Tensor,
        embedding: torch.Tensor | None = None,
        a_shared: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """按 conditioned 开关分派：条件化前向需 (embedding, a_shared)，普通前向只需 x。"""
        if self.conditioned:
            return self._forward_cond(x, embedding, a_shared)
        return self._forward_plain(x)


# ============================================================
# ResNet18CondNet
# ============================================================
class ResNet18CondNet(nn.Module):
    """
    条件化范围消融的统一模型。`location` 决定哪些卷积层挂 adapter（见 LOCATION_SCOPES），
    其余锁死（shared-A、conv rank=4、乘性/加性低秩、Δθ≈0 初始化、16→128 编码器）。

    默认条件编码器为三属性 PatientEmbedding（sex/race/age，用于 MIMIC/CheXpert）；单属性
    数据集用子类 ResNet18CondNetAge（HAM）/ ResNet18CondNetSkin（Fitzpatrick）替换编码器。

    forward 签名 (image, *attrs)：与 harness forward_* 回调一致。ERM（location="none"）忽略
    attrs、退化为纯 ResNet-18；conditioning cell 从 attrs 生成条件向量。

    Args:
        location         : {"none","head","deep","full"}，见 LOCATION_SCOPES。
        num_classes      : 分类头输出维度（二分类→1，配 BCEWithLogitsLoss）。
        num_sex/num_race/num_age: 三属性类别数（PatientEmbedding）。
        patient_embed_dim: 条件向量维度。
        rank             : conv 低秩分解的秩 k（锁死 4）。
        pretrained       : 是否载入 torchvision ImageNet 预训练 backbone。
        base_fc_seed     : canonical base fc 的确定性种子（build_base_fc_seed 产出）。传 None 时
                           用普通默认初始化（仅单模型 smoke test 用；跨 cell 比较必须传同一种子）。
        freeze_backbone  : 冻结 backbone 卷积/BN 仿射参数（对齐冻结 regime；仅影响梯度、不影响结构）。
    """

    def __init__(
        self,
        location: str = "full",
        num_classes: int = 1,
        num_sex: int = 2,
        num_race: int = 2,
        num_age: int = 2,
        patient_embed_dim: int = 128,
        rank: int = 4,
        pretrained: bool = True,
        base_fc_seed: int | None = None,
        freeze_backbone: bool = False,
    ) -> None:
        super().__init__()
        if location not in LOCATION_SCOPES:
            raise ValueError(f"未知 location {location!r}，合法：{tuple(LOCATION_SCOPES)}")
        self.location = location
        self.conditioned_stages, self.condition_fc = LOCATION_SCOPES[location]
        self.rank = rank
        self.out_dim = num_classes
        self.patient_embed_dim = patient_embed_dim
        self.in_channels_running = 64
        # 是否需要生成条件向量：任一 conv stage 被条件化，或 fc 被条件化
        self.needs_embedding = self.condition_fc or bool(self.conditioned_stages)

        # ============================================================
        # 1. 条件通路（仅在需要条件向量时建立）
        # ============================================================
        if self.needs_embedding:
            self.patient_embed = PatientEmbedding(
                num_sex=num_sex, num_race=num_race, num_age=num_age,
                cat_embed_dim=16, out_dim=patient_embed_dim,
            )
        else:
            self.patient_embed = None
        # shared-A 生成器：只为被条件化的 stage 的输出通道建立（精确 adapter 参数量）
        cond_channels = sorted({STAGE_CHANNELS[s] for s in self.conditioned_stages})
        self.A_gen = nn.ModuleDict({
            f"c{c}": nn.Linear(patient_embed_dim, c * rank) for c in cond_channels
        })

        # ============================================================
        # 2. Backbone（命名与 torchvision 严格对齐；stem 恒不条件化）
        # ============================================================
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer("layer1", out_channels=64, blocks=2, stride=1)
        self.layer2 = self._make_layer("layer2", out_channels=128, blocks=2, stride=2)
        self.layer3 = self._make_layer("layer3", out_channels=256, blocks=2, stride=2)
        self.layer4 = self._make_layer("layer4", out_channels=512, blocks=2, stride=2)

        # ============================================================
        # 3. 分类头（fc；条件化时挂加性低秩 adapter）
        # ============================================================
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)
        if self.condition_fc:
            self.fc_A_gen = nn.Linear(patient_embed_dim, num_classes * rank)
            self.fc_B_gen = nn.Linear(patient_embed_dim, rank * 512)
        else:
            self.fc_A_gen = None
            self.fc_B_gen = None

        # ============================================================
        # 4. 初始化：canonical base state（base fc 确定性）+ 预训练 backbone + Δθ≈0 adapter
        # ============================================================
        self.apply_canonical_base_state(base_fc_seed, pretrained=pretrained)
        if freeze_backbone:
            self._freeze_backbone()

    # ------------------------------------------------------------
    # 构造一个 stage；conditioned 由 stage 是否在 location 范围内决定
    # ------------------------------------------------------------
    def _make_layer(self, stage_name: str, out_channels: int, blocks: int, stride: int) -> nn.ModuleList:
        """构造一个 residual stage（downsample 命名与 torchvision 对齐 Sequential[conv,bn]）。"""
        conditioned = stage_name in self.conditioned_stages
        downsample = None
        if stride != 1 or self.in_channels_running != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels_running, out_channels,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        layers = nn.ModuleList()
        layers.append(CondBasicBlock(
            self.in_channels_running, out_channels, stride, downsample,
            patient_embed_dim=self.patient_embed_dim, rank=self.rank, conditioned=conditioned,
        ))
        self.in_channels_running = out_channels
        for _ in range(1, blocks):
            layers.append(CondBasicBlock(
                self.in_channels_running, out_channels, 1, None,
                patient_embed_dim=self.patient_embed_dim, rank=self.rank, conditioned=conditioned,
            ))
        return layers

    # ------------------------------------------------------------
    # canonical base state：确定性 base fc + Δθ≈0 adapter + 预训练 backbone
    # ------------------------------------------------------------
    def apply_canonical_base_state(self, base_fc_seed: int | None, pretrained: bool = True) -> None:
        """
        施加 canonical base state（见模块 docstring）：

          1. base fc：用独立、确定性的 torch.Generator（由 base_fc_seed 播种）初始化，
             与模块注册顺序无关——保证同一 (dataset,fold,seed) 下所有 cell 的 base fc 逐元素相等。
          2. adapter 生成器：shared-A / fc_A 全部置零（⇒ M=A·B=0、ΔW=0，Δθ=0 严格等价 base）；
             B 生成器 / fc_B 用 Kaiming（非零，保证 ∂M/∂A=Bᵀ≠0、梯度不死锁）。B/fc_B 的初始化
             也用同一确定性 generator，使整个 base state（含 adapter 初值）在给定种子下完全可复现。
          3. 预训练 backbone：形状匹配的 torchvision ImageNet 权重覆盖 conv/bn（跳过 fc 与生成器）。

        Args:
            base_fc_seed: build_base_fc_seed 产出的确定性种子；None 时用默认初始化（仅 smoke test）。
            pretrained  : 是否载入 ImageNet 预训练 backbone。
        """
        gen = None if base_fc_seed is None else torch.Generator().manual_seed(int(base_fc_seed))

        # (0) base backbone 确定性初始化（与 fc 用独立 generator，避免模块注册顺序错配）：
        #     使所有 cell 在同一 seed 下 base backbone 逐元素相等，无论 pretrained 与否；
        #     pretrained=True 时随后被 ImageNet 覆盖（无害）。base_fc_seed=None 时跳过（走默认 init）。
        if gen is not None:
            self._deterministic_backbone_init(int(base_fc_seed) ^ 0x9E3779B9)

        def _kaiming(weight: torch.Tensor) -> None:
            if gen is None:
                nn.init.kaiming_normal_(weight, a=0, mode="fan_in", nonlinearity="linear")
            else:
                # 复刻 kaiming_normal_(fan_in, linear)：std = 1/sqrt(fan_in)
                fan_in = weight.shape[1]
                with torch.no_grad():
                    weight.normal_(0.0, 1.0 / (fan_in ** 0.5), generator=gen)

        def _base_fc_init(linear: nn.Linear) -> None:
            # 复刻 nn.Linear 默认初始化（kaiming_uniform_(a=√5) + fan_in bound），但用独立 generator
            fan_in = linear.weight.shape[1]
            with torch.no_grad():
                if gen is None:
                    nn.init.kaiming_uniform_(linear.weight, a=5 ** 0.5)
                    fan = linear.weight.shape[1]
                    b = 1.0 / (fan ** 0.5)
                    linear.bias.uniform_(-b, b)
                else:
                    # kaiming_uniform_(a=√5): bound = sqrt(6/((1+5)*fan_in)) = 1/sqrt(fan_in)
                    w_bound = 1.0 / (fan_in ** 0.5)
                    linear.weight.uniform_(-w_bound, w_bound, generator=gen)
                    linear.bias.uniform_(-w_bound, w_bound, generator=gen)

        # (1) base fc 确定性初始化
        _base_fc_init(self.fc)

        # (2) adapter：shared-A / fc_A 置零；B / fc_B Kaiming（确定性）
        for m in self.A_gen.values():
            nn.init.zeros_(m.weight)
            nn.init.zeros_(m.bias)
        for blk in self._all_blocks():
            if not blk.conditioned:
                continue
            for b_gen in (blk.B_gen_conv1, blk.B_gen_conv2, blk.B_gen_down):
                if b_gen is None:
                    continue
                _kaiming(b_gen.weight)
                nn.init.zeros_(b_gen.bias)
        if self.condition_fc:
            nn.init.zeros_(self.fc_A_gen.weight)
            nn.init.zeros_(self.fc_A_gen.bias)
            _kaiming(self.fc_B_gen.weight)
            nn.init.zeros_(self.fc_B_gen.bias)

        # (3) 预训练 backbone 覆盖
        if pretrained:
            self._load_pretrained_backbone()

    def _load_pretrained_backbone(self) -> None:
        """把 torchvision ImageNet resnet18 中形状匹配的 backbone 权重 load 进来（跳过 fc 与生成器）。"""
        tv_state = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).state_dict()
        own_state = self.state_dict()
        load_state = {
            k: v for k, v in tv_state.items()
            if k in own_state and own_state[k].shape == v.shape and not k.startswith("fc.")
        }
        self.load_state_dict(load_state, strict=False)

    def _deterministic_backbone_init(self, seed: int) -> None:
        """
        用独立、确定性的 generator 初始化全部 backbone conv/bn（复刻 torchvision ResNet 默认初始化：
        conv 用 kaiming_normal_(fan_out, relu)、bn weight=1/bias=0、running stats 复位）。使所有 cell
        在同一 (dataset,fold,seed) 下 base backbone 逐元素相等——与模块注册顺序、adapter 数量无关。
        pretrained=True 时随后被 ImageNet 覆盖。
        """
        gen = torch.Generator().manual_seed(int(seed) & ((1 << 63) - 1))
        with torch.no_grad():
            for module in self.modules():
                if isinstance(module, nn.Conv2d):
                    # kaiming_normal_(fan_out, relu): std = sqrt(2 / fan_out)
                    fan_out = module.out_channels * module.kernel_size[0] * module.kernel_size[1]
                    module.weight.normal_(0.0, (2.0 / fan_out) ** 0.5, generator=gen)
                    if module.bias is not None:
                        module.bias.zero_()
                elif isinstance(module, nn.BatchNorm2d):
                    module.weight.fill_(1.0)
                    module.bias.zero_()
                    module.running_mean.zero_()
                    module.running_var.fill_(1.0)
                    module.num_batches_tracked.zero_()

    def _freeze_backbone(self) -> None:
        """冻结 backbone 卷积/BN 仿射参数（stem + 各 block backbone 层 + downsample）；生成器与 fc 保持可训练。"""
        for module in (self.conv1, self.bn1):
            for p in module.parameters():
                p.requires_grad = False
        for blk in self._all_blocks():
            backbone_modules: list[nn.Module] = [blk.conv1, blk.bn1, blk.conv2, blk.bn2]
            if blk.downsample is not None:
                backbone_modules.append(blk.downsample)
            for module in backbone_modules:
                for p in module.parameters():
                    p.requires_grad = False

    def _all_blocks(self) -> list[CondBasicBlock]:
        """按前向顺序返回全部 CondBasicBlock。"""
        return [*self.layer1, *self.layer2, *self.layer3, *self.layer4]

    # ------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------
    def forward(self, image: torch.Tensor, *attrs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image : [B, 3, 224, 224]。
            *attrs: 若干离散敏感属性（[B] long），按位置传给条件编码器；ERM/无条件时忽略。

        Returns:
            logits: [B, num_classes]。
        """
        embedding = None
        a_shared: dict[int, torch.Tensor] = {}
        if self.needs_embedding:
            embedding = self.patient_embed(*attrs)                 # [B, embed_dim]
            batch_size = embedding.shape[0]
            for c in (STAGE_CHANNELS[s] for s in self.conditioned_stages):
                a_shared[c] = self.A_gen[f"c{c}"](embedding).view(batch_size, c, self.rank)

        # stem（无 adapter）
        out = F.relu(self.bn1(self.conv1(image)), inplace=True)
        out = self.maxpool(out)

        # 4 个 residual stage：条件化 block 走 adapter 前向，其余走普通前向
        for stage_name, blocks in (
            ("layer1", self.layer1), ("layer2", self.layer2),
            ("layer3", self.layer3), ("layer4", self.layer4),
        ):
            c = STAGE_CHANNELS[stage_name]
            for blk in blocks:
                if blk.conditioned:
                    out = blk(out, embedding, a_shared[c])
                else:
                    out = blk(out)

        # Pool + fc（条件化时加性低秩 adapter）
        out = self.avgpool(out)
        out = torch.flatten(out, 1)                                # [B, 512]
        if self.condition_fc:
            batch_size = out.shape[0]
            a_fc = self.fc_A_gen(embedding).view(batch_size, self.out_dim, self.rank)
            b_fc = self.fc_B_gen(embedding).view(batch_size, self.rank, 512)
            delta_w = torch.bmm(a_fc, b_fc)                        # [B, out_dim, 512]
            return adapted_linear(out, self.fc.weight, delta_w, self.fc.bias)
        return self.fc(out)

    # ------------------------------------------------------------
    # 机制诊断：逐层有效偏移 ρ_l（device-aware，供训练时逐 epoch 探针）
    # ------------------------------------------------------------
    @torch.no_grad()
    def conditioning_rho(self, embeddings: torch.Tensor) -> dict[str, dict[str, float]]:
        """
        给定条件嵌入 `embeddings`（[K, embed_dim]，K = 属性取值数），算每个**条件化模块**的两个偏移量
        （对 K 个属性取值等权、先逐单位求 Frobenius norm 再平均）：

        · `rho`         = E_a‖ΔW(a)‖_F / ‖W‖_F                    —— 该层被改动了多少（总偏移量级）。
        · `rho_between` = sqrt(E_a‖ΔW(a) − mean_a ΔW‖_F²) / ‖W‖_F  —— 偏移中**随属性变化**的部分。

        与 `scripts/e1_rho_and_permutation.compute_rho` **同一数学**（该函数现委托本方法，单一事实来源）；
        区别仅在 device 由 `embeddings` 决定，故可在训练时（模型在 GPU）直接调用做逐 epoch 探针。
        conv 走乘性 ΔW=W⊙M（M=A·B）；fc 走加性 ΔW=A_fc·B_fc。ERM（无条件通路）返回空 dict。

        Args:
            embeddings: 条件编码器对全部属性取值的输出 [K, patient_embed_dim]，须与本模型同 device。

        Returns:
            {模块名 -> {"rho", "rho_between"}}，按前向顺序（conv 各层 + 可选 fc）。
        """
        k = embeddings.shape[0]
        out: dict[str, dict[str, float]] = {}

        def _record(name: str, w_sq: torch.Tensor | None, delta_or_mod: torch.Tensor,
                    den: torch.Tensor) -> None:
            x = delta_or_mod
            if w_sq is not None:                                  # 乘性：ΔW = W⊙M
                per = torch.sqrt(((x ** 2) * w_sq.unsqueeze(0)).sum(dim=(1, 2)))
                dev = x - x.mean(dim=0, keepdim=True)
                betw = torch.sqrt((((dev ** 2) * w_sq.unsqueeze(0)).sum(dim=(1, 2))).mean())
            else:                                                 # 加性：ΔW 直接给
                per = torch.sqrt((x ** 2).sum(dim=(1, 2)))
                dev = x - x.mean(dim=0, keepdim=True)
                betw = torch.sqrt(((dev ** 2).sum(dim=(1, 2))).mean())
            out[name] = {"rho": float((per / den).mean()), "rho_between": float(betw / den)}

        stage_of = {"layer1": self.layer1, "layer2": self.layer2,
                    "layer3": self.layer3, "layer4": self.layer4}
        for sname, blocks in stage_of.items():
            if sname not in self.conditioned_stages:
                continue
            c_out = STAGE_CHANNELS[sname]
            a_shared = self.A_gen[f"c{c_out}"](embeddings).view(k, c_out, self.rank)
            for bi, blk in enumerate(blocks):
                if not blk.conditioned:
                    continue
                targets = [(f"{sname}.{bi}.conv1", blk.conv1.weight, blk.B_gen_conv1, blk.in_channels),
                           (f"{sname}.{bi}.conv2", blk.conv2.weight, blk.B_gen_conv2, blk.out_channels)]
                if blk.B_gen_down is not None:
                    targets.append((f"{sname}.{bi}.downsample.0", blk.downsample[0].weight,
                                    blk.B_gen_down, blk.in_channels))
                for name, weight, bgen, c_in in targets:
                    b_mat = bgen(embeddings).view(k, self.rank, c_in)
                    modulation = torch.bmm(a_shared, b_mat)       # [K, C_out, C_in]
                    w_sq = (weight ** 2).sum(dim=(2, 3))          # [C_out, C_in] 逐通道对能量
                    _record(name, w_sq, modulation, torch.sqrt(w_sq.sum()))

        if self.condition_fc:
            a_fc = self.fc_A_gen(embeddings).view(k, self.out_dim, self.rank)
            b_fc = self.fc_B_gen(embeddings).view(k, self.rank, 512)
            delta_w = torch.bmm(a_fc, b_fc)                       # [K, out_dim, 512]
            _record("fc", None, delta_w, torch.linalg.norm(self.fc.weight))
        return out

    # ------------------------------------------------------------
    # 结构自省：adapter 挂载点、参数量、FLOPs
    # ------------------------------------------------------------
    def conditioned_module_names(self) -> list[str]:
        """
        返回本 cell 实际挂 adapter 的 backbone 模块名（含 downsample）＋（若条件化）"fc"，
        按前向顺序排列。供单元测试断言 C-Head ⊂ C-Deep ⊂ C-Full 的严格嵌套。
        """
        names: list[str] = []
        stage_of = {"layer1": self.layer1, "layer2": self.layer2,
                    "layer3": self.layer3, "layer4": self.layer4}
        for stage_name, blocks in stage_of.items():
            for i, blk in enumerate(blocks):
                if not blk.conditioned:
                    continue
                names.append(f"{stage_name}.{i}.conv1")
                names.append(f"{stage_name}.{i}.conv2")
                if blk.downsample is not None:
                    names.append(f"{stage_name}.{i}.downsample.0")
        if self.condition_fc:
            names.append("fc")
        return names

    def param_breakdown(self) -> tuple[int, int, int]:
        """返回 (总参数量, 超网络/adapter 参数量, backbone+fc 参数量)。"""
        adapter_params: list[nn.Parameter] = []
        if self.patient_embed is not None:
            adapter_params += list(self.patient_embed.parameters())
        adapter_params += [p for m in self.A_gen.values() for p in m.parameters()]
        if self.condition_fc:
            adapter_params += list(self.fc_A_gen.parameters())
            adapter_params += list(self.fc_B_gen.parameters())
        for blk in self._all_blocks():
            for b_gen in (blk.B_gen_conv1, blk.B_gen_conv2, blk.B_gen_down):
                if b_gen is not None:
                    adapter_params += list(b_gen.parameters())
        n_adapter = sum(p.numel() for p in adapter_params)
        n_total = sum(p.numel() for p in self.parameters())
        return n_total, n_adapter, n_total - n_adapter

    def adapter_flops_per_sample(self) -> int:
        """
        本 cell 每样本 adapter 的近似前向 MAC 数（不含 backbone 卷积主体，只算条件化引入的额外算量）：
        条件编码器 fuse MLP + shared-A 生成器 + 各条件化层 B 生成器 + A·B 组合 + 调制施加 + fc adapter。

        线性层 MAC = in_features × out_features；A·B 组合 MAC = C_out·rank·C_in；乘性调制施加
        (1+M)⊙W 的逐元素 MAC ≈ C_out·C_in·k·k。仅作范围档间「额外算量随范围增长」的量级对照，
        非精确 profiler 计数。

        Returns:
            每样本额外 MAC 数（int）。
        """
        if not self.needs_embedding:
            return 0
        rank = self.rank
        macs = 0
        # 条件编码器 fuse MLP（in_dim → out → out）
        pe = self.patient_embed
        fuse = pe.fuse
        macs += fuse[0].in_features * fuse[0].out_features
        macs += fuse[2].in_features * fuse[2].out_features
        # shared-A 生成器
        for c_str, gen in self.A_gen.items():
            macs += gen.in_features * gen.out_features
        # 各条件化层：B 生成器 + A·B 组合 + 调制施加
        kernel = {"conv1": 9, "conv2": 9, "downsample.0": 1}  # 3x3=9, 1x1 downsample=1
        for blk in self._all_blocks():
            if not blk.conditioned:
                continue
            c_in, c_out = blk.in_channels, blk.out_channels
            # conv1: C_in→ B; A·B = C_out·rank·C_in; 调制 = C_out·C_in·9
            macs += blk.B_gen_conv1.in_features * blk.B_gen_conv1.out_features
            macs += c_out * rank * c_in + c_out * c_in * kernel["conv1"]
            # conv2: 中间通道 c_out
            macs += blk.B_gen_conv2.in_features * blk.B_gen_conv2.out_features
            macs += c_out * rank * c_out + c_out * c_out * kernel["conv2"]
            if blk.B_gen_down is not None:
                macs += blk.B_gen_down.in_features * blk.B_gen_down.out_features
                macs += c_out * rank * c_in + c_out * c_in * kernel["downsample.0"]
        # fc adapter
        if self.condition_fc:
            macs += self.fc_A_gen.in_features * self.fc_A_gen.out_features
            macs += self.fc_B_gen.in_features * self.fc_B_gen.out_features
            macs += self.out_dim * rank * 512  # A·B
        return int(macs)


# ============================================================
# 单属性子类：HAM（age）/ Fitzpatrick（skin）
# ============================================================
class ResNet18CondNetAge(ResNet18CondNet):
    """
    HAM10000 版：条件编码器换成单一 age_group（0–3）AgeEmbedding，forward 签名 (image, age)。
    仅当被条件化（location != "none"）时才有编码器；ERM 无编码器（忽略 age）。
    """

    def __init__(
        self,
        location: str = "full",
        num_classes: int = 1,
        num_age: int = 4,
        patient_embed_dim: int = 128,
        rank: int = 4,
        pretrained: bool = True,
        base_fc_seed: int | None = None,
        freeze_backbone: bool = False,
    ) -> None:
        super().__init__(
            location=location, num_classes=num_classes, patient_embed_dim=patient_embed_dim,
            rank=rank, pretrained=pretrained, base_fc_seed=base_fc_seed,
            freeze_backbone=freeze_backbone,
        )
        if self.needs_embedding:
            self.patient_embed = AgeEmbedding(
                num_age=num_age, cat_embed_dim=16, out_dim=patient_embed_dim,
            )


class ResNet18CondNetSkin(ResNet18CondNet):
    """
    Fitzpatrick17k 版：条件编码器换成单一 skin（0–5）SkinEmbedding，forward 签名 (image, skin)。
    """

    def __init__(
        self,
        location: str = "full",
        num_classes: int = 1,
        num_skin: int = 6,
        patient_embed_dim: int = 128,
        rank: int = 4,
        pretrained: bool = True,
        base_fc_seed: int | None = None,
        freeze_backbone: bool = False,
    ) -> None:
        super().__init__(
            location=location, num_classes=num_classes, patient_embed_dim=patient_embed_dim,
            rank=rank, pretrained=pretrained, base_fc_seed=base_fc_seed,
            freeze_backbone=freeze_backbone,
        )
        if self.needs_embedding:
            self.patient_embed = SkinEmbedding(
                num_skin=num_skin, cat_embed_dim=16, out_dim=patient_embed_dim,
            )


# ============================================================
# 结构自检（E0.4）：Δθ≈0 等价 + 模块嵌套 assert + 参数量/FLOPs
# ============================================================
def _selftest() -> None:
    """
    验证（不下载预训练权重，仅测结构/等价/嵌套/参数量）：
      · 前向/反向 OK、logits 形状正确；
      · Δθ=0 严格等价：C-Head/C-Deep/C-Full 初始 logits 与 ERM 逐元素相等（canonical base fc）；
      · 模块嵌套：conditioned_module_names 满足 C-Head ⊂ C-Deep ⊂ C-Full（conv 集合）；
      · 打印各 cell adapter 参数量与每样本 adapter FLOPs。
    """
    torch.manual_seed(0)
    batch_size = 4
    image = torch.randn(batch_size, 3, 224, 224)
    age = torch.tensor([0, 1, 2, 3], dtype=torch.long)

    seed = build_base_fc_seed("ham10000", fold=0, seed=42)
    cells = {}
    for loc in ("none", "head", "deep", "full"):
        m = ResNet18CondNetAge(location=loc, num_age=4, pretrained=False, base_fc_seed=seed)
        m.eval()
        cells[loc] = m

    # --- 前向/反向 + logits 形状 ---
    logits = {}
    for loc, m in cells.items():
        out = m(image, age) if m.needs_embedding else m(image)
        assert out.shape == (batch_size, 1), (loc, out.shape)
        logits[loc] = out
        out.sum().backward()
    print("[OK] 四 cell 前向/反向通过，logits 形状均 (4, 1)")

    # --- Δθ=0 等价（分两层证明）---
    # (a) 权重侧严格零：shared-A / fc_A 置零 ⇒ 生成的 M=A·B、ΔW=A_fc·B_fc 逐元素恒为 0（bitwise）。
    for loc in ("head", "deep", "full"):
        m = cells[loc]
        emb = m.patient_embed(age)
        bs = emb.shape[0]
        if m.condition_fc:
            a_fc = m.fc_A_gen(emb)
            assert torch.count_nonzero(a_fc) == 0, f"{loc} fc_A_gen 输出非零 ⇒ ΔW≠0"
        for c in (STAGE_CHANNELS[s] for s in m.conditioned_stages):
            a_sh = m.A_gen[f"c{c}"](emb)
            assert torch.count_nonzero(a_sh) == 0, f"{loc} A_gen[c{c}] 输出非零 ⇒ M≠0"
    print("[OK] 权重侧 Δθ=0 严格成立：所有 shared-A / fc_A 生成的 M、ΔW 逐元素恒为 0（bitwise）")
    # (b) logits 侧近零：adapter 走 grouped-conv / bmm，与 ERM 的 plain conv/linear 是同一数学的
    #     不同数值实现，故 logits 仅差 float32 累加噪声（远小于 logit 量级 O(1)、亦远小于任何真实信号）。
    erm_logits = logits["none"]
    for loc in ("head", "deep", "full"):
        max_abs = (logits[loc] - erm_logits).abs().max().item()
        assert max_abs < 1e-4, f"{loc} 初始 logits 与 ERM 偏差 {max_abs:.2e} 超近零阈值（疑 Δθ≠0）"
        print(f"[OK] {loc:>4s} 初始 logits 与 ERM 近零一致（max|Δlogit|={max_abs:.2e}，仅 conv 实现浮点噪声）")

    # --- canonical base state：同 (fold,seed) 下所有 cell 的 base fc 逐元素相等 ---
    for loc in ("head", "deep", "full"):
        assert torch.equal(cells[loc].fc.weight, cells["none"].fc.weight), f"{loc} base fc.weight != ERM"
        assert torch.equal(cells[loc].fc.bias, cells["none"].fc.bias), f"{loc} base fc.bias != ERM"
    print("[OK] canonical base fc：head/deep/full 与 ERM 的 fc 逐元素相等（与模块注册顺序无关）")

    # --- 模块嵌套：conv 集合 C-Head ⊂ C-Deep ⊂ C-Full ---
    def conv_set(m: ResNet18CondNet) -> set[str]:
        return {n for n in m.conditioned_module_names() if n != "fc"}
    head_s, deep_s, full_s = conv_set(cells["head"]), conv_set(cells["deep"]), conv_set(cells["full"])
    assert head_s == set(), f"C-Head 不应有 conv adapter，实得 {head_s}"
    assert head_s < deep_s < full_s, "conv adapter 集合未严格嵌套 head ⊂ deep ⊂ full"
    # C-Deep 必含 layer4[0].downsample（HyperFusion 注入点），且 layer4 两 block 全 conv
    assert "layer4.0.downsample.0" in deep_s and "layer4.1.conv2" in deep_s
    # C-Full 含 layer2/3/4 downsample、layer1 无 downsample
    assert "layer2.0.downsample.0" in full_s and "layer1.0.conv1" in full_s
    assert not any(n.startswith("layer1") and "downsample" in n for n in full_s)
    for loc in ("head", "deep", "full"):
        assert cells[loc].condition_fc and "fc" in cells[loc].conditioned_module_names()
    print(f"[OK] 模块嵌套：head{sorted(head_s)} ⊂ deep({len(deep_s)} convs) ⊂ full({len(full_s)} convs)；fc 三档恒有")

    # --- 参数量 + FLOPs 报告 ---
    print("\n各 cell 参数量与每样本 adapter FLOPs：")
    print(f"  {'cell':>6s} | {'total':>12s} | {'adapter':>10s} | {'backbone+fc':>12s} | {'adapter MAC/样本':>16s}")
    for loc, cell_name in (("none", "ERM"), ("head", "C-Head"), ("deep", "C-Deep"), ("full", "C-Full")):
        m = cells[loc]
        n_total, n_adapter, n_bb = m.param_breakdown()
        flops = m.adapter_flops_per_sample()
        print(f"  {cell_name:>6s} | {n_total:>12,} | {n_adapter:>10,} | {n_bb:>12,} | {flops:>16,}")

    # --- 不同 (fold,seed) 的 base fc 应不同（种子确实起作用）---
    seed_b = build_base_fc_seed("ham10000", fold=1, seed=42)
    m_b = ResNet18CondNetAge(location="none", num_age=4, pretrained=False, base_fc_seed=seed_b)
    assert not torch.equal(m_b.fc.weight, cells["none"].fc.weight), "不同 fold 的 base fc 不应相同"
    print("\n[OK] 不同 (fold,seed) 派生不同 base fc（canonical 种子生效）")
    print("\nResNet18CondNet 结构自检全部通过 ✓")


if __name__ == "__main__":
    _selftest()
