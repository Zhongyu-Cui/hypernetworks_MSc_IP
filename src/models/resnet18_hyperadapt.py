"""
ImageNet 预训练 ResNet-18 + HyperAdapt for MIMIC-CXR Binary Classification
=========================================================================
在 image-only baseline (src/models/resnet18_pretrained.py) 的基础上，把整个
backbone 的卷积层与分类头都接上一个由「敏感属性」驱动的 HyperNetwork：超网络
依据 (sex, race, age_group) 逐样本生成 backbone 参数的 *残差偏移* Δθ_p，得到
个性化模型 f_p(x) = f(x; θ + Δθ_p)。

设计参考 drafts/ResNet18_hyperadapt.py（UTKFace 原型，对齐 Xu et al. 2026
"Patient-Conditioned Adaptive Offsets"），但做了三点适配以与 MIMIC baseline 对齐：

  1. Backbone 换成与 baseline 完全一致的 torchvision.models.resnet18 + ImageNet
     预训练权重（drafts 里是从零实现的 ResNet-18）。本模块自建一套与 torchvision
     命名严格对齐的层（conv1/bn1/layerX.i.conv1...），从而能直接把预训练 state_dict
     load 进来，确保图像通路与 baseline 严格可比。
  2. 条件属性从 (sex, race) 扩展为 (sex, race, age_group)，覆盖 MIMICCXRDataset
     返回的全部离散敏感属性；这也对应 baseline 报告里 EqOdds gap 最大的 Age 维度
     与最弱交叉组 Male & Non-White & >=60。
  3. forward 签名统一为 (image, sex, race, age_group)，与 ResNet18HyperHead 一致，
     训练循环可直接复用 (与 MICCCXRDataset.__getitem__ 返回顺序对齐)。

与 HyperHead 的区别（注入深度）：
  - HyperHead 只用超网络生成 **最后一层 fc** 的权重，backbone 特征是共享的——
    无法改变共享特征里已有的子群偏差（实测公平性未稳健提升）。
  - HyperAdapt 把超网络注入到 **每一个卷积层**（channel-wise 乘性调制）以及 fc
    （加性低秩更新），从特征提取阶段就按属性做个性化调整，注入容量更大。

低秩分解（控制超网络参数量、防过拟合）：
  · 卷积层（channel-wise multiplicative modulation）:
        M_p   = A_p · B_p           # [C_out, C_in]
        Θ'_p  = Θ * (1 + M_p)       # 对每个 (out, in) 通道对乘性缩放
  · 线性层 (fc, additive low-rank update):
        ΔW_p  = A_p · B_p           # [out_dim, 512]
        W'_p  = W + ΔW_p
  · Shared A-generator: 同一 output channel 大小 (64/128/256/512) 的所有卷积层
    共用一个 A 生成器，每层只保留独立的 B 生成器（论文 shared generation block）。

初始化保证「起始等价于预训练 baseline」：所有 shared-A 生成器置零 ⇒ M = A·B = 0、
ΔW = 0，于是 Θ' = Θ、W' = W，第 0 步前向严格等于纯 ImageNet 预训练 ResNet-18；
但 B 生成器用 Kaiming 初始化（非零），保证 ∂M/∂A = Bᵀ ≠ 0，超网络梯度不会死锁。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import ResNet18_Weights, resnet18

from src.models.soft_attr_embedding import (
    SoftAgeEmbedding,
    SoftPatientEmbedding,
    SoftSexAgeEmbedding,
    SoftSkinEmbedding,
)


# ============================================================
# Functional helpers: per-sample adapted conv / linear
# ============================================================
def adapted_conv2d(
    x: torch.Tensor,
    weight: torch.Tensor,
    modulation: torch.Tensor,
    stride: int,
    padding: int,
) -> torch.Tensor:
    """
    对每个样本使用其个性化 kernel 做卷积:

        kernel_p = weight * (1 + M_p)        # 各样本不同
        y_p      = conv2d(x_p, kernel_p)

    为了在一次调用里完成整个 batch, 用 grouped-conv 把 batch 维折叠成 group:
    把 weight 复制成 B 份并各自乘上调制, reshape 成 [B*C_out, C_in, K, K] 当作
    groups=B 的卷积核, 输入相应 reshape 成 [1, B*C_in, H, W]。

    Args:
        x         : [B, C_in,  H, W] 输入特征。
        weight    : [C_out, C_in, K, K] backbone 权重 (同 batch 共享)。
        modulation: [B, C_out, C_in] 逐样本通道对乘性调制 M_p。
        stride    : 卷积步幅。
        padding   : 卷积 padding。

    Returns:
        y         : [B, C_out, H', W'] 逐样本卷积结果。
    """
    batch_size, c_in, height, width = x.shape
    c_out, _, kernel_h, kernel_w = weight.shape

    # 1. 构造 per-sample kernel: [B, C_out, C_in, K, K]
    mod = (1.0 + modulation).unsqueeze(-1).unsqueeze(-1)   # [B, C_out, C_in, 1, 1]
    w_per_sample = weight.unsqueeze(0) * mod               # 广播 -> [B, C_out, C_in, K, K]

    # 2. 用 grouped conv 一次性算完: groups=B
    w_grouped = w_per_sample.reshape(batch_size * c_out, c_in, kernel_h, kernel_w)
    x_grouped = x.reshape(1, batch_size * c_in, height, width)  # 把 batch 折进 channel
    y = F.conv2d(x_grouped, w_grouped, bias=None,
                 stride=stride, padding=padding, groups=batch_size)
    out_h, out_w = y.shape[-2:]
    return y.reshape(batch_size, c_out, out_h, out_w)


def adapted_linear(
    x: torch.Tensor,
    weight: torch.Tensor,
    delta_weight: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    对每个样本使用其个性化权重做 linear:
        y_p = (W + ΔW_p) · x_p + b

    Args:
        x           : [B, d_in]           输入特征。
        weight      : [d_out, d_in]       backbone 共享权重。
        delta_weight: [B, d_out, d_in]    逐样本加性低秩更新 ΔW_p。
        bias        : [d_out] 或 None     共享偏置。

    Returns:
        y           : [B, d_out] 逐样本线性变换结果。
    """
    w_per_sample = weight.unsqueeze(0) + delta_weight      # [B, d_out, d_in]
    # bmm: [B, d_out, d_in] @ [B, d_in, 1] -> [B, d_out, 1]
    y = torch.bmm(w_per_sample, x.unsqueeze(-1)).squeeze(-1)
    if bias is not None:
        y = y + bias
    return y


# ============================================================
# Patient embedding (categorical attributes -> dense vector)
# ============================================================
class PatientEmbedding(nn.Module):
    """
    sex / race / age_group 均为离散敏感属性，各自用 nn.Embedding 编码，拼接后经一个
    小 MLP 融合，得到 patient profile vector (条件向量)。对应论文 Sec. II-C
    "Embedding Subgroup-Relevant Attributes" 的 categorical pathway。

    Args:
        num_sex       : sex 类别数 (MIMIC: Male/Female = 2)。
        num_race      : race 类别数 (MIMIC: White/Non-White = 2)。
        num_age       : age_group 类别数 (MIMIC: <thr / >=thr = 2)。
        cat_embed_dim : 每个属性 embedding 的维度。
        out_dim       : patient profile vector 的维度。
    """

    def __init__(
        self,
        num_sex: int = 2,
        num_race: int = 2,
        num_age: int = 2,
        cat_embed_dim: int = 16,
        out_dim: int = 128,
    ) -> None:
        super().__init__()
        self.sex_embed = nn.Embedding(num_sex, cat_embed_dim)
        self.race_embed = nn.Embedding(num_race, cat_embed_dim)
        self.age_embed = nn.Embedding(num_age, cat_embed_dim)
        # Embedding fusion MLP
        self.fuse = nn.Sequential(
            nn.Linear(3 * cat_embed_dim, out_dim),
            nn.ReLU(inplace=True),
            nn.Linear(out_dim, out_dim),
        )

    def forward(
        self, sex: torch.Tensor, race: torch.Tensor, age_group: torch.Tensor
    ) -> torch.Tensor:
        """sex / race / age_group: [B] (long) -> patient profile [B, out_dim]。"""
        s = self.sex_embed(sex)
        r = self.race_embed(race)
        a = self.age_embed(age_group)
        cond = torch.cat([s, r, a], dim=-1)
        return self.fuse(cond)                              # [B, out_dim]


class SkinEmbedding(nn.Module):
    """
    Fitzpatrick17k 的条件通路: 唯一敏感属性是肤色 skin (Fitzpatrick I–VI → 0–5)，
    单个离散属性 (6 类) 用 nn.Embedding 编码后经与 PatientEmbedding 同款的两层 fuse
    MLP 投到 profile vector。输出维度与 PatientEmbedding 一致 (out_dim)，因此可直接
    替换 ResNet18HyperAdapt 的 self.patient_embed 而复用其全部 HyperAdapt 机制。

    Args:
        num_skin     : 肤色类别数 (Fitzpatrick: I–VI = 6)。
        cat_embed_dim: skin embedding 维度。
        out_dim      : profile vector 维度 (须与模型 patient_embed_dim 一致)。
    """

    def __init__(
        self,
        num_skin: int = 6,
        cat_embed_dim: int = 16,
        out_dim: int = 128,
    ) -> None:
        super().__init__()
        self.skin_embed = nn.Embedding(num_skin, cat_embed_dim)
        # fuse 结构与 PatientEmbedding 对齐 (仅输入维度由 3*dim 变为 dim)
        self.fuse = nn.Sequential(
            nn.Linear(cat_embed_dim, out_dim),
            nn.ReLU(inplace=True),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, skin: torch.Tensor) -> torch.Tensor:
        """skin: [B] (long, 0–5) -> patient profile [B, out_dim]。"""
        return self.fuse(self.skin_embed(skin))             # [B, out_dim]


class AgeEmbedding(nn.Module):
    """
    HAM10000 的单属性条件通路: 这里只用 age_group 一个敏感属性条件化 (sex 轴在 HAM 的
    条件互信息诊断中读弱/null, age 轴读出可复现的 I(Y;age|X)>0, 故先只条件化 age)。
    单个离散属性 (4 有效组: 20-40/40-60/60-80/80+ → 0–3) 用 nn.Embedding 编码后经与
    SkinEmbedding / PatientEmbedding 同款的两层 fuse MLP 投到 profile vector。输出维度与
    PatientEmbedding 一致 (out_dim), 因此可直接替换 ResNet18HyperAdapt 的 self.patient_embed
    而复用其全部 HyperAdapt 机制。

    ⚠️ age_group==-1 (0-20 排除组) 不是合法 embedding 索引, 训练/评估前必须在数据侧过滤
       age_group>=0 (见 train_ham10000_hyperadapt.py 的 _filter_age_valid)。

    Args:
        num_age      : age_group 有效类别数 (HAM: 4)。
        cat_embed_dim: age embedding 维度。
        out_dim      : profile vector 维度 (须与模型 patient_embed_dim 一致)。
    """

    def __init__(
        self,
        num_age: int = 4,
        cat_embed_dim: int = 16,
        out_dim: int = 128,
    ) -> None:
        super().__init__()
        self.age_embed = nn.Embedding(num_age, cat_embed_dim)
        # fuse 结构与 SkinEmbedding 对齐 (单属性: 输入维度 = cat_embed_dim)
        self.fuse = nn.Sequential(
            nn.Linear(cat_embed_dim, out_dim),
            nn.ReLU(inplace=True),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, age_group: torch.Tensor) -> torch.Tensor:
        """age_group: [B] (long, 0–3) -> patient profile [B, out_dim]。"""
        return self.fuse(self.age_embed(age_group))         # [B, out_dim]


class SexAgeEmbedding(nn.Module):
    """
    HAM10000 的**双属性**条件通路: sex (2 类) + age_group (4 有效组) 各自 nn.Embedding 编码后
    拼接, 经与 PatientEmbedding / AgeEmbedding 同款的两层 fuse MLP 投到 profile vector。
    结构 = PatientEmbedding 去掉 race 通路 (输入 3*dim → 2*dim), 输出维度 out_dim 不变,
    因此可直接替换 ResNet18HyperAdapt 的 self.patient_embed 而复用全部 HyperAdapt 机制。

    **为什么需要这个双属性版**: 现有 HAM HN 三臂 (Head/Fusion/Adapt) 只条件化 age, 而
    worst-group 评估口径是 Sex / Age / Sex×Age 三套分组 (见 utils/ham10000_fairness.py)
    ——条件化口径与评估口径不对称。本类提供「条件化 = 评估分组变量全集」的对称性对照臂,
    与 age-only 臂的唯一差异就是条件通路多了 sex embedding (单变量对照)。

    ⚠️ age_group==-1 (0-20 排除组) 不是合法 embedding 索引, 训练/评估前须在数据侧过滤
       age_group>=0 (与 age-only 臂用同一 _filter_age_valid, 保证两臂样本集逐样本可比)。

    Args:
        num_sex      : sex 类别数 (HAM: Male/Female = 2)。
        num_age      : age_group 有效类别数 (HAM: 4)。
        cat_embed_dim: 每个属性 embedding 的维度 (与 PatientEmbedding 同为 16)。
        out_dim      : profile vector 维度 (须与模型 patient_embed_dim 一致)。
    """

    def __init__(
        self,
        num_sex: int = 2,
        num_age: int = 4,
        cat_embed_dim: int = 16,
        out_dim: int = 128,
    ) -> None:
        super().__init__()
        self.sex_embed = nn.Embedding(num_sex, cat_embed_dim)
        self.age_embed = nn.Embedding(num_age, cat_embed_dim)
        # fuse 结构与 PatientEmbedding 对齐 (双属性: 输入维度 = 2 * cat_embed_dim)
        self.fuse = nn.Sequential(
            nn.Linear(2 * cat_embed_dim, out_dim),
            nn.ReLU(inplace=True),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, sex: torch.Tensor, age_group: torch.Tensor) -> torch.Tensor:
        """sex: [B] (long, 0/1)、age_group: [B] (long, 0-3) -> patient profile [B, out_dim]。"""
        cond = torch.cat([self.sex_embed(sex), self.age_embed(age_group)], dim=-1)
        return self.fuse(cond)                              # [B, out_dim]


# ============================================================
# HyperAdapt BasicBlock (与 torchvision BasicBlock 命名对齐)
# ============================================================
class HyperBasicBlock(nn.Module):
    """
    结构与 torchvision.models.resnet.BasicBlock 完全相同 (conv1/bn1/conv2/bn2 +
    可选 downsample[conv,bn])，因此能直接 load ImageNet 预训练权重；但每个卷积层
    都接了一个 channel-wise 乘性调制:
                 Θ_ℓ -> Θ_ℓ * (1 + A · B_ℓ)
    其中 A 是 *跨层共享* 的 shared-A (从外部传入)，B_ℓ 是该层独有的生成器。

    Args:
        in_channels      : 输入通道数。
        out_channels     : 输出通道数。
        stride           : 第一个卷积 / downsample 的步幅。
        downsample       : 维度对齐用的 1x1 conv+bn (nn.Sequential) 或 None。
        patient_embed_dim: 条件向量维度。
        rank             : 低秩分解的秩 k。
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
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.rank = rank

        # ----- backbone 卷积层 (命名与 torchvision BasicBlock 一致) -----
        self.conv1 = nn.Conv2d(in_channels, out_channels,
                               kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels,
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

        # ----- 层独有的 B 生成器 (embedding -> rank * C_in) -----
        # conv1 的输入通道是 in_channels
        self.B_gen_conv1 = nn.Linear(patient_embed_dim, rank * in_channels)
        # conv2 的输入通道是 out_channels (block 内中间张量已升维)
        self.B_gen_conv2 = nn.Linear(patient_embed_dim, rank * out_channels)
        # downsample 1x1 conv 的输入通道是 in_channels
        if downsample is not None:
            self.B_gen_down = nn.Linear(patient_embed_dim, rank * in_channels)
        else:
            self.B_gen_down = None

        self._kaiming_init_B()

    def _kaiming_init_B(self) -> None:
        """
        B 生成器用 Kaiming 初始化 (非零)。配合外部 shared-A 全零, 初始有 M=A·B=0
        (等价纯 backbone), 但 ∂M/∂A = Bᵀ ≠ 0, 避免 A、B 同时为零导致 adapter 梯度
        死锁 (经典 LoRA 全零初始化坑)。
        """
        for m in (self.B_gen_conv1, self.B_gen_conv2, self.B_gen_down):
            if m is None:
                continue
            nn.init.kaiming_normal_(m.weight, a=0, mode="fan_in", nonlinearity="linear")
            nn.init.zeros_(m.bias)

    def _make_modulation(
        self, a_shared: torch.Tensor, b_gen: nn.Linear,
        embedding: torch.Tensor, c_in: int,
    ) -> torch.Tensor:
        """
        生成某层的 channel-wise 调制 M = A · B。

        Args:
            a_shared : [B, C_out, rank] 跨层共享的 A (外部传入)。
            b_gen    : nn.Linear(embed_dim -> rank * c_in) 层独有的 B 生成器。
            embedding: [B, embed_dim] 条件向量。
            c_in     : 该层输入通道数。

        Returns:
            M : [B, C_out, c_in] 逐样本通道对调制。
        """
        batch_size = embedding.shape[0]
        b_mat = b_gen(embedding).view(batch_size, self.rank, c_in)
        return torch.bmm(a_shared, b_mat)

    def forward(
        self, x: torch.Tensor, embedding: torch.Tensor, a_shared: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            x        : [B, C_in, H, W] 输入特征。
            embedding: [B, embed_dim] 条件向量。
            a_shared : [B, C_out, rank] 本 stage 共享的 A。

        Returns:
            [B, C_out, H', W'] 残差块输出。
        """
        identity = x

        # ---- conv1 with adapter ----
        m1 = self._make_modulation(a_shared, self.B_gen_conv1, embedding, self.in_channels)
        out = adapted_conv2d(x, self.conv1.weight, m1, stride=self.stride, padding=1)
        out = self.bn1(out)
        out = F.relu(out, inplace=True)

        # ---- conv2 with adapter ----
        m2 = self._make_modulation(a_shared, self.B_gen_conv2, embedding, self.out_channels)
        out = adapted_conv2d(out, self.conv2.weight, m2, stride=1, padding=1)
        out = self.bn2(out)

        # ---- downsample (也带 adapter) ----
        if self.downsample is not None:
            ds_conv: nn.Conv2d = self.downsample[0]
            ds_bn: nn.BatchNorm2d = self.downsample[1]
            md = self._make_modulation(a_shared, self.B_gen_down, embedding, self.in_channels)
            ds_stride = ds_conv.stride[0] if isinstance(ds_conv.stride, tuple) else ds_conv.stride
            identity = adapted_conv2d(x, ds_conv.weight, md, stride=ds_stride, padding=0)
            identity = ds_bn(identity)

        out = out + identity
        out = F.relu(out, inplace=True)
        return out


# ============================================================
# ResNet-18 (ImageNet 预训练) + HyperAdapt
# ============================================================
class ResNet18HyperAdapt(nn.Module):
    """
    与 MIMIC baseline 对齐的 ImageNet 预训练 ResNet-18 backbone，其每个卷积层
    (channel-wise 乘性调制) 与分类头 fc (加性低秩更新) 都由 HyperAdapt 超网络依据
    离散敏感属性 (sex / race / age_group) 逐样本调整。

    Args:
        num_classes      : 分类头输出维度。
            - 1 -> 配合 nn.BCEWithLogitsLoss (默认; No Finding U-Zeros 二分类)
            - k -> 配合 nn.CrossEntropyLoss
        num_sex, num_race, num_age: 三个离散敏感属性的类别数 (MIMIC 均为 2)。
        patient_embed_dim: patient profile vector 维度。
        rank             : 低秩分解的秩 k (越大注入容量越强、参数越多)。
        pretrained       : 是否载入 torchvision ImageNet 预训练 backbone 权重
                           (与 baseline 一致, 默认 True)。
        freeze_backbone  : 是否冻结特征提取 backbone (stem conv1/bn1 + 各 stage 内
                           每个 block 的 backbone conv1/bn1/conv2/bn2/downsample)。
                           置 True 时这些层的 requires_grad=False, 梯度只能流入超网络
                           生成器与任务头 fc——对齐论文「冻结 backbone, 只训 adapter」
                           的设定 (用于检验「联合微调时 backbone 吸走梯度、Δθ≈0」假设)。
                           注: 仅冻结仿射/卷积参数 (θ); BN running stats 仍随 CXR 分布
                           自适应, 避免 ImageNet BN 统计直接套用拖垮整体 AUC。
                           任务头 fc 因是新初始化的 (非预训练诊断头) 保持可训练。
    """

    # ResNet-18 各 stage 输出通道数
    STAGE_CHANNELS = (64, 128, 256, 512)

    def __init__(
        self,
        num_classes: int = 1,
        num_sex: int = 2,
        num_race: int = 2,
        num_age: int = 2,
        patient_embed_dim: int = 128,
        rank: int = 4,
        pretrained: bool = True,
        freeze_backbone: bool = False,
    ) -> None:
        super().__init__()
        self.rank = rank
        self.out_dim = num_classes
        self.patient_embed_dim = patient_embed_dim
        self.in_channels_running = 64

        # ============================================================
        # 1. Patient-Aware Modulation Net (超网络条件通路)
        # ============================================================
        # (1) 属性 -> patient profile vector
        self.patient_embed = PatientEmbedding(
            num_sex=num_sex, num_race=num_race, num_age=num_age,
            cat_embed_dim=16, out_dim=patient_embed_dim,
        )
        # (2) Shared A 生成器 —— 按 output channel 大小分组共享 (shared generation block)
        self.A_gen = nn.ModuleDict({
            f"c{c}": nn.Linear(patient_embed_dim, c * rank)
            for c in self.STAGE_CHANNELS
        })

        # ============================================================
        # 2. Backbone (ResNet-18, 命名与 torchvision 严格对齐)
        # ============================================================
        # Stem (不带 adapter, 论文明确说明 stem 保持原样)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # 4 个 residual stage, 每个 stage 2 个 HyperBasicBlock (与 ResNet-18 一致)
        self.layer1 = self._make_layer(out_channels=64, blocks=2, stride=1)
        self.layer2 = self._make_layer(out_channels=128, blocks=2, stride=2)
        self.layer3 = self._make_layer(out_channels=256, blocks=2, stride=2)
        self.layer4 = self._make_layer(out_channels=512, blocks=2, stride=2)

        # ============================================================
        # 3. 带 linear adapter 的分类头
        # ============================================================
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

        # FC adapter: ΔW = A_fc · B_fc (additive, low-rank)
        self.fc_A_gen = nn.Linear(patient_embed_dim, num_classes * rank)
        self.fc_B_gen = nn.Linear(patient_embed_dim, rank * 512)

        # ============================================================
        # 4. 参数初始化 + 载入预训练 backbone
        # ============================================================
        self._init_adapter_weights()
        if pretrained:
            self._load_pretrained_backbone()
        if freeze_backbone:
            self._freeze_backbone()

    # ------------------------------------------------------------
    # 构造一个 stage (n 个 HyperBasicBlock), 用 nn.ModuleList 保存
    # ------------------------------------------------------------
    def _make_layer(self, out_channels: int, blocks: int, stride: int) -> nn.ModuleList:
        """构造一个 residual stage; downsample 命名与 torchvision 对齐 (Sequential[conv,bn])。"""
        downsample = None
        if stride != 1 or self.in_channels_running != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels_running, out_channels,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

        layers = nn.ModuleList()
        layers.append(HyperBasicBlock(
            self.in_channels_running, out_channels, stride, downsample,
            patient_embed_dim=self.patient_embed_dim, rank=self.rank,
        ))
        self.in_channels_running = out_channels
        for _ in range(1, blocks):
            layers.append(HyperBasicBlock(
                self.in_channels_running, out_channels, 1, None,
                patient_embed_dim=self.patient_embed_dim, rank=self.rank,
            ))
        return layers

    # ------------------------------------------------------------
    # 初始化 adapter 生成器 (backbone 卷积权重稍后由预训练覆盖)
    # ------------------------------------------------------------
    def _init_adapter_weights(self) -> None:
        """
        初始化超网络相关参数, 保证「起始 Δθ ≈ 0、等价于纯预训练 backbone」:
          · 所有 shared-A 生成器置零 ⇒ M = A·B = 0 (卷积无调制)、ΔW = 0 (fc 无更新)。
          · B 生成器用 Kaiming (block 内已做 / 此处对 fc_B_gen 做), 保证梯度非零。
        backbone 的 Conv2d/BN 权重不在此初始化 (由 _load_pretrained_backbone 覆盖);
        base fc 用默认 Linear 初始化 (与 baseline 替换 fc 的做法一致)。
        """
        # patient_embed 内的 Linear / Embedding 用默认初始化即可
        # Shared A 生成器 (conv) 全部置零
        for m in self.A_gen.values():
            nn.init.zeros_(m.weight)
            nn.init.zeros_(m.bias)

        # FC adapter: A 置零保证初始 ΔW=0; B 用 Kaiming 保证梯度可流动
        nn.init.zeros_(self.fc_A_gen.weight)
        nn.init.zeros_(self.fc_A_gen.bias)
        nn.init.kaiming_normal_(self.fc_B_gen.weight, a=0, mode="fan_in", nonlinearity="linear")
        nn.init.zeros_(self.fc_B_gen.bias)

    # ------------------------------------------------------------
    # 载入 torchvision ImageNet 预训练 backbone 权重
    # ------------------------------------------------------------
    def _load_pretrained_backbone(self) -> None:
        """
        本模块的 backbone 层命名与 torchvision.models.resnet18 严格对齐
        (conv1/bn1, layerX.i.conv1/bn1/conv2/bn2/downsample.0/.1)，因此可直接把
        torchvision 预训练 state_dict 中形状匹配的键 load 进来。fc 因输出维度不同
        (1000 vs num_classes) 被自动跳过, 与 baseline 替换 fc 的设置一致;
        超网络生成器 (A_gen / B_gen / fc adapter / patient_embed) 不在预训练 sd 中,
        保留 _init_adapter_weights 的初始化。
        """
        tv_state = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).state_dict()
        own_state = self.state_dict()
        # 只载入「键存在且形状匹配」的 backbone 参数 (跳过 fc 与所有 adapter 生成器)
        load_state = {
            k: v for k, v in tv_state.items()
            if k in own_state and own_state[k].shape == v.shape
        }
        self.load_state_dict(load_state, strict=False)

    # ------------------------------------------------------------
    # 冻结特征提取 backbone (只训超网络生成器 + 任务头 fc)
    # ------------------------------------------------------------
    def _freeze_backbone(self) -> None:
        """
        冻结 backbone 特征提取部分的可学习参数 (requires_grad=False):
          · stem: conv1 / bn1
          · 各 stage 内每个 block 的 backbone conv1/bn1/conv2/bn2 与 downsample(conv+bn)

        *不* 冻结的部分 (保持可训练):
          · 超网络生成器: patient_embed / A_gen / 各 block 的 B_gen_* / fc_A_gen / fc_B_gen
          · 任务头 base fc (新初始化, 需学到 No Finding 诊断映射)

        说明: 这里只冻结仿射/卷积参数 θ; BN 的 running_mean/var 是 buffer (非梯度更新),
        仍随训练自适应 CXR 分布——避免把 ImageNet BN 统计硬套在 CXR 上拖垮整体 AUC,
        从而把实验变量隔离为「卷积/线性权重能否被梯度更新」(即梯度是否被逼入 adapter)。
        """
        # stem
        for module in (self.conv1, self.bn1):
            for p in module.parameters():
                p.requires_grad = False
        # 各 residual stage 内每个 block 的 backbone 层 (B_gen_* 生成器不在此列, 保持可训练)
        for blk in (*self.layer1, *self.layer2, *self.layer3, *self.layer4):
            backbone_modules: list[nn.Module] = [blk.conv1, blk.bn1, blk.conv2, blk.bn2]
            if blk.downsample is not None:
                backbone_modules.append(blk.downsample)
            for module in backbone_modules:
                for p in module.parameters():
                    p.requires_grad = False

    # ------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------
    def forward(
        self,
        image: torch.Tensor,
        *attrs: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            image : [B, 3, 224, 224] 图像 (CXR 经 Grayscale(3) 复制为 3 通道)。
            *attrs: 若干离散敏感属性，每个为 [B] long，按位置传给 self.patient_embed。
                - MIMIC (默认 PatientEmbedding): (sex, race, age_group)，0=Male/White/<thr 等。
                - Fitzpatrick (ResNet18HyperAdaptSkin 的 SkinEmbedding): 单个 (skin,)，0–5。
                forward 不关心属性语义，只把 *attrs 转交条件通路生成 embedding——这是
                单/多属性数据集能共用整套 HyperAdapt 机制 (A_gen/B_gen/逐样本卷积/fc) 的关键。

        Returns:
            logits: [B, num_classes]。
        """
        # 1. patient embedding (一次性算出条件向量)
        embedding = self.patient_embed(*attrs)                 # [B, embed_dim]
        batch_size = embedding.shape[0]

        # 2. 预生成 4 个 stage 的 shared-A。同 stage 内所有 block / conv 共用 A_c，
        #    但每个 conv 用各自的 B_ℓ 配出不同的 M = A·B_ℓ。
        a_shared = {
            c: self.A_gen[f"c{c}"](embedding).view(batch_size, c, self.rank)
            for c in self.STAGE_CHANNELS
        }

        # 3. Stem (无 adapter)
        out = self.conv1(image)
        out = self.bn1(out)
        out = F.relu(out, inplace=True)
        out = self.maxpool(out)

        # 4. 带 adapter 的 4 个 residual stage
        for blk in self.layer1:
            out = blk(out, embedding, a_shared[64])
        for blk in self.layer2:
            out = blk(out, embedding, a_shared[128])
        for blk in self.layer3:
            out = blk(out, embedding, a_shared[256])
        for blk in self.layer4:
            out = blk(out, embedding, a_shared[512])

        # 5. Pool + 带 adapter 的 fc
        out = self.avgpool(out)
        out = torch.flatten(out, 1)                            # [B, 512]

        a_fc = self.fc_A_gen(embedding).view(batch_size, self.out_dim, self.rank)
        b_fc = self.fc_B_gen(embedding).view(batch_size, self.rank, 512)
        delta_w = torch.bmm(a_fc, b_fc)                        # [B, out_dim, 512]

        logits = adapted_linear(out, self.fc.weight, delta_w, self.fc.bias)
        return logits

    # ------------------------------------------------------------
    # 工具: 分别统计 backbone / adapter 的参数量
    # ------------------------------------------------------------
    def param_breakdown(self) -> tuple[int, int, int]:
        """返回 (总参数量, 超网络/adapter 参数量, backbone 参数量)。"""
        adapter_modules = list(self.patient_embed.parameters())
        adapter_modules += [p for m in self.A_gen.values() for p in m.parameters()]
        adapter_modules += list(self.fc_A_gen.parameters())
        adapter_modules += list(self.fc_B_gen.parameters())
        for blk in (*self.layer1, *self.layer2, *self.layer3, *self.layer4):
            adapter_modules += list(blk.B_gen_conv1.parameters())
            adapter_modules += list(blk.B_gen_conv2.parameters())
            if blk.B_gen_down is not None:
                adapter_modules += list(blk.B_gen_down.parameters())
        n_adapter = sum(p.numel() for p in adapter_modules)
        n_total = sum(p.numel() for p in self.parameters())
        return n_total, n_adapter, n_total - n_adapter


# ============================================================
# ResNet-18 + HyperAdapt for Fitzpatrick17k (单一肤色属性)
# ============================================================
class ResNet18HyperAdaptSkin(ResNet18HyperAdapt):
    """
    Fitzpatrick17k 版 HyperAdapt: 复用 ResNet18HyperAdapt 的全部机制 (ImageNet 预训练
    backbone + 每个 conv 的 channel-wise 乘性调制 + fc 加性低秩更新 + Δθ≈0 初始化)，
    仅把 MIMIC 的三属性条件通路 (sex/race/age_group, PatientEmbedding) 替换为单一肤色
    属性 (skin ∈ {0..5}, SkinEmbedding)，以对齐 FitzpatrickDataset 的 (image, label, skin)。

    forward 签名 (image, skin) ——父类 forward 已泛化为 (image, *attrs)，此处单属性即
    self.patient_embed(skin)；其余前向/反向、param_breakdown 完全复用父类。

    Args:
        num_classes      : 分类头输出维度 (malignant 二分类 → 1，配 BCEWithLogitsLoss)。
        num_skin         : 肤色类别数 (Fitzpatrick I–VI = 6)。
        patient_embed_dim: profile vector 维度 (须与 SkinEmbedding.out_dim 一致)。
        rank             : 低秩分解的秩 k。
        pretrained       : 是否载入 ImageNet 预训练 backbone (与 baseline 一致, 默认 True)。
        freeze_backbone  : 是否冻结特征提取 backbone (只训超网络生成器 + 任务头 fc)。
    """

    def __init__(
        self,
        num_classes: int = 1,
        num_skin: int = 6,
        patient_embed_dim: int = 128,
        rank: int = 4,
        pretrained: bool = True,
        freeze_backbone: bool = False,
    ) -> None:
        # 先按父类构建完整 HyperAdapt (含预训练载入 / 冻结)，patient_embed 暂为
        # 默认 PatientEmbedding；随后整体替换为单属性 SkinEmbedding。
        super().__init__(
            num_classes=num_classes,
            patient_embed_dim=patient_embed_dim,
            rank=rank,
            pretrained=pretrained,
            freeze_backbone=freeze_backbone,
        )
        # 替换条件通路: 三属性 → 单一 6 类肤色 (输出维度不变, 不影响下游 A_gen/B_gen/fc)
        self.patient_embed = SkinEmbedding(
            num_skin=num_skin, cat_embed_dim=16, out_dim=patient_embed_dim,
        )


# ============================================================
# ResNet-18 + HyperAdapt for HAM10000 (单一 age 属性)
# ============================================================
class ResNet18HyperAdaptAge(ResNet18HyperAdapt):
    """
    HAM10000 版 HyperAdapt: 复用 ResNet18HyperAdapt 的全部机制 (ImageNet 预训练 backbone +
    每个 conv 的 channel-wise 乘性调制 + fc 加性低秩更新 + Δθ≈0 初始化)，仅把三属性条件通路
    (sex/race/age_group, PatientEmbedding) 替换为单一 age_group 属性 (age ∈ {0..3},
    AgeEmbedding)，以对齐 HAM10000Dataset 的 (image, label, sex, age_group)——这里**只条件化
    age**: HAM 条件互信息诊断里 sex 轴读弱/null、age 轴读出可复现 I(Y;age|X)>0, 故先只接 age。

    forward 签名 (image, age_group) ——父类 forward 已泛化为 (image, *attrs)，此处单属性即
    self.patient_embed(age_group)；其余前向/反向、param_breakdown 完全复用父类。

    ⚠️ 仅接受 age_group∈{0..3}; age_group==-1 (0-20 排除组) 须在数据侧先过滤 (训练脚本负责)。

    Args:
        num_classes      : 分类头输出维度 (malignant 二分类 → 1，配 BCEWithLogitsLoss)。
        num_age          : age_group 有效类别数 (HAM: 4)。
        patient_embed_dim: profile vector 维度 (须与 AgeEmbedding.out_dim 一致)。
        rank             : 低秩分解的秩 k。
        pretrained       : 是否载入 ImageNet 预训练 backbone (与 baseline 一致, 默认 True)。
        freeze_backbone  : 是否冻结特征提取 backbone (只训超网络生成器 + 任务头 fc)。
    """

    def __init__(
        self,
        num_classes: int = 1,
        num_age: int = 4,
        patient_embed_dim: int = 128,
        rank: int = 4,
        pretrained: bool = True,
        freeze_backbone: bool = False,
    ) -> None:
        # 先按父类构建完整 HyperAdapt (含预训练载入 / 冻结)，patient_embed 暂为
        # 默认 PatientEmbedding；随后整体替换为单属性 AgeEmbedding。
        super().__init__(
            num_classes=num_classes,
            patient_embed_dim=patient_embed_dim,
            rank=rank,
            pretrained=pretrained,
            freeze_backbone=freeze_backbone,
        )
        # 替换条件通路: 三属性 → 单一 4 类 age (输出维度不变, 不影响下游 A_gen/B_gen/fc)
        self.patient_embed = AgeEmbedding(
            num_age=num_age, cat_embed_dim=16, out_dim=patient_embed_dim,
        )


# ============================================================
# ResNet-18 + HyperAdapt for HAM10000 (sex + age 双属性，对称性对照臂)
# ============================================================
class ResNet18HyperAdaptSexAge(ResNet18HyperAdapt):
    """
    HAM10000 版 HyperAdapt 的**双属性 (sex + age_group)** 变体: 与 ResNet18HyperAdaptAge 的
    唯一差异是条件通路由 AgeEmbedding 换成 SexAgeEmbedding, HyperAdapt 机制 (每 conv 的
    channel-wise 乘性调制 + fc 加性低秩更新 + Δθ≈0 初始化 + 预训练 backbone) 完全复用父类。

    动机: age-only 三臂的条件化口径 (仅 age) 与 worst-group 评估口径 (Sex / Age / Sex×Age)
    不对称——sex 轴与交叉格上的最差子群, 模型从未拿到对应属性。本臂把条件输入补齐为评估分组
    变量全集, 作为对称性对照 (method 名 hyperadapt_sexage, 与 age-only 臂产物分开存放)。

    forward 签名 (image, sex, age_group)——父类 forward 已泛化为 (image, *attrs), 按位置
    转交 self.patient_embed(sex, age_group), 故无需覆盖 forward。

    ⚠️ 仅接受 age_group∈{0..3}; age_group==-1 须在数据侧先过滤 (训练脚本负责)。

    Args:
        num_classes      : 分类头输出维度 (malignant 二分类 → 1)。
        num_sex          : sex 类别数 (HAM: 2)。
        num_age          : age_group 有效类别数 (HAM: 4)。
        patient_embed_dim: profile vector 维度 (须与 SexAgeEmbedding.out_dim 一致)。
        rank             : 低秩分解的秩 k。
        pretrained       : 是否载入 ImageNet 预训练 backbone (与 baseline 一致, 默认 True)。
        freeze_backbone  : 是否冻结特征提取 backbone (只训超网络生成器 + 任务头 fc)。
    """

    def __init__(
        self,
        num_classes: int = 1,
        num_sex: int = 2,
        num_age: int = 4,
        patient_embed_dim: int = 128,
        rank: int = 4,
        pretrained: bool = True,
        freeze_backbone: bool = False,
    ) -> None:
        # 先按父类构建完整 HyperAdapt (含预训练载入 / 冻结)，随后替换条件通路
        super().__init__(
            num_classes=num_classes,
            patient_embed_dim=patient_embed_dim,
            rank=rank,
            pretrained=pretrained,
            freeze_backbone=freeze_backbone,
        )
        # 替换条件通路: 三属性 (sex/race/age) → 双属性 (sex/age)，输出维度不变
        self.patient_embed = SexAgeEmbedding(
            num_sex=num_sex, num_age=num_age, cat_embed_dim=16, out_dim=patient_embed_dim,
        )


# ============================================================
# ResNet-18 + HyperAdapt，条件输入为**属性概率**（Predicted-Attribute 线，实验 P）
# ============================================================
# 方案见 docs/predicted_attribute_hyperadapt_plan.md。与上面三个 GT 变体的唯一差异是条件通路
# 换成 soft_attr_embedding 的期望嵌入版（e = p̂ᵀE）：条件输入由属性索引 [B] 变为属性概率 [B, C]。
# p̂ 为 one-hot 时与 GT 版逐位等价，且参数命名一致 ⇒ 两版 state_dict 可互 load。


class ResNet18HyperAdaptSoftSkin(ResNet18HyperAdapt):
    """
    Fitzpatrick17k 的 **pred-attr** HyperAdapt：forward 签名 (image, skin_prob)，
    其中 skin_prob 为子群分类器 g 输出的 [B, 6] softmax（训练与测试两阶段都用它）。

    与 ResNet18HyperAdaptSkin 的唯一差异是 self.patient_embed 换成 SoftSkinEmbedding；
    A_gen / B_gen / 逐样本卷积 / fc 低秩更新 / Δθ≈0 初始化全部复用父类。

    Args:
        num_classes      : 分类头输出维度（malignant 二分类 → 1）。
        num_skin         : 肤色类别数（Fitzpatrick I–VI = 6）。
        patient_embed_dim: profile vector 维度。
        rank             : 低秩分解的秩 k。
        pretrained       : 是否载入 ImageNet 预训练 backbone。
        freeze_backbone  : 是否冻结特征提取 backbone。
    """

    def __init__(
        self,
        num_classes: int = 1,
        num_skin: int = 6,
        patient_embed_dim: int = 128,
        rank: int = 4,
        pretrained: bool = True,
        freeze_backbone: bool = False,
    ) -> None:
        super().__init__(
            num_classes=num_classes,
            patient_embed_dim=patient_embed_dim,
            rank=rank,
            pretrained=pretrained,
            freeze_backbone=freeze_backbone,
        )
        self.patient_embed = SoftSkinEmbedding(
            num_skin=num_skin, cat_embed_dim=16, out_dim=patient_embed_dim,
        )


class ResNet18HyperAdaptSoftAge(ResNet18HyperAdapt):
    """
    HAM10000 的 **pred-attr** HyperAdapt：forward 签名 (image, age_prob)，
    age_prob 为 g 输出的 [B, num_age] softmax。其余同 ResNet18HyperAdaptAge。

    Args:
        num_classes / num_age / patient_embed_dim / rank / pretrained / freeze_backbone:
            见 ResNet18HyperAdaptAge。
    """

    def __init__(
        self,
        num_classes: int = 1,
        num_age: int = 4,
        patient_embed_dim: int = 128,
        rank: int = 4,
        pretrained: bool = True,
        freeze_backbone: bool = False,
    ) -> None:
        super().__init__(
            num_classes=num_classes,
            patient_embed_dim=patient_embed_dim,
            rank=rank,
            pretrained=pretrained,
            freeze_backbone=freeze_backbone,
        )
        self.patient_embed = SoftAgeEmbedding(
            num_age=num_age, cat_embed_dim=16, out_dim=patient_embed_dim,
        )


class ResNet18HyperAdaptSoftSexAge(ResNet18HyperAdapt):
    """
    HAM10000 **双属性 pred-attr** HyperAdapt（实验 P 的 sex+age 条件化版）：forward 签名
    (image, sex_prob, age_prob)，两份 softmax 均由子群分类器 g 给出（训练与测试两阶段都用 p̂）。

    与 GT 版 `ResNet18HyperAdaptSexAge` 的唯一差异是条件通路换成 SoftSexAgeEmbedding
    （期望嵌入 e = p̂ᵀE），参数命名一致 ⇒ 两版 state_dict 可互 load，p̂ 为 one-hot 时逐位等价。
    与 age-only pred 版 `ResNet18HyperAdaptSoftAge` 的唯一差异是条件输入多了 sex 概率，
    使实验 P 的条件化口径与 worst-group 评估口径（Sex / Age / Sex×Age）对齐。

    Args:
        num_classes / num_sex / num_age / patient_embed_dim / rank / pretrained /
        freeze_backbone: 见 ResNet18HyperAdaptSexAge。
    """

    def __init__(
        self,
        num_classes: int = 1,
        num_sex: int = 2,
        num_age: int = 4,
        patient_embed_dim: int = 128,
        rank: int = 4,
        pretrained: bool = True,
        freeze_backbone: bool = False,
    ) -> None:
        super().__init__(
            num_classes=num_classes,
            patient_embed_dim=patient_embed_dim,
            rank=rank,
            pretrained=pretrained,
            freeze_backbone=freeze_backbone,
        )
        self.patient_embed = SoftSexAgeEmbedding(
            num_sex=num_sex, num_age=num_age, cat_embed_dim=16, out_dim=patient_embed_dim,
        )


class ResNet18HyperAdaptSoftPatient(ResNet18HyperAdapt):
    """
    MIMIC / CheXpert 的 **pred-attr** HyperAdapt：forward 签名
    (image, sex_prob, race_prob, age_prob)，三轴各一份 softmax。其余同 ResNet18HyperAdapt。

    Args:
        num_classes / num_sex / num_race / num_age / patient_embed_dim / rank /
        pretrained / freeze_backbone: 见 ResNet18HyperAdapt 与 PatientEmbedding。
    """

    def __init__(
        self,
        num_classes: int = 1,
        num_sex: int = 2,
        num_race: int = 2,
        num_age: int = 2,
        patient_embed_dim: int = 128,
        rank: int = 4,
        pretrained: bool = True,
        freeze_backbone: bool = False,
    ) -> None:
        super().__init__(
            num_classes=num_classes,
            patient_embed_dim=patient_embed_dim,
            rank=rank,
            pretrained=pretrained,
            freeze_backbone=freeze_backbone,
        )
        self.patient_embed = SoftPatientEmbedding(
            num_sex=num_sex, num_race=num_race, num_age=num_age,
            cat_embed_dim=16, out_dim=patient_embed_dim,
        )


# ============================================================
# 结构自检
# ============================================================
if __name__ == "__main__":
    torch.manual_seed(0)
    # 自检不下载预训练权重 (pretrained=False), 仅验证结构与前向/反向
    model = ResNet18HyperAdapt(num_classes=1, rank=4, pretrained=False)
    model.eval()                                          # eval 避免 BN 在小 batch 上不稳

    batch_size = 4
    image = torch.randn(batch_size, 3, 224, 224)
    sex = torch.tensor([0, 1, 0, 1], dtype=torch.long)
    race = torch.tensor([0, 1, 0, 1], dtype=torch.long)
    age = torch.tensor([0, 0, 1, 1], dtype=torch.long)

    logits = model(image, sex, race, age)
    print(f"[MIMIC 3-attr] Input  shape : {tuple(image.shape)}")   # (4, 3, 224, 224)
    print(f"[MIMIC 3-attr] Logits shape : {tuple(logits.shape)}")  # (4, 1)

    n_total, n_adapter, n_backbone = model.param_breakdown()
    print(f"Total    params : {n_total:>12,}")
    print(f"  Backbone      : {n_backbone:>12,}")
    print(f"  Adapter (phi) : {n_adapter:>12,}")

    # 反向传播检查
    loss = logits.sum()
    loss.backward()
    print("Backward pass OK.")

    # ---- Fitzpatrick 单属性 (skin) 变体自检 ----
    skin_model = ResNet18HyperAdaptSkin(num_classes=1, num_skin=6, rank=4, pretrained=False)
    skin_model.eval()
    skin = torch.tensor([0, 2, 4, 5], dtype=torch.long)
    skin_logits = skin_model(image, skin)
    print(f"\n[Fitz 1-attr ] Logits shape : {tuple(skin_logits.shape)}")  # (4, 1)
    n_total_s, n_adapter_s, n_backbone_s = skin_model.param_breakdown()
    print(f"Total params : {n_total_s:>12,} (backbone {n_backbone_s:,} + adapter {n_adapter_s:,})")
    skin_logits.sum().backward()
    print("Skin variant backward pass OK.")

    # ---- HAM10000 单属性 (age) 变体自检 ----
    age_model = ResNet18HyperAdaptAge(num_classes=1, num_age=4, rank=4, pretrained=False)
    age_model.eval()
    age = torch.tensor([0, 1, 2, 3], dtype=torch.long)   # 4 个有效 age 组 (无 -1)
    age_logits = age_model(image, age)
    print(f"\n[HAM 1-attr  ] Logits shape : {tuple(age_logits.shape)}")  # (4, 1)
    n_total_a, n_adapter_a, n_backbone_a = age_model.param_breakdown()
    print(f"Total params : {n_total_a:>12,} (backbone {n_backbone_a:,} + adapter {n_adapter_a:,})")
    age_logits.sum().backward()
    print("Age variant backward pass OK.")

    # ---- pred-attr (soft) 变体自检：one-hot 概率须与 GT 索引版给出同样的 logits ----
    soft_model = ResNet18HyperAdaptSoftSkin(num_classes=1, num_skin=6, rank=4, pretrained=False)
    soft_model.eval()
    # 两版参数名一致，直接把 GT 版权重整体搬过来，构成严格可比的等价性检验
    soft_model.load_state_dict(skin_model.state_dict())
    skin_onehot = torch.nn.functional.one_hot(skin, num_classes=6).float()
    with torch.no_grad():
        delta = (soft_model(image, skin_onehot) - skin_model(image, skin)).abs().max().item()
    print(f"\n[Fitz soft   ] one-hot 概率 vs GT 索引  max|Δlogit| = {delta:.3e}")
    assert delta < 1e-5, "soft 版在 one-hot 输入下未复现 GT 版 logits"
    # 软输入（非 one-hot）前向 + 反向
    soft_prob = torch.softmax(torch.randn(batch_size, 6), dim=-1)
    soft_logits = soft_model(image, soft_prob)
    print(f"[Fitz soft   ] Logits shape : {tuple(soft_logits.shape)}")
    soft_logits.sum().backward()
    print("Soft-skin variant backward pass OK.")
