"""
ImageNet 预训练 ResNet-18 + HyperFusion (MIP-init) for MIMIC-CXR Binary Classification
======================================================================================
在 image-only baseline (src/models/resnet18_pretrained.py) 的基础上, 仅在 backbone 的
**layer4[0].downsample 1x1 卷积** 处接一个由「敏感属性」驱动的 HyperNetwork: 超网络依据
(sex, race, age_group) 逐样本生成该 downsample 卷积权重的 *残差扰动* Δθ_p, 得到个性化
shortcut θ_ds = θ_0 + h(γ_p)。其余层 (stem、layer1~3、layer4[0] 主分支、layer4[1]、fc)
与 baseline 完全一致, 共享同一份权重。

设计参考 drafts/PreactivResNet18_hyperfusion_MIP.py (UTKFace 原型, 对齐 HyperFusion
论文 Duenias et al., Medical Image Analysis 102:103503, 2025), 但做了三点适配以与 MIMIC baseline 对齐:

  1. Backbone 换成与 baseline 完全一致的 torchvision.models.resnet18 + ImageNet 预训练
     权重 (drafts 里是从零实现的 PreactivResNet18)。本模块直接持有一个完整的 torchvision
     resnet18 作为 self.backbone, 因此预训练权重可无损 load, 图像通路与 baseline 严格可比。
  2. 条件属性从 (sex, race) 扩展为 (sex, race, age_group), 覆盖 MIMICCXRDataset 返回的
     全部离散敏感属性; 这也对应 baseline 报告里 EqOdds gap 最大的 Age 维度与最弱交叉组
     Male & Non-White & >=60。
  3. forward 签名统一为 (image, sex, race, age_group), 与 ResNet18HyperHead /
     ResNet18HyperAdapt 一致, 训练循环可直接复用 (与 MIMICCXRDataset.__getitem__ 返回
     顺序对齐: image, label, sex, race, age_group)。

为什么选 layer4[0].downsample (对齐 HyperFusion 论文的注入位置):
  - HyperFusion 论文把超网络接在网络较深、通道数最大的 shortcut 上 (此处 256->512,
    1x1 conv), 既能在高层语义特征上按属性做个性化重组, 又因 1x1 卷积参数量适中
    (512*256=131,072) 而便于超网络直接生成全权重。
  - 与 HyperAdapt「每个卷积层都低秩调制」相比, HyperFusion 是「单点、全权重生成」:
    注入点更集中、更深, 用以检验「集中在高层 shortcut 的强注入」对公平性的影响。

MIP 初始化 (Ortiz et al., ICLR 2024; ⚠ 展开式与完整出处待核实——HyperFusion 原文用的是
Chang et al. 2019 的方差对齐初始化, 不含 MIP) 保证「起始等价于
预训练 baseline」:
  - additive 重写: θ_ds = θ_0 + h(γ)
        · θ_0    : backbone 自带的 downsample 卷积权重 (已由 ImageNet 预训练填充, 可学基底)
        · h(γ)   : 超网络生成的扰动, 输出层权重显式缩小 mip_scale 倍 ⇒ 初始 h(γ) ≈ 0
    于是第 0 步前向 θ_ds ≈ θ_0, 严格等价于纯 ImageNet 预训练 ResNet-18, 训练从 baseline
    起步、平滑接入属性条件化。
  - E_L2 投影: patient profile 向量逐样本归一到固定 norm 球面, 切断属性 embedding 的
    magnitude proportionality, 稳定超网络输出尺度 (MIP 组件 1)。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import ResNet18_Weights, resnet18


# ============================================================
# Patient embedding (categorical attributes -> dense vector)
# ============================================================
class PatientEmbedding(nn.Module):
    """
    sex / race / age_group 均为离散敏感属性, 各自用 nn.Embedding 编码, 拼接后经一个
    小 MLP 融合, 得到 patient profile vector (条件向量 γ)。与 ResNet18HyperAdapt 中的
    PatientEmbedding 保持一致的写法, 保证不同变体的条件通路可比。

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


class AgeEmbedding(nn.Module):
    """
    HAM10000 的单属性条件通路: 只用 age_group 一个敏感属性 (4 有效组: 20-40/40-60/60-80/80+
    → 0–3)。结构与本模块 PatientEmbedding 对齐 (单 embedding → 两层 fuse MLP), 仅输入由
    「sex+race+age 三 embedding 拼接」缩为「单一 age embedding」; 输出维度 out_dim 与
    PatientEmbedding 一致, 可直接替换 ResNet18HyperFusion 的 self.patient_embed。

    为什么只用 age: HAM 条件互信息诊断 (作业 68297) 显示 sex 轴 I(Y;sex|X)≈0、age 轴读出
    可复现 I(Y;age|X)>0, 故先只接 age。⚠️ age_group==-1 (0-20 排除组) 非法索引, 须先过滤。

    Args:
        num_age      : age_group 有效类别数 (HAM: 4)。
        cat_embed_dim: age embedding 维度。
        out_dim      : patient profile 维度 (须与模型 patient_embed_dim 一致)。
    """

    def __init__(
        self,
        num_age: int = 4,
        cat_embed_dim: int = 16,
        out_dim: int = 128,
    ) -> None:
        super().__init__()
        self.age_embed = nn.Embedding(num_age, cat_embed_dim)
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
    拼接, 经与 PatientEmbedding / AgeEmbedding 同款的两层 fuse MLP 投到 patient profile。
    = PatientEmbedding 去掉 race 通路 (输入 3*dim → 2*dim); 输出维度 out_dim 不变, 可直接
    替换 ResNet18HyperFusion 的 self.patient_embed。

    为什么需要双属性版: HAM HN 三臂原本只条件化 age, 而 worst-group 评估口径是
    Sex / Age / Sex×Age——条件化口径与评估口径不对称。本类提供对称性对照臂的条件通路,
    与 age-only 臂的唯一差异是多一条 sex embedding。

    ⚠️ age_group==-1 (0-20 排除组) 非法索引, 须在数据侧先过滤 (与 age-only 臂同一过滤)。

    Args:
        num_sex      : sex 类别数 (HAM: 2)。
        num_age      : age_group 有效类别数 (HAM: 4)。
        cat_embed_dim: 每个属性 embedding 维度。
        out_dim      : patient profile 维度 (须与模型 patient_embed_dim 一致)。
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
        self.fuse = nn.Sequential(
            nn.Linear(2 * cat_embed_dim, out_dim),
            nn.ReLU(inplace=True),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, sex: torch.Tensor, age_group: torch.Tensor) -> torch.Tensor:
        """sex: [B] (long, 0/1)、age_group: [B] (long, 0-3) -> patient profile [B, out_dim]。"""
        cond = torch.cat([self.sex_embed(sex), self.age_embed(age_group)], dim=-1)
        return self.fuse(cond)                              # [B, out_dim]


class SkinEmbedding(nn.Module):
    """
    Fitzpatrick17k 的单属性条件通路: 只用肤色 skin (Fitzpatrick I–VI → 0–5) 一个敏感属性。
    结构与本模块 PatientEmbedding / AgeEmbedding 对齐 (单 embedding → 两层 fuse MLP), 仅输入由
    「sex+race+age 三 embedding 拼接」缩为「单一 skin embedding」; 输出维度 out_dim 与
    PatientEmbedding 一致, 可直接替换 ResNet18HyperFusion 的 self.patient_embed。

    注: Fitzpatrick 条件互信息 I(Y;skin|X)≈0 (作业 67913), 据核心论点 HN 在此注定无收益,
    本变体产出「信号缺失处的 null result」对照。

    Args:
        num_skin     : 肤色类别数 (Fitzpatrick I–VI = 6)。
        cat_embed_dim: skin embedding 维度。
        out_dim      : patient profile 维度 (须与模型 patient_embed_dim 一致)。
    """

    def __init__(
        self,
        num_skin: int = 6,
        cat_embed_dim: int = 16,
        out_dim: int = 128,
    ) -> None:
        super().__init__()
        self.skin_embed = nn.Embedding(num_skin, cat_embed_dim)
        self.fuse = nn.Sequential(
            nn.Linear(cat_embed_dim, out_dim),
            nn.ReLU(inplace=True),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, skin: torch.Tensor) -> torch.Tensor:
        """skin: [B] (long, 0–5) -> patient profile [B, out_dim]。"""
        return self.fuse(self.skin_embed(skin))             # [B, out_dim]


# ============================================================
# HyperFusion 超网络: patient profile -> downsample 卷积权重的扰动 Δθ
# ============================================================
class HyperFusionNet(nn.Module):
    """
    由 patient profile γ 生成 layer4[0].downsample 1x1 卷积权重的逐样本残差扰动 Δθ_p。
    downsample 卷积在 torchvision ResNet-18 中 bias=False, 故只生成 weight 扰动。

    MIP 设计:
      · E_L2 投影 (组件 1): γ -> sqrt(d) * γ / ||γ||, 逐样本归一到固定 norm 球面。
      · 输出层小尺度初始化 (组件 2): fc_out 权重按 Kaiming 后再统一缩 mip_scale 倍,
        使初始 Δθ ≈ 0; 隐藏层用普通 Kaiming, bias 全部置零。
      · additive 重写在外层 block 完成 (θ_ds = θ_0 + Δθ), 此处只产出 Δθ。

    Args:
        in_channels  : downsample 卷积输入通道数 (layer4: 256)。
        out_channels : downsample 卷积输出通道数 (layer4: 512)。
        embed_dim    : patient profile 维度。
        hidden_dim   : MLP 隐藏层维度。
        use_l2_norm  : 是否做 E_L2 投影 (MIP 组件 1)。
        mip_scale    : 输出层小尺度因子 (MIP 组件 2), 越小初始扰动越接近 0。
    """

    def __init__(
        self,
        in_channels: int = 256,
        out_channels: int = 512,
        embed_dim: int = 128,
        hidden_dim: int = 64,
        use_l2_norm: bool = True,
        mip_scale: float = 0.01,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.embed_dim = embed_dim
        self.use_l2_norm = use_l2_norm
        self.mip_scale = mip_scale

        # 1x1 卷积权重 numel = out_c * in_c (kernel 1x1), 无 bias
        weight_numel = out_channels * in_channels

        self.fc_hidden = nn.Linear(embed_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, weight_numel)

        self._init_mip()

    def _init_mip(self) -> None:
        """
        MIP 风格初始化:
          · 隐藏层普通 Kaiming, bias 置零。
          · 输出层 fc_out 先 Kaiming, 再统一缩 mip_scale 倍 (核心: 让初始 Δθ ≈ 0),
            bias 置零。
        """
        nn.init.kaiming_uniform_(self.fc_hidden.weight, a=5 ** 0.5)
        nn.init.zeros_(self.fc_hidden.bias)

        nn.init.kaiming_uniform_(self.fc_out.weight, a=5 ** 0.5)
        with torch.no_grad():
            self.fc_out.weight.mul_(self.mip_scale)   # MIP: 初始扰动 ≈ 0
        nn.init.zeros_(self.fc_out.bias)

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        """
        Args:
            embedding: [B, embed_dim] patient profile γ。

        Returns:
            delta_w: [B, out_channels, in_channels] 逐样本 downsample 卷积权重扰动。
        """
        cond = embedding
        # E_L2 投影: 切断 magnitude proportionality, 稳定输出尺度
        if self.use_l2_norm:
            norm = cond.norm(p=2, dim=-1, keepdim=True).clamp(min=1e-6)
            cond = cond * (self.embed_dim ** 0.5) / norm

        h = F.relu(self.fc_hidden(cond), inplace=True)
        out = self.fc_out(h)                              # [B, out_c * in_c]
        delta_w = out.view(-1, self.out_channels, self.in_channels)
        return delta_w


# ============================================================
# ResNet-18 (ImageNet 预训练) + HyperFusion
# ============================================================
class ResNet18HyperFusion(nn.Module):
    """
    与 MIMIC baseline 对齐的 ImageNet 预训练 ResNet-18, 仅在 layer4[0].downsample 的
    1x1 卷积处接 HyperFusion 超网络: 该 shortcut 卷积权重由 (sex / race / age_group)
    逐样本调整 (θ_ds = θ_0 + Δθ_p, MIP additive), 其余所有层与 baseline 共享同一份权重。

    self.backbone 直接持有一个完整的 torchvision.models.resnet18, 因此 ImageNet 预训练
    权重可无损载入, 图像通路与 baseline 严格可比; forward 中手动遍历各 stage, 只在
    layer4[0] 把 downsample 替换为超网络驱动的逐样本卷积。

    Args:
        num_classes      : 分类头输出维度。
            - 1 -> 配合 nn.BCEWithLogitsLoss (默认; No Finding U-Zeros 二分类)
            - k -> 配合 nn.CrossEntropyLoss
        num_sex, num_race, num_age: 三个离散敏感属性的类别数 (MIMIC 均为 2)。
        patient_embed_dim: patient profile vector 维度。
        hyper_hidden     : 超网络 MLP 隐藏层维度。
        mip_scale        : MIP 输出层小尺度因子 (越小初始越接近纯 baseline)。
        use_l2_norm      : 是否对 patient profile 做 E_L2 投影。
        pretrained       : 是否载入 torchvision ImageNet 预训练 backbone 权重
                           (与 baseline 一致, 默认 True)。
    """

    # 注入位置: layer4[0].downsample, 输入 256 通道 -> 输出 512 通道
    HYPER_IN_CHANNELS = 256
    HYPER_OUT_CHANNELS = 512

    def __init__(
        self,
        num_classes: int = 1,
        num_sex: int = 2,
        num_race: int = 2,
        num_age: int = 2,
        patient_embed_dim: int = 128,
        hyper_hidden: int = 64,
        mip_scale: float = 0.01,
        use_l2_norm: bool = True,
        pretrained: bool = True,
    ) -> None:
        super().__init__()

        # ============================================================
        # 1. Backbone (完整 torchvision ResNet-18, 与 baseline 完全一致)
        # ============================================================
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = resnet18(weights=weights)
        # 替换分类头: 1000 类 -> num_classes (与 baseline ResNet18Pretrained 一致)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)

        # layer4[0].downsample = Sequential(Conv2d(256,512,1,stride=2,bias=False),
        #                                    BatchNorm2d(512))
        # 取出 downsample 的步幅, 供 forward 里逐样本卷积复用 (theta_0 即该 conv.weight)
        hyper_block = self.backbone.layer4[0]
        assert hyper_block.downsample is not None, \
            "layer4[0] 必须含 downsample (256->512, stride=2) 才能接 HyperFusion"
        ds_conv = hyper_block.downsample[0]
        ds_stride = ds_conv.stride
        self.hyper_ds_stride = ds_stride[0] if isinstance(ds_stride, tuple) else ds_stride

        # ============================================================
        # 2. HyperFusion 条件通路
        # ============================================================
        self.patient_embed = PatientEmbedding(
            num_sex=num_sex, num_race=num_race, num_age=num_age,
            cat_embed_dim=16, out_dim=patient_embed_dim,
        )
        self.hyper_net = HyperFusionNet(
            in_channels=self.HYPER_IN_CHANNELS,
            out_channels=self.HYPER_OUT_CHANNELS,
            embed_dim=patient_embed_dim,
            hidden_dim=hyper_hidden,
            use_l2_norm=use_l2_norm,
            mip_scale=mip_scale,
        )

    # ------------------------------------------------------------
    # layer4[0]: 主分支正常, downsample 由超网络逐样本生成
    # ------------------------------------------------------------
    def _hyper_layer4_block0(
        self, x: torch.Tensor, embedding: torch.Tensor
    ) -> torch.Tensor:
        """
        复刻 torchvision BasicBlock.forward, 但 shortcut 的 1x1 卷积权重换成
        θ_0 + Δθ_p (逐样本)。用 grouped conv 把 batch 维折进通道维, 一次算完整个 batch。

        Args:
            x        : [B, 256, H, W] layer3 输出特征。
            embedding: [B, embed_dim] patient profile γ。

        Returns:
            [B, 512, H/2, W/2] layer4[0] 输出。
        """
        block = self.backbone.layer4[0]
        batch_size, c_in, height, width = x.shape

        # ---- shortcut 分支: θ_ds = θ_0 + Δθ_p (MIP additive) ----
        ds_conv: nn.Conv2d = block.downsample[0]
        ds_bn: nn.BatchNorm2d = block.downsample[1]
        theta0_w = ds_conv.weight                          # [512, 256, 1, 1] 预训练基底
        delta_w = self.hyper_net(embedding)                # [B, 512, 256]
        delta_w = delta_w.unsqueeze(-1).unsqueeze(-1)      # [B, 512, 256, 1, 1]
        full_w = theta0_w.unsqueeze(0) + delta_w           # [B, 512, 256, 1, 1]

        # grouped conv: groups=B, 把 batch 折进通道维, 每个样本用各自的 θ_ds 卷积
        x_grouped = x.reshape(1, batch_size * c_in, height, width)
        w_grouped = full_w.reshape(batch_size * self.HYPER_OUT_CHANNELS, c_in, 1, 1)
        identity = F.conv2d(
            x_grouped, w_grouped, bias=None,
            stride=self.hyper_ds_stride, padding=0, groups=batch_size,
        )
        out_h, out_w = identity.shape[-2:]
        identity = identity.reshape(batch_size, self.HYPER_OUT_CHANNELS, out_h, out_w)
        identity = ds_bn(identity)                         # downsample BN 保持共享

        # ---- 主分支 (与 baseline 完全一致, 共享权重) ----
        out = block.conv1(x)
        out = block.bn1(out)
        out = block.relu(out)
        out = block.conv2(out)
        out = block.bn2(out)

        out = out + identity
        out = block.relu(out)
        return out

    # ------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------
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
        bb = self.backbone

        # patient profile γ (一次性算出条件向量)
        embedding = self.patient_embed(sex, race, age_group)   # [B, embed_dim]

        # Stem + layer1~3 (与 baseline 完全一致)
        x = bb.conv1(image)
        x = bb.bn1(x)
        x = bb.relu(x)
        x = bb.maxpool(x)

        x = bb.layer1(x)
        x = bb.layer2(x)
        x = bb.layer3(x)

        # layer4[0]: 唯一的 hyperlayer (downsample 由超网络驱动)
        x = self._hyper_layer4_block0(x, embedding)
        # layer4[1]: 普通 block (共享 baseline 权重)
        x = bb.layer4[1](x)

        # Pool + 分类头 (与 baseline 完全一致)
        x = bb.avgpool(x)
        x = torch.flatten(x, 1)
        logits = bb.fc(x)
        return logits

    # ------------------------------------------------------------
    # 工具: 分别统计 backbone / 超网络的参数量
    # ------------------------------------------------------------
    def param_breakdown(self) -> tuple[int, int, int]:
        """返回 (总参数量, 超网络参数量, backbone 参数量)。"""
        hyper_modules = list(self.patient_embed.parameters())
        hyper_modules += list(self.hyper_net.parameters())
        n_hyper = sum(p.numel() for p in hyper_modules)
        n_total = sum(p.numel() for p in self.parameters())
        return n_total, n_hyper, n_total - n_hyper


# ============================================================
# ResNet-18 + HyperFusion for HAM10000 (单一 age 属性)
# ============================================================
class ResNet18HyperFusionAge(ResNet18HyperFusion):
    """
    HAM10000 版 HyperFusion: 复用 ResNet18HyperFusion 的全部机制 (仅 layer4[0].downsample
    1x1 conv 由超网络 MIP additive 逐样本生成 θ_ds=θ_0+Δθ、E_L2 投影、grouped conv、其余层
    与 baseline 共享), 仅把三属性 PatientEmbedding 替换为单一 age 的 AgeEmbedding, 并把
    forward 签名改为 (image, age_group)——**只条件化 age** (sex 轴诊断读弱/null, 先不接)。

    ⚠️ 仅接受 age_group∈{0..3}; age_group==-1 (0-20 排除组) 须在数据侧先过滤 (训练脚本负责)。

    Args:
        num_classes      : 分类头输出维度 (malignant 二分类 → 1)。
        num_age          : age_group 有效类别数 (HAM: 4)。
        patient_embed_dim: patient profile 维度 (须与 AgeEmbedding.out_dim 一致)。
        hyper_hidden     : 超网络 MLP 隐藏层维度。
        mip_scale        : MIP 输出层小尺度因子 (越小初始越接近纯 baseline)。
        use_l2_norm      : 是否对 patient profile 做 E_L2 投影。
        pretrained       : 是否载入 ImageNet 预训练 backbone (与 baseline 一致, 默认 True)。
    """

    def __init__(
        self,
        num_classes: int = 1,
        num_age: int = 4,
        patient_embed_dim: int = 128,
        hyper_hidden: int = 64,
        mip_scale: float = 0.01,
        use_l2_norm: bool = True,
        pretrained: bool = True,
    ) -> None:
        # 先按父类构建完整 HyperFusion (含预训练载入 / MIP 初始化)，patient_embed 暂为
        # 默认三属性 PatientEmbedding；随后整体替换为单属性 AgeEmbedding。
        super().__init__(
            num_classes=num_classes,
            patient_embed_dim=patient_embed_dim,
            hyper_hidden=hyper_hidden,
            mip_scale=mip_scale,
            use_l2_norm=use_l2_norm,
            pretrained=pretrained,
        )
        # 替换条件通路: 三属性 → 单一 4 类 age (输出维度不变, 不影响 hyper_net / 注入点)
        self.patient_embed = AgeEmbedding(
            num_age=num_age, cat_embed_dim=16, out_dim=patient_embed_dim,
        )

    def forward(self, image: torch.Tensor, age_group: torch.Tensor) -> torch.Tensor:
        """
        与父类 forward 同构，仅条件向量由单一 age 生成 (父类 forward 签名是三属性，此处覆盖)。

        Args:
            image    : [B, 3, 224, 224] 皮肤镜 RGB 图像。
            age_group: [B] long，0–3 (4 有效年龄组；-1 须已过滤)。

        Returns:
            logits: [B, num_classes]。
        """
        bb = self.backbone
        embedding = self.patient_embed(age_group)              # [B, embed_dim]

        x = bb.conv1(image)
        x = bb.bn1(x)
        x = bb.relu(x)
        x = bb.maxpool(x)

        x = bb.layer1(x)
        x = bb.layer2(x)
        x = bb.layer3(x)

        x = self._hyper_layer4_block0(x, embedding)            # 唯一注入点
        x = bb.layer4[1](x)

        x = bb.avgpool(x)
        x = torch.flatten(x, 1)
        logits = bb.fc(x)
        return logits


class ResNet18HyperFusionSexAge(ResNet18HyperFusion):
    """
    HAM10000 版 HyperFusion 的**双属性 (sex + age_group)** 变体: 与 ResNet18HyperFusionAge 的
    唯一差异是条件通路由 AgeEmbedding 换成 SexAgeEmbedding, HyperFusion 机制 (仅
    layer4[0].downsample 1x1 conv 由超网络 MIP additive 逐样本生成 θ_ds=θ_0+Δθ、E_L2 投影、
    grouped conv、其余层与 baseline 共享) 完全复用父类。

    动机: age-only 臂的条件化口径与 worst-group 评估口径 (Sex / Age / Sex×Age) 不对称,
    本臂把条件输入补齐为评估分组变量全集 (method 名 hyperfusion_sexage)。

    ⚠️ 仅接受 age_group∈{0..3}; -1 须在数据侧先过滤 (训练脚本负责)。

    Args:
        num_classes      : 分类头输出维度 (malignant 二分类 → 1)。
        num_sex          : sex 类别数 (HAM: 2)。
        num_age          : age_group 有效类别数 (HAM: 4)。
        patient_embed_dim: patient profile 维度 (须与 SexAgeEmbedding.out_dim 一致)。
        hyper_hidden     : 超网络 MLP 隐藏层维度。
        mip_scale        : MIP 输出层小尺度因子。
        use_l2_norm      : 是否对 patient profile 做 E_L2 投影。
        pretrained       : 是否载入 ImageNet 预训练 backbone (默认 True)。
    """

    def __init__(
        self,
        num_classes: int = 1,
        num_sex: int = 2,
        num_age: int = 4,
        patient_embed_dim: int = 128,
        hyper_hidden: int = 64,
        mip_scale: float = 0.01,
        use_l2_norm: bool = True,
        pretrained: bool = True,
    ) -> None:
        super().__init__(
            num_classes=num_classes,
            patient_embed_dim=patient_embed_dim,
            hyper_hidden=hyper_hidden,
            mip_scale=mip_scale,
            use_l2_norm=use_l2_norm,
            pretrained=pretrained,
        )
        # 替换条件通路: 三属性 → 双属性 (sex/age)，输出维度不变，注入点与 hyper_net 不受影响
        self.patient_embed = SexAgeEmbedding(
            num_sex=num_sex, num_age=num_age, cat_embed_dim=16, out_dim=patient_embed_dim,
        )

    def forward(
        self, image: torch.Tensor, sex: torch.Tensor, age_group: torch.Tensor
    ) -> torch.Tensor:
        """
        与 ResNet18HyperFusionAge.forward 同构，仅条件向量由 (sex, age) 双属性生成。

        Args:
            image    : [B, 3, 224, 224] 皮肤镜 RGB 图像。
            sex      : [B] long，0=Male / 1=Female。
            age_group: [B] long，0-3 (4 有效年龄组；-1 须已过滤)。

        Returns:
            logits: [B, num_classes]。
        """
        bb = self.backbone
        embedding = self.patient_embed(sex, age_group)          # [B, embed_dim]

        x = bb.conv1(image)
        x = bb.bn1(x)
        x = bb.relu(x)
        x = bb.maxpool(x)

        x = bb.layer1(x)
        x = bb.layer2(x)
        x = bb.layer3(x)

        x = self._hyper_layer4_block0(x, embedding)             # 唯一注入点
        x = bb.layer4[1](x)

        x = bb.avgpool(x)
        x = torch.flatten(x, 1)
        logits = bb.fc(x)
        return logits


class ResNet18HyperFusionSkin(ResNet18HyperFusion):
    """
    Fitzpatrick17k 版 HyperFusion: 复用 ResNet18HyperFusion 的全部机制 (仅 layer4[0].downsample
    1x1 conv 由超网络 MIP additive 逐样本生成 θ_ds=θ_0+Δθ、E_L2 投影、grouped conv、其余层
    与 baseline 共享), 仅把三属性 PatientEmbedding 替换为单一 skin 的 SkinEmbedding, 并把
    forward 签名改为 (image, skin)。与 ResNet18HyperFusionAge 同构，仅条件属性由 age 换成 skin。

    Args:
        num_classes      : 分类头输出维度 (malignant 二分类 → 1)。
        num_skin         : skin 有效类别数 (Fitzpatrick I–VI = 6)。
        patient_embed_dim: patient profile 维度 (须与 SkinEmbedding.out_dim 一致)。
        hyper_hidden     : 超网络 MLP 隐藏层维度。
        mip_scale        : MIP 输出层小尺度因子 (越小初始越接近纯 baseline)。
        use_l2_norm      : 是否对 patient profile 做 E_L2 投影。
        pretrained       : 是否载入 ImageNet 预训练 backbone (与 baseline 一致, 默认 True)。
    """

    def __init__(
        self,
        num_classes: int = 1,
        num_skin: int = 6,
        patient_embed_dim: int = 128,
        hyper_hidden: int = 64,
        mip_scale: float = 0.01,
        use_l2_norm: bool = True,
        pretrained: bool = True,
    ) -> None:
        # 先按父类构建完整 HyperFusion (含预训练载入 / MIP 初始化)，patient_embed 暂为
        # 默认三属性 PatientEmbedding；随后整体替换为单属性 SkinEmbedding。
        super().__init__(
            num_classes=num_classes,
            patient_embed_dim=patient_embed_dim,
            hyper_hidden=hyper_hidden,
            mip_scale=mip_scale,
            use_l2_norm=use_l2_norm,
            pretrained=pretrained,
        )
        # 替换条件通路: 三属性 → 单一 6 类肤色 (输出维度不变, 不影响 hyper_net / 注入点)
        self.patient_embed = SkinEmbedding(
            num_skin=num_skin, cat_embed_dim=16, out_dim=patient_embed_dim,
        )

    def forward(self, image: torch.Tensor, skin: torch.Tensor) -> torch.Tensor:
        """
        与父类 forward 同构，仅条件向量由单一 skin 生成 (父类 forward 签名是三属性，此处覆盖)。

        Args:
            image: [B, 3, 224, 224] 皮肤镜 RGB 图像。
            skin : [B] long，0–5 (6 肤色型)。

        Returns:
            logits: [B, num_classes]。
        """
        bb = self.backbone
        embedding = self.patient_embed(skin)                   # [B, embed_dim]

        x = bb.conv1(image)
        x = bb.bn1(x)
        x = bb.relu(x)
        x = bb.maxpool(x)

        x = bb.layer1(x)
        x = bb.layer2(x)
        x = bb.layer3(x)

        x = self._hyper_layer4_block0(x, embedding)            # 唯一注入点
        x = bb.layer4[1](x)

        x = bb.avgpool(x)
        x = torch.flatten(x, 1)
        logits = bb.fc(x)
        return logits


# ============================================================
# 结构自检
# ============================================================
if __name__ == "__main__":
    torch.manual_seed(0)
    # 自检不下载预训练权重 (pretrained=False), 仅验证结构与前向/反向
    model = ResNet18HyperFusion(num_classes=1, mip_scale=0.01, pretrained=False)
    model.eval()                                          # eval 避免 BN 在小 batch 上不稳

    batch_size = 4
    image = torch.randn(batch_size, 3, 224, 224)
    sex = torch.tensor([0, 1, 0, 1], dtype=torch.long)
    race = torch.tensor([0, 1, 0, 1], dtype=torch.long)
    age = torch.tensor([0, 0, 1, 1], dtype=torch.long)

    logits = model(image, sex, race, age)
    print(f"Input  shape : {tuple(image.shape)}")         # (4, 3, 224, 224)
    print(f"Logits shape : {tuple(logits.shape)}")        # (4, 1)
    print(f"Logit  stats : mean={logits.mean().item():+.3f}  std={logits.std().item():.3f}")

    n_total, n_hyper, n_backbone = model.param_breakdown()
    print(f"Total    params : {n_total:>12,}")
    print(f"  Backbone      : {n_backbone:>12,}")
    print(f"  HyperNet (phi): {n_hyper:>12,}")

    # 反向传播检查: 超网络与 theta_0 (downsample 基底) 都应拿到梯度
    loss = logits.sum()
    loss.backward()
    hyper_grad = sum(
        p.grad.abs().sum().item()
        for p in (*model.patient_embed.parameters(), *model.hyper_net.parameters())
        if p.grad is not None
    )
    theta0_grad = model.backbone.layer4[0].downsample[0].weight.grad.abs().sum().item()
    print(f"HyperNet gradient sum : {hyper_grad:.4f}")
    print(f"theta_0  gradient sum : {theta0_grad:.4f}")
    print("Backward pass OK." if hyper_grad > 0 and theta0_grad > 0 else
          "HyperNet/theta_0 没拿到梯度, 检查 forward 链路!")

    # ---- HAM10000 单属性 (age) 变体自检 ----
    age_model = ResNet18HyperFusionAge(num_classes=1, num_age=4, mip_scale=0.01, pretrained=False)
    age_model.eval()
    age_only = torch.tensor([0, 1, 2, 3], dtype=torch.long)   # 4 个有效 age 组 (无 -1)
    age_logits = age_model(image, age_only)
    print(f"\n[HAM age 1-attr] Logits : {tuple(age_logits.shape)}")  # (4, 1)
    n_total_a, n_hyper_a, n_backbone_a = age_model.param_breakdown()
    print(f"Total params : {n_total_a:>12,} (backbone {n_backbone_a:,} + hyper {n_hyper_a:,})")
    age_logits.sum().backward()
    a_hyper_grad = sum(
        p.grad.abs().sum().item()
        for p in (*age_model.patient_embed.parameters(), *age_model.hyper_net.parameters())
        if p.grad is not None
    )
    a_theta0_grad = age_model.backbone.layer4[0].downsample[0].weight.grad.abs().sum().item()
    print("Age variant backward pass OK." if a_hyper_grad > 0 and a_theta0_grad > 0 else
          "Age 变体 HyperNet/theta_0 没拿到梯度!")
