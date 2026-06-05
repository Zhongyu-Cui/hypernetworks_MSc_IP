"""
ResNet-18 with HyperAdapt
=========================

参考论文:
    "Patient-Conditioned Adaptive Offsets for Reliable Diagnosis across Subgroups"
    (Xu et al., 2026, arXiv:2601.13094)

核心思想:
---------
对于每个病人 p, 利用其属性向量 c_p (这里是 sex 和 race) 通过 hyper-adapter
h(c_p; phi) 生成 backbone 参数的 *残差偏移* Δθ_p, 得到个性化模型:

        f_p(x) = f(x;  θ + Δθ_p)

为了控制参数量并防止过拟合, Δθ 用低秩分解 + shared generation block 表示:

  · 卷积层 (channel-wise multiplicative modulation):
        M_p   = A_p · B_p          # [C_out, C_in]
        Θ'_p  = Θ * (1 + M_p)      # 对每个 (out, in) 通道对乘性缩放

  · 线性层 (additive low-rank update):
        ΔW_p  = A_p · B_p          # [d_out, d_in]
        W'_p  = W + ΔW_p

  · Shared A-generator: 同一个 output dim (eg. 64/128/256/512) 的所有层
    共用一个 A 生成器, 每层只保留独立的 B 生成器.

注意: 论文里 backbone 通常是冻结的预训练权重, 这里在 UTKFace 上从零训练,
      因此 backbone 权重也参与训练. Adapter 在初始化时让 Δθ ≈ 0,
      模型起始等价于普通 ResNet-18.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Functional helpers: per-sample adapted conv / linear
# ============================================================
def adapted_conv2d(x, weight, M, stride, padding):
    """
    对每个样本使用其个性化 kernel 做卷积:

        kernel_p = weight * (1 + M_p)        # 各样本不同
        y_p      = conv2d(x_p, kernel_p)

    为了在一次调用里完成 batch, 用 grouped-conv 把 batch 维度折叠成 group.

    Args:
        x      : [B, C_in,  H, W]
        weight : [C_out, C_in, K, K]   (backbone 权重, 同 batch 共享)
        M      : [B, C_out, C_in]      (per-sample modulation)
        stride : int
        padding: int
    Returns:
        y      : [B, C_out, H', W']
    """
    B, C_in, H, W = x.shape
    C_out, _, Kh, Kw = weight.shape

    # 1. 构造 per-sample kernel: [B, C_out, C_in, K, K]
    mod = (1.0 + M).unsqueeze(-1).unsqueeze(-1)         # [B, C_out, C_in, 1, 1]
    w_per_sample = weight.unsqueeze(0) * mod            # 广播得到 [B, C_out, C_in, K, K]

    # 2. 用 grouped conv 一次性算完: groups=B
    w_grouped = w_per_sample.reshape(B * C_out, C_in, Kh, Kw)
    x_grouped = x.reshape(1, B * C_in, H, W)            # 把 batch 折进 channel
    y = F.conv2d(x_grouped, w_grouped, bias=None,
                 stride=stride, padding=padding, groups=B)
    H_out, W_out = y.shape[-2:]
    return y.reshape(B, C_out, H_out, W_out)


def adapted_linear(x, weight, dW, bias=None):
    """
    对每个样本使用其个性化权重做 linear:
        y_p = (W + ΔW_p) · x_p + b

    Args:
        x      : [B, d_in]
        weight : [d_out, d_in]
        dW     : [B, d_out, d_in]
        bias   : [d_out] or None
    """
    w_per_sample = weight.unsqueeze(0) + dW             # [B, d_out, d_in]
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
    sex (2 类) 和 race (5 类) 都是离散属性, 各自用 nn.Embedding 编码,
    拼接后通过一个 MLP 融合, 得到 patient profile vector.

    这对应论文 Sec. II-C "Embedding Subgroup-Relevant Attributes" 中
    categorical pathway 的实现 (本任务不涉及连续属性).
    """
    def __init__(self, num_sex=2, num_race=5,
                 cat_embed_dim=16, out_dim=128):
        super().__init__()
        self.sex_embed = nn.Embedding(num_sex, cat_embed_dim)
        self.race_embed = nn.Embedding(num_race, cat_embed_dim)
        # Embedding fusion MLP
        self.fuse = nn.Sequential(
            nn.Linear(2 * cat_embed_dim, out_dim),
            nn.ReLU(inplace=True),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, sex, race):
        # sex, race: [B] (long)
        s = self.sex_embed(sex)
        r = self.race_embed(race)
        e = torch.cat([s, r], dim=-1)
        return self.fuse(e)                # [B, out_dim]


# ============================================================
# HyperAdapt BasicBlock
# ============================================================
class HyperBasicBlock(nn.Module):
    """
    与原版 ResNet-18 的 BasicBlock 结构相同, 但每个卷积层都接了一个
    channel-wise modulation:
                 Θ_ℓ -> Θ_ℓ * (1 + A · B_ℓ)

    其中 A 是 *跨层共享* 的 (从外部传入), B_ℓ 是层独有的.
    """
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None,
                 patient_embed_dim=128, rank=4):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.rank = rank

        # ----- backbone 卷积层 (与普通 ResNet 相同) -----
        self.conv1 = nn.Conv2d(in_channels, out_channels,
                               kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels,
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

        # ----- 层独有的 B 生成器 -----
        # conv1 的输入通道是 in_channels
        self.B_gen_conv1 = nn.Linear(patient_embed_dim, rank * in_channels)
        # conv2 的输入通道是 out_channels (block 内中间张量已经升维)
        self.B_gen_conv2 = nn.Linear(patient_embed_dim, rank * out_channels)
        # downsample 1x1 conv 的输入通道是 in_channels
        if downsample is not None:
            self.B_gen_down = nn.Linear(patient_embed_dim, rank * in_channels)
        else:
            self.B_gen_down = None

        # ----- 初始化 B 生成器 -----
        # 注意 LoRA 经典坑: 如果 A、B 都初始化为 0, 那么 M = A·B = 0,
        # 同时 ∂M/∂A = B^T = 0, ∂M/∂B = A^T = 0, adapter 的梯度永远是 0,
        # 整个 hyper-adapter 模块 *死锁* 学不动.
        # 标准做法: 让其中一个用普通初始化, 另一个置零, 这样:
        #   · 初始仍有 M = A·B = 0 (保证 "起始等价于纯 ResNet-18")
        #   · grad 不会全为 0,  B (这里是 A_shared) 能学起来,
        #     等 A 非零后 B 也跟上.
        # 这里选择 A_shared 全置零 (在 ResNet18HyperHead._init_weights 中做),
        # 而 B_gen_* 使用 Kaiming. 这样 ∂M/∂A_shared = B^T 非零,
        # adapter 能正常获得梯度.
        self._kaiming_init_B()

    def _kaiming_init_B(self):
        for m in [self.B_gen_conv1, self.B_gen_conv2, self.B_gen_down]:
            if m is None:
                continue
            nn.init.kaiming_normal_(m.weight, a=0, mode="fan_in", nonlinearity="linear")
            nn.init.zeros_(m.bias)

    def _make_M(self, A_shared, B_gen, embedding, c_in):
        """
        A_shared : [B, C_out, rank]      (跨层共享, 从外部传入)
        B_gen    : nn.Linear(embed_dim -> rank*c_in)
        embedding: [B, embed_dim]
        Returns  : M = A · B,  shape [B, C_out, c_in]
        """
        B = embedding.shape[0]
        Bmat = B_gen(embedding).view(B, self.rank, c_in)
        return torch.bmm(A_shared, Bmat)

    def forward(self, x, embedding, A_shared):
        identity = x

        # ---- conv1 with adapter ----
        M1 = self._make_M(A_shared, self.B_gen_conv1, embedding, self.in_channels)
        out = adapted_conv2d(x, self.conv1.weight, M1,
                             stride=self.stride, padding=1)
        out = self.bn1(out)
        out = F.relu(out, inplace=True)

        # ---- conv2 with adapter ----
        M2 = self._make_M(A_shared, self.B_gen_conv2, embedding, self.out_channels)
        out = adapted_conv2d(out, self.conv2.weight, M2,
                             stride=1, padding=1)
        out = self.bn2(out)

        # ---- downsample (also adapted) ----
        if self.downsample is not None:
            ds_conv: nn.Conv2d = self.downsample[0]
            ds_bn:   nn.BatchNorm2d = self.downsample[1]
            Md = self._make_M(A_shared, self.B_gen_down, embedding, self.in_channels)
            ds_stride = ds_conv.stride[0] if isinstance(ds_conv.stride, tuple) else ds_conv.stride
            identity = adapted_conv2d(x, ds_conv.weight, Md,
                                      stride=ds_stride, padding=0)
            identity = ds_bn(identity)

        out = out + identity
        out = F.relu(out, inplace=True)
        return out


# ============================================================
# ResNet-18 + HyperAdapt
# ============================================================
class ResNet18HyperHead(nn.Module):
    """
    ResNet-18 backbone + HyperAdapt 模块.

    Args:
        out_dim          : 输出 logit 维度 (BCE 二分类用 1)
        num_sex, num_race: 离散属性的类别数
        patient_embed_dim: patient profile vector 维度
        rank             : 低秩分解的秩 k
    """

    # ResNet-18 在每个 stage 的输出通道数
    STAGE_CHANNELS = (64, 128, 256, 512)

    def __init__(self, out_dim=1, num_sex=2, num_race=5,
                 patient_embed_dim=128, rank=4):
        super().__init__()
        self.rank = rank
        self.out_dim = out_dim
        self.patient_embed_dim = patient_embed_dim
        self.in_channels_running = 64

        # ============================================================
        # 1. Patient-Aware Modulation Net
        # ============================================================
        # (1) 属性 -> patient profile vector
        self.patient_embed = PatientEmbedding(
            num_sex=num_sex, num_race=num_race,
            cat_embed_dim=16, out_dim=patient_embed_dim,
        )

        # (2) Shared A generators —— 按 output channel 大小分组共享
        #     这就是论文 Sec. II-D-3 的 shared generation block
        self.A_gen = nn.ModuleDict({
            f"c{c}": nn.Linear(patient_embed_dim, c * rank)
            for c in self.STAGE_CHANNELS
        })

        # ============================================================
        # 2. Backbone (ResNet-18)
        # ============================================================
        # Stem (不带 adapter, 论文明确说明 stem 保持原样)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # 4 个 residual stage, 每个 stage 2 个 HyperBasicBlock
        self.layer1 = self._make_layer(out_channels=64,  blocks=2, stride=1)
        self.layer2 = self._make_layer(out_channels=128, blocks=2, stride=2)
        self.layer3 = self._make_layer(out_channels=256, blocks=2, stride=2)
        self.layer4 = self._make_layer(out_channels=512, blocks=2, stride=2)

        # ============================================================
        # 3. Classification head with linear adapter
        # ============================================================
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, out_dim)

        # FC adapter: ΔW = A_fc · B_fc  (additive, low-rank)
        self.fc_A_gen = nn.Linear(patient_embed_dim, out_dim * rank)
        self.fc_B_gen = nn.Linear(patient_embed_dim, rank * 512)

        # ============================================================
        # 4. 参数初始化
        # ============================================================
        self._init_weights()

    # ------------------------------------------------------------
    # 构造一个 stage (n 个 HyperBasicBlock)
    # ------------------------------------------------------------
    def _make_layer(self, out_channels, blocks, stride):
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
        return layers   # nn.ModuleList

    # ------------------------------------------------------------
    # 初始化: backbone 用 Kaiming, 所有 adapter 生成器初始化为 0
    # ------------------------------------------------------------
    def _init_weights(self):
        # backbone (Conv2d / BN)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, a=0, mode="fan_in", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # Shared A generators 全部置零 (这样 M=A·B=0, 初始无 modulation)
        for m in self.A_gen.values():
            nn.init.zeros_(m.weight)
            nn.init.zeros_(m.bias)

        # 注: 每个 block 的 B_gen_* 已经在 HyperBasicBlock 里用 Kaiming 初始化过.
        # 由于 A_shared 全部为 0, 此时 M = A·B = 0, 等价于不带 adapter 的纯 backbone;
        # 但梯度可以正常流动 (∂M/∂A = B^T 非零), adapter 能正常学习.

        # FC adapter: A 置零即可保证初始 ΔW=0
        nn.init.zeros_(self.fc_A_gen.weight); nn.init.zeros_(self.fc_A_gen.bias)
        # B 用正常初始化 (避免 A,B 都死掉, 这样 A 一旦学起来就能反向传梯度)
        nn.init.kaiming_normal_(self.fc_B_gen.weight, nonlinearity="linear")
        nn.init.zeros_(self.fc_B_gen.bias)

    # ------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------
    def forward(self, x, sex, race):
        """
        Args:
            x   : [B, 3, 224, 224]
            sex : [B]  (long, 0=male, 1=female)
            race: [B]  (long, 0..4)
        Returns:
            logits: [B, out_dim]
        """
        # 1. patient embedding (one shot)
        embedding = self.patient_embed(sex, race)         # [B, embed_dim]
        Bsz = embedding.shape[0]

        # 2. 预先生成 4 个 stage 的 shared-A. 每个 stage 内的所有 block / conv
        #    都共用 A_c, 但每个 conv 用自己的 B_ℓ 配出不同的 M = A·B_ℓ.
        A_shared = {
            c: self.A_gen[f"c{c}"](embedding).view(Bsz, c, self.rank)
            for c in self.STAGE_CHANNELS
        }

        # 3. Stem (no adapter)
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out, inplace=True)
        out = self.maxpool(out)

        # 4. Adapted residual stages
        for blk in self.layer1:
            out = blk(out, embedding, A_shared[64])
        for blk in self.layer2:
            out = blk(out, embedding, A_shared[128])
        for blk in self.layer3:
            out = blk(out, embedding, A_shared[256])
        for blk in self.layer4:
            out = blk(out, embedding, A_shared[512])

        # 5. Pool + adapted FC
        out = self.avgpool(out)
        out = torch.flatten(out, 1)                       # [B, 512]

        A_fc = self.fc_A_gen(embedding).view(Bsz, self.out_dim, self.rank)
        B_fc = self.fc_B_gen(embedding).view(Bsz, self.rank, 512)
        dW   = torch.bmm(A_fc, B_fc)                       # [B, out_dim, 512]

        logits = adapted_linear(out, self.fc.weight, dW, self.fc.bias)
        return logits

    # ------------------------------------------------------------
    # 工具: 分别统计 backbone / adapter 的参数量
    # ------------------------------------------------------------
    def param_breakdown(self):
        adapter_modules = list(self.patient_embed.parameters())
        adapter_modules += [p for m in self.A_gen.values() for p in m.parameters()]
        adapter_modules += list(self.fc_A_gen.parameters())
        adapter_modules += list(self.fc_B_gen.parameters())
        for blk in [*self.layer1, *self.layer2, *self.layer3, *self.layer4]:
            adapter_modules += list(blk.B_gen_conv1.parameters())
            adapter_modules += list(blk.B_gen_conv2.parameters())
            if blk.B_gen_down is not None:
                adapter_modules += list(blk.B_gen_down.parameters())
        n_adapter = sum(p.numel() for p in adapter_modules)
        n_total   = sum(p.numel() for p in self.parameters())
        return n_total, n_adapter, n_total - n_adapter


# ============================================================
# 自检
# ============================================================
if __name__ == "__main__":
    torch.manual_seed(0)
    model = ResNet18HyperHead(out_dim=1, num_sex=2, num_race=5,
                              patient_embed_dim=128, rank=4)
    model.eval()                                          # 用 eval 避免 BN 在 1-sample 上炸
    x   = torch.randn(4, 3, 224, 224)
    sex = torch.tensor([0, 1, 0, 1], dtype=torch.long)
    race= torch.tensor([0, 1, 2, 4], dtype=torch.long)

    with torch.no_grad():
        logits = model(x, sex, race)

    print(f"Input  shape : {tuple(x.shape)}")             # (4, 3, 224, 224)
    print(f"Logits shape : {tuple(logits.shape)}")        # (4, 1)

    n_total, n_adapter, n_backbone = model.param_breakdown()
    print(f"Total    params : {n_total:>12,}")
    print(f"  Backbone      : {n_backbone:>12,}")
    print(f"  Adapter (phi) : {n_adapter:>12,}")

    # 一致性测试: 由于初始化让 M=0 / dW=0, 输出应等于把所有 adapter 拆掉的纯 ResNet-18
    print("\n[Sanity] At init,  sigmoid(logits) =", torch.sigmoid(logits).flatten().tolist())