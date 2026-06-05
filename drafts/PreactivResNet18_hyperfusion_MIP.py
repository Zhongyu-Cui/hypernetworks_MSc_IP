"""
PreactivResNet18 + HyperFusion (MIP-init) for UTKFace
======================================================
基于 PreactivResNet18.py 的骨架, 在 layer4 第一个 PreactivResBlock 的
downsample 1x1 conv 上接超网络, 由 (sex, race) 生成卷积权重.

设计对齐:
    - 主网络结构: PreactivResNet18, init_features=16, dropout 同原版
    - Hyperlayer 位置: 仅 layer4[0].downsample (与 HyperFusion 论文一致,
      对应 hyper_embedding_models=[None, None, hyper_embd_tab] 中的第3项)
    - 超网络风格: 借鉴 ResNet18_hyperhead.HyperNet (Embedding + MLP)
    - 端到端训练, 不依赖预训练嵌入 (UTKFace 属性信息量很少)

MIP 初始化 (Ortiz et al., ICLR 2024):
    - E_L2 投影: sex/race embedding 拼接后归一到固定 norm 球面
    - additive 重写: theta = theta_0 + h(gamma)
        - theta_0: 独立可学基底, 用普通 Kaiming-conv 初始化
        - h(gamma): 超网络生成的扰动, 输出层 std 缩小 mip_scale 倍
    - 初始化阶段 h(gamma) ≈ 0, 主网络等价于普通 PreactivResNet18

Forward:
    image -> backbone (前 3 个 stage + layer4 第二个 block 留到最后)
    layer4[0]: 主分支正常前向, downsample 用 (sex, race) 生成的权重
    layer4[1]: 普通 block
    -> head -> logits
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Dim-aware module 工厂 (与 PreactivResNet18.py 对齐)
# ============================================================
def _nd_modules(dim):
    if dim == 2:
        return nn.Conv2d, nn.BatchNorm2d, nn.Dropout2d, nn.MaxPool2d, nn.AdaptiveAvgPool2d
    if dim == 3:
        return nn.Conv3d, nn.BatchNorm3d, nn.Dropout3d, nn.MaxPool3d, nn.AdaptiveAvgPool3d
    raise ValueError(f"dim must be 2 or 3, got {dim}")


def conv_bn_relu(in_channels, out_channels, dim=2,
                 kernel_size=3, stride=1, padding=1,
                 bn_momentum=0.05, conv_bias=True):
    Conv, BN, *_ = _nd_modules(dim)
    return nn.Sequential(
        Conv(in_channels, out_channels, kernel_size,
             stride=stride, padding=padding, bias=conv_bias),
        BN(out_channels, momentum=bn_momentum),
        nn.ReLU(inplace=True),
    )


# ============================================================
# 普通 PreactivResBlock (与原版完全一致)
# ============================================================
class PreactivResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dim=2,
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

        out = out + identity
        return out


# ============================================================
# 超网络 (借鉴 ResNet18_hyperhead.HyperNet 风格)
# ============================================================
class AttrHyperNet(nn.Module):
    """
    输入: sex [B] (long, 0/1), race [B] (long, 0~4)
    输出: delta_w [B, out_c, in_c], delta_b [B, out_c]
          —— 注意是"扰动" (delta), 不是完整权重, 配合 MIP additive 使用

    Args:
        in_channels  : 主网络 downsample 的输入通道数 (=8f)
        out_channels : 主网络 downsample 的输出通道数 (=16f)
        num_sex / num_race      : 类别数
        sex_embed / race_embed  : embedding 维度
        hidden_dim   : MLP 隐藏层
        use_l2_norm  : 是否做 E_L2 投影 (MIP 组件 1)
        mip_scale    : 输出层小尺度因子 (MIP 组件 2)
    """

    def __init__(self, in_channels, out_channels,
                 num_sex=2, num_race=5,
                 sex_embed=4, race_embed=8,
                 hidden_dim=64,
                 use_l2_norm=True, mip_scale=0.01):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.cond_dim = sex_embed + race_embed
        self.use_l2_norm = use_l2_norm
        self.mip_scale = mip_scale

        self.sex_emb = nn.Embedding(num_sex, sex_embed)
        self.race_emb = nn.Embedding(num_race, race_embed)

        # downsample 是 1x1 卷积: 权重 shape = [out_c, in_c, 1, 1]
        # 参数总数 = out_c * in_c * 1 * 1 + out_c (bias)
        weight_numel = out_channels * in_channels
        bias_numel = out_channels

        self.fc_hidden = nn.Linear(self.cond_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, weight_numel + bias_numel)

        self._init_mip()

    def _init_mip(self):
        """
        MIP 风格初始化:
            - 隐藏层用普通 Kaiming
            - 输出层 fc_out 的权重显式缩到 mip_scale 倍, 让初始 delta ≈ 0
            - 输出层 bias 置零
            - Embedding 用普通正态初始化 (PyTorch 默认)
        """
        nn.init.kaiming_uniform_(self.fc_hidden.weight, a=5 ** 0.5)
        nn.init.zeros_(self.fc_hidden.bias)

        # 关键: 输出层小尺度初始化, 这是 MIP 让 delta ≈ 0 的核心机制
        # 先按 Kaiming 初始化, 再统一缩 mip_scale 倍
        nn.init.kaiming_uniform_(self.fc_out.weight, a=5 ** 0.5)
        with torch.no_grad():
            self.fc_out.weight.mul_(self.mip_scale)
        nn.init.zeros_(self.fc_out.bias)

    def forward(self, sex, race):
        s = self.sex_emb(sex)            # [B, sex_embed]
        r = self.race_emb(race)          # [B, race_embed]
        cond = torch.cat([s, r], dim=1)  # [B, cond_dim]

        # E_L2 投影: cond -> sqrt(d) * cond / ||cond||
        # 对每个样本独立归一化, 切断 magnitude proportionality
        if self.use_l2_norm:
            norm = cond.norm(p=2, dim=-1, keepdim=True).clamp(min=1e-6)
            cond = cond * (self.cond_dim ** 0.5) / norm

        h = F.relu(self.fc_hidden(cond), inplace=True)
        out = self.fc_out(h)             # [B, weight_numel + bias_numel]

        wn = self.out_channels * self.in_channels
        delta_w = out[:, :wn].view(-1, self.out_channels, self.in_channels, 1, 1) \
                  if False else out[:, :wn]   # 先保留扁平, forward 里再 reshape
        delta_w = delta_w.view(-1, self.out_channels, self.in_channels)
        delta_b = out[:, wn:]            # [B, out_c]
        return delta_w, delta_b


# ============================================================
# Hyper 版 PreactivResBlock (仅 downsample 是 hyper)
# ============================================================
class HyperPreactivResBlock(PreactivResBlock):
    """
    与 PreactivResBlock 完全一致, 只是 downsample 分支的 1x1 卷积权重
    被改写成 theta_0 + h((sex, race)).

    实现策略 (MIP additive):
        - self.downsample[1] 仍然是普通 Conv2d, 它的 weight 就是 theta_0 (基底)
        - hyper_net 生成 delta_w 和 delta_b, forward 时手动加上, 用 F.conv2d
        - downsample[0] (BN) 保持不变
    """

    def __init__(self, in_channels, out_channels, dim=2,
                 stride=1, dropout=0.0,
                 bn_momentum=0.05, conv_bias=True,
                 hyper_net=None):
        super().__init__(in_channels, out_channels, dim=dim,
                         stride=stride, dropout=dropout,
                         bn_momentum=bn_momentum, conv_bias=conv_bias)
        assert self.downsample is not None, \
            "HyperPreactivResBlock 需要 stride!=1 或 in!=out, 才能有 downsample"
        assert dim == 2, "本文件只实现 2D 版本 (UTKFace)"
        self.dim = dim
        self.stride = stride
        self.hyper_net = hyper_net

        # 缓存 downsample 的两个分量, 方便 forward 里手动调用
        # downsample[0] = BN, downsample[1] = 1x1 Conv (theta_0)
        self.ds_bn = self.downsample[0]
        self.ds_conv = self.downsample[1]

    def forward(self, x, sex=None, race=None):
        """
        与原 PreactivResBlock.forward 同结构, 只是 downsample 分支换成 hyper 版本.
        sex / race: [B] long, 用于驱动超网络.
        """
        # ========= Downsample 分支 (hyper 版本) =========
        ds = self.ds_bn(x)                       # 先 BN (theta_0 的一部分)

        # 主权重: theta_0 (普通 Conv2d 的权重)
        theta0_w = self.ds_conv.weight           # [out_c, in_c, 1, 1]
        theta0_b = self.ds_conv.bias             # [out_c] 或 None

        # 生成扰动
        delta_w, delta_b = self.hyper_net(sex, race)
        # delta_w: [B, out_c, in_c]  -> 加最后两个 1x1 维度
        delta_w = delta_w.unsqueeze(-1).unsqueeze(-1)   # [B, out_c, in_c, 1, 1]
        # delta_b: [B, out_c]

        # 逐样本卷积: 不同样本用不同的 (theta_0 + delta) 做 1x1 conv
        # 用 grouped conv 技巧批量处理: 把 batch 维拼到通道维
        B, C_in, H, W = ds.shape
        out_c = self.ds_conv.out_channels

        # full_w[i] = theta0_w + delta_w[i]
        full_w = theta0_w.unsqueeze(0) + delta_w   # [B, out_c, in_c, 1, 1]
        if theta0_b is not None:
            full_b = theta0_b.unsqueeze(0) + delta_b   # [B, out_c]
        else:
            full_b = delta_b

        # reshape 成 grouped conv 的形式: 把 B 折叠进通道维
        # input  : [1, B*C_in, H, W]
        # weight : [B*out_c, C_in, 1, 1]   (groups=B)
        # bias   : [B*out_c]
        # output : [1, B*out_c, H', W'] -> [B, out_c, H', W']
        ds_grouped = ds.reshape(1, B * C_in, H, W)
        w_grouped = full_w.reshape(B * out_c, C_in, 1, 1)
        b_grouped = full_b.reshape(B * out_c)

        identity = F.conv2d(ds_grouped, w_grouped, b_grouped,
                            stride=self.stride, padding=0, groups=B)
        identity = identity.reshape(B, out_c,
                                    identity.shape[-2], identity.shape[-1])

        # ========= 主分支 (与原 PreactivResBlock.forward 一致) =========
        out = self.bn1(x)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.conv2(out)

        out = out + identity
        return out


# ============================================================
# HyperFusion-style PreactivResNet18 (UTKFace, 2D)
# ============================================================
class HyperFusionPreactivResNet18(nn.Module):
    """
    HyperFusion 风格的 PreactivResNet18:
        - 前 3 个 stage 是普通 PreactivResBlock
        - layer4[0]: HyperPreactivResBlock (downsample 由超网络驱动)
        - layer4[1]: 普通 PreactivResBlock
        - 分类头: 与原版一致

    forward(image, sex, race) -> logits.
    """

    _STAGE_DROPOUTS = (0.1, 0.2, 0.2, 0.3)

    def __init__(self, in_channels=3, n_outputs=1,
                 init_features=16, bn_momentum=0.1,
                 num_sex=2, num_race=5,
                 sex_embed=4, race_embed=8,
                 hyper_hidden=64, mip_scale=0.01, use_l2_norm=True):
        super().__init__()
        self.in_channels_running = init_features
        f = init_features
        self.bn_momentum = bn_momentum
        dim = 2

        _, _, _, MaxPool, GAP = _nd_modules(dim)

        # Stem
        self.stem = conv_bn_relu(in_channels, f, dim=dim, bn_momentum=bn_momentum)
        self.maxpool = MaxPool(kernel_size=2, stride=2)

        # Stage 1~3: 普通 block
        self.layer1 = self._make_layer(2 * f,  blocks=2, stride=1,
                                       dropout=self._STAGE_DROPOUTS[0])
        self.layer2 = self._make_layer(4 * f,  blocks=2, stride=2,
                                       dropout=self._STAGE_DROPOUTS[1])
        self.layer3 = self._make_layer(8 * f,  blocks=2, stride=2,
                                       dropout=self._STAGE_DROPOUTS[2])

        # ====== Stage 4: layer4[0] 是 hyper 版本, layer4[1] 是普通版本 ======
        # layer4[0]: 8f -> 16f, stride=2, 有 downsample (这是唯一的 hyperlayer)
        self.hyper_net = AttrHyperNet(
            in_channels=8 * f, out_channels=16 * f,
            num_sex=num_sex, num_race=num_race,
            sex_embed=sex_embed, race_embed=race_embed,
            hidden_dim=hyper_hidden,
            use_l2_norm=use_l2_norm, mip_scale=mip_scale,
        )
        self.layer4_block0 = HyperPreactivResBlock(
            in_channels=8 * f, out_channels=16 * f, dim=dim,
            stride=2, dropout=self._STAGE_DROPOUTS[3],
            bn_momentum=bn_momentum, hyper_net=self.hyper_net,
        )
        # layer4[1]: 16f -> 16f, stride=1, 没有 downsample, 普通 block
        self.layer4_block1 = PreactivResBlock(
            in_channels=16 * f, out_channels=16 * f, dim=dim,
            stride=1, dropout=self._STAGE_DROPOUTS[3],
            bn_momentum=bn_momentum,
        )

        # 分类头 (与原版一致)
        self.gap = GAP(1)
        self.linear_drop1 = nn.Dropout(0.6)
        self.fc1 = nn.Linear(16 * f, 4 * f)
        self.relu = nn.ReLU()
        self.linear_drop2 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(4 * f, n_outputs)

        self._init_main_weights()

    def _make_layer(self, out_channels, blocks, stride, dropout):
        layers = [PreactivResBlock(
            self.in_channels_running, out_channels,
            dim=2, stride=stride, dropout=dropout,
            bn_momentum=self.bn_momentum,
        )]
        self.in_channels_running = out_channels
        for _ in range(1, blocks):
            layers.append(PreactivResBlock(
                out_channels, out_channels,
                dim=2, stride=1, dropout=dropout,
                bn_momentum=self.bn_momentum,
            ))
        return nn.Sequential(*layers)

    def _init_main_weights(self):
        """
        主网络初始化, 不覆盖 hyper_net 的初始化 (那部分已在 AttrHyperNet._init_mip 完成).
        Linear 保留 PyTorch 默认 (避免 fc2 fan_out 爆炸问题, 见 PreactivResNet18 注释).
        """
        for name, m in self.named_modules():
            # 跳过 hyper_net 内部模块, 它已经自己初始化过了
            if name.startswith("hyper_net"):
                continue
            if isinstance(m, (nn.Conv2d, nn.Conv3d)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm3d)):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, image, sex, race):
        x = self.stem(image)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        # layer4[0]: hyper block (需要 sex, race)
        x = self.layer4_block0(x, sex=sex, race=race)
        # layer4[1]: 普通 block
        x = self.layer4_block1(x)

        x = self.gap(x)
        x = torch.flatten(x, 1)

        x = self.linear_drop1(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.linear_drop2(x)
        x = self.fc2(x)
        return x


# ============================================================
# 结构自检
# ============================================================
if __name__ == "__main__":
    model = HyperFusionPreactivResNet18(
        in_channels=3, n_outputs=1, init_features=16,
        num_sex=2, num_race=5,
        mip_scale=0.01, use_l2_norm=True,
    )

    B = 4
    image = torch.randn(B, 3, 224, 224)
    sex = torch.randint(0, 2, (B,))
    race = torch.randint(0, 5, (B,))

    logits = model(image, sex, race)
    print(f"Image  : {tuple(image.shape)}")
    print(f"Sex    : {tuple(sex.shape)}")
    print(f"Race   : {tuple(race.shape)}")
    print(f"Logits : {tuple(logits.shape)}")     # (4, 1)
    print(f"Logit  stats: mean={logits.mean().item():+.3f}  std={logits.std().item():.3f}")
    print(f"  (MIP small-init 下, 初始 logits 应当跟普通 PreactivResNet18 同尺度)")

    # 参数量分解
    hyper_params = sum(p.numel() for n, p in model.named_parameters()
                       if n.startswith("hyper_net"))
    main_params = sum(p.numel() for n, p in model.named_parameters()
                      if not n.startswith("hyper_net"))
    print(f"\nMain network params : {main_params:,}")
    print(f"HyperNet params     : {hyper_params:,}")
    print(f"Total params        : {main_params + hyper_params:,}")

    # 验证: 反向传播能否走通 hyper_net
    loss = logits.sum()
    loss.backward()
    n_hyper_grad = sum(p.grad.abs().sum().item()
                       for n, p in model.named_parameters()
                       if n.startswith("hyper_net") and p.grad is not None)
    print(f"\nHyperNet gradient sum: {n_hyper_grad:.4f}")
    print("✅ Backward pass OK." if n_hyper_grad > 0 else
          "⚠️  HyperNet 没拿到梯度, 检查 forward 链路!")

    # 验证 MIP additive: theta_0 也应该拿到梯度
    theta0_grad = model.layer4_block0.ds_conv.weight.grad.abs().sum().item()
    print(f"theta_0 (ds_conv.weight) gradient sum: {theta0_grad:.4f}")