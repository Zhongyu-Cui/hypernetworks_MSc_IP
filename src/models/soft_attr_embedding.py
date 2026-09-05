"""
软属性条件通路：期望嵌入（Expected Embedding）
================================================
方案见 `docs/predicted_attribute_hyperadapt_plan.md` §3.3。HyperAdapt 原本的条件通路是
`nn.Embedding` 查表（离散属性索引 -> 稠密向量），本模块把它换成**概率加权的嵌入期望**：

    e = p̂ᵀ E            # p̂ ∈ R^{B×C}（子群分类器 g 的 softmax），E = embedding.weight ∈ R^{C×d}

数学性质（本设计的全部要点）：
  1. **p̂ 为 one-hot 时逐位等价于查表**，因此 GT 臂与 pred 臂共用同一模型类与同一 state_dict
     schema——已训好的 GT-HyperAdapt checkpoint 可零重训直接吃软属性（方案 §6 Stage 0）；
  2. 参数名与 GT 版（`resnet18_hyperadapt.SkinEmbedding` / `AgeEmbedding` / `PatientEmbedding`）
     **严格一致**（`<axis>_embed.weight`、`fuse.*`），故两版 state_dict 可互相 load；
  3. p̂ 的锐度直接决定 e 在嵌入空间的位置：近 one-hot → e≈某子群嵌入（等价硬属性），
     平缓 → e 落在若干子群嵌入之间。这正是「软」条件化的全部含义，也是 g 需要温度校准的原因
     （见 `src/training/attr_predictor.py`）。

本模块只负责条件通路；HyperAdapt 的 A_gen/B_gen/逐样本卷积/fc 机制完全复用，
由 `resnet18_hyperadapt.ResNet18HyperAdaptSoft*` 三个子类替换 `self.patient_embed` 接入。
"""

from __future__ import annotations

import torch
import torch.nn as nn


def expected_embedding(probs: torch.Tensor, embedding: nn.Embedding) -> torch.Tensor:
    """
    概率加权的嵌入期望 e = p̂ᵀE。

    Args:
        probs    : [B, C] 各类别概率（每行非负、和为 1；one-hot 时退化为普通查表）。
        embedding: nn.Embedding(C, d)，其 weight 即 E。

    Returns:
        [B, d] 条件向量。

    Raises:
        ValueError: probs 维度不是 2 维，或类别数与 embedding 不匹配。
    """
    if probs.dim() != 2:
        raise ValueError(f"probs 应为 [B, C] 二维张量，实得 shape={tuple(probs.shape)}")
    num_classes = embedding.weight.shape[0]
    if probs.shape[1] != num_classes:
        raise ValueError(
            f"probs 的类别数 {probs.shape[1]} 与 embedding 的 {num_classes} 不一致"
        )
    # dtype 对齐：probs 由 dataloader 给出 float32，embedding 权重同为 float32
    return probs.to(embedding.weight.dtype) @ embedding.weight


class SoftSkinEmbedding(nn.Module):
    """
    Fitzpatrick17k 的**软**条件通路：输入 skin 的 softmax 分布 [B, 6] 而非索引 [B]。
    参数命名与 `resnet18_hyperadapt.SkinEmbedding` 严格一致（skin_embed / fuse），
    两者 state_dict 可互 load。

    Args:
        num_skin     : 肤色类别数（Fitzpatrick I–VI = 6）。
        cat_embed_dim: skin embedding 维度。
        out_dim      : profile vector 维度（须与模型 patient_embed_dim 一致）。
    """

    def __init__(self, num_skin: int = 6, cat_embed_dim: int = 16, out_dim: int = 128) -> None:
        super().__init__()
        self.skin_embed = nn.Embedding(num_skin, cat_embed_dim)
        self.fuse = nn.Sequential(
            nn.Linear(cat_embed_dim, out_dim),
            nn.ReLU(inplace=True),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, skin_prob: torch.Tensor) -> torch.Tensor:
        """skin_prob: [B, 6] 概率 -> patient profile [B, out_dim]。"""
        return self.fuse(expected_embedding(skin_prob, self.skin_embed))


class SoftAgeEmbedding(nn.Module):
    """
    HAM10000 / PAPILA 的**软** age 条件通路（对应 `resnet18_hyperadapt.AgeEmbedding`）。

    Args:
        num_age      : age_group 有效类别数（HAM: 4；PAPILA: 2）。
        cat_embed_dim: age embedding 维度。
        out_dim      : profile vector 维度。
    """

    def __init__(self, num_age: int = 4, cat_embed_dim: int = 16, out_dim: int = 128) -> None:
        super().__init__()
        self.age_embed = nn.Embedding(num_age, cat_embed_dim)
        self.fuse = nn.Sequential(
            nn.Linear(cat_embed_dim, out_dim),
            nn.ReLU(inplace=True),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, age_prob: torch.Tensor) -> torch.Tensor:
        """age_prob: [B, num_age] 概率 -> patient profile [B, out_dim]。"""
        return self.fuse(expected_embedding(age_prob, self.age_embed))


class SoftSexAgeEmbedding(nn.Module):
    """
    HAM10000 的**软**双属性条件通路（对应 `resnet18_hyperadapt.SexAgeEmbedding`）：
    sex / age_group 各给一份 softmax，各自取嵌入期望后拼接过同款 fuse MLP。

    参数命名与 GT 版严格一致（`sex_embed` / `age_embed` / `fuse.*`），两版 state_dict 可互 load；
    两份 p̂ 均为 one-hot 时与 GT 查表逐位等价（见本模块 `_selftest`）。

    Args:
        num_sex      : sex 类别数（HAM: Male/Female = 2）。
        num_age      : age_group 有效类别数（HAM: 4）。
        cat_embed_dim: 每个属性 embedding 的维度。
        out_dim      : profile vector 维度。
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

    def forward(self, sex_prob: torch.Tensor, age_prob: torch.Tensor) -> torch.Tensor:
        """sex_prob [B, num_sex] + age_prob [B, num_age] -> patient profile [B, out_dim]。"""
        cond = torch.cat(
            [
                expected_embedding(sex_prob, self.sex_embed),
                expected_embedding(age_prob, self.age_embed),
            ],
            dim=-1,
        )
        return self.fuse(cond)


class SoftPatientEmbedding(nn.Module):
    """
    MIMIC / CheXpert 的**软**三轴条件通路（对应 `resnet18_hyperadapt.PatientEmbedding`）：
    sex / race / age_group 各给一份 softmax，各自取嵌入期望后拼接过同款 fuse MLP。

    Args:
        num_sex / num_race / num_age: 各轴类别数。
        cat_embed_dim: 每轴 embedding 维度。
        out_dim      : profile vector 维度。
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
        self.fuse = nn.Sequential(
            nn.Linear(3 * cat_embed_dim, out_dim),
            nn.ReLU(inplace=True),
            nn.Linear(out_dim, out_dim),
        )

    def forward(
        self, sex_prob: torch.Tensor, race_prob: torch.Tensor, age_prob: torch.Tensor
    ) -> torch.Tensor:
        """三轴概率（各 [B, C_i]）-> patient profile [B, out_dim]。"""
        cond = torch.cat(
            [
                expected_embedding(sex_prob, self.sex_embed),
                expected_embedding(race_prob, self.race_embed),
                expected_embedding(age_prob, self.age_embed),
            ],
            dim=-1,
        )
        return self.fuse(cond)


# ============================================================
# 自检：one-hot 输入必须与 GT 查表逐位一致（方案 P0.1 的验收条件）
# ============================================================
def _selftest() -> None:
    """对四个 Soft* 通路验证「one-hot 输入 ≡ nn.Embedding 查表」，误差须 < 1e-6。"""
    from src.models.resnet18_hyperadapt import (
        AgeEmbedding, PatientEmbedding, SexAgeEmbedding, SkinEmbedding,
    )

    torch.manual_seed(0)
    batch = 32

    def one_hot(idx: torch.Tensor, num: int) -> torch.Tensor:
        return torch.nn.functional.one_hot(idx, num_classes=num).float()

    # --- skin (Fitzpatrick, 6 类) ---
    gt_skin, soft_skin = SkinEmbedding(num_skin=6), SoftSkinEmbedding(num_skin=6)
    soft_skin.load_state_dict(gt_skin.state_dict())   # 键完全一致，可直接互 load
    idx = torch.randint(0, 6, (batch,))
    diff = (gt_skin(idx) - soft_skin(one_hot(idx, 6))).abs().max().item()
    print(f"  SkinEmbedding   one-hot 等价性 max|Δ| = {diff:.3e}")
    assert diff < 1e-6

    # --- age (HAM, 4 类) ---
    gt_age, soft_age = AgeEmbedding(num_age=4), SoftAgeEmbedding(num_age=4)
    soft_age.load_state_dict(gt_age.state_dict())
    idx = torch.randint(0, 4, (batch,))
    diff = (gt_age(idx) - soft_age(one_hot(idx, 4))).abs().max().item()
    print(f"  AgeEmbedding    one-hot 等价性 max|Δ| = {diff:.3e}")
    assert diff < 1e-6

    # --- sex+age (HAM 双属性对照臂, 2×4) ---
    gt_sa, soft_sa = SexAgeEmbedding(num_sex=2, num_age=4), SoftSexAgeEmbedding(num_sex=2, num_age=4)
    soft_sa.load_state_dict(gt_sa.state_dict())
    s_idx, a_idx = torch.randint(0, 2, (batch,)), torch.randint(0, 4, (batch,))
    diff = (
        gt_sa(s_idx, a_idx) - soft_sa(one_hot(s_idx, 2), one_hot(a_idx, 4))
    ).abs().max().item()
    print(f"  SexAgeEmbedding one-hot 等价性 max|Δ| = {diff:.3e}")
    assert diff < 1e-6

    # --- patient (MIMIC, 2×2×2) ---
    gt_pat, soft_pat = PatientEmbedding(), SoftPatientEmbedding()
    soft_pat.load_state_dict(gt_pat.state_dict())
    s, r, a = (torch.randint(0, 2, (batch,)) for _ in range(3))
    diff = (
        gt_pat(s, r, a) - soft_pat(one_hot(s, 2), one_hot(r, 2), one_hot(a, 2))
    ).abs().max().item()
    print(f"  PatientEmbedding one-hot 等价性 max|Δ| = {diff:.3e}")
    assert diff < 1e-6

    # --- 软输入确实落在子群嵌入之间（不是退化到某一类）---
    p = torch.full((1, 6), 1.0 / 6)
    e_uniform = expected_embedding(p, soft_skin.skin_embed)
    e_mean = soft_skin.skin_embed.weight.mean(dim=0, keepdim=True)
    assert (e_uniform - e_mean).abs().max().item() < 1e-6
    print("  均匀分布 -> 嵌入均值 ✓")
    print("soft_attr_embedding 自检通过。")


if __name__ == "__main__":
    _selftest()
