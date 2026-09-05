"""
命名超网络的逐层条件化强度 rho / rho_between（三变体 × 四数据集，纯 CPU、无需图像）
=====================================================================================
补齐第 5 章 §5.6.1 的缺口：`conditioning_rho` 此前只对消融的 `ResNet18CondNet` 在
HAM/MIMIC 上算过，**三个命名模型从未计算**，导致附录 `sec:validity` 的
「computed for every trained conditioned model」无产物支撑。

**定义与 `ResNet18CondNet.conditioning_rho` 完全一致**（单一事实来源，见该方法 docstring）：

    rho_l         = E_a‖ΔW_l(a)‖_F / ‖W_l‖_F                      该层被改动了多少
    rho_between_l = sqrt(E_a‖ΔW_l(a) − mean_a ΔW_l‖_F²) / ‖W_l‖_F  偏移中随属性变化的部分

对 K 个属性取值**等权**（避免大子群主导），先逐单位求 Frobenius norm 再平均。
三个变体的 ΔW 各自不同，故分别实现：

· **HyperAdapt** —— conv 为乘性 `ΔW = W ⊙ (A·B)`，fc 为加性 `ΔW = A_fc·B_fc`。与消融同形。
· **HyperFusion** —— 仅 `layer4[0].downsample` 一处，超网络输出**加性残差** Δθ，
  分母取 backbone 自带的 θ_0。
· **HyperHead** —— ⚠️ **rho 无定义**：该变体**直接生成**整个 fc 权重而非在某个 base 上加偏移，
  没有可作分母的 W_l。故只报 `rho_between`，分母改用**跨属性平均权重**的范数
  `‖mean_a W(a)‖_F`——它量的仍是「产出的头随属性分化了多少，相对于头自身的尺度」，
  与其他两个变体的 rho_between 同量纲、可并列阅读，但**不可与它们的 rho 混读**。

用法（秒级，CPU）：
    PYTHONPATH=. python scripts/named_hn_rho.py
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import torch

from src.models.resnet18_hyperadapt import (
    ResNet18HyperAdapt, ResNet18HyperAdaptSexAge, ResNet18HyperAdaptSkin,
)
from src.models.resnet18_hyperfusion import (
    ResNet18HyperFusion, ResNet18HyperFusionSexAge, ResNet18HyperFusionSkin,
)
from src.models.resnet18_hyperhead import (
    ResNet18HyperHead, ResNet18HyperHeadSexAge, ResNet18HyperHeadSkin,
)
from src.paths import OUTPUTS_DIR

OUTPUTS = OUTPUTS_DIR
N_FOLDS = 5
SEED_OF_FOLD = lambda fold: 42 + fold          # noqa: E731  与 cv5 训练脚本一致

# 每个数据集的 (输出子目录, 属性取值网格, 各方法的类与构造参数)
GRIDS = {
    "HAM10000":      [(s, a) for s in range(2) for a in range(4)],      # sex × age_group
    "Fitzpatrick17k": [(k,) for k in range(6)],                         # skin
    "MIMIC-CXR":     list(itertools.product(range(2), range(2), range(2))),   # sex × race × age
    "CheXpert":      list(itertools.product(range(2), range(2), range(2))),
}
SUBDIR = {"HAM10000": "ham10000", "Fitzpatrick17k": "fitzpatrick",
          "MIMIC-CXR": "mimic_cxr", "CheXpert": "chexpert_cxr"}
# (方法键前缀, 类)；HAM 用 sex+age 双轴臂，其余用各自的单/三属性臂
CLASSES = {
    "HAM10000": {"hyperhead_sexage": (ResNet18HyperHeadSexAge, {}),
                 "hyperfusion_sexage": (ResNet18HyperFusionSexAge, {}),
                 "hyperadapt_sexage": (ResNet18HyperAdaptSexAge, {})},
    "Fitzpatrick17k": {"hyperhead": (ResNet18HyperHeadSkin, {}),
                       "hyperfusion": (ResNet18HyperFusionSkin, {}),
                       "hyperadapt": (ResNet18HyperAdaptSkin, {})},
    "MIMIC-CXR": {"hyperhead": (ResNet18HyperHead, {}),
                  "hyperfusion": (ResNet18HyperFusion, {}),
                  "hyperadapt": (ResNet18HyperAdapt, {})},
    "CheXpert": {"hyperhead": (ResNet18HyperHead, {}),
                 "hyperfusion": (ResNet18HyperFusion, {}),
                 "hyperadapt": (ResNet18HyperAdapt, {})},
}


def _record(x: torch.Tensor, den: torch.Tensor,
            w_sq: torch.Tensor | None = None) -> dict[str, float]:
    """
    由 [K, ...] 的偏移张量算 (rho, rho_between)，口径与 CondNet.conditioning_rho 逐行一致。

    Args:
        x: [K, C_out, C_in] 偏移。乘性时是调制 M，加性时是 ΔW 本身。
        den: 标量分母 ‖W‖_F。
        w_sq: 乘性情形下的 [C_out, C_in] 逐通道对权重能量；加性情形传 None。

    Returns:
        {"rho", "rho_between"}。
    """
    if w_sq is not None:
        per = torch.sqrt(((x ** 2) * w_sq.unsqueeze(0)).sum(dim=(1, 2)))
        dev = x - x.mean(dim=0, keepdim=True)
        betw = torch.sqrt((((dev ** 2) * w_sq.unsqueeze(0)).sum(dim=(1, 2))).mean())
    else:
        per = torch.sqrt((x ** 2).sum(dim=(1, 2)))
        dev = x - x.mean(dim=0, keepdim=True)
        betw = torch.sqrt(((dev ** 2).sum(dim=(1, 2))).mean())
    return {"rho": float((per / den).mean()), "rho_between": float(betw / den)}


@torch.no_grad()
def rho_hyperadapt(model: ResNet18HyperAdapt, emb: torch.Tensor) -> dict[str, dict[str, float]]:
    """HyperAdapt 的逐层 rho：conv 乘性 W⊙(A·B)，fc 加性 A_fc·B_fc。"""
    k = emb.shape[0]
    out: dict[str, dict[str, float]] = {}
    for sname, blocks, c_out in (("layer1", model.layer1, 64), ("layer2", model.layer2, 128),
                                 ("layer3", model.layer3, 256), ("layer4", model.layer4, 512)):
        a_shared = model.A_gen[f"c{c_out}"](emb).view(k, c_out, model.rank)
        for bi, blk in enumerate(blocks):
            targets = [(f"{sname}.{bi}.conv1", blk.conv1.weight, blk.B_gen_conv1, blk.in_channels),
                       (f"{sname}.{bi}.conv2", blk.conv2.weight, blk.B_gen_conv2, blk.out_channels)]
            if blk.B_gen_down is not None:
                targets.append((f"{sname}.{bi}.downsample.0", blk.downsample[0].weight,
                                blk.B_gen_down, blk.in_channels))
            for name, weight, bgen, c_in in targets:
                b_mat = bgen(emb).view(k, model.rank, c_in)
                mod = torch.bmm(a_shared, b_mat)
                w_sq = (weight ** 2).sum(dim=(2, 3))
                out[name] = _record(mod, torch.sqrt(w_sq.sum()), w_sq)
    a_fc = model.fc_A_gen(emb).view(k, model.fc.out_features, model.rank)
    b_fc = model.fc_B_gen(emb).view(k, model.rank, 512)
    out["fc"] = _record(torch.bmm(a_fc, b_fc), torch.linalg.norm(model.fc.weight))
    return out


@torch.no_grad()
def rho_hyperfusion(model: ResNet18HyperFusion, emb: torch.Tensor) -> dict[str, dict[str, float]]:
    """HyperFusion 的单一注入点：layer4[0].downsample 的加性残差 Δθ，分母取 θ_0。"""
    delta = model.hyper_net(emb)                                  # [K, 512, 256]
    theta0 = model.backbone.layer4[0].downsample[0].weight        # [512, 256, 1, 1]
    return {"layer4.0.downsample.0": _record(delta, torch.linalg.norm(theta0))}


@torch.no_grad()
def rho_hyperhead(model: torch.nn.Module, *attrs: torch.Tensor) -> dict[str, dict[str, float]]:
    """
    HyperHead 直接生成 fc 权重，没有可作分母的 base ⇒ **rho 无定义**，只报 rho_between，
    分母改用跨属性平均权重的范数。
    """
    weight, _ = model.hyper(*attrs)                               # [K, C, 512]
    mean_w = weight.mean(dim=0, keepdim=True)
    betw = torch.sqrt(((weight - mean_w) ** 2).sum(dim=(1, 2)).mean())
    return {"fc": {"rho": float("nan"),
                   "rho_between": float(betw / torch.linalg.norm(mean_w[0]))}}


def stage_of(layer: str) -> str:
    """把层名归并到 stage（layer1--layer4 或 fc）。"""
    return "fc" if layer == "fc" else layer.split(".")[0]


def main() -> None:
    results: dict[str, dict] = {}
    for ds, grid in GRIDS.items():
        cfg = json.loads((OUTPUTS / SUBDIR[ds] / "cv5" / "selected_configs.json").read_text())
        cfg = cfg.get("config", cfg)
        attrs = [torch.tensor([g[i] for g in grid], dtype=torch.long) for i in range(len(grid[0]))]
        for method, (cls, kwargs) in CLASSES[ds].items():
            per_run: list[dict[str, dict[str, float]]] = []
            for fold in range(N_FOLDS):
                ck = (OUTPUTS / SUBDIR[ds] / "cv5" /
                      f"{method}_{cfg[method]}_seed{SEED_OF_FOLD(fold)}_best_overall.pth")
                if not ck.exists():
                    print(f"[跳过] 缺 {ck.name}")
                    continue
                model = cls(**kwargs)
                state = torch.load(ck, map_location="cpu", weights_only=False)
                state = state.get("model_state", state.get("state_dict", state))
                model.load_state_dict(state)
                model.eval()
                if "hyperhead" in method:
                    per_run.append(rho_hyperhead(model, *attrs))
                else:
                    emb = model.patient_embed(*attrs)
                    per_run.append(rho_hyperfusion(model, emb) if "hyperfusion" in method
                                   else rho_hyperadapt(model, emb))
            if not per_run:
                continue
            layers = per_run[0].keys()
            agg = {ly: {q: float(np.nanmean([r[ly][q] for r in per_run]))
                        for q in ("rho", "rho_between")} for ly in layers}
            by_stage: dict[str, list[float]] = {}
            for ly, v in agg.items():
                by_stage.setdefault(stage_of(ly), []).append(v["rho_between"])
            results.setdefault(ds, {})[method] = {
                "n_folds": len(per_run), "config": cfg[method],
                "per_layer": agg,
                "stage_rho_between": {k: float(np.mean(v)) for k, v in by_stage.items()},
            }
            summary = "  ".join(f"{k}={v:.4f}" for k, v in
                                results[ds][method]["stage_rho_between"].items())
            print(f"{ds:<16}{method:<20}folds={len(per_run)}  {summary}")
    out = OUTPUTS / "analysis" / "named_hn_rho.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("\n已落盘 ->", out)


if __name__ == "__main__":
    main()
