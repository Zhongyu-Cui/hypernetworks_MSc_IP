"""
通路归因审计的统一汇报逻辑（两个数据集共用）
============================================
被 scripts/viz_hyperadapt_hyperplane_{ham,mimic}.py 调用，把三项审计打印成表并返回可落盘字典：

  1. **分解顺序敏感性**：order-1 / order-2 / Shapley 三口径下的 fc 占比（单一顺序的占比是
     路径依赖读数，交互项 <Δw, Δf> 全给谁会改变结论 ⇒ 必须以 Shapley 为主口径）。
  2. **AUC（排序）口径归因**：在参照组 logit 上只叠加一条通路，看 AUC 动多少。这与条件化消融
     `docs/conditioning_ablation_e1_pathway_knockout.md` 的 knockout 实验同构，可直接对账。
  3. **平移 vs 重排**：每条通路贡献的 mean/std，以及该贡献单独作为分数的 AUC（≈0.5 ⇒ 与标签
     无关，只是平移或噪声，不可能改排序）。

设计动机：logit 幅度口径（mean |贡献|）与 AUC 口径回答的是**不同问题**，可以给出方向相反的答案
——一条通路可以把 logit 推得很远却完全不改排序。两者并列报告才不会把结论讲过头。
"""

from __future__ import annotations

import numpy as np

from src.utils.hyperplane_viz import audit_decomposition_orders, audit_pathway_auc_attribution


def _fc_share(fc_term: np.ndarray, conv_term: np.ndarray, g: int) -> float:
    """某组在某口径下的 fc 占比 = mean|fc| / (mean|fc| + mean|conv|)。"""
    fa, ca = float(np.abs(fc_term[g]).mean()), float(np.abs(conv_term[g]).mean())
    return fa / (fa + ca + 1e-12)


def run_pathway_audit(
    feats: np.ndarray, w_eff: np.ndarray, logits: np.ndarray, labels: np.ndarray,
    ref_idx: int, group_labels: dict[int, str],
) -> dict:
    """
    跑完三项审计并打印人读表格。

    Args:
        feats  : [G, N, 512] 反事实特征；w_eff: [G, 512]；logits: [G, N]；labels: [N]。
        ref_idx: 参照组下标；group_labels: 组下标 -> 显示名。

    Returns:
        可直接写入 metrics JSON 的嵌套字典（顺序敏感性 / AUC 归因 / 汇总判读量）。
    """
    n_groups = w_eff.shape[0]
    others = [g for g in range(n_groups) if g != ref_idx]
    orders = audit_decomposition_orders(feats, w_eff, logits, ref_idx)

    # ---- 1. 分解顺序敏感性 ----
    print("\n--- 审计① 分解顺序敏感性（fc 占比；交互项归属不同 ⇒ 单一顺序是路径依赖读数）---")
    print(f"{'subgroup':<28s} {'order1':>8s} {'order2':>8s} {'Shapley':>8s} "
          f"{'|interaction|':>14s} {'I/|Δlogit|':>11s}")
    order_rows: dict[str, dict[str, float]] = {}
    for g in others:
        sh_fc, sh_cv = orders["fc_shapley"], orders["conv_shapley"]
        s1 = _fc_share(orders["fc_order1"], orders["conv_order1"], g)
        s2 = _fc_share(orders["fc_order2"], orders["conv_order2"], g)
        ssh = _fc_share(sh_fc, sh_cv, g)
        i_abs = float(np.abs(orders["interaction"][g]).mean())
        d_abs = float(np.abs(logits[g] - logits[ref_idx]).mean())
        order_rows[group_labels[g]] = {
            "fc_share_order1": s1, "fc_share_order2": s2, "fc_share_shapley": ssh,
            "mean_abs_interaction": i_abs, "interaction_over_mean_abs_delta_logit":
                i_abs / (d_abs + 1e-12),
            "mean_abs_fc_shapley": float(np.abs(sh_fc[g]).mean()),
            "mean_abs_conv_shapley": float(np.abs(sh_cv[g]).mean()),
        }
        print(f"{group_labels[g]:<28s} {s1:>7.1%} {s2:>8.1%} {ssh:>8.1%} "
              f"{i_abs:>14.4f} {i_abs / (d_abs + 1e-12):>10.2f}x")
    print(f"三口径恒等式残差: order1={float(orders['residual_order1']):.2e}  "
          f"order2={float(orders['residual_order2']):.2e}  "
          f"shapley={float(orders['residual_shapley']):.2e}（均应在 float32 误差量级）")

    # ---- 2 & 3. AUC 口径归因 + 平移 vs 重排（主口径用 Shapley）----
    auc_attr = audit_pathway_auc_attribution(
        orders["fc_shapley"], orders["conv_shapley"], logits, labels, ref_idx)
    print("\n--- 审计② AUC（排序）口径归因，与消融 knockout 同构（Shapley 项）---")
    print(f"{'subgroup':<28s} {'AUC_ref':>8s} {'AUC_full':>9s} {'fc_only':>9s} {'conv_only':>10s} "
          f"{'ΔAUC_fc':>9s} {'ΔAUC_conv':>10s}")
    for g in others:
        a = auc_attr[g]
        print(f"{group_labels[g]:<28s} {a['auc_ref']:>8.4f} {a['auc_full']:>9.4f} "
              f"{a['auc_fc_only']:>9.4f} {a['auc_conv_only']:>10.4f} "
              f"{a['auc_fc_only'] - a['auc_ref']:>+9.4f} "
              f"{a['auc_conv_only'] - a['auc_ref']:>+10.4f}")

    print("\n--- 审计③ 平移 vs 重排（贡献的 mean/std，及该贡献单独作分数的 AUC；≈0.5 = 与标签无关）---")
    print(f"{'subgroup':<28s} {'fc mean':>9s} {'fc std':>8s} {'AUC(fc)':>8s} "
          f"{'conv mean':>10s} {'conv std':>9s} {'AUC(conv)':>10s}")
    for g in others:
        a = auc_attr[g]
        print(f"{group_labels[g]:<28s} {a['fc_mean']:>+9.4f} {a['fc_std']:>8.4f} "
              f"{a['auc_of_fc_term']:>8.4f} {a['conv_mean']:>+10.4f} {a['conv_std']:>9.4f} "
              f"{a['auc_of_conv_term']:>10.4f}")

    # ---- 汇总判读量（跨组平均，便于一句话结论）----
    summary = {
        "mean_fc_share_order1": float(np.mean([order_rows[group_labels[g]]["fc_share_order1"]
                                               for g in others])),
        "mean_fc_share_order2": float(np.mean([order_rows[group_labels[g]]["fc_share_order2"]
                                               for g in others])),
        "mean_fc_share_shapley": float(np.mean([order_rows[group_labels[g]]["fc_share_shapley"]
                                                for g in others])),
        "max_abs_delta_auc_fc_only": float(np.max([abs(auc_attr[g]["auc_fc_only"]
                                                       - auc_attr[g]["auc_ref"]) for g in others])),
        "max_abs_delta_auc_conv_only": float(np.max([abs(auc_attr[g]["auc_conv_only"]
                                                         - auc_attr[g]["auc_ref"])
                                                     for g in others])),
        "max_abs_auc_of_fc_term_minus_half": float(np.max([abs(auc_attr[g]["auc_of_fc_term"] - 0.5)
                                                           for g in others])),
        "max_abs_auc_of_conv_term_minus_half": float(
            np.max([abs(auc_attr[g]["auc_of_conv_term"] - 0.5) for g in others])),
    }
    print(f"\n汇总: fc 占比 order1={summary['mean_fc_share_order1']:.1%} / "
          f"order2={summary['mean_fc_share_order2']:.1%} / "
          f"Shapley={summary['mean_fc_share_shapley']:.1%}  |  "
          f"max|ΔAUC| fc_only={summary['max_abs_delta_auc_fc_only']:.4f} "
          f"conv_only={summary['max_abs_delta_auc_conv_only']:.4f}")

    return {
        "order_sensitivity": order_rows,
        "auc_attribution": {group_labels[g]: auc_attr[g] for g in others},
        "summary": summary,
        "residuals": {k: float(v) for k, v in orders.items() if k.startswith("residual_")},
    }
