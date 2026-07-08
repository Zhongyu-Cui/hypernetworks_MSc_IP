"""
合成信号剂量-反应汇总与绘图（评审补强 R1.4）
============================================
读取 R1.3 各信号档（η）× 各方法 × 5 seed 的 test 预测，计算 (HN − ERM) 的 Overall / worst-group
（子群 = A_syn）增益，画「增益 vs 实测 I(Y;A_syn|X)」剂量-反应曲线，检验增益是否随信号单调上升、
0 档（η=0.5）是否打平。这是 R1 唯一直接回答「信号存在是否**充分**导致 HN 增益」的因果证据。

**口径统一**：
  - 主指标选择 = overall-selection checkpoint（与 D3 主口径一致）。
  - worst-group = A_syn 两组 AUC 的最小值（`synthetic_fairness.worst_case_auc`）。
  - 信号强度 x 轴 = R1.2 标定表的**实测** I(Y;A_syn|X)（int 口径，nats），非名义 η。
  - **ERM 跨档不变**（忽略 A_syn）：ERM 只训一次（η=0.5 目录），各档 worst-group 由该档 A_syn
    叠加到 ERM 预测上重算（A_syn 由 y_true 确定性重建，与 HN 所见逐元素一致，脚本内断言校验）。
  - 增益的不确定度 = 跨 5 seed 配对（HN_seed − ERM_seed），报均值 ± 95%CI（Student-t）。

方法对标（R1.4）：spurious-correlation 谱系的「指标 vs 强度」扫描（Spawrious fine control）。

运行：轻量（读 npz + 算 AUC + 画图），实验室机器直接跑即可。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from src.datasets.synthetic_attribute import inject_synthetic_attribute, make_split_rng
from src.training.harness.predictions import load_predictions
from src.training.harness.run import SEEDS_CONFIRM
from src.training.harness.significance import mean_ci
from src.utils.synthetic_fairness import worst_case_auc

OUTPUT_ROOT = Path("/vol/biomedic2/bglocker_studproj/zc125/outputs/mimic_synth")
CALIB_PATH = Path("/vol/biomedic2/bglocker_studproj/zc125/outputs/mimic_cxr/synthetic_signal_calibration.json")

# 方法 → C2 Pareto 选定 config_tag（R1.3 复用）
METHOD_CONFIG = {
    "erm": "lr3e-04_wd1e-03",
    "hyperhead": "lr1e-04_wd1e-04",
    "hyperfusion": "lr1e-04_wd1e-04",
    "hyperadapt": "lr3e-04_wd1e-04",
}
HN_METHODS = ("hyperhead", "hyperfusion", "hyperadapt")
ETAS = (0.5, 0.3621, 0.2942, 0.2096)           # R1.2 标定的 4 档（0.5=负控制）
ERM_ETA = 0.5                                    # ERM 跨档不变，统一从此档目录读
SELECTION = "overall"


def eta_tag(eta: float) -> str:
    """信号档目录标签（与训练脚本一致）。"""
    return f"eta{eta:.4f}"


def load_calibration() -> dict[float, float]:
    """读 R1.2 标定表，返回 {η: 实测 I(Y;A_syn|X) nats}（0 档=0，其余取复核实测均值）。"""
    with open(CALIB_PATH, "r", encoding="utf-8") as f:
        calib = json.load(f)
    out = {0.5: 0.0}
    for entry in calib["calibration_table"]:
        v = entry.get("verification")
        if v is not None:
            out[round(float(v["eta_star"]), 4)] = float(v["measured_int_nats_mean"])
    return out


def _pred_path(eta: float, method: str, seed: int) -> Path:
    """某 (η, method, seed) 的 overall-selection test 预测 npz 路径。"""
    tag = METHOD_CONFIG[method]
    from src.training.harness.predictions import default_prediction_path
    return default_prediction_path(OUTPUT_ROOT / eta_tag(eta), method, tag, seed, SELECTION)


def _asyn_for_test(y_true: np.ndarray, eta: float) -> np.ndarray:
    """由 test 真实标签确定性重建该档 A_syn（与 MIMICSynthDataset 逐元素一致：同 CSV 序、同种子）。"""
    return inject_synthetic_attribute(y_true.astype(np.int64), eta, make_split_rng(eta, "test"))


def per_seed_metrics(eta: float, method: str) -> dict[int, dict] | None:
    """
    某 (η, method) 的逐 seed {Overall AUC, worst(A_syn) AUC}。缺文件返回 None（供部分汇总）。

    ERM：从 ERM_ETA 目录读同一批预测，A_syn 用当前档 eta 重建叠加。
    HN ：从当前档目录读，直接用存储的 attr_a_syn（并断言与重建一致，交叉校验 ERM 叠加合法性）。
    """
    src_eta = ERM_ETA if method == "erm" else eta
    out: dict[int, dict] = {}
    for seed in SEEDS_CONFIRM:
        path = _pred_path(src_eta, method, seed)
        if not path.exists():
            return None
        y_true, y_score, attrs = load_predictions(path)
        a_syn = _asyn_for_test(y_true, eta)
        if method != "erm":
            # HN 存了 A_syn：断言重建与存储一致（确定性契约 + ERM 叠加合法性的守护）
            stored = attrs["a_syn"].astype(np.int64)
            if not np.array_equal(stored, a_syn):
                raise AssertionError(f"{method} η={eta} seed{seed}: 重建 A_syn 与存储不一致")
        out[seed] = {
            "overall_auc": float(roc_auc_score(y_true, y_score)),
            "worst_auc": float(worst_case_auc(y_true, y_score, a_syn)),
        }
    return out


def aggregate() -> dict:
    """对每 (η, method) 聚合 Overall/worst 的 mean±95%CI；对 HN 计算 vs ERM 的配对增益 mean±95%CI。"""
    calib = load_calibration()
    result: dict = {"calibration_nats": {str(k): v for k, v in calib.items()},
                    "per_dose": {}, "missing": []}

    # 先取 ERM 各档（同一批预测叠加不同档 A_syn）逐 seed 指标
    erm_by_dose = {}
    for eta in ETAS:
        m = per_seed_metrics(eta, "erm")
        if m is None:
            result["missing"].append(f"erm(η={eta})")
        erm_by_dose[eta] = m

    for eta in ETAS:
        nats = calib.get(round(eta, 4))
        dose_entry: dict = {"eta": eta, "measured_nats": nats, "methods": {}}
        erm = erm_by_dose[eta]
        for method in ("erm",) + HN_METHODS:
            m = erm if method == "erm" else per_seed_metrics(eta, method)
            if m is None:
                result["missing"].append(f"{method}(η={eta})")
                continue
            seeds = sorted(m)
            ov = [m[s]["overall_auc"] for s in seeds]
            wc = [m[s]["worst_auc"] for s in seeds]
            entry = {
                "n_seed": len(seeds),
                "overall": _meanci_dict(ov),
                "worst": _meanci_dict(wc),
            }
            # HN vs ERM 配对增益（同 seed 对齐；ERM 该档指标已叠加对应 A_syn）
            if method != "erm" and erm is not None:
                common = [s for s in seeds if s in erm]
                d_ov = [m[s]["overall_auc"] - erm[s]["overall_auc"] for s in common]
                d_wc = [m[s]["worst_auc"] - erm[s]["worst_auc"] for s in common]
                entry["gain_overall_vs_erm"] = _meanci_dict(d_ov)
                entry["gain_worst_vs_erm"] = _meanci_dict(d_wc)
            dose_entry["methods"][method] = entry
        result["per_dose"][eta_tag(eta)] = dose_entry

    result["monotonicity"] = _monotonicity(result)
    return result


def _meanci_dict(vals: list[float]) -> dict:
    """mean_ci → 可 JSON 化 dict。"""
    if not vals:
        return {"mean": None, "ci_low": None, "ci_high": None, "n": 0}
    ci = mean_ci(vals)
    return {"mean": ci.mean, "ci_low": ci.ci_low, "ci_high": ci.ci_high,
            "half_width": ci.half_width, "n": ci.n}


def _monotonicity(result: dict) -> dict:
    """
    对每个 HN 方法，检验 (Overall/worst) 增益是否随实测 nats 单调上升 + 0 档是否打平。

    单调性用 Spearman 秩相关（nats vs 增益均值）；0 档打平用其增益 95%CI 是否含 0。
    """
    from scipy.stats import spearmanr
    mono: dict = {}
    for method in HN_METHODS:
        xs, g_ov, g_wc, zero_ov_ci, zero_wc_ci = [], [], [], None, None
        for eta in ETAS:
            de = result["per_dose"][eta_tag(eta)]
            me = de["methods"].get(method)
            nats = de["measured_nats"]
            if me is None or nats is None or "gain_overall_vs_erm" not in me:
                continue
            xs.append(nats)
            g_ov.append(me["gain_overall_vs_erm"]["mean"])
            g_wc.append(me["gain_worst_vs_erm"]["mean"])
            if abs(eta - 0.5) < 1e-9:
                zero_ov_ci = [me["gain_overall_vs_erm"]["ci_low"], me["gain_overall_vs_erm"]["ci_high"]]
                zero_wc_ci = [me["gain_worst_vs_erm"]["ci_low"], me["gain_worst_vs_erm"]["ci_high"]]
        entry: dict = {"n_doses": len(xs)}
        if len(xs) >= 3:
            rho_ov, p_ov = spearmanr(xs, g_ov)
            rho_wc, p_wc = spearmanr(xs, g_wc)
            entry.update({
                "spearman_overall": {"rho": float(rho_ov), "p": float(p_ov)},
                "spearman_worst": {"rho": float(rho_wc), "p": float(p_wc)},
            })
        if zero_ov_ci is not None:
            entry["zero_dose_overall_ci_includes_0"] = bool(zero_ov_ci[0] <= 0 <= zero_ov_ci[1])
            entry["zero_dose_worst_ci_includes_0"] = bool(zero_wc_ci[0] <= 0 <= zero_wc_ci[1])
        mono[method] = entry
    return mono


def plot(result: dict, out_png: Path) -> None:
    """画剂量-反应：两面板（Overall 增益 / worst 增益）vs 实测 nats，一线一 HN 方法 + CI 误差棒。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharex=True)
    colors = {"hyperhead": "#4C78A8", "hyperfusion": "#F58518", "hyperadapt": "#54A24B"}
    # 英文标签，避免默认字体缺 CJK 字形导致图上出现方框
    for ax, metric, title in zip(
        axes, ("gain_overall_vs_erm", "gain_worst_vs_erm"),
        ("(a) Overall AUC gain (HN - ERM)", "(b) Worst-group(A_syn) AUC gain (HN - ERM)"),
    ):
        for method in HN_METHODS:
            xs, ys, errs = [], [], []
            for eta in ETAS:
                de = result["per_dose"][eta_tag(eta)]
                me = de["methods"].get(method)
                if me is None or de["measured_nats"] is None or metric not in me:
                    continue
                xs.append(de["measured_nats"]); ys.append(me[metric]["mean"])
                errs.append(me[metric]["half_width"] or 0.0)
            if xs:
                order = np.argsort(xs)
                xs, ys, errs = np.array(xs)[order], np.array(ys)[order], np.array(errs)[order]
                ax.errorbar(xs, ys, yerr=errs, marker="o", capsize=3, label=method, color=colors[method])
        ax.axhline(0, color="grey", lw=0.8, ls="--")
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("measured I(Y;A_syn|X)  (nats)")
        ax.set_ylabel("AUC gain (mean +/- 95% CI)")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
    fig.suptitle("R1 synthetic-signal dose-response: HN gain vs conditional MI (MIMIC-CXR, A_syn side-channel)", fontsize=12)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    print(f"图已保存: {out_png}")


def main() -> None:
    """汇总 → 落 JSON → 画图 → 打印可读小结。"""
    result = aggregate()
    out_json = OUTPUT_ROOT / "dose_response_summary.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"汇总已保存: {out_json}")
    if result["missing"]:
        print(f"⚠️  缺失（训练未完成或文件未落盘）: {result['missing']}")

    print(f"\n{'='*78}\n剂量-反应汇总（overall-selection；增益=HN−ERM 配对 5-seed 均值±95%CI）\n{'='*78}")
    for eta in ETAS:
        de = result["per_dose"][eta_tag(eta)]
        print(f"\nη={eta} (实测 {de['measured_nats']} nats):")
        for method in ("erm",) + HN_METHODS:
            me = de["methods"].get(method)
            if me is None:
                print(f"  {method:12s} [缺]"); continue
            line = f"  {method:12s} Overall {me['overall']['mean']:.4f} worst {me['worst']['mean']:.4f}"
            if "gain_overall_vs_erm" in me:
                go, gw = me["gain_overall_vs_erm"], me["gain_worst_vs_erm"]
                line += (f"  | ΔOverall {go['mean']:+.4f}[{go['ci_low']:+.4f},{go['ci_high']:+.4f}]"
                         f"  Δworst {gw['mean']:+.4f}[{gw['ci_low']:+.4f},{gw['ci_high']:+.4f}]")
            print(line)

    print(f"\n{'='*78}\n单调性检验（Spearman: 增益 vs 实测 nats；0 档打平 = CI 含 0）\n{'='*78}")
    for method, mo in result["monotonicity"].items():
        so = mo.get("spearman_overall"); sw = mo.get("spearman_worst")
        z = mo.get("zero_dose_overall_ci_includes_0")
        if so:
            print(f"  {method:12s} Overall ρ={so['rho']:+.3f}(p={so['p']:.3f})  "
                  f"worst ρ={sw['rho']:+.3f}(p={sw['p']:.3f})  0档Overall打平={z}")
        else:
            print(f"  {method:12s} [档位不足 3，暂不算单调性]")

    if not result["missing"]:
        plot(result, OUTPUT_ROOT / "dose_response_curve.png")


if __name__ == "__main__":
    main()
