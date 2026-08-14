"""
测试集预测落盘（比较协议 A0 延伸，服务 A2/A4/A5/D）
================================================
A2 显著性（多-seed CI / 配对 bootstrap / DeLong）、A4 ROC 后处理、A5 OOD、D 汇总，都需要**每个 run
在 test 集上的逐样本预测**（y_true、模型分数、敏感属性）。训练 harness 的 `run_training` 原本只在
test 集打印指标、不落盘预测，构成公共缺口。本模块提供预测的**统一存取**：

  - `save_predictions` / `load_predictions`：单 run 的 (y_true, y_score, attrs) ↔ .npz。
  - `default_prediction_path`：按 run_id + selection 命名，与 checkpoint / val 日志同址可寻。

**存什么分数**：存**logits**（原始模型输出）。AUC / DeLong 对单调变换不敏感（logits 与 sigmoid 概率
等价）；ROC 后处理需概率，可在消费侧用 `roc.logits_to_prob` 还原。故存 logits 是无损、通用的选择。

**attrs 编码**：把属性字典（{sex/race/age/skin 的子集 -> [N] int 数组}）逐键存为 `attr_<name>`，
并存一份键清单 `attr_keys`，读回时还原为同键字典——与 subgroup_auc_vector / 各 fairness 模块的 kwargs
完全对齐，可直接 splat。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.training.harness.run import Selection, SELECTIONS, run_id


def default_prediction_path(
    output_dir: Path | str, method: str, config_tag: str, seed: int, selection: Selection,
) -> Path:
    """
    某 run、某 model selection 的 test 预测路径：
    `<output_dir>/predictions/<run_id>_<selection>.npz`。

    与 checkpoint（`<run_id>_best_<selection>.pth`）、val 日志（`val_logs/<run_id>.jsonl`）同一 run_id
    基名，A2/A4/A5/D 可按 (method, config, seed, selection) 无歧义寻址。

    Args:
        output_dir: 该数据集 outputs 根。
        method / config_tag / seed: 见 run.run_id。
        selection : "overall" / "worstcase"。

    Returns:
        .npz 路径（不创建目录）。
    """
    if selection not in SELECTIONS:
        raise ValueError(f"未知 selection {selection!r}，合法：{SELECTIONS}")
    return Path(output_dir) / "predictions" / f"{run_id(method, config_tag, seed)}_{selection}.npz"


def val_prediction_path(
    output_dir: Path | str, method: str, config_tag: str, seed: int,
) -> Path:
    """
    某 run 的 **val** 预测路径：`<output_dir>/predictions/<run_id>_val_overall.npz`。

    ROC 后处理（A4 / C*.8）需 val 预测定 deprived 子群与选 margin θ。val 预测始终取 overall-selection
    checkpoint，语义单一，故不走 default_prediction_path 的两 selection 校验（overall/worstcase），
    单独命名以免污染 SELECTIONS 遍历（checkpoint id 生成等）。

    Args:
        output_dir: 该数据集 outputs 根。
        method / config_tag / seed: 见 run.run_id。

    Returns:
        .npz 路径（不创建目录）。
    """
    return Path(output_dir) / "predictions" / f"{run_id(method, config_tag, seed)}_val_overall.npz"


def save_predictions(
    path: Path | str,
    y_true: np.ndarray,
    y_score: np.ndarray,
    attrs: dict[str, np.ndarray],
    extra: dict[str, np.ndarray] | None = None,
) -> None:
    """
    把单 run 的 test 预测存为 .npz（不存在则建目录）。

    Args:
        path   : 目标 .npz（通常来自 default_prediction_path）。
        y_true : [N] 0/1 真实标签。
        y_score: [N] 模型 logits（原始输出，见模块 docstring）。
        attrs  : {属性名 -> [N] 数组}（sex/race/age/skin 的子集）。**仅放会被 splat 进
                 subgroup_auc_vector / fairness 模块的分组属性**。
        extra  : 可选的**逐样本附加列**（{名 -> [N] 数组}），存为 `extra_<名>`，**不进 attrs**、
                 不会被 splat 成 fairness kwargs。用途：cluster bootstrap 需要的聚类键
                 （CXR `patient_id` / HAM `lesion_id`）——条件化范围消融 plan §3.3 要求
                 「患者/病灶级配对 cluster bootstrap」，而 patient_id 不是分组属性，
                 混进 attrs 会让 subgroup_auc_vector 收到非法 kwarg。
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, np.ndarray] = {
        "y_true": np.asarray(y_true),
        "y_score": np.asarray(y_score),
        "attr_keys": np.array(list(attrs.keys()), dtype=object),
    }
    for k, v in attrs.items():
        payload[f"attr_{k}"] = np.asarray(v)
    if extra:
        payload["extra_keys"] = np.array(list(extra.keys()), dtype=object)
        for k, v in extra.items():
            payload[f"extra_{k}"] = np.asarray(v)
    np.savez(path, **payload)


def load_extra(path: Path | str) -> dict[str, np.ndarray]:
    """
    读回 save_predictions 的 `extra` 附加列（无则返回空字典）。

    与 load_predictions 分开是为**向后兼容**：既有 .npz 无 `extra_keys` 键，load_predictions
    的 3 元组签名保持不变，既有消费方（cv_oof_report / significance / roc）零改动。

    Args:
        path: .npz 路径。

    Returns:
        {名 -> [N] 数组}；文件不含 extra 时为 {}。
    """
    with np.load(path, allow_pickle=True) as data:
        if "extra_keys" not in data:
            return {}
        keys = [str(k) for k in data["extra_keys"]]
        return {k: data[f"extra_{k}"] for k in keys}


def load_predictions(
    path: Path | str,
) -> "tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]":
    """
    读回单 run 的 test 预测。

    Args:
        path: .npz 路径。

    Returns:
        (y_true, y_score, attrs_dict)；attrs_dict 键与保存时一致，可直接 splat 进 subgroup_auc_vector。
    """
    with np.load(path, allow_pickle=True) as data:
        y_true = data["y_true"]
        y_score = data["y_score"]
        keys = [str(k) for k in data["attr_keys"]]
        attrs = {k: data[f"attr_{k}"] for k in keys}
    return y_true, y_score, attrs


# ============================================================
# self-test：round-trip + 路径命名
# ============================================================
def _selftest() -> None:
    """验证 save/load round-trip（含多属性）、路径命名、目录自建。"""
    import tempfile

    rng = np.random.default_rng(0)
    n = 500
    y = rng.integers(0, 2, n)
    score = rng.standard_normal(n)
    attrs = {"sex": rng.integers(0, 2, n), "race": rng.integers(0, 2, n), "age": rng.integers(0, 2, n)}

    with tempfile.TemporaryDirectory() as td:
        p = default_prediction_path(td, "hyperfusion", "lr1e-04_wd1e-04", 44, "overall")
        assert p.name == "hyperfusion_lr1e-04_wd1e-04_seed44_overall.npz", p.name
        assert p.parent.name == "predictions"
        assert not p.parent.exists()          # 尚未创建
        save_predictions(p, y, score, attrs)
        assert p.exists()                      # 目录+文件已建

        y2, s2, a2 = load_predictions(p)
        assert np.array_equal(y2, y) and np.allclose(s2, score)
        assert set(a2.keys()) == set(attrs.keys())
        for k in attrs:
            assert np.array_equal(a2[k], attrs[k]), k

        # 单属性数据集（fitzpatrick: skin）
        p2 = default_prediction_path(td, "erm", "lr3e-05_wd1e-03", 42, "worstcase")
        save_predictions(p2, y, score, {"skin": rng.integers(0, 6, n)})
        _, _, a3 = load_predictions(p2)
        assert list(a3.keys()) == ["skin"]

        # 两 selection 不撞名
        assert default_prediction_path(td, "erm", "lr3e-05_wd1e-03", 42, "overall") != p2

        # 非法 selection 报错
        try:
            default_prediction_path(td, "erm", "lr3e-05_wd1e-03", 42, "bogus")  # type: ignore[arg-type]
            raise AssertionError("非法 selection 应报错")
        except ValueError:
            pass

        # --- extra 附加列（cluster bootstrap 的 patient_id / lesion_id）---
        assert load_extra(p) == {}, "未写 extra 的文件应返回空字典（向后兼容）"
        pid = rng.integers(0, 100, n)
        p3 = default_prediction_path(td, "erm", "lr1e-04_wd1e-04", 42, "overall")
        save_predictions(p3, y, score, attrs, extra={"patient_id": pid})
        # extra 不污染 attrs（否则会被 splat 成 subgroup_auc_vector 的非法 kwarg）
        _, _, a4 = load_predictions(p3)
        assert set(a4.keys()) == set(attrs.keys()), f"extra 不应混入 attrs，实得 {sorted(a4)}"
        ex = load_extra(p3)
        assert set(ex.keys()) == {"patient_id"} and np.array_equal(ex["patient_id"], pid)

    print("predictions self-test 全部通过 ✓（round-trip 多/单属性 + 路径命名 + 目录自建 + "
          "selection 校验 + extra 附加列与向后兼容）")


if __name__ == "__main__":
    _selftest()
