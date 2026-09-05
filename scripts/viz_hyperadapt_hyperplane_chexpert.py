"""
HyperAdapt(sex,race,age) 分类超平面的逐子群可视化（CheXpert，8 子群）
====================================================================
与 scripts/viz_hyperadapt_hyperplane_mimic.py **完全同一套流程**（共享
src/utils/hyperplane_cxr_runner.py）：CheXpert 与 MIMIC 来自同一份 CXR7-1M master CSV、
schema 同构，共用 MIMICCXRDataset 与 ResNet18HyperAdapt(sex×race×age = 8 子群)。

CheXpert 特有注意
-----------------
· 标签为官方 CheXbert **impression 段**口径，正例（健康）率仅 ~8%（MIMIC ~30%）；
  ⇒ 分层抽样的 label=1 格在小子群里会取满不足配额（如 Female/Non-White/>=60 全 split 仅 86 张），
  各格实际样本数不等，逐组 AUC 是在这个**平衡子样本**上算的，**不等于** split 全量指标。
· 16-bit PNG 坑与 MIMIC 相同，由 MIMICCXRDataset 内部处理（禁用 convert("L")）。
· cv5 选中配置为 lr3e-04_wd1e-04（见 outputs/chexpert_cxr/cv5/selected_configs.json），
  与 MIMIC 的 lr1e-04_wd1e-03 不同；seed42 ↔ fold0，须用对应 fold 的 held-out test。

运行（medimg env，实验室机器即可，纯推理）：
    PYTHONPATH=. python scripts/viz_hyperadapt_hyperplane_chexpert.py \
        --ckpt "$HN_OUTPUTS"/chexpert_cxr/cv5/hyperadapt_lr3e-04_wd1e-04_seed42_best_overall.pth \
        --split_dir data/splits/chexpert_nofinding/cv5/fold0

输出：outputs/chexpert_cxr/hyperplane_viz/ 下 3 张 PNG + metrics JSON + arrays npz。
"""

from __future__ import annotations

from pathlib import Path

from src.utils.hyperplane_cxr_runner import CXRVizSpec, parse_args, run_hyperplane_viz
from src.paths import OUTPUTS_DIR

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS_ROOT = OUTPUTS_DIR

SPEC = CXRVizSpec(
    dataset_key="chexpert_cxr",
    display_name="CheXpert",
    config_path=REPO_ROOT / "configs" / "chexpert_baseline.yaml",
    # cv5 选中配置（cv5/selected_configs.json: hyperadapt = lr3e-04_wd1e-04），seed42 ↔ fold0
    default_ckpt=(OUTPUTS_ROOT / "chexpert_cxr" / "cv5"
                  / "hyperadapt_lr3e-04_wd1e-04_seed42_best_overall.pth"),
    default_split_dir="data/splits/chexpert_nofinding/cv5/fold0",
    default_out_dir=OUTPUTS_ROOT / "chexpert_cxr" / "hyperplane_viz",
    default_tag="chexpert_fold0_seed42",
)


def main() -> None:
    """解析 CLI 并运行 CheXpert 的超平面可视化流程。"""
    run_hyperplane_viz(SPEC, parse_args(SPEC))


if __name__ == "__main__":
    main()
