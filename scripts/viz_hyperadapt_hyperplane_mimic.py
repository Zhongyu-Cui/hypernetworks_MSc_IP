"""
HyperAdapt(sex,race,age) 分类超平面的逐子群可视化（MIMIC-CXR，8 子群）
=====================================================================
与 scripts/viz_hyperadapt_hyperplane_ham.py 同方法，换成 MIMIC-CXR No Finding 二分类、
三属性条件通路（PatientEmbedding: sex × race × age_group = **8 个子群**）。

流程实现全部在 src/utils/hyperplane_cxr_runner.py（与 CheXpert 共用，两库在本可视化上同构：
同一 Dataset 类 / 同一模型类 / 同一 eval transform，仅 config、split、checkpoint、输出路径不同）；
数学与渲染在 src/utils/hyperplane_viz.py；方法学规格见 docs/hyperplane_subgroup_visualisation.md。

运行（medimg env，实验室机器即可，纯推理）：
    PYTHONPATH=. python scripts/viz_hyperadapt_hyperplane_mimic.py \
        --ckpt "$HN_OUTPUTS"/mimic_cxr/cv5/hyperadapt_lr1e-04_wd1e-03_seed42_best_overall.pth \
        --split_dir data/splits/mimic_cxr_nofinding/cv5/fold0

输出：outputs/mimic_cxr/hyperplane_viz/ 下 3 张 PNG + metrics JSON + arrays npz。
"""

from __future__ import annotations

from pathlib import Path

from src.utils.hyperplane_cxr_runner import CXRVizSpec, parse_args, run_hyperplane_viz
from src.paths import OUTPUTS_DIR

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS_ROOT = OUTPUTS_DIR

SPEC = CXRVizSpec(
    dataset_key="mimic_cxr",
    display_name="MIMIC-CXR",
    config_path=REPO_ROOT / "configs" / "mimic_cxr_baseline.yaml",
    # cv5 选中配置（cv5/selected_configs.json: hyperadapt = lr1e-04_wd1e-03），seed42 ↔ fold0
    default_ckpt=(OUTPUTS_ROOT / "mimic_cxr" / "cv5"
                  / "hyperadapt_lr1e-04_wd1e-03_seed42_best_overall.pth"),
    default_split_dir="data/splits/mimic_cxr_nofinding/cv5/fold0",
    default_out_dir=OUTPUTS_ROOT / "mimic_cxr" / "hyperplane_viz",
    default_tag="mimic_fold0_seed42",
)


def main() -> None:
    """解析 CLI 并运行 MIMIC-CXR 的超平面可视化流程。"""
    run_hyperplane_viz(SPEC, parse_args(SPEC))


if __name__ == "__main__":
    main()
