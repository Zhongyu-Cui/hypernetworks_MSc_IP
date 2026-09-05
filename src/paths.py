"""
统一路径解析：把仓库内所有绝对路径集中到这一处
=================================================

改造前各脚本把开发机的绝对路径（`/vol/biomedic2/.../zc125/...`）写死在文件里，仓库
换一台机器就跑不动。本模块把路径分成三类，全部支持环境变量覆盖，缺省值与原开发机
布局一致，因此既有作业脚本与既有产物路径不受影响：

1. **仓库内路径**（`REPO_ROOT` / `SPLITS_DIR`）——由本文件位置推导，永远正确，无需配置。
2. **产物路径**（`OUTPUTS_DIR` / `LOGS_DIR`）——checkpoint、预测与分析 JSON 的写入位置。
   体积大，按项目约定放在仓库外；用 `HN_WORK_ROOT` 指到自己的工作目录即可。
3. **只读原始数据路径**（`CXR_DATA_ROOT` 等）——各公开数据集在本机的位置，因机器而异，
   每个数据集单独一个环境变量。

环境变量（全部可选，未设时用缺省值）：

===================  ==========================================================
`HN_WORK_ROOT`       产物根目录，缺省 = 仓库的祖父目录（即 `<root>/code/<repo>` 的 `<root>`）
`HN_OUTPUTS`         直接指定 outputs 目录，优先于 `HN_WORK_ROOT`
`HN_LOGS`            直接指定 logs 目录，优先于 `HN_WORK_ROOT`
`HN_CXR_ROOT`        CXR7-1M 根（含 `cxr/cxr7-1m_master.csv`，MIMIC-CXR 与 CheXpert 图像）
`HN_CHEXPERT_META`   CheXpert-Plus 元数据目录（含 `chexbert_labels.zip`）
`HN_HAM_ROOT`        HAM10000 根（含 `HAM10000_metadata.csv` 与 `images/`）
`HN_FITZ_ROOT`       Fitzpatrick17k 根（含 `fitzpatrick17k.csv` 与 `preproc_224x224/`）
===================  ==========================================================

用法::

    from src.paths import OUTPUTS_DIR, SPLITS_DIR
    ckpt_dir = OUTPUTS_DIR / "ham10000" / "cv5"
"""

from __future__ import annotations

import os
from pathlib import Path

__all__ = [
    "REPO_ROOT", "SPLITS_DIR", "WORK_ROOT", "OUTPUTS_DIR", "LOGS_DIR",
    "CXR_DATA_ROOT", "CHEXPERT_META_ROOT", "HAM10000_ROOT", "FITZPATRICK_ROOT",
]


def _env_path(name: str, default: Path) -> Path:
    """读环境变量作为路径，未设或为空串时返回 default（不检查存在性，便于纯分析场景）。"""
    raw = os.environ.get(name, "").strip()
    return Path(raw).expanduser().resolve() if raw else default


# ------------------------------------------------------------------
# 1. 仓库内路径：由本文件位置推导（src/paths.py -> 仓库根）
# ------------------------------------------------------------------
REPO_ROOT: Path = Path(__file__).resolve().parents[1]
SPLITS_DIR: Path = REPO_ROOT / "data" / "splits"

# ------------------------------------------------------------------
# 2. 产物路径：缺省沿用「<work_root>/code/<repo>」的项目布局，即仓库的祖父目录
# ------------------------------------------------------------------
WORK_ROOT: Path = _env_path("HN_WORK_ROOT", REPO_ROOT.parents[1])
OUTPUTS_DIR: Path = _env_path("HN_OUTPUTS", WORK_ROOT / "outputs")
LOGS_DIR: Path = _env_path("HN_LOGS", WORK_ROOT / "logs")

# ------------------------------------------------------------------
# 3. 只读原始数据路径：缺省值为原开发机（Imperial BioMedIA 共享盘）上的位置
# ------------------------------------------------------------------
# CXR7-1M：MIMIC-CXR 与 CheXpert-Plus 的图像与 master CSV 均在此根下
CXR_DATA_ROOT: Path = _env_path("HN_CXR_ROOT", Path("/vol/biodata/projects/chai/data"))
# CheXpert-Plus 的 CheXbert 标签压缩包所在目录（impression 段标签来源）
CHEXPERT_META_ROOT: Path = _env_path(
    "HN_CHEXPERT_META", Path("/vol/biodata/data/chest_xray/CheXpert-Plus")
)
HAM10000_ROOT: Path = _env_path("HN_HAM_ROOT", Path("/vol/biodata/data/HAM10000"))
FITZPATRICK_ROOT: Path = _env_path("HN_FITZ_ROOT", Path("/vol/biodata/data/fitzpatrick17k"))
