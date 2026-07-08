"""
MIMIC-CXR + 合成侧信道属性 A_syn 的 Dataset（评审补强 R1.3）
=========================================================
在 MIMIC-CXR No Finding 二分类之上，为每个样本附加**确定性**的合成属性
`A_syn = Y ⊕ Bernoulli(η)`（见 src/datasets/synthetic_attribute.py）。图像与标签完全沿用
MIMICCXRDataset（同 16-bit 加载、同 transform），**A_syn 不改动图像**、仅作 HN 的条件输入
与公平性子群键。

**确定性契约（正确性核心）**：A_syn 只依赖 (η, split, 样本在该 split CSV 中的行序)，由
`make_split_rng(eta, split)` 播种一次性生成。因此训练循环、conditional-MI 标定（R1.2）、
剂量-反应汇总（R1.4）读到的 A_syn 逐元素一致——三处必须对齐，否则信号强度对不上。
CSV 行序即 pandas 读入顺序（与 MIMICCXRDataset 一致），DataLoader shuffle 只打乱**取用顺序**、
不改变行↔A_syn 绑定（A_syn 按 idx 索引）。

每个 __getitem__ 返回 (image, label, a_syn)，与单属性 HN 通路（forward_asyn / *Age(num_age=2)）对齐。
"""

from pathlib import Path

import numpy as np

from src.datasets.mimic_cxr_dataset import MIMICCXRDataset
from src.datasets.synthetic_attribute import inject_synthetic_attribute, make_split_rng


class MIMICSynthDataset(MIMICCXRDataset):
    """
    Args:
        csv_path   : MIMIC split CSV 路径（train/val/test.csv，字段同 MIMICCXRDataset）。
        eta        : 合成属性翻转率 η ∈ [0, 0.5]（信号强度，见 R1.2 标定表）。
        split      : 'train'/'val'/'test'，并入 A_syn 播种，使三集 A_syn 独立且确定。
        transform  : 图像 transform（同 MIMICCXRDataset）。
        image_size : '512x512'（默认）或 '224x224'。
        age_threshold: age 二分箱阈值（仅为兼容父类签名；本任务不使用 age 作为条件）。

    每个 __getitem__ 返回 (image, label, a_syn)：
        image : Tensor [C,H,W]
        label : int (0/1)  No Finding 标签
        a_syn : int (0/1)  合成侧信道属性 = label ⊕ Bernoulli(η)
    """

    def __init__(
        self,
        csv_path: str | Path,
        eta: float,
        split: str,
        transform=None,
        image_size: str = "512x512",
        age_threshold: int = 60,
    ):
        super().__init__(csv_path, transform=transform, image_size=image_size,
                         age_threshold=age_threshold)
        self.eta = eta
        self.split = split
        # 一次性确定性生成 A_syn（按 CSV 行序，与 idx 对齐）
        self.a_syn = inject_synthetic_attribute(
            self.labels, eta, make_split_rng(eta, split)
        ).astype(np.int64)

    def __getitem__(self, idx: int):
        # 复用父类的 16-bit 图像加载 + transform，只取 image 与 label，附加 A_syn
        image, label, _sex, _race, _age = super().__getitem__(idx)
        return image, label, int(self.a_syn[idx])


# ============================================================
# 自检：确定性 + A_syn 与标签的经验翻转率 ≈ η
# ============================================================
if __name__ == "__main__":
    REPO_ROOT = Path(__file__).resolve().parents[2]
    SPLIT_DIR = REPO_ROOT / "data" / "splits" / "mimic_cxr_nofinding"

    for eta in (0.5, 0.3621, 0.2096):
        ds = MIMICSynthDataset(SPLIT_DIR / "val.csv", eta=eta, split="val", transform=None)
        emp_flip = float(np.mean(ds.a_syn != ds.labels))
        # 确定性：重建一次应逐元素一致
        ds2 = MIMICSynthDataset(SPLIT_DIR / "val.csv", eta=eta, split="val", transform=None)
        assert np.array_equal(ds.a_syn, ds2.a_syn), "A_syn 必须确定"
        print(f"η={eta:.4f}  n={len(ds):,}  经验翻转率={emp_flip:.4f}（应≈η）  确定性 ✓")
