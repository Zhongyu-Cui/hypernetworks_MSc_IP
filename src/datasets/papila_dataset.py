"""
PAPILA 青光眼二分类 Dataset（MEDFAIR 对齐）。

数据来源:
    split CSV 由 scripts/build_papila_splits.py 生成，位于
    data/splits/papila/{train,val,test}.csv，每行含:
        image_path : str    图像绝对路径（默认指向 data/processed 下 256×256 RGB JPEG 缓存；
                            PRERESIZE_256=False 时指向原始 2576×1934 大图）
        label      : int    0=非青光眼, 1=青光眼（排除 suspect，Maron 风格二分类 / MEDFAIR）
        sex        : int    0=Male, 1=Female（敏感属性）
        age        : float  原始年龄（连续值，备用）
        age_group  : int    敏感属性年龄组：<=60->0, >60->1（对齐 MEDFAIR「0-60 / 60+」）
        joint      : int    sex*2+age_group ∈ {0,1,2,3}（交叉子群，备用）
        pid / eye / diagnosis : 备用列（训练时不使用；pid 用于防泄漏审计）

与 HAM10000 Dataset 的关键差异:
    1. 眼底自然 RGB，直接 .convert("RGB")，**不需要** CXR 那套 16-bit 处理（与 HAM/Fitz 同）。
       若 image_path 指向 256 缓存，transform 里的 Resize(256,256) 为幂等空操作。
    2. 敏感属性为 sex（2 类）+ age_group（**仅 2 类**，MEDFAIR 对 PAPILA 即二分），
       区别于 HAM 的 age_group 4 类。
    3. 样本极小（420 眼-图），开启 cache=True 可整库常驻内存（256 缓存约 82 MB）。

用法:
    每个 __getitem__ 返回 (image, label, sex, age_group)。属性融合模型迁到 PAPILA 时把
    forward 适配为 (image, sex, age_group)——即把原 (image, sex, race) 的 race 槽位填 age_group
    （PAPILA 无种族标注，与 HAM 同一惯例）。baseline（image-only）训练不受影响。
"""

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


class PAPILADataset(Dataset):
    """
    Args:
        csv_path  : split CSV 路径 (train.csv / val.csv / test.csv)
        transform : 图像 transform (作用在 PIL Image 上)
        cache     : 是否把解码后的 PIL Image 整库缓存进内存（PAPILA 极小，推荐 True）

    每个 __getitem__ 返回: (image, label, sex, age_group)
        image     : Tensor [C, H, W]
        label     : int (0/1)  0=非青光眼, 1=青光眼
        sex       : int (0/1)  0=Male, 1=Female
        age_group : int (0/1)  <=60=0, >60=1
    """

    def __init__(self, csv_path: str | Path, transform=None, cache: bool = False):
        df = pd.read_csv(csv_path)
        required_cols = {"image_path", "label", "sex", "age_group"}
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(f"CSV 缺少必要的列: {missing}")

        self.image_paths = df["image_path"].astype(str).values
        self.labels = df["label"].astype(np.int64).values
        self.sexes = df["sex"].astype(np.int64).values
        self.age_groups = df["age_group"].astype(np.int64).values
        self.transform = transform
        self.cache = cache
        # 整库内存缓存（PIL RGB Image），仅在 cache=True 时填充
        self._image_cache: list[Image.Image] | None = None
        if cache:
            self._image_cache = [
                Image.open(p).convert("RGB") for p in self.image_paths
            ]

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        # 眼底为自然 RGB JPEG，直接 convert("RGB")——无需 CXR 的 16-bit min-max 处理。
        if self._image_cache is not None:
            image = self._image_cache[idx]
        else:
            image = Image.open(self.image_paths[idx]).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return (
            image,
            int(self.labels[idx]),
            int(self.sexes[idx]),
            int(self.age_groups[idx]),
        )


# ============================================================
# 自检
# ============================================================
if __name__ == "__main__":
    from torchvision import transforms

    SPLIT_DIR = Path(
        "/vol/biomedic2/bglocker_studproj/zc125/code/hypernetworks_MSc_IP"
        "/data/splits/papila"
    )
    eval_tf = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    for split in ("train", "val", "test"):
        ds = PAPILADataset(SPLIT_DIR / f"{split}.csv", transform=eval_tf, cache=True)
        img, label, sex, age_group = ds[0]
        print(
            f"{split:<6s} n={len(ds):>3d}  sample: img={tuple(img.shape)} "
            f"label={label} sex={sex} age_group={age_group}"
        )
