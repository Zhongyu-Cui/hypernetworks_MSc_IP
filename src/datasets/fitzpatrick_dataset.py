"""
Fitzpatrick17k malignant 二分类 Dataset（MEDFAIR 对齐）。

数据来源:
    split CSV 由 scripts/build_fitzpatrick_splits.py 生成，位于
    data/splits/fitzpatrick17k/{train,val,test}.csv，每行含:
        image_path : str    图像绝对路径（指向 preproc_224x224/{md5hash}.jpg）
        label      : int    0=benign/non-neoplastic, 1=malignant
        skin       : int    0–5（Fitzpatrick 肤色 I–VI），敏感属性（6 个子群）
        md5hash    : str    图像唯一标识（训练时不使用）
        three_partition_label / nine_partition_label : str  备用粒度标签（训练时不使用）

与 MIMIC-CXR Dataset 的关键差异:
    Fitzpatrick17k 是自然 RGB 图像（preproc_224x224 为 uint8, 0~255），
    直接 .convert("RGB") 加载即可，**不需要** CXR 那套 16-bit min-max 归一化。

用法:
    每个 __getitem__ 返回 (image, label, skin)。当前 forward 签名为单一敏感属性 skin，
    与 MIMIC 的 (image, label, sex, race, age_group) 不同——属性融合模型尚未做单属性
    接口适配（见仓库 CLAUDE.md），baseline（image-only）训练不受影响。
"""

from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset

import numpy as np
import pandas as pd
from src.paths import SPLITS_DIR


class FitzpatrickDataset(Dataset):
    """
    Args:
        csv_path  : split CSV 路径 (train.csv / val.csv / test.csv)
        transform : 图像 transform (作用在 PIL Image 上)

    每个 __getitem__ 返回: (image, label, skin)
        image : Tensor [C, H, W]
        label : int (0/1)  0=benign/non-neoplastic, 1=malignant
        skin  : int (0–5)  Fitzpatrick 肤色 I–VI 敏感属性
    """

    def __init__(self, csv_path: str | Path, transform=None):
        df = pd.read_csv(csv_path)
        required_cols = {"image_path", "label", "skin"}
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(f"CSV 缺少必要的列: {missing}")

        self.image_paths = df["image_path"].astype(str).values
        self.labels = df["label"].astype(np.int64).values
        self.skins = df["skin"].astype(np.int64).values
        self.transform = transform

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        # Fitzpatrick17k preproc_224x224 为自然 RGB JPEG（uint8, 0~255），
        # 直接 convert("RGB") 即可——无需 CXR 的 16-bit min-max 处理。
        image = Image.open(self.image_paths[idx]).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return (
            image,
            int(self.labels[idx]),
            int(self.skins[idx]),
        )


# ============================================================
# 自检
# ============================================================
if __name__ == "__main__":
    from torchvision import transforms

    SPLIT_DIR = SPLITS_DIR / "fitzpatrick17k"

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    ds = FitzpatrickDataset(csv_path=SPLIT_DIR / "val.csv", transform=transform)
    print(f"Dataset size: {len(ds)}")

    image, label, skin = ds[0]
    print(f"Image : shape={tuple(image.shape)}, dtype={image.dtype}")
    print(f"        min={image.min():.3f}, max={image.max():.3f}")
    print(f"Label : {label}  (0=benign/non-neoplastic, 1=malignant)")
    print(f"Skin  : {skin}  (0–5 = Fitzpatrick I–VI)")

    print("\n--- 分布 ---")
    unique, counts = np.unique(ds.labels, return_counts=True)
    print(f"Label : {dict(zip(unique.tolist(), counts.tolist()))}")
    unique, counts = np.unique(ds.skins, return_counts=True)
    print(f"Skin  : {dict(zip(unique.tolist(), counts.tolist()))}")
