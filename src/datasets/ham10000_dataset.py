"""
HAM10000 malignant 二分类 Dataset（MEDFAIR 对齐）。

数据来源:
    split CSV 由 scripts/build_ham10000_splits.py 生成，位于
    data/splits/ham10000/{train,val,test}.csv，每行含:
        image_path : str    图像绝对路径（指向 images/{image_id}.jpg，600×450 RGB JPEG）
        label      : int    0=benign, 1=malignant（malignant={akiec, mel}，Maron 2019/MEDFAIR）
        sex        : int    0=Male, 1=Female（敏感属性）
        age        : float  原始年龄（连续值，备用）
        age_group  : int    敏感属性年龄组：20-40=0/40-60=1/60-80=2/80+=3；0-20=-1（排除组）
        dx / dx_type / lesion_id / image_id : 备用列（训练时不使用）

与 Fitzpatrick17k Dataset 的关键差异:
    1. 源图为 600×450（≠224），需 resize：transform 里走 MEDFAIR 的
       Resize(256,256)→Crop(224)（Fitzpatrick 源图已 224 故省去 resize）。两者都是自然 RGB，
       直接 .convert("RGB") 加载即可，**不需要** CXR 那套 16-bit min-max 归一化。
    2. 敏感属性为 sex（2 类）+ age_group（4 有效组）两轴，Fitzpatrick 为单一肤色轴 6 子群。

用法:
    每个 __getitem__ 返回 (image, label, sex, age_group)。属性融合模型迁到 HAM 时需把
    forward 适配为 (image, sex, age_group) 双属性通路（参照 Fitzpatrick 单属性适配的方式）；
    baseline（image-only）训练不受影响。

    age_group==-1（0-20 排除组）样本仍保留在 CSV 中（供 sex 任务使用全部 9948 张）；
    做「按年龄分组」的训练/公平性评估时应在上层过滤 age_group>=0（对齐 MEDFAIR 排除 0-20）。
"""

from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset

import numpy as np
import pandas as pd


class HAM10000Dataset(Dataset):
    """
    Args:
        csv_path  : split CSV 路径 (train.csv / val.csv / test.csv)
        transform : 图像 transform (作用在 PIL Image 上)

    每个 __getitem__ 返回: (image, label, sex, age_group)
        image     : Tensor [C, H, W]
        label     : int (0/1)  0=benign, 1=malignant
        sex       : int (0/1)  0=Male, 1=Female
        age_group : int        20-40=0/40-60=1/60-80=2/80+=3；0-20=-1（排除组）
    """

    def __init__(self, csv_path: str | Path, transform=None):
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

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        # HAM10000 images/ 为自然 RGB JPEG（600×450, uint8, 0~255），
        # 直接 convert("RGB") 即可——无需 CXR 的 16-bit min-max 处理。
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
        "/vol/biomedic2/bglocker_studproj/zc125/code/hypernetworks_MSc_IP/data/splits/ham10000"
    )

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    ds = HAM10000Dataset(csv_path=SPLIT_DIR / "val.csv", transform=transform)
    print(f"Dataset size: {len(ds)}")

    image, label, sex, age_group = ds[0]
    print(f"Image     : shape={tuple(image.shape)}, dtype={image.dtype}")
    print(f"            min={image.min():.3f}, max={image.max():.3f}")
    print(f"Label     : {label}  (0=benign, 1=malignant)")
    print(f"Sex       : {sex}  (0=Male, 1=Female)")
    print(f"Age group : {age_group}  (20-40=0/40-60=1/60-80=2/80+=3; 0-20=-1)")

    print("\n--- 分布 ---")
    for name, arr in (("Label", ds.labels), ("Sex", ds.sexes), ("AgeGroup", ds.age_groups)):
        unique, counts = np.unique(arr, return_counts=True)
        print(f"{name:<9s}: {dict(zip(unique.tolist(), counts.tolist()))}")
