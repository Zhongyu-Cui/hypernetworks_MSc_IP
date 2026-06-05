"""
UTKFace CSV Dataset
===================
数据来源:
    https://www.kaggle.com/datasets/nipunarora8/age-gender-and-ethnicity-face-data-csv

文件格式:
    单个 CSV 文件 (约 191MB, 27305 行), 列:
        age       : int,    0-116
        gender    : int,    0=male, 1=female
        ethnicity : int,    0=White, 1=Black, 2=Asian, 3=Indian, 4=Others
        img_name  : str,    原文件名 (我们不用)
        pixels    : str,    空格分隔的 2304 个像素值 (48x48 灰度图, 0-255)

使用:
    从 Kaggle 下载后, 解压得到 age_gender.csv, 放在 ./data/ 目录下。
"""

import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


class UTKFaceCSVDataset(Dataset):
    """
    Args:
        csv_path : CSV 文件路径
        transform: 图像 transform (作用在 PIL Image 上)
        cache    : 是否把所有像素一次性解析成 uint8 数组存内存中
                   (默认 True, 会多占约 60MB 内存, 但训练时每个 epoch 省很多时间)

    每个 __getitem__ 返回: (image, age_group, sex, race)
        image     : Tensor [C, H, W]
        age_group : int  (0/1)   年龄分组 (二分类):
                        0: [0, 35)
                        1: [35, +inf)
        sex       : int  (0/1)      -- 对应 CSV 里的 gender 列
        race      : int  (0~4)      -- 对应 CSV 里的 ethnicity 列
    """

    IMG_SIZE = 48  # 数据集固定 48x48 灰度
    AGE_BINS = [35]  # 分界点, 2 组: [0,35), [35,+inf)

    def __init__(self, csv_path, transform=None, cache=True):
        df = pd.read_csv(csv_path)

        # --- 基本合法性过滤 ---
        required_cols = {"age", "gender", "ethnicity", "pixels"}
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(f"CSV 缺少必要的列: {missing}")

        df = df.dropna(subset=["age", "gender", "ethnicity", "pixels"])
        df = df[
            (df["age"].between(0, 120)) &
            (df["gender"].isin([0, 1])) &
            (df["ethnicity"].isin([0, 1, 2, 3, 4]))
        ].reset_index(drop=True)

        self.ages = df["age"].astype(np.int32).values
        # 把 age 离散成 5 个分组 (0-4). np.digitize 左闭右开
        self.age_groups = np.digitize(self.ages, self.AGE_BINS).astype(np.int64)
        self.sexes = df["gender"].astype(np.int64).values
        self.races = df["ethnicity"].astype(np.int64).values
        self.pixels_raw = df["pixels"].values       # np.ndarray of str

        self.transform = transform
        self.cache = cache

        if cache:
            # 预先解析所有图像为 uint8, shape: [N, 48, 48]
            # 每张图 2304 字节, 27k 张约 60MB
            print(f"[UTKFaceCSV] Caching {len(self.pixels_raw)} images into memory...")
            self.images = np.stack(
                [self._parse_pixels(s) for s in self.pixels_raw], axis=0,
            )
            self.pixels_raw = None  # 释放原始字符串占用的内存
            print(f"[UTKFaceCSV] Done. Array shape: {self.images.shape}, "
                  f"dtype: {self.images.dtype}")
        else:
            self.images = None

    @classmethod
    def _parse_pixels(cls, s):
        """把 '0 1 2 ...' 这样的字符串转成 [48, 48] uint8 数组."""
        arr = np.fromstring(s, sep=" ", dtype=np.uint8)
        return arr.reshape(cls.IMG_SIZE, cls.IMG_SIZE)

    def __len__(self):
        return len(self.ages)

    def __getitem__(self, idx):
        # 获取 48x48 灰度 uint8 数组
        if self.cache:
            arr = self.images[idx]
        else:
            arr = self._parse_pixels(self.pixels_raw[idx])

        # 转成 PIL "L" 模式 (灰度), 后续用 Grayscale(3) 扩成 3 通道
        image = Image.fromarray(arr, mode="L")

        if self.transform is not None:
            image = self.transform(image)

        return (
            image,
            int(self.age_groups[idx]),   # 年龄组 (0-4)
            int(self.sexes[idx]),
            int(self.races[idx]),
        )


# ============================================================
# 自检
# ============================================================
if __name__ == "__main__":
    from torchvision import transforms

    CSV_PATH = "./data/age_gender.csv"

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    ds = UTKFaceCSVDataset(csv_path=CSV_PATH, transform=transform)
    print(f"\nDataset size: {len(ds)}")

    img, age_group, sex, race = ds[0]
    print(f"Image : shape={tuple(img.shape)}, dtype={img.dtype}")
    print(f"        min={img.min():.3f}, max={img.max():.3f}")
    print(f"AgeGrp: {age_group}  (age={ds.ages[0]})")
    print(f"Sex   : {sex}  (0=male, 1=female)")
    print(f"Race  : {race} (0=White, 1=Black, 2=Asian, 3=Indian, 4=Others)")

    # 一些分布信息, 有助于检查数据合不合理
    print("\n--- Label distribution ---")
    ages = ds.ages
    print(f"Age     : min={ages.min()}, max={ages.max()}, "
          f"mean={ages.mean():.1f}, median={np.median(ages):.0f}")
    unique, counts = np.unique(ds.age_groups, return_counts=True)
    group_names = ["[0,35)", "[35,+)"]
    print("AgeGrp  :", {group_names[g]: int(c) for g, c in zip(unique, counts)})
    unique, counts = np.unique(ds.sexes, return_counts=True)
    print(f"Sex     : {dict(zip(unique.tolist(), counts.tolist()))}")
    unique, counts = np.unique(ds.races, return_counts=True)
    print(f"Race    : {dict(zip(unique.tolist(), counts.tolist()))}")
