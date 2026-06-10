"""
MIMIC-CXR No Finding 二分类 Dataset（MEDFAIR 对齐）。

数据来源:
    split CSV 由 scripts/build_mimic_splits_nofinding.py 生成（U-Zeros 标签策略），
    位于 data/splits/mimic_cxr_nofinding/{train,val,test}.csv，每行含:
        image_path : str    图像绝对路径（默认指向 512x512 PNG）
        label      : int    0=有病灶, 1=无病灶/健康 (No Finding Positive)
        sex        : int    0=Male, 1=Female
        race       : int    0=White, 1=Non-White
        age        : float  患者年龄
        view       : str    拍摄体位 (PA / AP)
        patient_id : int    患者 ID（用于核对 split 完整性，训练时不使用）

    age_group 由原始 age 按 age_threshold 在线二分箱得到（< threshold -> 0，
    >= threshold -> 1），阈值来自 configs/mimic_cxr_baseline.yaml 的
    attributes.age.age_threshold（默认 60）。

用法:
    与 UTKFaceCSVDataset 对齐，forward 签名一致，便于直接复用所有
    属性融合模型 (ResNet18AttrConcat / HyperHead / HyperAdapt / HyperFusion)。
"""

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


class MIMICCXRDataset(Dataset):
    """
    Args:
        csv_path      : split CSV 路径 (train.csv / val.csv / test.csv)
        transform     : 图像 transform (作用在 PIL Image 上)
        image_size    : 图像分辨率，'512x512'（默认，与 CSV 中路径一致）或 '224x224'
                        （会将路径中的 '512x512' 替换为 '224x224'）
        age_threshold : age_group 二分箱阈值（< threshold -> 0, >= threshold -> 1）

    每个 __getitem__ 返回: (image, label, sex, race, age_group)
        image     : Tensor [C, H, W]
        label     : int (0/1)  No Finding 二分类标签 (U-Zeros 策略)
                        0: 有病灶（Unknown/未提及 视为有病灶）
                        1: 无病灶/健康（No Finding Positive）
        sex       : int (0/1)  0=Male, 1=Female
        race      : int (0/1)  0=White, 1=Non-White
        age_group : int (0/1)  0: age < age_threshold, 1: age >= age_threshold
    """

    VALID_IMAGE_SIZES = ("512x512", "224x224")

    def __init__(
        self,
        csv_path: str | Path,
        transform=None,
        image_size: str = "512x512",
        age_threshold: int = 60,
    ):
        if image_size not in self.VALID_IMAGE_SIZES:
            raise ValueError(f"image_size 必须是 {self.VALID_IMAGE_SIZES} 之一，得到 {image_size!r}")

        df = pd.read_csv(csv_path)
        required_cols = {"image_path", "label", "sex", "race", "age"}
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(f"CSV 缺少必要的列: {missing}")

        self.image_paths = df["image_path"].astype(str).values
        self.labels = df["label"].astype(np.int64).values
        self.sexes = df["sex"].astype(np.int64).values
        self.races = df["race"].astype(np.int64).values
        self.age_groups = (df["age"].astype(float).values >= age_threshold).astype(np.int64)

        self.transform = transform
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.labels)

    def _resolve_image_path(self, image_path: str) -> Path:
        """把 CSV 中的 512x512 路径按需替换为 224x224 分辨率路径。"""
        path = Path(image_path)
        if self.image_size == "224x224":
            path = Path(str(path).replace("512x512", "224x224"))
        return path

    def __getitem__(self, idx: int):
        path = self._resolve_image_path(self.image_paths[idx])
        # CXR7-1M 的 PNG 是 16-bit 图像（PIL mode 'I;16'，真实像素范围 0~65535）。
        # 直接 .convert("L") 会触发 PIL 有损的 I;16->L 转换，把 ~3 万灰度级压成
        # 10~120 级（严重色阶断裂），导致模型欠拟合、AUC 卡在 ~0.67。
        # 正确做法：按真实动态范围逐图 min-max 归一化到 [0,1] 后转 8-bit，再交给下游
        # transform（Grayscale(3) 会复制为 3 通道以兼容 ImageNet 预训练 backbone）。
        arr = np.array(Image.open(path)).astype(np.float32)
        arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
        image = Image.fromarray((arr * 255.0).astype(np.uint8), mode="L")

        if self.transform is not None:
            image = self.transform(image)

        return (
            image,
            int(self.labels[idx]),
            int(self.sexes[idx]),
            int(self.races[idx]),
            int(self.age_groups[idx]),
        )


# ============================================================
# 自检
# ============================================================
if __name__ == "__main__":
    from torchvision import transforms

    SPLIT_DIR = Path(
        "/vol/biomedic2/bglocker_studproj/zc125/code/hypernetworks_MSc_IP/data/splits/mimic_cxr_nofinding"
    )

    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    ds = MIMICCXRDataset(csv_path=SPLIT_DIR / "val.csv", transform=transform)
    print(f"Dataset size: {len(ds)}")

    image, label, sex, race, age_group = ds[0]
    print(f"Image     : shape={tuple(image.shape)}, dtype={image.dtype}")
    print(f"            min={image.min():.3f}, max={image.max():.3f}")
    print(f"Label     : {label}  (0=有病灶, 1=无病灶/健康)")
    print(f"Sex       : {sex}  (0=Male, 1=Female)")
    print(f"Race      : {race}  (0=White, 1=Non-White)")
    print(f"Age group : {age_group}  (0=<60, 1=>=60)")

    print("\n--- Label distribution ---")
    unique, counts = np.unique(ds.labels, return_counts=True)
    print(f"Label     : {dict(zip(unique.tolist(), counts.tolist()))}")
    unique, counts = np.unique(ds.sexes, return_counts=True)
    print(f"Sex       : {dict(zip(unique.tolist(), counts.tolist()))}")
    unique, counts = np.unique(ds.races, return_counts=True)
    print(f"Race      : {dict(zip(unique.tolist(), counts.tolist()))}")
    unique, counts = np.unique(ds.age_groups, return_counts=True)
    print(f"Age group : {dict(zip(unique.tolist(), counts.tolist()))}")
