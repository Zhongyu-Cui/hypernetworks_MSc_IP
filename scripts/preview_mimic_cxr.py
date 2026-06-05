"""
MIMIC-CXR 数据集快速预览脚本。

功能：
  1. 打印数据集基本统计（样本量、split 分布、人口统计、标签分布）
  2. 随机抽取若干图像，排成网格保存为 PNG
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

BASE_PATH = Path("/vol/biodata/projects/chai/data")
CSV_PATH = BASE_PATH / "cxr" / "cxr7-1m_master.csv"
OUTPUT_PATH = Path("/vol/biomedic2/bglocker_studproj/zc125/outputs/preview_mimic_cxr.png")

LABEL_COLS = [
    "Atelectasis", "Cardiomegaly", "Consolidation", "Edema",
    "Enlarged Cardiomediastinum", "Fracture", "Lung Lesion", "Lung Opacity",
    "No Finding", "Pleural Effusion", "Pleural Other", "Pneumonia",
    "Pneumothorax", "Support Devices",
]

N_SAMPLE_IMAGES = 12   # 预览图中展示的图像张数
GRID_COLS = 4


def load_mimic_df(csv_path: Path) -> pd.DataFrame:
    """
    从 master CSV 加载并过滤出 MIMIC-CXR 子集。

    Args:
        csv_path: cxr7-1m_master.csv 的完整路径。

    Returns:
        仅含 MIMIC-CXR 行的 DataFrame。
    """
    print(f"读取 CSV：{csv_path}")
    df = pd.read_csv(csv_path, low_memory=False)
    df_mimic = df[df["Dataset_name"] == "MIMIC-CXR"].reset_index(drop=True)
    print(f"MIMIC-CXR 总样本数：{len(df_mimic):,}")
    return df_mimic


def print_statistics(df: pd.DataFrame) -> None:
    """
    打印 split、人口统计与标签的分布统计。

    Args:
        df: MIMIC-CXR DataFrame。
    """
    # Split 分布
    print("\n── Split 分布 ──")
    print(df["Split"].value_counts().to_string())

    # 性别分布
    print("\n── 性别分布 ──")
    print(df["Sex_name"].value_counts().to_string())

    # 体位分布（正位 / 侧位等）
    print("\n── 拍摄体位分布 ──")
    print(df["View_name"].value_counts().to_string())

    # 年龄摘要（仅数值行）
    age_series = pd.to_numeric(df["Age"], errors="coerce").dropna()
    print(f"\n── 年龄摘要（n={len(age_series):,}）──")
    print(age_series.describe().round(1).to_string())

    # 标签正例率（忽略 uncertainty=-1，只统计 1 为正）
    print("\n── 各标签正例率（正例 / 有效标签）──")
    for col in LABEL_COLS:
        if col not in df.columns:
            continue
        valid = df[col][df[col] != -1]  # 排除 uncertainty
        pos_rate = (valid == 1).mean()
        print(f"  {col:<30s}: {pos_rate:.1%}  (n={len(valid):,})")


def load_image(image_path_relative: str, size: str = "512x512") -> np.ndarray:
    """
    从 CXR7-1M 加载单张图像并归一化至 [0, 1]。

    Args:
        image_path_relative: master CSV 中 ImagePath 列的相对路径。
        size: 分辨率字符串，'512x512' 或 '224x224'。

    Returns:
        归一化至 [0, 1] 的 float32 二维 numpy 数组。
    """
    full_path = BASE_PATH / image_path_relative
    if size == "224x224":
        full_path = Path(str(full_path).replace("512x512", "224x224"))
    img = np.array(Image.open(full_path)).astype(np.float32)
    # 逐图归一化，消除不同设备/数据集间的灰度范围差异
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    return img


def save_image_grid(df: pd.DataFrame, n: int, grid_cols: int, output_path: Path) -> None:
    """
    随机抽取 n 张图像，排成网格并保存。

    Args:
        df: MIMIC-CXR DataFrame。
        n: 要展示的图像总数。
        grid_cols: 网格列数。
        output_path: 输出 PNG 路径。
    """
    sample = df.sample(n=n, random_state=42).reset_index(drop=True)
    grid_rows = int(np.ceil(n / grid_cols))

    fig, axes = plt.subplots(grid_rows, grid_cols, figsize=(grid_cols * 3, grid_rows * 3))
    axes = np.array(axes).flatten()

    for idx, row in sample.iterrows():
        ax = axes[idx]
        try:
            img = load_image(row["ImagePath"])
            ax.imshow(img, cmap="gray", vmin=0, vmax=1)
        except FileNotFoundError:
            ax.text(0.5, 0.5, "图像缺失", ha="center", va="center", transform=ax.transAxes)

        # 标注：体位 / 性别 / 年龄
        view = row.get("View_name", "")
        sex = row.get("Sex_name", "")
        age = row.get("Age", "")
        ax.set_title(f"{view} | {sex} | {age}y", fontsize=8)
        ax.axis("off")

    # 隐藏多余子图
    for ax in axes[n:]:
        ax.axis("off")

    plt.suptitle(f"MIMIC-CXR random sample (n={n})", fontsize=12, y=1.01)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\n图像网格已保存至：{output_path}")
    plt.close()


def main() -> None:
    """主函数：加载数据、打印统计、保存图像预览。"""
    df_mimic = load_mimic_df(CSV_PATH)
    print_statistics(df_mimic)
    save_image_grid(df_mimic, n=N_SAMPLE_IMAGES, grid_cols=GRID_COLS, output_path=OUTPUT_PATH)


if __name__ == "__main__":
    main()
