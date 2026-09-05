"""
Fitzpatrick17k 预处理（MEDFAIR 对齐版）：生成 malignant 二分类任务的 train/val/test split CSV。

与 MEDFAIR（Zong et al., ICLR 2023）的对应关系
─────────────────────────────────────────────
  目标标签  : malignant（Positive=1 表示「恶性」；non-neoplastic / benign 合并为 0）
               —— 对齐 MEDFAIR：将 three_partition_label 三分类压成二分类，
               "non-neoplastic" 与 "benign" → benign(0)，"malignant" → malignant(1)。
  敏感属性  : fitzpatrick_scale 肤色 I–VI（编码 0–5，共 6 个子群），
               排除 -1（未知，缺失敏感属性）—— 与 MEDFAIR 一致，排除后恰好 16,012 行
               （论文 Table 1）。MEDFAIR 用的是 fitzpatrick_scale（非 fitzpatrick_centaur），
               已由排除 -1 后的计数 16,012 与各型 %Images / %Malignant（Table A4）逐一反推确证。
  数据划分  : 图像级随机 80/10/10（论文 B.1.2「randomly split」，2D 数据集 80/10/10）。
               Fitzpatrick17k 无 patient_id（每个 md5hash 唯一），无法做患者级去泄漏，
               与 MIMIC/CheXpert 的患者级 split 不同——这是数据集本身的限制。
  图像变换  : 见 configs/fitzpatrick_baseline.yaml。本项目采用「224 原生」方案：
               源图已是 preproc_224x224，不做 MEDFAIR 的 Resize(256)→Crop(224)（会先上采样
               再裁剪，引入无信息插值），直接在 224 上 RandomHorizontalFlip + RandomRotation(±15°)；
               ImageNet 均值/标准差归一化。皮损无解剖学左右约束，故保留水平翻转（与 MEDFAIR 一致，
               但与本项目 CXR 流程「不翻转」相反）。

与 MIMIC No Finding 版（build_mimic_splits_nofinding.py）的关键差异
─────────────────────────────────────────────────────────────────
  1. 模态：自然 RGB JPEG（uint8, 0~255），无需 CXR 那套 16-bit min-max 加载。
  2. 敏感属性：单一肤色轴 6 个子群（vs sex×race×age 三个二值轴的交叉）。
  3. 划分：图像级随机 80/10/10（vs CXR7-1M 内置患者级 Split 列）。
  4. 输出目录：data/splits/fitzpatrick17k/ vs data/splits/mimic_cxr_nofinding/。

MEDFAIR 偏差（对照论文指标时需关注）
──────────────────────────────────
  图像分辨率：MEDFAIR 在 256×256 上 RandomCrop(224)；本项目源图为 224，直接 224 原生增强
    （省去无意义上采样）。两者差异极小，但精确复现 AUC 时需注意。
  QC 过滤：本脚本默认「不过滤」qc 列，以对齐论文计数 16,012（论文未做 QC 过滤）。
    如需更干净的训练集，可置 DROP_QC_WRONGLY_LABELLED=True 丢弃 17 张 "3 Wrongly labelled"，
    但此时样本数将不再等于 16,012。

输出
────
  data/splits/fitzpatrick17k/{train,val,test}.csv
  每行字段：image_path（绝对路径）, label, skin, md5hash,
            three_partition_label, nine_partition_label
"""

from pathlib import Path

import numpy as np
import pandas as pd
from src.paths import FITZPATRICK_ROOT, SPLITS_DIR

RANDOM_STATE = 42

BASE_PATH = FITZPATRICK_ROOT
CSV_PATH = BASE_PATH / "fitzpatrick17k.csv"
IMG_DIR = BASE_PATH / "preproc_224x224"   # 已预处理 224x224 RGB JPEG（uint8）
OUTPUT_DIR = SPLITS_DIR / "fitzpatrick17k"

# three_partition_label → 二值 label（malignant=1，其余=0）
MALIGNANT_LABEL = "malignant"

# 肤色子群名（编码 0–5 对应 Fitzpatrick I–VI）
SKIN_NAMES = {0: "I", 1: "II", 2: "III", 3: "IV", 4: "V", 5: "VI"}

# 划分比例（2D 数据集，MEDFAIR 80/10/10）
SPLIT_FRACTIONS = {"train": 0.8, "val": 0.1, "test": 0.1}

# 默认不丢弃 QC 标记，以对齐论文计数 16,012
DROP_QC_WRONGLY_LABELLED = False


def load_fitzpatrick(csv_path: Path) -> pd.DataFrame:
    """
    加载 Fitzpatrick17k master CSV。

    Args:
        csv_path: fitzpatrick17k.csv 的完整路径。

    Returns:
        完整 DataFrame（16,577 行）。
    """
    print(f"读取 CSV：{csv_path}")
    df = pd.read_csv(csv_path)
    print(f"Fitzpatrick17k 总样本数：{len(df):,}")
    return df


def filter_valid_skin(df: pd.DataFrame) -> pd.DataFrame:
    """
    排除 fitzpatrick_scale 缺失（-1）的样本，对齐 MEDFAIR「removing those missing
    sensitive attributes」。排除后样本数应恰为 16,012（论文 Table 1）。

    Args:
        df: 输入 DataFrame，需含 'fitzpatrick_scale' 列。

    Returns:
        仅含 fitzpatrick_scale ∈ {1..6} 的 DataFrame。
    """
    n_before = len(df)
    df = df[df["fitzpatrick_scale"].between(1, 6)].reset_index(drop=True)
    print(
        f"排除 fitzpatrick_scale=-1（缺失敏感属性）：{n_before - len(df):,} 行"
        f"（剩余 {len(df):,}）"
    )
    assert len(df) == 16012, (
        f"对齐 MEDFAIR 失败：排除 -1 后应为 16,012 行，实际 {len(df):,}。"
        "请检查数据集版本。"
    )
    return df


def maybe_drop_qc(df: pd.DataFrame) -> pd.DataFrame:
    """
    可选：丢弃 qc 列标记为 "3 Wrongly labelled" 的样本。

    默认关闭（DROP_QC_WRONGLY_LABELLED=False）以对齐论文计数；若开启则样本数 < 16,012。

    Args:
        df: 输入 DataFrame，需含 'qc' 列。

    Returns:
        过滤后的 DataFrame。
    """
    if not DROP_QC_WRONGLY_LABELLED:
        return df
    n_before = len(df)
    df = df[df["qc"] != "3 Wrongly labelled"].reset_index(drop=True)
    print(f"丢弃 qc='3 Wrongly labelled'：{n_before - len(df):,} 行（剩余 {len(df):,}）")
    return df


def map_label(df: pd.DataFrame) -> pd.DataFrame:
    """
    将 three_partition_label 三分类映射为 malignant 二分类。

    MEDFAIR：将 "non-neoplastic" 与 "benign" 视为 benign(0)，"malignant" 视为 1。

    Args:
        df: 输入 DataFrame，需含 'three_partition_label' 列。

    Returns:
        新增 'label'（int，0=benign/non-neoplastic, 1=malignant）的 DataFrame。
    """
    df = df.copy()
    df["label"] = (df["three_partition_label"] == MALIGNANT_LABEL).astype("int64")
    pos_rate = df["label"].mean()
    n_pos = int(df["label"].sum())
    print(
        f"标签映射（malignant=1）：恶性占比 {pos_rate:.2%}"
        f"（恶性 {n_pos:,} / 良性+非肿瘤 {len(df) - n_pos:,}，共 {len(df):,}）"
    )
    return df


def encode_skin(df: pd.DataFrame) -> pd.DataFrame:
    """
    将 fitzpatrick_scale（1–6）编码为 skin（0–5），作为 6 类敏感属性。

    Args:
        df: 输入 DataFrame，需含 'fitzpatrick_scale' 列（已保证 ∈ {1..6}）。

    Returns:
        新增 'skin'（int, 0–5）列的 DataFrame。
    """
    df = df.copy()
    df["skin"] = (df["fitzpatrick_scale"].astype("int64") - 1)
    return df


def build_full_image_path(md5hash: str) -> str:
    """
    把 md5hash 拼接为 preproc_224x224 下的完整图像绝对路径。

    Args:
        md5hash: master CSV 'md5hash' 列（与图像文件名 1:1 对应）。

    Returns:
        图像的绝对路径字符串。
    """
    return str(IMG_DIR / f"{md5hash}.jpg")


def verify_images_exist(df: pd.DataFrame) -> None:
    """
    校验每行对应的图像文件均存在。任意缺失将抛出 AssertionError。

    Args:
        df: 含 'md5hash' 列的 DataFrame。
    """
    exists = df["md5hash"].map(lambda h: (IMG_DIR / f"{h}.jpg").exists())
    n_missing = int((~exists).sum())
    assert n_missing == 0, f"有 {n_missing} 张图像在 {IMG_DIR} 中缺失"
    print(f"图像存在性校验通过：{len(df):,} 张图像全部存在于 {IMG_DIR}")


def build_split_frame(df: pd.DataFrame) -> pd.DataFrame:
    """
    抽取下游 Dataset 类需要的列，生成最终输出 DataFrame。

    Args:
        df: 已完成标签与属性编码的 DataFrame。

    Returns:
        含 image_path / label / skin / md5hash / three_partition_label /
        nine_partition_label 列的 DataFrame。
    """
    out = pd.DataFrame({
        "image_path": df["md5hash"].map(build_full_image_path),
        "label": df["label"],
        "skin": df["skin"],
        "md5hash": df["md5hash"],
        "three_partition_label": df["three_partition_label"],
        "nine_partition_label": df["nine_partition_label"],
    })
    return out


def split_random(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """
    图像级随机 80/10/10 划分（对齐 MEDFAIR 2D 数据集设定）。

    用固定 RANDOM_STATE 打乱后按比例切分，保证可复现。Fitzpatrick17k 无 patient_id，
    每个 md5hash 唯一，故图像级划分无患者泄漏问题。

    Args:
        df: 待划分的完整 DataFrame。

    Returns:
        {"train": ..., "val": ..., "test": ...} 三个互斥子集。
    """
    shuffled = df.sample(frac=1.0, random_state=RANDOM_STATE).reset_index(drop=True)
    n = len(shuffled)
    n_train = int(n * SPLIT_FRACTIONS["train"])
    n_val = int(n * SPLIT_FRACTIONS["val"])
    return {
        "train": shuffled.iloc[:n_train].reset_index(drop=True),
        "val": shuffled.iloc[n_train:n_train + n_val].reset_index(drop=True),
        "test": shuffled.iloc[n_train + n_val:].reset_index(drop=True),
    }


def write_splits(splits: dict[str, pd.DataFrame], output_dir: Path) -> None:
    """
    把各 split 写出 CSV，并打印每个 split 的标签与肤色子群分布。

    Args:
        splits: split_random() 的返回值。
        output_dir: 输出目录。
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, subset in splits.items():
        out_path = output_dir / f"{name}.csv"
        subset.to_csv(out_path, index=False)
        skin_dist = dict(subset["skin"].value_counts().sort_index())
        skin_str = ", ".join(
            f"{SKIN_NAMES.get(k, k)}={v:,}" for k, v in skin_dist.items()
        )
        print(
            f"  {name:<6s}: {len(subset):>6,} 行  "
            f"malignant 率 {subset['label'].mean():.2%}  "
            f"skin=({skin_str})"
        )


def main() -> None:
    """主函数：依次执行筛选、标签处理、属性编码、随机划分写出。"""
    df = load_fitzpatrick(CSV_PATH)
    df = filter_valid_skin(df)
    df = maybe_drop_qc(df)
    df = map_label(df)
    df = encode_skin(df)

    split_df = build_split_frame(df)
    verify_images_exist(split_df)

    print(f"\n图像级随机 80/10/10 划分（seed={RANDOM_STATE}）并写出至：{OUTPUT_DIR}")
    splits = split_random(split_df)
    write_splits(splits, OUTPUT_DIR)

    print("\n完成。配套图像 transform（训练集，224 原生）：")
    print(
        "  transforms.Compose([\n"
        "      transforms.RandomHorizontalFlip(),\n"
        "      transforms.RandomRotation(degrees=15),\n"
        "      transforms.ToTensor(),\n"
        "      transforms.Normalize(mean=[0.485, 0.456, 0.406],\n"
        "                           std=[0.229, 0.224, 0.225]),\n"
        "  ])"
    )
    print(
        "  验证/测试集：\n"
        "  transforms.Compose([\n"
        "      transforms.ToTensor(),\n"
        "      transforms.Normalize(mean=[0.485, 0.456, 0.406],\n"
        "                           std=[0.229, 0.224, 0.225]),\n"
        "  ])"
    )


if __name__ == "__main__":
    main()
