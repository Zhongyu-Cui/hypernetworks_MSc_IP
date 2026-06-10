"""
MIMIC-CXR 预处理（MEDFAIR 对齐版）：生成 No Finding 二分类任务的 train/val/test split CSV。

与 MEDFAIR（Xu et al., ICLR 2023）的对应关系
─────────────────────────────────────────────
  目标标签  : No Finding（Positive=1 表示「无病灶/健康」，Negative/Unknown=0 表示「有病灶」）
  标签策略  : U-Zeros（CXR7-1M 中 No Finding 的 Unknown(0) 即「报告未提及/有其他病灶」，
               视为 Negative；Positive(2) 即「显式无病灶」，视为 Positive）
  体位过滤  : 仅保留正面位（PA / AP），与 MEDFAIR 一致（LATERAL 剔除）
  Sex 编码  : Male=0, Female=1；排除 Unknown（与 MEDFAIR 一致）
  Race 编码 : 二分类 White=0, Non-White=1；排除 Unknown Race（无法归入 White/Non-White）
               ── 与 MEDFAIR "White vs. non-White" 对齐；与现有 Pleural Effusion 版
               的四分类（White/Black/Asian/Unknown）不同
  数据划分  : 使用 CXR7-1M 自带的患者级 Split 列（已验证无跨 split 患者重叠）
               比例约 88%/3%/9%，与 MEDFAIR 的 80/10/10 略有差异（见下方「MEDFAIR 偏差」）
  图像变换  : 训练集 RandomResizedCrop(224, scale=(0.875, 1.0))（等价于先 256 resize
               后随机 224 crop）；小角度旋转 ±15°；ImageNet 均值/标准差归一化
               ── 本项目「不使用水平翻转」（胸片有解剖学左右约束，MEDFAIR 使用但本项目
               在 Pleural Effusion 实验中已确认不翻转，详见仓库 CLAUDE.md）

与现有 build_mimic_splits.py（Pleural Effusion 版）的关键差异
─────────────────────────────────────────────────────────────
  1. 目标标签：No Finding（无病灶=1）vs Pleural Effusion（胸腔积液=1）
  2. Race 编码：二分类 White/Non-White（排除 Unknown）vs 四分类（保留 Unknown=3）
  3. 输出目录：data/splits/mimic_cxr_nofinding/ vs data/splits/mimic_cxr/

MEDFAIR 偏差（对照论文指标时需关注）
──────────────────────────────────
  Split 比例：CXR7-1M 内置约 88/3/9，MEDFAIR 为 80/10/10。
    val 集约 3%（≈7,000 行），早停分辨率相对有限。
    如需严格复现可在 write_splits() 之前调用 resplit_patients_80_10_10()（函数
    已在本文件底部提供，默认不调用，保留 CXR7-1M 内置 split）。
  水平翻转：MEDFAIR 使用，本项目不使用。两者 AUC 差距通常 < 0.5%，对公平性比较
    影响可忽略，但做精确对照复现时需注意。

输出
────
  data/splits/mimic_cxr_nofinding/{train,val,test}.csv          人群真实分布
  data/splits/mimic_cxr_nofinding/train_oversampled.csv         训练专用：正例过采样至 50/50
  每行字段：image_path（绝对路径）, label, sex, race, age, view, patient_id
"""

from pathlib import Path

import numpy as np
import pandas as pd

RANDOM_STATE = 42

BASE_PATH = Path("/vol/biodata/projects/chai/data")
CSV_PATH = BASE_PATH / "cxr" / "cxr7-1m_master.csv"
OUTPUT_DIR = Path(
    "/vol/biomedic2/bglocker_studproj/zc125/code/hypernetworks_MSc_IP"
    "/data/splits/mimic_cxr_nofinding"
)

TARGET_LABEL = "No Finding"
FRONTAL_VIEWS = ["PA", "AP"]

# CXR7-1M 中 No Finding 列只有 0=Unknown（报告未提及/有其他病灶）和 2=Positive（显式无病灶）
# 没有 1=Negative（显式有病灶但未提 No Finding）的实例。
# U-Zeros：Unknown(0) → 0（视为有病灶），Positive(2) → 1（无病灶/健康）
RAW_LABEL_TO_BINARY = {0: 0, 1: 0, 2: 1}

SEX_TO_CODE = {"Male": 0, "Female": 1}

# MEDFAIR 对齐：White=0, 所有非 White 合并为 1（Non-White）；Unknown 排除
RACE_TO_CODE_MEDFAIR = {"White": 0, "Black": 1, "Asian": 1}

SPLIT_NAME_MAP = {"train": "train", "valid": "val", "test": "test"}


def load_mimic_frontal(csv_path: Path) -> pd.DataFrame:
    """
    加载 master CSV，筛选出 MIMIC-CXR 中的正面位（PA/AP）影像。

    Args:
        csv_path: cxr7-1m_master.csv 的完整路径。

    Returns:
        仅含 MIMIC-CXR 正面位影像的 DataFrame。
    """
    print(f"读取 CSV：{csv_path}")
    df = pd.read_csv(csv_path, low_memory=False)
    df = df[df["Dataset_name"] == "MIMIC-CXR"]
    print(f"MIMIC-CXR 总样本数：{len(df):,}")

    df = df[df["View_name"].isin(FRONTAL_VIEWS)].reset_index(drop=True)
    print(f"正面位（PA/AP）样本数：{len(df):,}")
    return df


def apply_u_zeros(df: pd.DataFrame, label_col: str) -> pd.DataFrame:
    """
    对目标标签应用 U-Zeros 策略。

    对 No Finding 而言，CXR7-1M 中只有 Unknown(0)（报告未提及 No Finding，
    即存在其他病灶）和 Positive(2)（显式无病灶）两种值。U-Zeros 将 Unknown
    视为 Negative（0），Positive 视为 Positive（1）。

    Args:
        df: 输入 DataFrame，需含目标标签列。
        label_col: 目标标签列名（如 "No Finding"）。

    Returns:
        新增 'label' 列（int，0=有病灶, 1=无病灶/健康）的 DataFrame。
    """
    df = df.copy()
    df["label"] = df[label_col].map(RAW_LABEL_TO_BINARY).astype("int64")
    pos_rate = df["label"].mean()
    n_pos = df["label"].sum()
    n_neg = len(df) - n_pos
    print(
        f"U-Zeros 重映射 '{label_col}'：正例（无病灶）占比 {pos_rate:.1%}"
        f"（正例 {n_pos:,} / 负例 {n_neg:,}，共 {len(df):,}）"
    )
    return df


def encode_attributes_medfair(df: pd.DataFrame) -> pd.DataFrame:
    """
    对齐 MEDFAIR：Sex（Male=0, Female=1）与 Race（White=0, Non-White=1），
    排除 Sex=Unknown 及 Race=Unknown 的样本。

    MEDFAIR 使用白人 vs 非白人二分类，因此 Black/Asian/... 均归入 Non-White(1)。
    Unknown Race 无法归入两类，排除以避免引入噪声。

    Args:
        df: 输入 DataFrame，需含 'Sex_name' 与 'Race_name' 列。

    Returns:
        新增 'sex'（0/1）与 'race'（0/1）整数列的 DataFrame。
    """
    n_before = len(df)
    df = df[df["Sex_name"].isin(SEX_TO_CODE.keys())].copy()
    print(f"排除 Sex=Unknown：{n_before - len(df):,} 行（剩余 {len(df):,}）")

    n_before = len(df)
    df = df[df["Race_name"].isin(RACE_TO_CODE_MEDFAIR.keys())].copy()
    print(
        f"排除 Race=Unknown（无法归入 White/Non-White）：{n_before - len(df):,} 行"
        f"（剩余 {len(df):,}）"
    )

    df["sex"] = df["Sex_name"].map(SEX_TO_CODE).astype("int64")
    # race 二分类：White=0, Non-White（Black/Asian）=1
    df["race"] = df["Race_name"].map(RACE_TO_CODE_MEDFAIR).astype("int64")
    return df


def build_full_image_path(image_path_relative: str) -> str:
    """
    把 master CSV 中的相对路径拼接为完整的图像绝对路径。

    Args:
        image_path_relative: master CSV 'ImagePath' 列的相对路径。

    Returns:
        图像的绝对路径字符串。
    """
    return str(BASE_PATH / image_path_relative)


def build_split_frame(df: pd.DataFrame) -> pd.DataFrame:
    """
    抽取下游 Dataset 类需要的列，生成最终输出 DataFrame。

    Args:
        df: 已完成标签与属性编码的 DataFrame。

    Returns:
        含 image_path / label / sex / race / age / view / patient_id / split 列的 DataFrame。
    """
    out = pd.DataFrame({
        "image_path": df["ImagePath"].map(build_full_image_path),
        "label": df["label"],
        "sex": df["sex"],
        "race": df["race"],
        "age": df["Age"],
        "view": df["View_name"],
        "patient_id": df["PatientID"],
        "split": df["Split"],
    })
    return out


def verify_no_patient_overlap(df: pd.DataFrame) -> None:
    """
    验证三个 split 之间无患者级重叠（patient_id 互不相交）。
    任意重叠将抛出 AssertionError。

    Args:
        df: 含 'split'（train/valid/test）与 'patient_id' 列的 DataFrame。
    """
    train_pts = set(df[df["split"] == "train"]["patient_id"])
    val_pts   = set(df[df["split"] == "valid"]["patient_id"])
    test_pts  = set(df[df["split"] == "test"]["patient_id"])

    tv = train_pts & val_pts
    tt = train_pts & test_pts
    vt = val_pts   & test_pts
    assert not tv, f"train/val 存在 {len(tv)} 名患者重叠"
    assert not tt, f"train/test 存在 {len(tt)} 名患者重叠"
    assert not vt, f"val/test 存在 {len(vt)} 名患者重叠"
    print(f"患者级重叠验证通过：train {len(train_pts):,} / val {len(val_pts):,} / test {len(test_pts):,} 名患者，无交集")


def write_splits(df: pd.DataFrame, output_dir: Path) -> None:
    """
    按 'split' 列把 DataFrame 拆分为 train/val/test 并分别写出 CSV。

    Args:
        df: 含 'split' 列（取值 train/valid/test）的 DataFrame。
        output_dir: 输出目录。
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    for raw_name, out_name in SPLIT_NAME_MAP.items():
        subset = df[df["split"] == raw_name].drop(columns=["split"]).reset_index(drop=True)
        out_path = output_dir / f"{out_name}.csv"
        subset.to_csv(out_path, index=False)
        race_dist = dict(subset["race"].value_counts().sort_index())
        race_label = {0: "White", 1: "Non-White"}
        race_str = ", ".join(f"{race_label.get(k, k)}={v:,}" for k, v in race_dist.items())
        print(
            f"  {out_name:<6s}: {len(subset):>7,} 行  "
            f"正例率 {subset['label'].mean():.1%}  "
            f"sex=(M={subset['sex'].eq(0).sum():,}, F={subset['sex'].eq(1).sum():,})  "
            f"race=({race_str})"
        )


def main() -> None:
    """主函数：依次执行筛选、标签处理、属性编码、划分写出。"""
    df = load_mimic_frontal(CSV_PATH)
    df = apply_u_zeros(df, TARGET_LABEL)
    df = encode_attributes_medfair(df)

    split_df = build_split_frame(df)
    verify_no_patient_overlap(split_df)

    print(f"\n按 master CSV 自带 Split 列拆分并写出至：{OUTPUT_DIR}")
    write_splits(split_df, OUTPUT_DIR)

    print("\n完成。配套图像 transform（训练集）：")
    print(
        "  transforms.Compose([\n"
        "      transforms.Grayscale(num_output_channels=3),\n"
        "      transforms.Resize(256),\n"
        "      transforms.RandomCrop(224),\n"
        "      transforms.RandomRotation(degrees=15),\n"
        "      transforms.ToTensor(),\n"
        "      transforms.Normalize(mean=[0.485, 0.456, 0.406],\n"
        "                           std=[0.229, 0.224, 0.225]),\n"
        "  ])"
    )
    print(
        "  验证/测试集：\n"
        "  transforms.Compose([\n"
        "      transforms.Grayscale(num_output_channels=3),\n"
        "      transforms.Resize(256),\n"
        "      transforms.CenterCrop(224),\n"
        "      transforms.ToTensor(),\n"
        "      transforms.Normalize(mean=[0.485, 0.456, 0.406],\n"
        "                           std=[0.229, 0.224, 0.225]),\n"
        "  ])"
    )


if __name__ == "__main__":
    main()
