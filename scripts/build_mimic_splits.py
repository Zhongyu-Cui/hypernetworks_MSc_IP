"""
# DEPRECATED: 已被 scripts/build_mimic_splits_nofinding.py（MEDFAIR 对齐，No Finding 任务）取代。

MIMIC-CXR 预处理：生成 Pleural Effusion 二分类任务的 train/val/test split CSV。

处理流程：
  1. 从 master CSV 中筛选出 MIMIC-CXR 子集
  2. 只保留正面位影像（PA / AP），剔除 LATERAL 与 Unknown 体位
  3. 标签策略 U-Zeros：master CSV 中 Pleural Effusion 的 "Unknown"(=0) 100%
     来自上游 v6_labelling CSV 中的 NaN（报告未提及该病灶），即放射科医生未在
     报告中专门提及 Pleural Effusion。CXR7-1M 的标注流程只产生二元标签（0=Negative,
     1=Positive）加 NaN（未提及），**不存在** CheXpert 原始 labeler 的 -1（uncertain）
     值。65% 的"未提及"率完全符合临床实际："未提及 ≈ 不存在"，视为 Negative。
     最终二分类编码：{Unknown(0) -> Negative(0), Negative(1) -> Negative(0), Positive(2) -> Positive(1)}
  4. 将 Sex / Race 编码为整数，与 UTKFace 数据集的属性编码风格对齐
  5. 按 master CSV 自带的 Split 列（已做患者级划分）拆分并写出三个 CSV
  6. 仅对训练集做随机过采样（正例 label=1 重复采样至与负例持平），
     额外写出 train_oversampled.csv（详见 oversample_train_split 决策记录）

输出：
  data/splits/mimic_cxr/{train,val,test}.csv          人群真实分布（验证/测试用，亦保留作训练集的真实分布记录）
  data/splits/mimic_cxr/train_oversampled.csv          训练专用：正例过采样至 50/50（详见下方决策记录）
  每行包含：image_path（绝对路径）, label, sex, race, age, view, patient_id
"""

from pathlib import Path

import pandas as pd

RANDOM_STATE = 42

BASE_PATH = Path("/vol/biodata/projects/chai/data")
CSV_PATH = BASE_PATH / "cxr" / "cxr7-1m_master.csv"
OUTPUT_DIR = Path(
    "/vol/biomedic2/bglocker_studproj/zc125/code/hypernetworks_MSc_IP/data/splits/mimic_cxr"
)

TARGET_LABEL = "Pleural Effusion"
FRONTAL_VIEWS = ["PA", "AP"]

# master CSV 中 Pleural Effusion 列的原始编码：0=Unknown(未提及/不确定), 1=Negative, 2=Positive
# U-Zeros 策略：Unknown 视为 Negative，{Unknown: 0, Negative: 1} -> 0，{Positive: 2} -> 1
RAW_LABEL_TO_BINARY = {0: 0, 1: 0, 2: 1}

SEX_TO_CODE = {"Male": 0, "Female": 1}
# Unknown race 保留为单独编码 3，便于后续按需排除或做消融实验
RACE_TO_CODE = {"White": 0, "Black": 1, "Asian": 2, "Unknown": 3}

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
    对目标标签应用 U-Zeros 策略：把 {Unknown(0), Negative(1)} 都视为
    Negative，{Positive(2)} 视为 Positive，重映射为二分类 {0, 1}。

    master CSV 中 "Unknown"(0) 100% 来自上游标注 CSV 的 NaN（报告未提及该病灶）。
    CXR7-1M 标注流程只产生二元标签（0=Negative, 1=Positive）加 NaN（未提及），
    不存在 CheXpert 原始 labeler 的 -1（uncertain）值。"未提及 ≈ 不存在"是
    标准医学文本标注假设，将其归为 Negative 以保留全量样本并维持接近真实患病率的正例率。

    Args:
        df: 输入 DataFrame，需含目标标签列。
        label_col: 目标标签列名。

    Returns:
        新增 'label' 列（int，0=Negative, 1=Positive）的 DataFrame。
    """
    df = df.copy()
    df["label"] = df[label_col].map(RAW_LABEL_TO_BINARY).astype("int64")
    print(
        f"U-Zeros 重映射 '{label_col}' 标签（Unknown/Negative -> 0, Positive -> 1）："
        f"正例占比 {df['label'].mean():.1%}（n={len(df):,}）"
    )
    return df


def encode_attributes(df: pd.DataFrame) -> pd.DataFrame:
    """
    将 Sex / Race 编码为整数，对齐 UTKFace 数据集的属性编码风格。

    Args:
        df: 输入 DataFrame，需含 'Sex_name' 与 'Race_name' 列。

    Returns:
        新增 'sex' 与 'race' 整数列的 DataFrame；剔除 Sex 为 Unknown 的样本
        （Race 的 Unknown 保留为编码 3，数量较多，留待按需处理）。
    """
    n_before = len(df)
    df = df[df["Sex_name"].isin(SEX_TO_CODE.keys())].copy()
    print(f"剔除 Sex 为 Unknown 的样本：{n_before - len(df):,}（剩余 {len(df):,}）")

    df["sex"] = df["Sex_name"].map(SEX_TO_CODE).astype("int64")
    df["race"] = df["Race_name"].map(RACE_TO_CODE).astype("int64")
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
    抽取下游 Dataset 类需要的列，生成最终输出的 DataFrame。

    Args:
        df: 已完成标签与属性编码的 DataFrame。

    Returns:
        含 image_path / label / sex / race / age / view / patient_id 列的 DataFrame。
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
        print(
            f"  {out_name:<6s}: {len(subset):>7,} 行 -> {out_path} "
            f"(正例占比 {subset['label'].mean():.1%}, "
            f"sex={dict(subset['sex'].value_counts().sort_index())}, "
            f"race={dict(subset['race'].value_counts().sort_index())})"
        )


def oversample_train_split(train_df: pd.DataFrame, output_dir: Path) -> None:
    """
    仅对训练集做随机过采样，缓解 Pleural Effusion 正例率仅约 24% 的类别不平衡
    （决策记录见模块 CLAUDE.md「训练集过采样」一节：baseline 训练中 BCE loss
    未做不平衡处理，导致模型几乎只预测多数类，TPR 低至 ~0.01~0.02）。

    做法：对正例（label=1）行有放回随机重复采样，使其数量与负例持平
    （约 3.26x），与全部负例拼接后打乱写出，得到一份正例占比 ~50% 的训练专用
    CSV。**只处理训练集**——验证 / 测试集必须保持人群真实分布，否则 AUC 等
    指标将不可信，也无法与 baseline 公平对比。

    注意：原始 train.csv 不被覆盖或丢弃，继续作为该 split 人群真实分布
    （正例率 23.5%）的记录保留；过采样结果写入新文件 train_oversampled.csv，
    训练脚本按需切换读取（见 configs/mimic_cxr_baseline.yaml 的 data.train_csv）。

    Args:
        train_df: 原始（未过采样）训练集 DataFrame，需含 'label' 列。
        output_dir: 输出目录（与 train.csv 相同）。
    """
    positives = train_df[train_df["label"] == 1]
    negatives = train_df[train_df["label"] == 0]

    positives_oversampled = positives.sample(
        n=len(negatives), replace=True, random_state=RANDOM_STATE,
    )
    balanced = (
        pd.concat([negatives, positives_oversampled])
        .sample(frac=1.0, random_state=RANDOM_STATE)
        .reset_index(drop=True)
    )

    out_path = output_dir / "train_oversampled.csv"
    balanced.to_csv(out_path, index=False)
    print(
        f"\n训练集随机过采样（正例 label=1，有放回采样至与负例持平，random_state={RANDOM_STATE}）：\n"
        f"  原始 train.csv     : {len(train_df):>7,} 行（正例占比 {train_df['label'].mean():.1%}，"
        f"正例 {len(positives):,} / 负例 {len(negatives):,}）\n"
        f"  train_oversampled  : {len(balanced):>7,} 行（正例占比 {balanced['label'].mean():.1%}） "
        f"-> {out_path}"
    )


def main() -> None:
    """主函数：依次执行筛选、标签处理、属性编码、拆分写出、训练集过采样。"""
    df = load_mimic_frontal(CSV_PATH)
    df = apply_u_zeros(df, TARGET_LABEL)
    df = encode_attributes(df)

    split_df = build_split_frame(df)
    print(f"\n按 master CSV 自带 Split 列拆分并写出至：{OUTPUT_DIR}")
    write_splits(split_df, OUTPUT_DIR)

    train_df = split_df[split_df["split"] == "train"].drop(columns=["split"]).reset_index(drop=True)
    oversample_train_split(train_df, OUTPUT_DIR)


if __name__ == "__main__":
    main()
