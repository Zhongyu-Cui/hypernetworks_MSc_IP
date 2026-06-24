"""
HAM10000 预处理（MEDFAIR 对齐版）：生成 benign/malignant 二分类任务的 train/val/test split CSV。

与 MEDFAIR（Zong et al., ICLR 2023）的对应关系
─────────────────────────────────────────────
  目标标签  : malignant（Positive=1 表示「恶性」）。遵循 Maron et al. (2019) / MEDFAIR，
               把 7 类 dx 压成二分类：
                 malignant(1) = {akiec, mel}
                 benign(0)    = {bcc, bkl, df, nv, vasc}
               ⚠️ bcc（基底细胞癌）医学上其实是恶性，但 Maron/MEDFAIR 将其归入 benign，
                  为与 MEDFAIR 可比，保留此归类。清洗后恶性率约 14.5%。
  样本筛选  : 丢弃 sex=='unknown'（57 行）或 age 缺失（57 行，交集 47）——并集 67 行，
               丢弃后恰好剩 9948 行（与论文 Table 1 / A2 一致，下方有硬断言对齐）。
  敏感属性  : sex（Male=0, Female=1）与 age（分箱）。本数据集无肤色/种族标注。
               年龄分箱对齐 MEDFAIR Table A16：4 个有效组 {20-40,40-60,60-80,80+}=0..3，
               0-20（仅约 2.4%≈241 张，论文「too few samples」）编码为 -1，
               在按年龄分组的公平性评估时过滤掉（与 Fitzpatrick 排除 skin==-1 同一惯例）。
               sex 任务用全部 9948 张；age 任务用 age_group>=0 的约 9707 张。
  数据划分  : **lesion 级 GroupShuffleSplit 80/10/10**（同一 lesion_id 整组只进一个 split）。
               这一点**偏离 MEDFAIR-HAM 的图像级随机划分**：HAM 有 7470 个 lesion 对应 10015 张图，
               45% 的图属于「同病灶多图」，图像级随机划分会让同病灶图跨 split 泄漏、指标虚高。
               本方案与 MEDFAIR 对 CheXpert/MIMIC/PAPILA 的「同病人不跨 split」原则一致，是更正确的做法。
               置 GROUP_BY_LESION=False 可退回图像级随机 80/10/10，用于严格复现论文数字。
  图像变换  : 见 configs/ham10000_baseline.yaml。源图为 600×450（≠224），降采样到 224 合法，
               故对齐 MEDFAIR 的 Resize(256,256)→Crop(224)（区别于 Fitzpatrick 的「224 原生」，
               后者源图已 224 才省去 resize）；皮损无左右约束，保留水平翻转；ImageNet 归一化。

与 Fitzpatrick17k 版（build_fitzpatrick_splits.py）的关键差异
─────────────────────────────────────────────────────────────
  1. 源分辨率：HAM 为 600×450（需 resize），Fitz 为 224 原生（不 resize）。
  2. 敏感属性：HAM 为 sex(2) + age(4 有效组) 两轴，Fitz 为单一肤色轴 6 子群。
  3. 划分：HAM 用 lesion 级分组划分（有 lesion_id），Fitz 用图像级随机（无 patient_id）。
  4. 对账锚点：HAM 断言 9948，Fitz 断言 16012。

输出
────
  data/splits/ham10000/{train,val,test}.csv
  每行字段：image_path（绝对路径）, label, sex, age, age_group,
            dx, dx_type, lesion_id, image_id
"""

from pathlib import Path

import numpy as np
import pandas as pd

RANDOM_STATE = 42

BASE_PATH = Path("/vol/biodata/data/HAM10000")
CSV_PATH = BASE_PATH / "HAM10000_metadata.csv"
IMG_DIR = BASE_PATH / "images"            # 已解压的原始 600×450 RGB JPEG（uint8）
OUTPUT_DIR = Path(
    "/vol/biomedic2/bglocker_studproj/zc125/code/hypernetworks_MSc_IP"
    "/data/splits/ham10000"
)

# 二分类标签映射（Maron et al. 2019 / MEDFAIR）
MALIGNANT_DX = {"akiec", "mel"}

# sex 编码（与 MIMIC 一致：Male=0, Female=1）
SEX_CODE = {"male": 0, "female": 1}

# 年龄分箱（左闭右开）：[0,20)→-1(排除), [20,40)→0, [40,60)→1, [60,80)→2, [80,inf)→3
AGE_BINS = [0, 20, 40, 60, 80, np.inf]
AGE_GROUP_CODES = [-1, 0, 1, 2, 3]
AGE_GROUP_NAMES = {0: "20-40", 1: "40-60", 2: "60-80", 3: "80+", -1: "0-20(excl)"}

# 划分比例（2D 数据集，MEDFAIR 80/10/10）
SPLIT_FRACTIONS = {"train": 0.8, "val": 0.1, "test": 0.1}

# 划分策略：True=lesion 级分组（推荐，无泄漏）；False=图像级随机（严格复现 MEDFAIR）
GROUP_BY_LESION = True


def load_metadata(csv_path: Path) -> pd.DataFrame:
    """
    加载 HAM10000 master CSV。

    Args:
        csv_path: HAM10000_metadata.csv 的完整路径。

    Returns:
        完整 DataFrame（10,015 行）。
    """
    print(f"读取 CSV：{csv_path}")
    df = pd.read_csv(csv_path)
    print(f"HAM10000 总样本数：{len(df):,}（lesion_id 数：{df['lesion_id'].nunique():,}）")
    return df


def filter_valid_attributes(df: pd.DataFrame) -> pd.DataFrame:
    """
    丢弃缺失敏感属性的样本，对齐 MEDFAIR「removing those missing sensitive attributes」。

    丢弃 sex=='unknown' 或 age 缺失的行；丢弃后样本数应恰为 9948（论文 Table 1）。

    Args:
        df: 输入 DataFrame，需含 'sex' 与 'age' 列。

    Returns:
        敏感属性完整的 DataFrame。
    """
    n_before = len(df)
    keep = (df["sex"] != "unknown") & df["age"].notna()
    df = df[keep].reset_index(drop=True)
    print(
        f"排除 sex='unknown' 或 age 缺失（缺失敏感属性）：{n_before - len(df):,} 行"
        f"（剩余 {len(df):,}）"
    )
    assert len(df) == 9948, (
        f"对齐 MEDFAIR 失败：清洗后应为 9,948 行，实际 {len(df):,}。请检查数据集版本。"
    )
    return df


def map_label(df: pd.DataFrame) -> pd.DataFrame:
    """
    将 7 类 dx 映射为 malignant 二分类（Maron 2019 / MEDFAIR）。

    Args:
        df: 输入 DataFrame，需含 'dx' 列。

    Returns:
        新增 'label'（int，1=malignant, 0=benign）的 DataFrame。
    """
    df = df.copy()
    df["label"] = df["dx"].isin(MALIGNANT_DX).astype("int64")
    pos_rate = df["label"].mean()
    n_pos = int(df["label"].sum())
    print(
        f"标签映射（malignant=1，={sorted(MALIGNANT_DX)}）：恶性占比 {pos_rate:.2%}"
        f"（恶性 {n_pos:,} / 良性 {len(df) - n_pos:,}，共 {len(df):,}）"
    )
    return df


def encode_attributes(df: pd.DataFrame) -> pd.DataFrame:
    """
    编码敏感属性：sex（Male=0, Female=1）与 age_group（4 有效组，0-20 标 -1 排除）。

    Args:
        df: 输入 DataFrame，需含 'sex' 与 'age' 列（已保证非缺失）。

    Returns:
        新增 'sex'（覆盖为 0/1）与 'age_group'（-1/0/1/2/3）列的 DataFrame。
    """
    df = df.copy()
    df["sex"] = df["sex"].map(SEX_CODE).astype("int64")
    # 左闭右开分箱得到 bin 索引 0..4（对应 [0,20)..[80,inf)），再显式映射为编码：
    # bin 0([0,20)) -> -1（排除组），bin 1..4 -> 0..3
    age_idx = pd.cut(df["age"], bins=AGE_BINS, right=False, labels=False)
    code_map = {0: -1, 1: 0, 2: 1, 3: 2, 4: 3}
    df["age_group"] = age_idx.map(code_map).astype("int64")

    sex_dist = {("M" if k == 0 else "F"): int((df["sex"] == k).sum()) for k in (0, 1)}
    ag_dist = {AGE_GROUP_NAMES[k]: int((df["age_group"] == k).sum())
               for k in sorted(df["age_group"].unique())}
    print(f"sex 编码（M=0,F=1）：{sex_dist}")
    print(f"age_group 编码（0-20 标 -1 排除）：{ag_dist}")
    return df


def build_full_image_path(image_id: str) -> str:
    """
    把 image_id 拼接为 images/ 下的完整图像绝对路径。

    Args:
        image_id: master CSV 'image_id' 列（如 ISIC_0027419），与图像文件名 1:1 对应。

    Returns:
        图像的绝对路径字符串。
    """
    return str(IMG_DIR / f"{image_id}.jpg")


def verify_images_exist(df: pd.DataFrame) -> None:
    """
    校验每行对应的图像文件均存在。任意缺失将抛出 AssertionError。

    Args:
        df: 含 'image_id' 列的 DataFrame。
    """
    exists = df["image_id"].map(lambda i: (IMG_DIR / f"{i}.jpg").exists())
    n_missing = int((~exists).sum())
    assert n_missing == 0, f"有 {n_missing} 张图像在 {IMG_DIR} 中缺失"
    print(f"图像存在性校验通过：{len(df):,} 张图像全部存在于 {IMG_DIR}")


def build_split_frame(df: pd.DataFrame) -> pd.DataFrame:
    """
    抽取下游 Dataset 类需要的列，生成最终输出 DataFrame。

    Args:
        df: 已完成标签与属性编码的 DataFrame。

    Returns:
        含 image_path / label / sex / age / age_group / dx / dx_type /
        lesion_id / image_id 列的 DataFrame。
    """
    out = pd.DataFrame({
        "image_path": df["image_id"].map(build_full_image_path),
        "label": df["label"],
        "sex": df["sex"],
        "age": df["age"],
        "age_group": df["age_group"],
        "dx": df["dx"],
        "dx_type": df["dx_type"],
        "lesion_id": df["lesion_id"],
        "image_id": df["image_id"],
    })
    return out


def split_grouped_by_lesion(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """
    lesion 级分组划分：同一 lesion_id 整组只进一个 split，按图像数逼近 80/10/10。

    打乱唯一 lesion_id（固定 seed），按累计图像数走到 80% / 90% 截断点切分，
    保证可复现且无同病灶图像跨 split 泄漏。

    Args:
        df: 待划分的完整 DataFrame（需含 'lesion_id'）。

    Returns:
        {"train": ..., "val": ..., "test": ...} 三个互斥子集（lesion 不重叠）。
    """
    rng = np.random.default_rng(RANDOM_STATE)
    # 转成 numpy object 数组再洗牌：df.unique() 可能返回 Arrow 扩展数组，
    # 直接 rng.shuffle 不保证正确（view 语义可能产生重复）。
    lesions = np.asarray(df["lesion_id"].unique(), dtype=object)
    rng.shuffle(lesions)

    # 每个 lesion 的图像数，按打乱后顺序累计
    sizes = df.groupby("lesion_id").size()
    cum = sizes.reindex(lesions).cumsum().values
    n_total = len(df)
    train_cut = SPLIT_FRACTIONS["train"] * n_total
    val_cut = (SPLIT_FRACTIONS["train"] + SPLIT_FRACTIONS["val"]) * n_total

    train_lesions = set(lesions[cum <= train_cut])
    val_lesions = set(lesions[(cum > train_cut) & (cum <= val_cut)])
    # 剩余归 test，保证三者并集覆盖全部 lesion
    assigned = train_lesions | val_lesions
    test_lesions = set(lesions) - assigned

    masks = {
        "train": df["lesion_id"].isin(train_lesions),
        "val": df["lesion_id"].isin(val_lesions),
        "test": df["lesion_id"].isin(test_lesions),
    }
    return {name: df[m].reset_index(drop=True) for name, m in masks.items()}


def split_random(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """
    图像级随机 80/10/10 划分（严格复现 MEDFAIR；有同病灶图像跨 split 泄漏风险）。

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


def assert_no_lesion_leakage(splits: dict[str, pd.DataFrame]) -> None:
    """
    校验任意两个 split 之间不存在共享的 lesion_id（仅在 lesion 级划分时有意义）。

    Args:
        splits: split 字典。
    """
    sets = {name: set(s["lesion_id"]) for name, s in splits.items()}
    for a in ("train", "val", "test"):
        for b in ("val", "test"):
            if a >= b:
                continue
            overlap = sets[a] & sets[b]
            assert not overlap, f"{a} 与 {b} 共享 {len(overlap)} 个 lesion_id（存在泄漏）"
    print("lesion 级无泄漏校验通过：train/val/test 三者 lesion_id 互不重叠")


def write_splits(splits: dict[str, pd.DataFrame], output_dir: Path) -> None:
    """
    把各 split 写出 CSV，并打印每个 split 的规模、恶性率与子群分布。

    Args:
        splits: 划分函数的返回值。
        output_dir: 输出目录。
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, subset in splits.items():
        out_path = output_dir / f"{name}.csv"
        subset.to_csv(out_path, index=False)
        sex_dist = {("M" if k == 0 else "F"): int((subset["sex"] == k).sum())
                    for k in (0, 1)}
        ag_dist = {AGE_GROUP_NAMES[k]: int((subset["age_group"] == k).sum())
                   for k in sorted(subset["age_group"].unique())}
        print(
            f"  {name:<6s}: {len(subset):>6,} 行  "
            f"malignant 率 {subset['label'].mean():.2%}  "
            f"sex={sex_dist}  age_group={ag_dist}"
        )


def main() -> None:
    """主函数：依次执行筛选、标签处理、属性编码、划分、写出。"""
    df = load_metadata(CSV_PATH)
    df = filter_valid_attributes(df)
    df = map_label(df)
    df = encode_attributes(df)

    split_df = build_split_frame(df)
    verify_images_exist(split_df)

    strategy = "lesion 级 GroupShuffleSplit" if GROUP_BY_LESION else "图像级随机"
    print(f"\n{strategy} 80/10/10 划分（seed={RANDOM_STATE}）并写出至：{OUTPUT_DIR}")
    if GROUP_BY_LESION:
        splits = split_grouped_by_lesion(split_df)
        assert_no_lesion_leakage(splits)
    else:
        splits = split_random(split_df)
    write_splits(splits, OUTPUT_DIR)

    print("\n完成。配套图像 transform（源图 600×450，需 resize，对齐 MEDFAIR）：")
    print(
        "  训练集：\n"
        "  transforms.Compose([\n"
        "      transforms.Resize((256, 256)),\n"
        "      transforms.RandomCrop(224),\n"
        "      transforms.RandomHorizontalFlip(),\n"
        "      transforms.RandomRotation(degrees=15),\n"
        "      transforms.ToTensor(),\n"
        "      transforms.Normalize(mean=[0.485, 0.456, 0.406],\n"
        "                           std=[0.229, 0.224, 0.225]),\n"
        "  ])\n"
        "  验证/测试集：\n"
        "  transforms.Compose([\n"
        "      transforms.Resize((256, 256)),\n"
        "      transforms.CenterCrop(224),\n"
        "      transforms.ToTensor(),\n"
        "      transforms.Normalize(mean=[0.485, 0.456, 0.406],\n"
        "                           std=[0.229, 0.224, 0.225]),\n"
        "  ])"
    )


if __name__ == "__main__":
    main()
