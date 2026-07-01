"""
PAPILA 预处理（MEDFAIR 对齐版）：生成青光眼二分类任务的 train/val/test split CSV，
并（可选）把原始大图预降采样为 256×256 缓存以降低集群 I/O。

方案文档：docs/papila_preprocessing_plan.md

与 MEDFAIR（Zong et al., ICLR 2023）的对应关系
─────────────────────────────────────────────
  目标标签  : 青光眼二分类。label=1 ⇔ Diagnosis==1(glaucomatous)，
              label=0 ⇔ Diagnosis==0(non-glaucomatous)。
              **排除 suspect（Diagnosis==2）**——对齐 MEDFAIR「exclude the suspect label class」。
              排除后恰为 420 眼-图 / 210 患者 / 阳性 87（阳性率 20.7%），与论文 Table A3/B.1.2 一致
              （被排除的 34 患者恰为「双眼均 suspect」，68=34×2，故 420=210×2）。
  建模单元  : **单眼图**（OD 右眼 / OS 左眼 各算一个样本）。临床表按眼分两个 xlsx，
              性别/年龄左右眼一致；7 个患者左右眼诊断标签不一致，按眼各自保留。
  样本筛选  : Age/Gender 零缺失（不同于 HAM 需丢 67 行），无需额外清洗。
  敏感属性  : sex（Male=0, Female=1）与 age（在 60 岁二分）。本数据集**无种族/肤色标注**。
              age_group: Age<=60 -> 0，Age>60 -> 1（对齐 MEDFAIR「0-60 / 60+」，边界 60 归组 0）。
              joint = sex*2 + age_group ∈ {0,1,2,3}，供 WeightedRandomSampler 与分组评估。
              sex 编码经青光眼率与 MEDFAIR Table A3 对齐验证（M:24.0% / F:19.0%）。
  数据划分  : **患者级 GroupShuffleSplit 70/10/20**（groups=pid，固定 seed）。
              比例用 **MEDFAIR 的 70/10/20**（注意区别于 HAM/Fitz 的 80/10/10）。
              同一患者双眼绝不跨 split——与 HAM「按 lesion_id 分组」、MIMIC「同病人不跨 split」同理
              （左右眼高度相关，图像级随机划分会泄漏、虚高指标）。
  图像变换  : 见 configs/papila_baseline.yaml。源图 2576×1934（≠224），降采样合法，
              对齐 MEDFAIR 的 Resize(256,256)→Crop(224)；自然 RGB，**无 16-bit 处理**。

与 HAM10000 版（build_ham10000_splits.py）的关键差异
─────────────────────────────────────────────────────
  1. 数据源：xlsx（左右眼两文件）而非单 CSV；A 列(ID)无表头名需 rename。
  2. 划分键：患者 pid（双眼）而非病灶 lesion_id；比例 70/10/20 而非 80/10/10。
  3. 敏感属性：sex(2) + age(2 箱) 两轴；age 只二分（MEDFAIR 对 PAPILA 即二分）。
  4. 对账锚点：断言 420 眼-图 / 210 患者 / 阳性 87。
  5. 额外步骤：可选预降采样 256×256 缓存（原图 1.2GB，解码偏重）。

输出
────
  data/splits/papila/{train,val,test}.csv
  每行字段：image_path（绝对路径，指向 256 缓存或原图）, label, sex, age, age_group,
            joint, pid, eye, diagnosis
  （可选）data/processed/papila/fundus_256/RET<ID><OD|OS>.jpg  256×256 RGB 缓存
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

RANDOM_STATE = 42

BASE_PATH = Path("/vol/biodata/data/PAPILA")
CLINICAL_DIR = BASE_PATH / "ClinicalData"
IMG_DIR = BASE_PATH / "FundusImages"            # 原始 2576×1934 RGB JPEG（uint8）

REPO_ROOT = Path(
    "/vol/biomedic2/bglocker_studproj/zc125/code/hypernetworks_MSc_IP"
)
OUTPUT_DIR = REPO_ROOT / "data/splits/papila"

# 256×256 预降采样缓存目录（个人目录 data/processed，禁止改动只读原始数据；不进仓库）
RESIZE_CACHE_DIR = Path(
    "/vol/biomedic2/bglocker_studproj/zc125/data/processed/papila/fundus_256"
)

# 划分比例（MEDFAIR 对 PAPILA 用 70/10/20）
SPLIT_FRACTIONS = {"train": 0.7, "val": 0.1, "test": 0.2}

# 预降采样开关：True 则生成 256×256 缓存，并把 image_path 指向缓存（推荐，降 I/O）
PRERESIZE_256 = True
RESIZE_TO = (256, 256)

# 对账锚点（MEDFAIR Table A3 / B.1.2）
EXPECT_N_IMAGES = 420
EXPECT_N_PATIENTS = 210
EXPECT_N_POSITIVE = 87


def load_clinical(eye: str) -> pd.DataFrame:
    """
    加载单眼临床表 xlsx 并抽取建模所需列。

    Args:
        eye: 'od'（右眼）或 'os'（左眼）。

    Returns:
        含 pid / eye / Age / Gender / Diagnosis 列的 DataFrame（仅数据行）。
    """
    path = CLINICAL_DIR / f"patient_data_{eye}.xlsx"
    # 合并表头占前 2 行，真正列名在第 2 行（header=1）；A 列(ID)无表头名需重命名。
    df = pd.read_excel(path, header=1)
    df = df.rename(columns={df.columns[0]: "ID"})
    # 仅保留以 '#' 开头的真实患者行（丢掉第 3 行的 'ID' 占位等）
    df = df[df["ID"].astype(str).str.startswith("#")].copy()
    df["pid"] = df["ID"].str.replace("#", "", regex=False).str.zfill(3)
    df["eye"] = "OD" if eye == "od" else "OS"
    out = df[["pid", "eye", "Age", "Gender", "Diagnosis"]].copy()
    out["Age"] = out["Age"].astype(float)
    out["Gender"] = out["Gender"].astype(int)
    out["Diagnosis"] = out["Diagnosis"].astype(int)
    return out


def load_all_eyes() -> pd.DataFrame:
    """
    合并左右眼临床表为「单眼图」长表（488 行 = 244 患者 × 双眼）。

    Returns:
        合并后的长表 DataFrame。
    """
    long = pd.concat([load_clinical("od"), load_clinical("os")], ignore_index=True)
    print(f"读取临床表：{len(long)} 眼-图（{long['pid'].nunique()} 患者）")
    diag = long["Diagnosis"].value_counts().sort_index().to_dict()
    print(f"全量 Diagnosis 分布 0健康/1青光眼/2可疑：{diag}")
    n_missing = int(long[["Age", "Gender"]].isna().any(axis=1).sum())
    assert n_missing == 0, f"存在 {n_missing} 行缺失 Age/Gender（PAPILA 预期为 0）"
    return long


def filter_binary(long: pd.DataFrame) -> pd.DataFrame:
    """
    排除 suspect（Diagnosis==2），生成青光眼二分类标签。

    Args:
        long: 全量长表。

    Returns:
        含 label（0/1）列的二分类 DataFrame（预期 420 行 / 210 患者）。
    """
    df = long[long["Diagnosis"].isin([0, 1])].copy()
    df["label"] = (df["Diagnosis"] == 1).astype("int64")
    n_pos = int(df["label"].sum())
    print(
        f"排除 suspect 后：{len(df)} 眼-图 / {df['pid'].nunique()} 患者，"
        f"阳性(青光眼) {n_pos}（阳性率 {df['label'].mean():.3f}）"
    )
    assert len(df) == EXPECT_N_IMAGES, (
        f"对齐 MEDFAIR 失败：应为 {EXPECT_N_IMAGES} 眼-图，实际 {len(df)}"
    )
    assert df["pid"].nunique() == EXPECT_N_PATIENTS, (
        f"对齐 MEDFAIR 失败：应为 {EXPECT_N_PATIENTS} 患者，实际 {df['pid'].nunique()}"
    )
    assert n_pos == EXPECT_N_POSITIVE, (
        f"对齐 MEDFAIR 失败：应为 {EXPECT_N_POSITIVE} 阳性，实际 {n_pos}"
    )
    return df


def encode_attributes(df: pd.DataFrame) -> pd.DataFrame:
    """
    编码敏感属性：sex（Gender 原始即 0=男/1=女）、age_group（60 岁二分）、joint。

    Args:
        df: 二分类 DataFrame，需含 Gender / Age 列。

    Returns:
        新增 sex / age_group / joint 列的 DataFrame。
    """
    df = df.copy()
    # Gender 原始编码 0=男/1=女（已由青光眼率与 MEDFAIR Table A3 对齐验证）
    df["sex"] = df["Gender"].astype("int64")
    # age_group: <=60 -> 0，>60 -> 1（对齐 MEDFAIR「0-60 / 60+」，边界 60 归组 0）
    df["age_group"] = (df["Age"] > 60).astype("int64")
    df["joint"] = (df["sex"] * 2 + df["age_group"]).astype("int64")

    sex_dist = {("M" if k == 0 else "F"): int((df["sex"] == k).sum()) for k in (0, 1)}
    ag_dist = {("<=60" if k == 0 else ">60"): int((df["age_group"] == k).sum())
               for k in (0, 1)}
    print(f"sex 编码（M=0,F=1）：{sex_dist}")
    print(f"age_group 编码（<=60->0,>60->1）：{ag_dist}")
    for col, name in [("sex", "性别"), ("age_group", "年龄箱")]:
        rates = df.groupby(col)["label"].mean().round(3).to_dict()
        print(f"  子群青光眼率 by {name}：{rates}")
    return df


def maybe_build_resize_cache(df: pd.DataFrame) -> dict[str, str]:
    """
    （可选）把原始大图降采样为 256×256 RGB JPEG 缓存，返回 img 文件名 -> 缓存绝对路径映射。

    Args:
        df: 含 pid / eye 列的 DataFrame。

    Returns:
        {原始文件名: 缓存绝对路径} 映射；PRERESIZE_256=False 时返回空 dict。
    """
    if not PRERESIZE_256:
        return {}
    RESIZE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    mapping: dict[str, str] = {}
    n_done = 0
    for pid, eye in zip(df["pid"], df["eye"]):
        fname = f"RET{pid}{eye}.jpg"
        dst = RESIZE_CACHE_DIR / fname
        mapping[fname] = str(dst)
        if dst.exists():
            continue
        # 自然 RGB，convert("RGB") 即可——无需 CXR 的 16-bit min-max 处理。
        # Resize 到正方形 256×256（与 MEDFAIR/HAM 同：会压扁长宽比，保持口径一致）。
        img = Image.open(IMG_DIR / fname).convert("RGB").resize(RESIZE_TO, Image.BILINEAR)
        img.save(dst, quality=95)
        n_done += 1
    print(
        f"256×256 缓存：新生成 {n_done} 张，复用 {len(mapping) - n_done} 张 -> {RESIZE_CACHE_DIR}"
    )
    return mapping


def build_split_frame(df: pd.DataFrame, cache_map: dict[str, str]) -> pd.DataFrame:
    """
    抽取下游 Dataset 类需要的列，image_path 指向缓存（若启用）或原图。

    Args:
        df: 已完成标签与属性编码的 DataFrame。
        cache_map: 原始文件名 -> 256 缓存路径的映射（可为空）。

    Returns:
        最终输出 DataFrame。
    """
    fnames = "RET" + df["pid"] + df["eye"] + ".jpg"
    if cache_map:
        image_path = fnames.map(cache_map)
    else:
        image_path = fnames.map(lambda f: str(IMG_DIR / f))
    out = pd.DataFrame({
        "image_path": image_path.values,
        "label": df["label"].values,
        "sex": df["sex"].values,
        "age": df["Age"].values,
        "age_group": df["age_group"].values,
        "joint": df["joint"].values,
        "pid": df["pid"].values,
        "eye": df["eye"].values,
        "diagnosis": df["Diagnosis"].values,
    })
    return out


def verify_images_exist(out: pd.DataFrame) -> None:
    """
    校验每行 image_path 指向的文件均存在。

    Args:
        out: 含 image_path 列的输出 DataFrame。
    """
    exists = out["image_path"].map(lambda p: Path(p).exists())
    n_missing = int((~exists).sum())
    assert n_missing == 0, f"有 {n_missing} 张图像缺失"
    print(f"图像存在性校验通过：{len(out)} 张全部存在")


def split_grouped_by_patient(out: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """
    患者级 GroupShuffleSplit 70/10/20：同一 pid 双眼整组只进一个 split。

    先切出 test(20%)，再从剩余 80% 中按 1/8 切出 val（=总体 10%），其余为 train。

    Args:
        out: 待划分的完整输出 DataFrame（需含 pid）。

    Returns:
        {"train": ..., "val": ..., "test": ...} 三个互斥子集（pid 不重叠）。
    """
    groups = out["pid"].values
    gss_test = GroupShuffleSplit(
        n_splits=1, test_size=SPLIT_FRACTIONS["test"], random_state=RANDOM_STATE
    )
    trainval_idx, test_idx = next(gss_test.split(out, groups=groups))
    trainval = out.iloc[trainval_idx].reset_index(drop=True)
    test = out.iloc[test_idx].reset_index(drop=True)

    # val 占总体 10% -> 占 trainval(80%) 的 1/8 = 0.125
    val_frac_within = SPLIT_FRACTIONS["val"] / (
        SPLIT_FRACTIONS["train"] + SPLIT_FRACTIONS["val"]
    )
    gss_val = GroupShuffleSplit(
        n_splits=1, test_size=val_frac_within, random_state=RANDOM_STATE
    )
    train_idx, val_idx = next(
        gss_val.split(trainval, groups=trainval["pid"].values)
    )
    train = trainval.iloc[train_idx].reset_index(drop=True)
    val = trainval.iloc[val_idx].reset_index(drop=True)
    return {"train": train, "val": val, "test": test}


def assert_no_patient_leakage(splits: dict[str, pd.DataFrame]) -> None:
    """
    校验任意两个 split 之间不存在共享的 pid（双眼防泄漏）。

    Args:
        splits: split 字典。
    """
    sets = {name: set(s["pid"]) for name, s in splits.items()}
    for a in ("train", "val", "test"):
        for b in ("val", "test"):
            if a >= b:
                continue
            overlap = sets[a] & sets[b]
            assert not overlap, f"{a} 与 {b} 共享 {len(overlap)} 个 pid（存在泄漏）"
    print("患者级无泄漏校验通过：train/val/test 三者 pid 互不重叠")


def write_splits(splits: dict[str, pd.DataFrame], output_dir: Path) -> None:
    """
    写出各 split CSV，并打印规模、阳性率与子群分布。

    Args:
        splits: 划分结果。
        output_dir: 输出目录。
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, subset in splits.items():
        subset.to_csv(output_dir / f"{name}.csv", index=False)
        sex_dist = {("M" if k == 0 else "F"): int((subset["sex"] == k).sum())
                    for k in (0, 1)}
        ag_dist = {("<=60" if k == 0 else ">60"): int((subset["age_group"] == k).sum())
                   for k in (0, 1)}
        print(
            f"  {name:<6s}: {len(subset):>4d} 眼-图 / {subset['pid'].nunique():>3d} 患者  "
            f"青光眼率 {subset['label'].mean():.2%}  sex={sex_dist}  age={ag_dist}"
        )


def build_cv_folds(out: pd.DataFrame, n_folds: int) -> None:
    """
    患者级 GroupKFold 交叉验证划分：每折以 1 折作 test(≈1/K)，其余按 70/10/20 口径从
    train+val 中再用 GroupShuffleSplit 切出 val（占总体≈10%），其余为 train。

    PAPILA n=420 极小、单次划分方差大（见 docs/papila_preprocessing_plan.md §2/§6），
    K 折合并是更稳健的评估口径，也是条件互信息 I(Y;A|X) 估计保统计功效的前提。
    GroupKFold 确定性（不依赖 random_state）；val 的切分用固定 RANDOM_STATE。

    Args:
        out: 完整输出 DataFrame（需含 pid）。
        n_folds: 折数（推荐 5）。

    输出:
        data/splits/papila/cv{K}/fold{k}/{train,val,test}.csv
    """
    gkf = GroupKFold(n_splits=n_folds)
    groups = out["pid"].values
    cv_root = OUTPUT_DIR / f"cv{n_folds}"
    val_frac_within = SPLIT_FRACTIONS["val"] / (
        SPLIT_FRACTIONS["train"] + SPLIT_FRACTIONS["val"]
    )
    print(f"\n生成 {n_folds} 折患者级 GroupKFold -> {cv_root}")
    for k, (trainval_idx, test_idx) in enumerate(gkf.split(out, groups=groups)):
        trainval = out.iloc[trainval_idx].reset_index(drop=True)
        test = out.iloc[test_idx].reset_index(drop=True)
        gss_val = GroupShuffleSplit(
            n_splits=1, test_size=val_frac_within, random_state=RANDOM_STATE
        )
        tr_idx, va_idx = next(gss_val.split(trainval, groups=trainval["pid"].values))
        splits = {
            "train": trainval.iloc[tr_idx].reset_index(drop=True),
            "val": trainval.iloc[va_idx].reset_index(drop=True),
            "test": test,
        }
        assert_no_patient_leakage(splits)
        fold_dir = cv_root / f"fold{k}"
        print(f"[fold {k}]")
        write_splits(splits, fold_dir)


def main() -> None:
    """主函数：读取 -> 二分类筛选 -> 属性编码 -> (可选)缓存 -> 划分 -> 写出。"""
    parser = argparse.ArgumentParser(description="构建 PAPILA 青光眼二分类 split")
    parser.add_argument(
        "--cv", type=int, default=0,
        help="K 折患者级 GroupKFold（推荐 5）；0=只生成单次 70/10/20 划分（默认）。"
             "二者可叠加：>0 时同时生成单次划分与 cv{K}/ 折划分。",
    )
    args = parser.parse_args()

    long = load_all_eyes()
    df = filter_binary(long)
    df = encode_attributes(df)
    cache_map = maybe_build_resize_cache(df)
    out = build_split_frame(df, cache_map)
    verify_images_exist(out)

    # 始终生成单次 70/10/20 划分（与 MEDFAIR 数字对表用）
    splits = split_grouped_by_patient(out)
    assert_no_patient_leakage(splits)
    print(f"写出单次 split CSV -> {OUTPUT_DIR}")
    write_splits(splits, OUTPUT_DIR)

    # 可选：K 折 CV（小样本稳健评估，推荐）
    if args.cv and args.cv >= 2:
        build_cv_folds(out, args.cv)

    print("完成。")


if __name__ == "__main__":
    main()
