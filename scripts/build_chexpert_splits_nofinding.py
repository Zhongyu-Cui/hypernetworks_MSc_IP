"""
CheXpert 预处理（MEDFAIR 对齐版）：生成 No Finding 二分类任务的 train/val/test split CSV。

结构与 build_mimic_splits_nofinding.py 同构（CheXpert-Plus 在 CXR7-1M master CSV 里
与 MIMIC-CXR 字段完全一致），仅改 Dataset_name / 输出目录，并提供两个开关。

⚠️ 标签来源修正（2026-06-29，弃用 master 的 findings 段标签）
─────────────────────────────────────────────
  旧版本直接用 master CSV 的 `No Finding` 列（U-Zeros {0:0,2:1}），正例率 75.7%。
  经核查，master 的 No Finding 来自 CheXbert 在**报告 findings 段**上的标注，而
  CheXpert 报告诊断内容多在 impression 段、findings 段常为空 → CheXbert 默认判
  No Finding=Positive，导致 75.7% 的「健康」是空 findings 段伪影，**不是疾病真值**。
  （旧版「No Finding=2 仅 0.6% 含病理阳性」的交叉验证是循环论证：标签与病理同源于
  同一段，findings 段空则全部为空，必然一致。）

  本版改用官方 chexbert_labels.zip 的 **report_fixed.json（全报告标注）**，与 MIMIC
  官方 `mimic-cxr-2.0.0-chexpert.csv`（全报告 CheXpert labeler）**同口径**，使
  MIMIC↔CheXpert OOD 对照在标签定义上一致。全报告正例率 ~5.8%（接近原版 CheXpert）。
  可用 LABEL_SECTION 切到 "impression"（~8.4%，≈ 原版 CheXpert 论文口径）或回退
  "findings"（旧伪影口径，仅供对照）。

  目标标签  : No Finding（report 段 Positive(1.0)→健康(1)，其余/缺失→有病灶(0)）
  体位过滤  : 仅保留正面位（PA / AP），LATERAL 剔除
  Sex 编码  : Male=0, Female=1；排除 Unknown
  Race 编码 : 二分类 White=0, Non-White(Asian/Black)=1；排除 Unknown Race
  数据划分  : 使用 CXR7-1M 自带的患者级 Split 列（已验证无跨 split 患者重叠）

⚠️ 与 MIMIC 的关键差异（务必注意）
─────────────────────────────────────────────
  类别平衡：全报告口径下 CheXpert 正例率 ~5.8%（少数类=健康），MIMIC ~33%。
  二者标签法一致（全报告 CheXbert），患病率差异是真实人群差异（CheXpert 门诊偏多）。
  过采样按「少数类」实现，在 CheXpert 上等价于上采样「健康(label=1)」。

  Race=Unknown 占比高（27%，52k 行）。EXCLUDE_RACE_UNKNOWN 控制是否排除：
    True  → 与 MIMIC race 任务对齐（White/Non-White 二分类），样本 ~138,644
    False → 仅 sex/age 任务时保留全部 ~190,836（race 列含 -1 占位，下游需过滤）

开关
────
  EXCLUDE_RACE_UNKNOWN : 默认 True（MEDFAIR race 对齐）
  WRITE_OVERSAMPLED    : 默认 False（先只产出人群真实分布的 split，不写重采样 CSV）

图像加载/Transform 与 MIMIC 完全一致（16-bit PNG，禁用 convert("L")；不水平翻转），
详见仓库 CLAUDE.md「图像加载」节，本脚本只负责生成 split CSV。

输出
────
  data/splits/chexpert_nofinding/{train,val,test}.csv
  data/splits/chexpert_nofinding/train_oversampled.csv   仅当 WRITE_OVERSAMPLED=True
  每行字段：image_path（绝对路径）, label, sex, race, age, view, patient_id
"""

import argparse
import json
import zipfile
from pathlib import Path

import pandas as pd
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from src.paths import CHEXPERT_META_ROOT, CXR_DATA_ROOT, SPLITS_DIR

RANDOM_STATE = 42
CV_VAL_FRAC = 0.10   # CV 每折 val 目标占总体比例（对齐 HAM/Fitz CV：val≈10%，train≈其余）

BASE_PATH = CXR_DATA_ROOT
CSV_PATH = BASE_PATH / "cxr" / "cxr7-1m_master.csv"
OUTPUT_DIR = SPLITS_DIR / "chexpert_nofinding"

DATASET_NAME = "CheXpert-Plus"
TARGET_LABEL = "No Finding"
FRONTAL_VIEWS = ["PA", "AP"]

# ── 标签来源（官方 CheXbert 标注，按报告段区分）──────────────────────────
# master CSV 的 No Finding 列用的是 findings 段（75.7% 伪影），本脚本改读官方 zip。
CHEXBERT_ZIP = CHEXPERT_META_ROOT / "chexbert_labels.zip"
SECTION_TO_JSONL = {
    "report": "report_fixed.json",          # 全报告（与 MIMIC 官方同口径）正例率 ~5.8% —— 默认
    "impression": "impression_fixed.json",  # 仅 impression（≈ 原版 CheXpert 论文）正例率 ~8.4%
    "findings": "findings_fixed.json",       # 仅 findings（master 旧伪影口径）正例率 ~75.7%，已弃用
}
LABEL_SECTION = "impression"   # impression 段（≈ 原版 CheXpert 论文口径，正例率 ~8.4%）；
                               # 注：report 全报告口径(5.8%)经实测训练坍缩，改用 impression
                               # （2026-06-29，详见 outputs/chexpert_cxr/CLAUDE.md 台账）

# ── 开关 ──────────────────────────────────────────────────────────────
EXCLUDE_RACE_UNKNOWN = True   # True: 排除 Race=Unknown，与 MEDFAIR White/Non-White 对齐
WRITE_OVERSAMPLED = False     # False: 先只产出无重采样的 split（按需改 True 生成 train_oversampled.csv）

SEX_TO_CODE = {"Male": 0, "Female": 1}

# White=0，所有非 White（Asian/Black）合并为 1（Non-White）；Unknown 视开关排除或置 -1
RACE_TO_CODE_MEDFAIR = {"White": 0, "Black": 1, "Asian": 1}

SPLIT_NAME_MAP = {"train": "train", "valid": "val", "test": "test"}


def load_chexpert_frontal(csv_path: Path) -> pd.DataFrame:
    """
    加载 master CSV，筛选出 CheXpert-Plus 中的正面位（PA/AP）影像。

    Args:
        csv_path: cxr7-1m_master.csv 的完整路径。

    Returns:
        仅含 CheXpert-Plus 正面位影像的 DataFrame。
    """
    print(f"读取 CSV：{csv_path}")
    df = pd.read_csv(csv_path, low_memory=False)
    df = df[df["Dataset_name"] == DATASET_NAME]
    print(f"{DATASET_NAME} 总样本数：{len(df):,}")

    df = df[df["View_name"].isin(FRONTAL_VIEWS)].reset_index(drop=True)
    print(f"正面位（PA/AP）样本数：{len(df):,}")
    return df


def _master_path_to_label_key(image_path_relative: str) -> str:
    """
    把 master ImagePath 转成与官方 CheXbert 标签 path_to_image 对齐的 key。

    master : cxr/chexpert/512x512/train/patientXXXX/studyY/viewZ_frontal.png
    标签   :                       train/patientXXXX/studyY/viewZ_frontal.jpg

    Args:
        image_path_relative: master CSV 'ImagePath' 列的相对路径。

    Returns:
        与 chexbert_labels.zip 内 path_to_image 一致的相对路径字符串。
    """
    tail = image_path_relative.split("512x512/", 1)[1]
    return tail.replace(".png", ".jpg")


def load_no_finding_labels(section: str) -> dict:
    """
    从官方 chexbert_labels.zip 读取指定报告段的 No Finding CheXbert 标注。

    每个 JSONL 行形如 {"path_to_image": "...jpg", "No Finding": 1.0/None, ...}，
    其中 No Finding==1.0 表示报告该段判为「无异常发现/健康」，None 表示未判为健康。

    Args:
        section: 报告段名，取 SECTION_TO_JSONL 的 key（report/impression/findings）。

    Returns:
        {path_to_image: No Finding 原始值(1.0 或 None)} 的字典。
    """
    member = SECTION_TO_JSONL[section]
    labels: dict = {}
    with zipfile.ZipFile(CHEXBERT_ZIP) as zf, zf.open(member) as fh:
        for raw in fh:
            obj = json.loads(raw)
            labels[obj["path_to_image"]] = obj.get("No Finding")
    print(f"读取官方 CheXbert 标签 [{section} 段]：{member}（{len(labels):,} 条）")
    return labels


def apply_report_labels(df: pd.DataFrame, section: str) -> pd.DataFrame:
    """
    用官方 CheXbert 报告段标注（默认全报告）重建 No Finding 二分类标签。

    替代旧的 U-Zeros（master findings 段，75.7% 伪影）。No Finding=Positive(1.0)→
    健康(1)，其余（None/未判为健康）→ 有病灶(0)。按 ImagePath 与官方标签 1:1 对齐，
    要求全部匹配（否则抛 AssertionError）。

    Args:
        df: 仅含 CheXpert-Plus 正面位影像的 DataFrame，需含 'ImagePath' 列。
        section: 报告段名（report/impression/findings）。

    Returns:
        新增 'label' 列（int，0=有病灶, 1=无病灶/健康）的 DataFrame。
    """
    df = df.copy()
    label_map = load_no_finding_labels(section)
    keys = df["ImagePath"].map(_master_path_to_label_key)

    matched = keys.isin(label_map)
    assert matched.all(), (
        f"{(~matched).sum():,} 行 ImagePath 无法匹配官方 CheXbert 标签 path_to_image，"
        f"请检查 key 拼接逻辑或标签来源。"
    )

    nf = keys.map(label_map)
    # No Finding==1.0（该段判为健康）→ 1；其余（None）→ 0（有病灶）
    df["label"] = (nf == 1.0).astype("int64")
    pos_rate = df["label"].mean()
    n_pos = int(df["label"].sum())
    n_neg = len(df) - n_pos
    print(
        f"全报告标注重建 '{TARGET_LABEL}' [{section} 段]：正例（无病灶/健康）占比 {pos_rate:.1%}"
        f"（正例 {n_pos:,} / 负例 {n_neg:,}，共 {len(df):,}）"
    )
    return df


def encode_attributes_medfair(df: pd.DataFrame) -> pd.DataFrame:
    """
    对齐 MEDFAIR：Sex（Male=0, Female=1）与 Race（White=0, Non-White=1）。

    Sex=Unknown 始终排除。Race=Unknown 视 EXCLUDE_RACE_UNKNOWN：
      True  → 排除该行；
      False → 保留并把 race 置 -1（占位，下游按需过滤），用于仅 sex/age 任务。

    Args:
        df: 输入 DataFrame，需含 'Sex_name' 与 'Race_name' 列。

    Returns:
        新增 'sex'（0/1）与 'race'（0/1，或 -1）整数列的 DataFrame。
    """
    n_before = len(df)
    df = df[df["Sex_name"].isin(SEX_TO_CODE.keys())].copy()
    print(f"排除 Sex=Unknown：{n_before - len(df):,} 行（剩余 {len(df):,}）")

    if EXCLUDE_RACE_UNKNOWN:
        n_before = len(df)
        df = df[df["Race_name"].isin(RACE_TO_CODE_MEDFAIR.keys())].copy()
        print(
            f"排除 Race=Unknown（无法归入 White/Non-White）：{n_before - len(df):,} 行"
            f"（剩余 {len(df):,}）"
        )
    else:
        print("保留 Race=Unknown（race 置 -1，仅供 sex/age 任务；race 任务须先过滤）")

    df["sex"] = df["Sex_name"].map(SEX_TO_CODE).astype("int64")
    # Unknown 在保留模式下映射为 -1
    df["race"] = df["Race_name"].map(RACE_TO_CODE_MEDFAIR).fillna(-1).astype("int64")
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
        race_label = {-1: "Unknown", 0: "White", 1: "Non-White"}
        race_str = ", ".join(f"{race_label.get(k, k)}={v:,}" for k, v in race_dist.items())
        print(
            f"  {out_name:<6s}: {len(subset):>7,} 行  "
            f"正例率(无病灶) {subset['label'].mean():.1%}  "
            f"sex=(M={subset['sex'].eq(0).sum():,}, F={subset['sex'].eq(1).sum():,})  "
            f"race=({race_str})"
        )


def oversample_train_split(train_df: pd.DataFrame, output_dir: Path) -> None:
    """
    仅对训练集做随机过采样（按少数类）以缓解类别不平衡，写出 train_oversampled.csv。

    全报告标注下 CheXpert 少数类是「健康(label=1)」（正例率 ~5.8%），与 MIMIC 同向。
    这里按「少数类」上采样——等价于上采样「健康(label=1)」，使两类持平至约 50/50。
    **只处理训练集**，验证/测试集保持人群真实分布。原始 train.csv 不被覆盖。

    由 WRITE_OVERSAMPLED 开关控制是否调用（默认 False，不生成此文件）。

    Args:
        train_df: 原始（未过采样）训练集 DataFrame，需含 'label' 列。
        output_dir: 输出目录（与 train.csv 相同）。
    """
    pos_rate = train_df["label"].mean()
    minority_label = 1 if pos_rate < 0.5 else 0
    majority_label = 1 - minority_label

    minority = train_df[train_df["label"] == minority_label]
    majority = train_df[train_df["label"] == majority_label]

    minority_oversampled = minority.sample(
        n=len(majority), replace=True, random_state=RANDOM_STATE,
    )
    balanced = (
        pd.concat([majority, minority_oversampled])
        .sample(frac=1.0, random_state=RANDOM_STATE)
        .reset_index(drop=True)
    )

    out_path = output_dir / "train_oversampled.csv"
    balanced.to_csv(out_path, index=False)
    print(
        f"\n训练集随机过采样（少数类 label={minority_label}，有放回采样至与多数类持平，"
        f"random_state={RANDOM_STATE}）：\n"
        f"  原始 train.csv     : {len(train_df):>7,} 行（正例率 {pos_rate:.1%}，"
        f"少数类 {len(minority):,} / 多数类 {len(majority):,}）\n"
        f"  train_oversampled  : {len(balanced):>7,} 行（正例率 {balanced['label'].mean():.1%}） "
        f"-> {out_path}"
    )


def _write_cv_fold(splits: dict, fold_dir: Path) -> None:
    """把某折的 {train,val,test} DataFrame（已去 split 列语义）写出为 CSV（丢弃 split 列）。"""
    fold_dir.mkdir(parents=True, exist_ok=True)
    for name, subset in splits.items():
        subset.drop(columns=["split"], errors="ignore").reset_index(drop=True).to_csv(
            fold_dir / f"{name}.csv", index=False)
    n = {k: len(v) for k, v in splits.items()}
    print(f"  {fold_dir.name}: train {n['train']:,} / val {n['val']:,} / test {n['test']:,}  "
          f"（test 正例率 {splits['test']['label'].mean():.1%}）")


def build_cv_folds(out: pd.DataFrame, n_folds: int) -> None:
    """
    患者级 GroupKFold 交叉验证划分（CV-OOF 口径铺开）：把全数据集按 patient_id 分 K 折，每折 1 折作
    test（≈1/K），其余 K-1 折为 train+val，再用 GroupShuffleSplit 从中按患者切出 val（占总体≈10%）。

    动机（`docs/oof_selection_rollout_plan.md`）：单-split CheXpert test（12,198）的少数子群（如
    Non-White×age 交叉）评估 n 有限；5 折 disjoint test 池化成全 ~138,644 OOF 后，少数子群评估 n
    放大约 K 倍，把 worst-group 从评估-n 噪声地板抬起。同一 patient_id 整组只进一折的 test/val/train
    （GroupKFold + GroupShuffleSplit 均按 patient_id 分组），杜绝同患者多片跨 split 泄漏。

    ⚠️ 与单-split 完全隔离：CV 折写入 cv{K}/ 子目录；训练时 output_dir 亦重定向到
    outputs/chexpert_cxr/cv{K}/，不覆盖已冻结的单-split 结果。忽略 master 自带 Split 列，按 patient_id
    重新分折（CV 需全数据集重划，built-in split 只是其中一种划法）。

    Args:
        out: 完整输出 DataFrame（需含 'patient_id'，全 ~138,644 行）。
        n_folds: 折数（本方案用 5）。

    输出:
        data/splits/chexpert_nofinding/cv{K}/fold{k}/{train,val,test}.csv
    """
    gkf = GroupKFold(n_splits=n_folds)
    groups = out["patient_id"].values
    cv_root = OUTPUT_DIR / f"cv{n_folds}"
    # val 目标占总体≈CV_VAL_FRAC：trainval=(K-1)/K，故 val 占 trainval 比例 = CV_VAL_FRAC / ((K-1)/K)
    val_frac_within = CV_VAL_FRAC / (1.0 - 1.0 / n_folds)
    print(f"\n患者级 {n_folds} 折 GroupKFold -> {cv_root}"
          f"（每折 test≈{100 / n_folds:.0f}%，val≈{CV_VAL_FRAC:.0%}，train≈其余）")
    for k, (trainval_idx, test_idx) in enumerate(gkf.split(out, groups=groups)):
        trainval = out.iloc[trainval_idx].reset_index(drop=True)
        test = out.iloc[test_idx].reset_index(drop=True)
        gss_val = GroupShuffleSplit(n_splits=1, test_size=val_frac_within, random_state=RANDOM_STATE)
        tr_idx, va_idx = next(gss_val.split(trainval, groups=trainval["patient_id"].values))
        splits = {
            "train": trainval.iloc[tr_idx].reset_index(drop=True),
            "val": trainval.iloc[va_idx].reset_index(drop=True),
            "test": test,
        }
        # 折内患者无跨 train/val/test 泄漏
        tr_p, va_p, te_p = (set(splits[s]["patient_id"]) for s in ("train", "val", "test"))
        assert not (tr_p & va_p) and not (tr_p & te_p) and not (va_p & te_p), f"fold{k} 患者跨 split 泄漏"
        _write_cv_fold(splits, cv_root / f"fold{k}")
    # 交叉校验：K 折 test 并集恰覆盖全部（disjoint 且完整），OOF 池化正确性前提
    test_sizes = [len(pd.read_csv(cv_root / f"fold{k}" / "test.csv")) for k in range(n_folds)]
    assert sum(test_sizes) == len(out), (
        f"{n_folds} 折 test 并集应=全数据集 {len(out)}，实得 {sum(test_sizes)}（{test_sizes}）")
    print(f"OOF 完整性校验通过：{n_folds} 折 test 并集 = {sum(test_sizes)} = 全数据集 {len(out)}")


def main() -> None:
    """主函数：依次执行筛选、标签处理、属性编码、划分写出（过采样、CV 折按开关）。"""
    parser = argparse.ArgumentParser(description="构建 CheXpert No Finding 二分类 split")
    parser.add_argument(
        "--cv", type=int, default=0,
        help="K 折患者级 GroupKFold（CV-OOF 用 5）；0=只用 master 自带单-split（默认）。"
             "二者可叠加：>0 时同时生成单-split 与 cv{K}/ 折划分（互不覆盖）。",
    )
    args = parser.parse_args()

    df = load_chexpert_frontal(CSV_PATH)
    df = apply_report_labels(df, LABEL_SECTION)
    df = encode_attributes_medfair(df)

    split_df = build_split_frame(df)
    verify_no_patient_overlap(split_df)

    print(f"\n按 master CSV 自带 Split 列拆分并写出至：{OUTPUT_DIR}")
    write_splits(split_df, OUTPUT_DIR)

    # 可选：K 折患者级 GroupKFold CV（CV-OOF 口径铺开；与单-split 隔离写 cv{K}/）
    if args.cv and args.cv >= 2:
        build_cv_folds(split_df, args.cv)

    if WRITE_OVERSAMPLED:
        train_df = split_df[split_df["split"] == "train"].drop(columns=["split"]).reset_index(drop=True)
        oversample_train_split(train_df, OUTPUT_DIR)
    else:
        print("\nWRITE_OVERSAMPLED=False：本次不生成 train_oversampled.csv（仅人群真实分布 split）。")

    print("\n完成。图像加载/Transform 与 MIMIC 一致（16-bit PNG，禁用 convert(\"L\")；不水平翻转），")
    print("详见仓库 CLAUDE.md「图像加载」节与 src/datasets/mimic_cxr_dataset.py。")


if __name__ == "__main__":
    main()
