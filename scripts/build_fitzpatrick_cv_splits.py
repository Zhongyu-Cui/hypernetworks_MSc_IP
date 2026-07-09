"""
Fitzpatrick17k 5 折 StratifiedKFold split（实验 F CV-OOF）
=========================================================
把单-split 的 train/val/test 池化回全 16012 行，按 skin×label（12 层）分层做 5 折 StratifiedKFold：
每折 test = held-out 1/5，其余 4/5 再分层切 90/10 得 train/val。5 折 disjoint test 池化 = 全 16012 OOF，
把肤色 VI（n≈635）等小子群评估 n 放大 ~5×，消除单-split worst-group 的评估-n 噪声地板。

Fitzpatrick 无 patient 分组（每 md5hash 唯一），故用普通 StratifiedKFold（区别于 HAM 的 lesion 级
GroupKFold）。fold k ↔ 训练 seed 42+k（对齐 slurm/c1_ham_cv_oof.sh 约定，避免 run_id 覆盖）。

输出：data/splits/fitzpatrick17k/cv5/fold{0..4}/{train,val,test}.csv（列与单-split 完全一致）。
"""
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit

REPO_ROOT = Path(__file__).resolve().parents[1]
SPLIT_DIR = REPO_ROOT / "data" / "splits" / "fitzpatrick17k"
K = 5
VAL_FRAC = 0.10          # 每折 train 中再切出的 val 比例（对剩余 4/5 分层抽）
SEED = 42


def load_full() -> pd.DataFrame:
    """池化单-split 三个 CSV → 全 16012 行（断言对齐论文计数）。"""
    parts = [pd.read_csv(SPLIT_DIR / f"{s}.csv") for s in ("train", "val", "test")]
    df = pd.concat(parts, ignore_index=True)
    assert len(df) == 16012, f"池化后应为 16012 行，实得 {len(df)}"
    assert df["md5hash"].is_unique, "md5hash 应唯一（无 patient 泄漏问题）"
    return df


def strat_key(df: pd.DataFrame) -> pd.Series:
    """分层键 = skin×label（12 层），同时保 6 肤色与 malignant 平衡。"""
    return df["skin"].astype(int) * 2 + df["label"].astype(int)


def build() -> None:
    """生成 cv5/fold{k}/{train,val,test}.csv，并打印每折规模与肤色分布核对。"""
    df = load_full().reset_index(drop=True)
    y = strat_key(df)
    skf = StratifiedKFold(n_splits=K, shuffle=True, random_state=SEED)

    for k, (rest_idx, test_idx) in enumerate(skf.split(df, y)):
        test_df = df.iloc[test_idx]
        rest_df = df.iloc[rest_idx]
        # 剩余 4/5 再分层切 train/val（val=10%）
        sss = StratifiedShuffleSplit(n_splits=1, test_size=VAL_FRAC, random_state=SEED)
        tr_rel, val_rel = next(sss.split(rest_df, strat_key(rest_df)))
        train_df = rest_df.iloc[tr_rel]
        val_df = rest_df.iloc[val_rel]

        # 无泄漏断言：三集 md5hash 互斥
        s_tr, s_va, s_te = set(train_df.md5hash), set(val_df.md5hash), set(test_df.md5hash)
        assert not (s_tr & s_va) and not (s_tr & s_te) and not (s_va & s_te), f"fold{k} 三集 md5 重叠"

        out = SPLIT_DIR / f"cv{K}" / f"fold{k}"
        out.mkdir(parents=True, exist_ok=True)
        train_df.to_csv(out / "train.csv", index=False)
        val_df.to_csv(out / "val.csv", index=False)
        test_df.to_csv(out / "test.csv", index=False)

        skin_te = test_df["skin"].value_counts().sort_index().to_dict()
        print(f"fold{k}: train={len(train_df):>5} val={len(val_df):>4} test={len(test_df):>4} "
              f"| test malignant={test_df.label.mean():.3f} | test skin(I..VI)={skin_te}")

    # 5 折 test 池化 = 全 16012、disjoint 断言
    pooled = pd.concat([pd.read_csv(SPLIT_DIR / f"cv{K}" / f"fold{k}" / "test.csv")
                        for k in range(K)], ignore_index=True)
    assert len(pooled) == 16012 and pooled["md5hash"].is_unique, "5 折 test 池化未覆盖全集或有重叠"
    print(f"\n✅ 5 折 test 池化 = {len(pooled)} 行 disjoint，覆盖全 16012 → CV-OOF 就绪")


if __name__ == "__main__":
    build()
