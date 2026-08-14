"""
DeepView 决策边界可视化：HAM10000 age-conditioned HyperAdapt 各年龄组对比
==========================================================================
用一个「紧凑版 DeepView」（Schulz et al., IJCAI 2020, arXiv:1909.09154）把
ResNet18HyperAdaptAge（只条件化 age_group∈{0..3}）在 HAM10000 malignant 二分类上的
**决策边界随年龄组的移动**可视化出来。

DeepView 三部件（本脚本自实现，仅依赖 umap-learn）：
  1. Fisher 判别距离：两样本的距离 = 沿其连线插值、累加相邻点上网络输出分布的
     Fisher-Rao 测地距离 d(p,q)=2·arccos(Σ√(p·q))。二分类下这条距离在决策边界
     (p≈0.5) 附近急剧增大——正是它让「边界」在 2D 里显形（而普通像素/特征距离不会）。
  2. 判别式降维：把该距离矩阵喂给 UMAP(metric='precomputed') 得 2D 正向投影。
  3. 决策背景：DeepView 原法是「2D 网格→逆映射回合成输入→过网络」。**本脚本做了一处
     贴合皮肤镜数据的偏离**：改为把「各 age 条件下真实样本的模型恶性概率」做 kNN 空间
     插值成决策场（= 逆映射取近邻的等价形式）。原因——皮肤镜图 kNN 平均后恶性特征被糊平、
     合成输入概率整体塌到良性、背景无边界（实测验证）；直接插值真实样本概率则与阈值 τ
     同分布、边界真实可见。背景色交界=决策边界；散点色（真标签）≠背景色=被分错样本。

⚠️ 已针对「DeepView 的坑」做的处理（对齐前期讨论）：
  · **同坐标系（最关键）**：HyperAdapt 会调制整个 backbone，不同 age 连特征都变，
    4 个 age 各是一张不同的网络。为可比，本脚本**只用一个锚点 age（--anchor_age，默认
    最大组 40-60=1）拟合一次投影与逆映射**，四个面板共享**完全相同的点坐标**，
    仅把「给背景网格分类时所用的 age 条件」在 0..3 间切换——于是面板间唯一变量就是
    条件化 age，直接读出边界如何为各年龄组移动。
  · **O(N²)**：Fisher 距离对样本对数平方，故 --n_samples 默认 150（分层采样保证两类
    与四个 age 组都有覆盖），前向全程 no_grad + 微批，可在单卡几分钟内跑完。
  · **逆映射是合成的、会糊**：背景是「网络对这些合成输入的判定」，非真实图像边界——
    仅作趋势定性，定量结论回到解析超平面/子群 AUC。
  · **2D 投影有损 / 随机性**：固定 --seed 与 umap random_state；换 seed 边界会变，
    重要结论需多 seed 稳健。
  · **控变量**：同一批样本、同一 UMAP 超参、同一网格范围跨面板严格一致。
  · HAM 为自然 RGB，无 CXR 16-bit / convert("L") 问题；age_group==-1 已过滤。

用法（在 medimg 环境的 GPU 机器上；纯推理，非训练）：
    python scripts/deepview_ham_hyperadapt.py \
        --ckpt outputs/ham10000/hyperadapt_lr3e-04_wd1e-04_seed42_best_overall.pth

输出：outputs/ham10000/deepview/deepview_hyperadapt_age.png（四面板）+ 同名 .npz（坐标/预测）。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import umap
import yaml
from matplotlib.lines import Line2D
from PIL import Image
from torchvision import transforms

from src.models.resnet18_hyperadapt import ResNet18HyperAdaptAge

def setup_cjk_font() -> None:
    """注册系统 Noto Sans CJK 字体，避免图中中文渲染成方框（无则回退英文由调用方处理）。"""
    from matplotlib import font_manager

    for f in font_manager.fontManager.ttflist:
        if "NotoSansCJK" in f.fname or "DroidSansFallback" in f.fname:
            plt.rcParams["font.family"] = f.name
            plt.rcParams["axes.unicode_minus"] = False
            return


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "configs" / "ham10000_baseline.yaml"
DEFAULT_CKPT = (
    REPO_ROOT.parent.parent
    / "outputs" / "ham10000" / "hyperadapt_lr3e-04_wd1e-04_seed42_best_overall.pth"
)
OUTPUT_DIR = Path("/vol/biomedic2/bglocker_studproj/zc125/outputs/ham10000/deepview")

NUM_AGE = 4
AGE_LABELS = {0: "20-40", 1: "40-60", 2: "60-80", 3: "80+"}
# 类别配色（benign=蓝、malignant=红），背景与散点共用同一语义 → 点色≠背景色即分错
CLASS_COLORS = np.array([[0.13, 0.40, 0.67], [0.70, 0.09, 0.17]])  # RdBu 两端
AGE_MARKERS = {0: "o", 1: "s", 2: "^", 3: "D"}


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    p = argparse.ArgumentParser(description="DeepView boundary viz for HAM HyperAdapt (age)")
    p.add_argument("--ckpt", type=Path, default=DEFAULT_CKPT,
                   help="HyperAdapt(age) checkpoint (纯 state_dict)。默认选中配置 seed42 best_overall。")
    p.add_argument("--split", type=str, default="test", choices=["train", "val", "test"],
                   help="用哪个 split 采样可视化点（默认 test）。")
    p.add_argument("--n_samples", type=int, default=150,
                   help="参与可视化的样本数（Fisher 距离 O(N²)，默认 150，按 label×age 分层采样）。")
    p.add_argument("--anchor_age", type=int, default=1, choices=list(range(NUM_AGE)),
                   help="拟合投影/逆映射所用的锚点 age 条件（默认 1=40-60 最大组）。四面板共享此坐标。")
    p.add_argument("--fisher_steps", type=int, default=5,
                   help="Fisher 距离沿连线的插值步数 S（默认 5）。")
    p.add_argument("--grid", type=int, default=45, help="背景决策图的网格分辨率 G×G（默认 45）。")
    p.add_argument("--knn", type=int, default=6, help="逆映射 kNN 的近邻数（默认 6）。")
    p.add_argument("--micro_batch", type=int, default=64,
                   help="前向微批大小（HyperAdapt 逐样本卷积核显存大，默认 64）。")
    p.add_argument("--seed", type=int, default=42, help="随机种子（采样 + UMAP）。")
    p.add_argument("--out", type=Path, default=OUTPUT_DIR / "deepview_hyperadapt_age.png",
                   help="输出 PNG 路径（同名 .npz 存坐标/预测）。")
    return p.parse_args()


def load_model(ckpt_path: Path, device: torch.device) -> ResNet18HyperAdaptAge:
    """构建 ResNet18HyperAdaptAge 并载入 checkpoint（纯 state_dict，strict）。"""
    model = ResNet18HyperAdaptAge(
        num_classes=1, num_age=NUM_AGE, rank=4, patient_embed_dim=128,
        pretrained=False,  # 权重整体由 checkpoint 覆盖，无需下载 ImageNet
        freeze_backbone=False,
    )
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(state, strict=True)  # 键完全匹配（含 BN running stats）
    return model.to(device).eval()


def build_eval_transform() -> transforms.Compose:
    """HAM 评估 transform（Resize(256,256)→CenterCrop(224)→ToTensor→ImageNet Normalize），与训练严格一致。"""
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    ev = cfg["transforms"]["eval"]
    norm = cfg["transforms"]["normalize"]
    return transforms.Compose([
        transforms.Resize(tuple(ev["resize"])),
        transforms.CenterCrop(ev["center_crop"]),
        transforms.ToTensor(),
        transforms.Normalize(mean=norm["mean"], std=norm["std"]),
    ])


def load_samples(
    split: str, n_samples: int, transform: transforms.Compose, rng: np.random.Generator,
) -> tuple[torch.Tensor, np.ndarray, np.ndarray]:
    """
    从 HAM split CSV 分层采样 n_samples 张图（过滤 age_group>=0；按 label×age_group 分层，
    尽量覆盖两类与四个年龄组），返回 (images[N,3,224,224], labels[N], age_groups[N])。
    """
    import pandas as pd

    csv_path = REPO_ROOT / "data" / "splits" / "ham10000" / f"{split}.csv"
    df = pd.read_csv(csv_path)
    df = df[df["age_group"].astype(int) >= 0].reset_index(drop=True)  # 排除 0-20

    # 按 (label, age_group) 分层比例采样，保证少数类(恶性)与稀缺年龄组(80+)都出现
    strata = df.groupby(["label", "age_group"]).indices
    picks: list[int] = []
    per = max(1, n_samples // max(1, len(strata)))
    for _, idx in strata.items():
        take = min(len(idx), per)
        picks.extend(rng.choice(idx, size=take, replace=False).tolist())
    # 不足则从剩余随机补齐
    if len(picks) < n_samples:
        remain = np.setdiff1d(np.arange(len(df)), np.array(picks, dtype=int))
        extra = rng.choice(remain, size=min(len(remain), n_samples - len(picks)), replace=False)
        picks.extend(extra.tolist())
    picks = picks[:n_samples]

    sub = df.iloc[picks].reset_index(drop=True)
    imgs = torch.stack([
        transform(Image.open(pth).convert("RGB")) for pth in sub["image_path"].astype(str)
    ])
    labels = sub["label"].astype(int).to_numpy()
    ages = sub["age_group"].astype(int).to_numpy()
    return imgs, labels, ages


@torch.no_grad()
def predict_prob(
    model: ResNet18HyperAdaptAge, imgs: torch.Tensor, age: int,
    device: torch.device, micro_batch: int,
) -> torch.Tensor:
    """在固定 age 条件下对一批图像做前向，返回恶性概率 p1=sigmoid(logit)（微批以省显存）。"""
    out: list[torch.Tensor] = []
    for i in range(0, imgs.shape[0], micro_batch):
        chunk = imgs[i:i + micro_batch].to(device, non_blocking=True)
        age_t = torch.full((chunk.shape[0],), age, dtype=torch.long, device=device)
        logits = model(chunk, age_t)                    # [b, 1]
        out.append(torch.sigmoid(logits.squeeze(1)).cpu())
    return torch.cat(out)


@torch.no_grad()
def fisher_distance_matrix(
    model: ResNet18HyperAdaptAge, imgs: torch.Tensor, anchor_age: int,
    steps: int, device: torch.device, micro_batch: int,
) -> np.ndarray:
    """
    计算 N×N 的 Fisher 判别距离矩阵（锚点 age 条件下）。

    对每一对 (i,j)：沿输入空间连线取 S 个插值点 x_t=(1-t)x_i+t x_j，得恶性概率 p_t，
    组成分布 [1-p_t, p_t]；相邻分布间用 Fisher-Rao 测地距离
        d(p,q) = 2·arccos( Σ_c √(p_c·q_c) )
    累加得该对的判别距离。二分类下这条距离在跨越 p≈0.5（决策边界）时最大。

    Args:
        imgs      : [N,3,224,224]（CPU）。
        anchor_age: 拟合投影所用的固定 age 条件。
        steps     : 插值步数 S。
    Returns:
        D : [N,N] 对称距离矩阵（对角 0）。
    """
    n = imgs.shape[0]
    ts = torch.linspace(0.0, 1.0, steps).view(steps, 1, 1, 1, 1)  # [S,1,1,1,1] 广播到 [S,P,3,H,W]
    iu, ju = np.triu_indices(n, k=1)                            # 上三角对索引
    dist = np.zeros(len(iu), dtype=np.float64)

    # 每个「样本对块」构造 P×S 张插值图后统一微批前向
    pairs_per_chunk = max(1, micro_batch * 4 // steps)
    for start in range(0, len(iu), pairs_per_chunk):
        ii = iu[start:start + pairs_per_chunk]
        jj = ju[start:start + pairs_per_chunk]
        xi = imgs[ii].unsqueeze(0)                    # [1,P,3,224,224]
        xj = imgs[jj].unsqueeze(0)
        interp = (1.0 - ts) * xi + ts * xj            # [S,P,3,224,224]
        s_dim, p_dim = interp.shape[0], interp.shape[1]
        flat = interp.reshape(s_dim * p_dim, *interp.shape[2:])
        p1 = predict_prob(model, flat, anchor_age, device, micro_batch)  # [S*P]
        p1 = p1.reshape(s_dim, p_dim).clamp(1e-6, 1 - 1e-6)              # [S,P]
        # Bhattacharyya 系数 → Fisher-Rao 距离，沿 S 相邻求和
        sp1, sp0 = p1.sqrt(), (1.0 - p1).sqrt()
        bc = sp1[:-1] * sp1[1:] + sp0[:-1] * sp0[1:]                     # [S-1,P]
        d_pair = 2.0 * torch.arccos(bc.clamp(-1.0, 1.0))                 # [S-1,P]
        dist[start:start + p_dim] = d_pair.sum(dim=0).numpy()

    dmat = np.zeros((n, n), dtype=np.float64)
    dmat[iu, ju] = dist
    dmat[ju, iu] = dist
    return dmat


def fit_projection(dmat: np.ndarray, seed: int) -> np.ndarray:
    """用预计算 Fisher 距离矩阵拟合 UMAP，得 2D 正向投影 [N,2]（固定 random_state 保可复现）。"""
    reducer = umap.UMAP(
        n_components=2, metric="precomputed", random_state=seed,
        n_neighbors=min(15, dmat.shape[0] - 1), min_dist=0.4,
    )
    return reducer.fit_transform(dmat)


def interpolate_field(
    grid_xy: np.ndarray, emb: np.ndarray, values: np.ndarray, knn: int,
) -> np.ndarray:
    """
    把定义在样本 2D 投影上的标量场（这里=某 age 条件下各样本的恶性概率）kNN 加权插值到网格。

    这是 DeepView「2D 网格 → 逆映射 → 过网络」在**逆映射取近邻**下的等价决策场；相比对
    kNN 平均出的合成皮肤镜图重新分类，直接插值真实样本的模型概率**避免了图像平均把恶性
    特征糊平、背景整体塌成良性**的问题，且与阈值 τ 同分布，边界真实可见。

    Args:
        grid_xy : [G², 2] 网格点坐标。
        emb     : [N, 2] 样本 2D 投影。
        values  : [N] 每个样本的标量值（恶性概率）。
        knn     : 近邻数（softmax 距离权重，带宽=近邻距离中位数）。
    Returns:
        [G²] 网格上的插值概率。
    """
    from scipy.spatial import cKDTree

    tree = cKDTree(emb)
    k = min(knn, emb.shape[0])
    dists, idx = tree.query(grid_xy, k=k)               # [G²,k]
    if k == 1:
        dists, idx = dists[:, None], idx[:, None]
    dists = np.maximum(dists, 1e-8)
    weights = np.exp(-dists / (np.median(dists) + 1e-8))
    weights /= weights.sum(axis=1, keepdims=True)
    return (weights * values[idx]).sum(axis=1)          # [G²]


def render(
    emb: np.ndarray, labels: np.ndarray, ages: np.ndarray,
    grid_probs: dict[int, np.ndarray], xx: np.ndarray, yy: np.ndarray,
    anchor_age: int, auc_anchor: float, tau: float, out_path: Path,
) -> None:
    """
    四面板渲染：共享坐标/散点，只切换背景决策图的 age 条件。

    背景按**共享的操作阈值 τ**（Youden J，从锚点预测一次性算定）分区：p1>τ 判恶性(红)、
    否则良性(蓝)；确定度=|p1-τ| 归一化（跨面板同一尺度，保证可比）→ 透明度。
    黑色虚线 = 决策边界等值线 p1=τ；面板间该线的移动即「边界随 age 组的变化」。
    用 τ 而非 0.5：恶性仅约 14%，sigmoid 几乎不过 0.5，硬阈 0.5 会让整幅背景同色、边界消失。
    """
    extent = (xx.min(), xx.max(), yy.min(), yy.max())
    # 跨面板共享的确定度尺度：所有 age 网格上 |p-τ| 的 95 分位（控变量，使面板透明度可比）
    cert_scale = float(np.percentile(
        np.abs(np.stack(list(grid_probs.values())) - tau), 95)) + 1e-8
    fig, axes = plt.subplots(2, 2, figsize=(13, 12))

    for ax, g in zip(axes.ravel(), range(NUM_AGE)):
        p1 = grid_probs[g]                              # [G,G] 恶性概率
        certainty = np.clip(np.abs(p1 - tau) / cert_scale, 0.0, 1.0)
        pred_cls = (p1 > tau).astype(int)
        rgba = np.zeros((*p1.shape, 4))
        rgba[..., :3] = CLASS_COLORS[pred_cls]
        rgba[..., 3] = 0.25 + 0.50 * certainty          # 确定度→透明度
        ax.imshow(rgba, extent=extent, origin="lower", aspect="auto", interpolation="bilinear")
        # 决策边界等值线 p1=τ
        ax.contour(xx, yy, p1, levels=[tau], colors="black", linewidths=1.4, linestyles="--")

        # 散点：真标签配色（与背景同语义），marker 形状=真实 age 组；白边突出
        for a in range(NUM_AGE):
            m = ages == a
            if not m.any():
                continue
            ax.scatter(emb[m, 0], emb[m, 1], c=CLASS_COLORS[labels[m]],
                       marker=AGE_MARKERS[a], s=48, edgecolors="white", linewidths=0.8, zorder=3)

        frac_mal = float((pred_cls == 1).mean())
        tag = "（锚点=拟合此坐标系）" if g == anchor_age else ""
        ax.set_title(f"背景条件 age_group = {g} ({AGE_LABELS[g]}){tag}\n"
                     f"网格判为恶性(p>τ)占比 {frac_mal:.0%}", fontsize=11)
        ax.set_xticks([]); ax.set_yticks([])

    # 图例
    cls_handles = [Line2D([0], [0], marker="o", color="w", markerfacecolor=CLASS_COLORS[c],
                          markersize=10, label=lab)
                   for c, lab in enumerate(["benign (0)", "malignant (1)"])]
    age_handles = [Line2D([0], [0], marker=AGE_MARKERS[a], color="w", markerfacecolor="0.4",
                          markeredgecolor="white", markersize=10, label=f"真实 age {AGE_LABELS[a]}")
                   for a in range(NUM_AGE)]
    fig.legend(handles=cls_handles + age_handles, loc="lower center", ncol=6, fontsize=9,
               frameon=False, bbox_to_anchor=(0.5, -0.01))

    fig.suptitle(
        f"DeepView 决策边界：HAM10000 · HyperAdapt(age) · 四年龄组对比（τ={tau:.3f}）\n"
        f"坐标系由锚点 age={anchor_age}({AGE_LABELS[anchor_age]}) 的 Fisher 距离一次性拟合并跨面板共享；"
        f"面板间唯一变量=背景分类所用 age 条件 | 锚点子集 AUC={auc_anchor:.3f}，操作阈值 τ={tau:.3f}(Youden)\n"
        "背景=各 age 下真实样本模型恶性概率的 kNN 插值决策场（仅定性）；点色≠背景色处即被分错样本",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0.03, 1, 0.94))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"[saved] {out_path}")


def main() -> None:
    args = parse_args()
    setup_cjk_font()
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}  ckpt={args.ckpt.name}  anchor_age={args.anchor_age}")

    model = load_model(args.ckpt, device)
    transform = build_eval_transform()
    imgs, labels, ages = load_samples(args.split, args.n_samples, transform, rng)
    print(f"采样 {imgs.shape[0]} 张（{args.split}）  label 分布={np.bincount(labels)}  "
          f"age 分布={np.bincount(ages, minlength=NUM_AGE)}")

    # 锚点子集 AUC（sanity check：确认权重/前向正常）
    from sklearn.metrics import roc_auc_score, roc_curve
    # 每个 age 条件下各样本的恶性概率（各 N 次前向）；背景决策场由它们插值而来
    sample_probs = {
        g: predict_prob(model, imgs, g, device, args.micro_batch).numpy() for g in range(NUM_AGE)
    }
    p_anchor = sample_probs[args.anchor_age]
    auc_anchor = float(roc_auc_score(labels, p_anchor)) if len(set(labels)) == 2 else float("nan")
    # 操作阈值 τ：锚点预测上的 Youden J 最优点，一次性算定后跨面板共享（背景决策区/边界线都用它）
    fpr, tpr, thr = roc_curve(labels, p_anchor)
    tau = float(thr[np.argmax(tpr - fpr)])
    print(f"锚点 age={args.anchor_age} 子集 AUC={auc_anchor:.3f}  Youden τ={tau:.3f}")
    for g in range(NUM_AGE):
        auc_g = roc_auc_score(labels, sample_probs[g]) if len(set(labels)) == 2 else float("nan")
        print(f"  age={g} 子集 AUC={auc_g:.3f}  恶性概率均值={sample_probs[g].mean():.3f}")

    # 1) Fisher 距离（锚点条件） → 2) UMAP 投影（一次，跨面板共享）
    print("计算 Fisher 判别距离矩阵 ...")
    dmat = fisher_distance_matrix(model, imgs, args.anchor_age, args.fisher_steps,
                                  device, args.micro_batch)
    print("拟合 UMAP 投影 ...")
    emb = fit_projection(dmat, args.seed)

    # 3) 网格（一次，跨面板共享）：每个 age 条件的背景=该 age 下样本恶性概率的 kNN 插值场
    pad = 0.08 * (emb.max(0) - emb.min(0))
    x_lin = np.linspace(emb[:, 0].min() - pad[0], emb[:, 0].max() + pad[0], args.grid)
    y_lin = np.linspace(emb[:, 1].min() - pad[1], emb[:, 1].max() + pad[1], args.grid)
    xx, yy = np.meshgrid(x_lin, y_lin)
    grid_xy = np.column_stack([xx.ravel(), yy.ravel()])

    grid_probs: dict[int, np.ndarray] = {}
    for g in range(NUM_AGE):
        field = interpolate_field(grid_xy, emb, sample_probs[g], args.knn)
        grid_probs[g] = field.reshape(args.grid, args.grid)
        print(f"  背景 age={g} 判恶性(p>τ)占比={float((grid_probs[g] > tau).mean()):.0%}")

    render(emb, labels, ages, grid_probs, xx, yy, args.anchor_age, auc_anchor, tau, args.out)

    # 存中间量供复现/进一步分析
    npz_path = args.out.with_suffix(".npz")
    np.savez_compressed(
        npz_path, emb=emb, labels=labels, ages=ages, p_anchor=p_anchor, tau=tau,
        grid_x=x_lin, grid_y=y_lin,
        **{f"grid_prob_age{g}": grid_probs[g] for g in range(NUM_AGE)},
    )
    print(f"[saved] {npz_path}")


if __name__ == "__main__":
    main()
