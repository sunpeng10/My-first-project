"""通用工具：归一化、Top-K overlap、Spearman、随机种子、中文字体、绘图。"""
import random

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats as sps

from . import config


# ---------------------------------------------------------------------------
# 基础工具
# ---------------------------------------------------------------------------
def set_seed(seed=config.SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def normalize_minmax(x):
    """min-max 归一化；若所有值相同则返回全 0（避免 NaN）。"""
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return x
    xmin, xmax = np.nanmin(x), np.nanmax(x)
    if not np.isfinite(xmin) or not np.isfinite(xmax) or (xmax - xmin) < 1e-12:
        return np.zeros_like(x)
    out = (x - xmin) / (xmax - xmin)
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def spearman_corr(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 2:
        return float("nan")
    return float(sps.spearmanr(a[mask], b[mask]).correlation)


def top_k_uids(score_map, k):
    """score_map: {uid: score} -> 按 score 降序取前 k 个 uid。"""
    items = sorted(score_map.items(), key=lambda kv: kv[1], reverse=True)
    return [uid for uid, _ in items[:k]]


def overlap_ratio(list_a, list_b):
    if not list_a:
        return 0.0
    return len(set(list_a) & set(list_b)) / len(list_a)


# ---------------------------------------------------------------------------
# 中文字体
# ---------------------------------------------------------------------------
def setup_fonts():
    import matplotlib.font_manager as fm
    available = {f.name for f in fm.fontManager.ttflist}
    chosen = None
    for f in config.CN_FONTS:
        if f in available:
            chosen = f
            break
    if chosen is not None:
        plt.rcParams["font.sans-serif"] = [chosen]
    plt.rcParams["axes.unicode_minus"] = False
    return chosen


def _short(s, n=14):
    s = str(s)
    return s if len(s) <= n else s[: n - 1] + "…"


# ---------------------------------------------------------------------------
# 绘图
# ---------------------------------------------------------------------------
def plot_top_nodes(top_df, topk, out_path):
    """Top-K GAT 核心节点横向柱状图。"""
    df = top_df.head(topk).iloc[::-1]  # 反转让最高分在最上方
    setup_fonts()
    fig, ax = plt.subplots(figsize=(9, max(5, topk * 0.42)))
    labels = [f"{_short(u)} (uid {uid})" for u, uid in zip(df["username"], df["uid"])]
    ax.barh(labels, df["gat_score"], color="#4C72B0", alpha=0.85)
    ax.set_xlabel("GAT score（core=1 概率）")
    ax.set_title(f"GAT Top {topk} 核心传播节点")
    ax.set_xlim(0, 1)
    for i, v in enumerate(df["gat_score"]):
        ax.text(v + 0.01, i, f"{v:.3f}", va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_pagerank_comparison(gat_top, pr_top, topk, out_path):
    """并排对比 GAT Top-K 与 PageRank Top-K。"""
    setup_fonts()
    g = gat_top.head(topk).iloc[::-1]
    p = pr_top.head(topk).iloc[::-1]

    fig, axes = plt.subplots(1, 2, figsize=(16, max(5, topk * 0.42)))

    gl = [f"{_short(u)} (uid {uid})" for u, uid in zip(g["username"], g["uid"])]
    axes[0].barh(gl, g["gat_score"], color="#4C72B0", alpha=0.85)
    axes[0].set_xlabel("GAT score")
    axes[0].set_title(f"GAT Top {topk}")
    axes[0].set_xlim(0, 1)

    pl = [f"{_short(u)} (uid {uid})" for u, uid in zip(p["username"], p["uid"])]
    axes[1].barh(pl, p["pagerank"], color="#DD8452", alpha=0.85)
    axes[1].set_xlabel("PageRank")
    axes[1].set_title(f"PageRank Top {topk}")

    fig.suptitle("GAT vs PageRank 核心节点对比", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
