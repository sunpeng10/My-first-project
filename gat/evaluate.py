"""评估：分类指标 + Top-K overlap + Spearman 相关。

关键说明：由于 label 是网络指标构造的 pseudo-label，分类 Accuracy 不是唯一评价。
额外输出 GAT 与 PageRank / Core Score 的 Top-K overlap 与 Spearman 相关，
以衡量 GAT 学到的核心节点排序与结构指标的契合程度。
"""
import numpy as np
import torch
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score)

from . import config
from .utils import overlap_ratio, spearman_corr, top_k_uids


def compute_metrics(model, data, device, labels_df, uid_order):
    """返回 (metrics_dict, gat_score, predicted_label)。

    - gat_score: softmax 后 core=1 的概率（与 uid_order 对齐）
    """
    model.eval()
    with torch.no_grad():
        logits = model(data.x.to(device), data.edge_index.to(device))
        probs = torch.softmax(logits, dim=1)
        gat_score = probs[:, 1].cpu().numpy()          # core=1 概率
        pred = logits.argmax(dim=1).cpu().numpy()

    y = data.y.cpu().numpy()
    test_mask = data.test_mask.cpu().numpy()

    y_test = y[test_mask]
    pred_test = pred[test_mask]
    prob_test = gat_score[test_mask]

    acc = accuracy_score(y_test, pred_test)
    prec = precision_score(y_test, pred_test, pos_label=1, zero_division=0)
    rec = recall_score(y_test, pred_test, pos_label=1, zero_division=0)
    f1 = f1_score(y_test, pred_test, pos_label=1, zero_division=0)
    roc_auc = roc_auc_score(y_test, prob_test) if len(set(y_test)) > 1 else float("nan")

    # ---- 排名对比（在全部子图节点上比较）----
    uid_order = list(uid_order)
    gat_score_map = dict(zip(uid_order, gat_score))
    pr_map = labels_df["pagerank"].to_dict()
    core_map = labels_df["core_score"].to_dict()

    overlaps = {}
    for k in config.OVERLAP_KS:
        gat_top = top_k_uids(gat_score_map, k)
        pr_top = top_k_uids(pr_map, k)
        core_top = top_k_uids(core_map, k)
        overlaps[f"gat_vs_pagerank_overlap_at_{k}"] = round(
            overlap_ratio(gat_top, pr_top), 4)
        overlaps[f"gat_vs_corescore_overlap_at_{k}"] = round(
            overlap_ratio(gat_top, core_top), 4)

    # Spearman：GAT score vs PageRank（全子图节点）
    spearman_pr = spearman_corr(
        np.array([gat_score_map[u] for u in uid_order]),
        np.array([pr_map.get(u, 0.0) for u in uid_order]),
    )

    metrics = {
        "test_accuracy": round(float(acc), 4),
        "test_precision": round(float(prec), 4),
        "test_recall": round(float(rec), 4),
        "test_f1": round(float(f1), 4),
        "roc_auc": round(float(roc_auc), 4) if np.isfinite(roc_auc) else None,
        "spearman_gat_vs_pagerank": round(float(spearman_pr), 4),
    }
    metrics.update(overlaps)

    return metrics, gat_score, pred
