"""pseudo-label 构造（弱监督，非人工 Ground Truth）。

core_score = alpha*norm(pagerank) + beta*norm(in_degree) + gamma*norm(weighted_in_degree)
Top 10% -> core_label = 1，其余 0。

注意：这是网络指标构造的弱标签，用于验证 GAT 流程，不是真实核心节点标注。
"""
import numpy as np
import pandas as pd

from . import config
from .utils import normalize_minmax


def build_pseudo_labels(node_metrics, comp_nodes):
    """基于全图网络指标构造 core_score 与 pseudo_label。

    返回 (labels_df, threshold)：
    - labels_df: index=uid，列 [pagerank, in_degree, weighted_in_degree, core_score, pseudo_label]
    """
    nm = node_metrics.set_index("uid")
    uids = list(comp_nodes)

    pagerank = nm["pagerank"].reindex(uids).fillna(0).astype(float).values
    in_degree = nm["in_degree"].reindex(uids).fillna(0).astype(float).values
    weighted_in = nm["weighted_in_degree"].reindex(uids).fillna(0).astype(float).values

    norm_pr = normalize_minmax(pagerank)
    norm_in = normalize_minmax(in_degree)
    norm_win = normalize_minmax(weighted_in)

    core_score = (
        config.CORE_ALPHA * norm_pr
        + config.CORE_BETA * norm_in
        + config.CORE_GAMMA * norm_win
    )

    # Top 10% -> 1
    threshold = float(np.quantile(core_score, 1.0 - config.TOP_PERCENT))
    degenerate = False
    if threshold <= 0.0:
        # 退化情形：超过 90% 的节点 core_score 恰为 0（叶子评论者，无入边）。
        # 此时 "Top 10%" 会混入大量并列 0 分节点，标签失去意义。
        # 退化为：core_score > 0（即收到过 >=1 条评论）作为正样本。
        degenerate = True
        threshold = 0.0
        pseudo_label = (core_score > 0.0).astype(int)
    else:
        pseudo_label = (core_score >= threshold).astype(int)

    df = pd.DataFrame(
        {
            "uid": uids,
            "pagerank": pagerank,
            "in_degree": in_degree,
            "weighted_in_degree": weighted_in,
            "core_score": core_score,
            "pseudo_label": pseudo_label,
        }
    ).set_index("uid")

    return df, threshold, degenerate
