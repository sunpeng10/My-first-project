"""节点特征构建。

只使用不会造成 target leakage 的特征（见 config.FEATURE_COLUMNS）。
绝不使用 pagerank / in_degree / weighted_in_degree / core_score。
"""
import os

import numpy as np
import pandas as pd

from . import config


def load_node_tables():
    """读取已有节点特征表。若文件缺失则报错并提示。"""
    required = {
        "node_metrics": config.NODE_METRICS,
        "user_features": config.USER_FEATURES,
    }
    missing = [k for k, p in required.items() if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(
            "缺少节点特征文件: " + ", ".join(required[k] for k in missing)
        )

    node_metrics = pd.read_csv(config.NODE_METRICS)
    user_features = pd.read_csv(config.USER_FEATURES)
    node_metrics["uid"] = node_metrics["uid"].astype("int64")
    user_features["uid"] = user_features["uid"].astype("int64")
    return node_metrics, user_features


def load_author_uids():
    """原创作者 uid 集合（来自 weibo 作者名单与帖子作者字段）。"""
    authors = set()
    if os.path.exists(config.USER_MAPPING):
        um = pd.read_csv(config.USER_MAPPING)
        if "user_id" in um.columns:
            authors.update(um["user_id"].dropna().astype("int64").tolist())
    if os.path.exists(config.POST_FEATURES):
        pf = pd.read_csv(config.POST_FEATURES)
        if "author_uid" in pf.columns:
            authors.update(pf["author_uid"].dropna().astype("int64").tolist())
    return authors


def build_feature_frame(edges_df, node_metrics, user_features, author_uids, comp_nodes):
    """为最大弱连通分量内的每个 uid 构建特征 DataFrame。

    返回 (feature_df, feature_cols)：
    - feature_df: index=uid，列为 config.FEATURE_COLUMNS
    - feature_cols: 实际用于训练的特征列（剔除常数特征后）
    """
    uids = list(comp_nodes)

    nm = node_metrics.set_index("uid")
    uf = user_features.set_index("uid")

    # 出度 / 加权出度 来自 comment_network_node_metrics.csv
    out_degree = nm["out_degree"].reindex(uids).fillna(0).astype(float)
    weighted_out = nm["weighted_out_degree"].reindex(uids).fillna(0).astype(float)

    # 帖子参与数量 来自 heterogeneous_graph_user_features.csv
    posts_commented = uf["posts_commented"].reindex(uids).fillna(0).astype(float)
    posts_authored = uf["posts_authored"].reindex(uids).fillna(0).astype(float)

    # 作者身份特征
    author_flag = np.array([1 if u in author_uids else 0 for u in uids], dtype=float)

    # 平均评论文本长度（每个 source_uid 发出评论的平均字符数）
    text = edges_df[["source_uid", "comment_text"]].copy()
    text["source_uid"] = text["source_uid"].astype("int64")
    text["length"] = text["comment_text"].fillna("").astype(str).str.len()
    avg_len = text.groupby("source_uid")["length"].mean()
    avg_comment_length = avg_len.reindex(uids).fillna(0).astype(float)

    df = pd.DataFrame(
        {
            "uid": uids,
            "out_degree": out_degree.values,
            "weighted_out_degree": weighted_out.values,
            "posts_commented": posts_commented.values,
            "posts_authored": posts_authored.values,
            "author_flag": author_flag,
            "avg_comment_length": avg_comment_length.values,
        }
    ).set_index("uid")

    return df, list(config.FEATURE_COLUMNS)


def preprocess_features(feature_df, feature_cols):
    """log1p 变换 + 剔除常数特征 + z-score 标准化。"""
    X = feature_df[feature_cols].copy()

    # log1p 压缩重尾计数特征
    for c in config.LOG1P_FEATURES:
        if c in X.columns:
            X[c] = np.log1p(X[c].astype(float).clip(lower=0))

    X = X.astype(float)
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # 剔除常数（零方差）特征，避免标准化时产生 NaN
    std = X.std(axis=0)
    kept = [c for c in X.columns if std[c] > 1e-9]
    dropped = [c for c in X.columns if std[c] <= 1e-9]

    X_kept = X[kept].values
    mean = X_kept.mean(axis=0)
    sd = X_kept.std(axis=0)
    sd[sd < 1e-9] = 1.0
    X_scaled = (X_kept - mean) / sd

    return X_scaled.astype(np.float32), kept, dropped
