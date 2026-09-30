"""图构造：从 comment_edges.csv 构造 User→User 有向评论网络。

- 方向严格保留：source_uid(评论者) -> target_uid(微博作者)
- 同一 (source,target) 出现多次 -> 统计 weight（评论次数），不丢信息
- 使用最大弱连通分量作为 GAT 实验子图（不删除原始数据）
"""
import os

import networkx as nx
import pandas as pd

from . import config


def load_comment_edges():
    """读取原始评论边数据（只读，不改动文件）。"""
    if not os.path.exists(config.COMMENT_EDGES):
        raise FileNotFoundError(f"评论边文件不存在: {config.COMMENT_EDGES}")
    df = pd.read_csv(config.COMMENT_EDGES)
    # UID 必须是真实的 source_uid / target_uid，不通过 username 猜测
    df["source_uid"] = df["source_uid"].astype("Int64")
    df["target_uid"] = df["target_uid"].astype("Int64")
    return df


def build_graph(edges_df):
    """构造完整有向加权图，并返回统计信息与最大弱连通分量节点。"""
    G = nx.DiGraph()

    source = edges_df["source_uid"].dropna().astype("int64")
    target = edges_df["target_uid"].dropna().astype("int64")

    all_nodes = sorted(set(source.tolist()) | set(target.tolist()))
    G.add_nodes_from(all_nodes)

    # 统计 (source, target) 出现次数作为 weight（评论次数）
    weighted = edges_df.groupby(["source_uid", "target_uid"]).size().reset_index(name="weight")
    G.add_weighted_edges_from(
        zip(weighted["source_uid"], weighted["target_uid"], weighted["weight"])
    )

    # self-loop 统计（不静默删除，仅记录）
    self_loop_count = int((edges_df["source_uid"] == edges_df["target_uid"]).sum())

    # 最大弱连通分量
    wccs = list(nx.weakly_connected_components(G))
    wccs.sort(key=len, reverse=True)
    largest = sorted(wccs[0]) if wccs else []

    stats = {
        "original_nodes": G.number_of_nodes(),
        "original_edges_unique": G.number_of_edges(),
        "original_edge_rows": int(len(edges_df)),
        "self_loop_rows": self_loop_count,
        "multi_comment_pairs": int((weighted["weight"] > 1).sum()),
        "max_edge_weight": int(weighted["weight"].max()) if len(weighted) else 0,
        "num_wcc": len(wccs),
        "largest_wcc_size": len(largest),
        "component_ratio": round(len(largest) / G.number_of_nodes(), 4) if G.number_of_nodes() else 0.0,
    }

    return G, largest, stats


def subgraph_edges(G, comp_nodes):
    """返回最大弱连通分量内部的有向唯一边列表 [(src_uid, tgt_uid, weight), ...]。

    顺序稳定（按 (src, tgt) 排序），保证与后续 edge_index / attention 对齐。
    """
    comp_set = set(comp_nodes)
    edges = []
    for src, tgt, w in G.edges(data="weight"):
        if src in comp_set and tgt in comp_set:
            edges.append((int(src), int(tgt), int(w)))
    edges.sort(key=lambda e: (e[0], e[1]))
    return edges
