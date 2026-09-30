# -*- coding: utf-8 -*-
"""
构建第一版真实微博作者 Mention Network（第三阶段 · 第三步）
============================================================

输入（只读）：
  - graph/post_authors.csv          image_id, weibo_url, author_uid, text
  - graph/mention_relations_raw.csv image_id, source_uid, mentioned_username, mention_count, relation
  - D:\\WeiboCrawler\\data\\weibo_user_mapping.csv  user_id, user_name, weibo_url

输出：
  - graph/nodes.csv               node_id, user_name, post_count
  - graph/edges.csv               source, target, relation, weight（聚合，主边表）
  - graph/edges_post_detail.csv   source, target, post_id, mention_count（post 级明细）
  - graph/node_centrality.csv     node_id, user_name, post_count, in_degree, out_degree, degree, pagerank, betweenness
  - graph/top_nodes.csv           ranking_type, rank, node_id, user_name, value
  - graph/network_statistics.json 拓扑统计

约束：仅使用已可靠映射的 UID→UID mention 关系；47 个外部账号不生成 Edge；不猜测。
"""

import csv
import json
import os
from collections import Counter, defaultdict

import networkx as nx

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # visual_saliency_analysis/
GRAPH = os.path.join(BASE, "graph")
POST_AUTHORS = os.path.join(GRAPH, "post_authors.csv")
MENTION_RAW = os.path.join(GRAPH, "mention_relations_raw.csv")
MAPPING = r"D:\WeiboCrawler\data\weibo_user_mapping.csv"


def read_csv(path):
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path, fieldnames, rows):
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def main():
    posts = read_csv(POST_AUTHORS)
    mentions = read_csv(MENTION_RAW)
    mapping = read_csv(MAPPING)

    # 1. uid → user_name（已验证映射）
    uid2name = {r["user_id"]: r["user_name"] for r in mapping}
    name2uid = {r["user_name"]: r["user_id"] for r in mapping}

    # 2. nodes：403 个作者，post_count = 发帖数（不丢任何一个）
    post_count = Counter(r["author_uid"] for r in posts)
    node_ids = sorted(post_count.keys())
    nodes = [
        {"node_id": uid, "user_name": uid2name.get(uid, ""), "post_count": post_count[uid]}
        for uid in node_ids
    ]

    # 3. edges：仅可靠映射的 mention 行 → UID→UID
    raw_edges = []  # (source, target, post_id, mention_count)
    for m in mentions:
        mname = m["mentioned_username"]
        target_uid = name2uid.get(mname)  # 只有精确匹配到映射才生成边
        if not target_uid:
            continue
        raw_edges.append({
            "source": m["source_uid"],
            "target": target_uid,
            "post_id": m["image_id"],
            "mention_count": int(m.get("mention_count", 1) or 1),
        })

    # 4. 聚合 (source, target) → weight
    agg = defaultdict(int)
    post_detail = []
    for e in raw_edges:
        key = (e["source"], e["target"])
        agg[key] += e["mention_count"]
        post_detail.append(e)

    edges = [
        {"source": s, "target": t, "relation": "mention", "weight": agg[(s, t)]}
        for (s, t) in sorted(agg.keys())
    ]

    # 5. 构建有向图
    G = nx.DiGraph()
    G.add_nodes_from(node_ids)
    for e in edges:
        G.add_edge(e["source"], e["target"], weight=e["weight"], relation=e["relation"])

    n = G.number_of_nodes()
    m = G.number_of_edges()
    self_loops = [e for e in G.edges() if e[0] == e[1]]
    isolated = [v for v in G.nodes() if G.degree(v) == 0]
    active = n - len(isolated)

    # 连通分量（弱连通）
    wcc = list(nx.weakly_connected_components(G))
    largest_comp = max(wcc, key=len) if wcc else set()
    largest_ratio = len(largest_comp) / n if n else 0

    # 6. 中心性
    in_deg = dict(G.in_degree())
    out_deg = dict(G.out_degree())
    deg = dict(G.degree())
    pagerank = nx.pagerank(G)
    betweenness = nx.betweenness_centrality(G)

    centrality_rows = [
        {
            "node_id": v,
            "user_name": uid2name.get(v, ""),
            "post_count": post_count.get(v, 0),
            "in_degree": in_deg[v],
            "out_degree": out_deg[v],
            "degree": deg[v],
            "pagerank": round(pagerank[v], 8),
            "betweenness": round(betweenness[v], 8),
        }
        for v in node_ids
    ]

    # 7. Top-K（Top 20 by Degree / PageRank / Betweenness）
    def top_rows(metric, key):
        ranked = sorted(centrality_rows, key=lambda r: (-r[key], r["node_id"]))
        return [
            {"ranking_type": metric, "rank": i + 1, "node_id": r["node_id"],
             "user_name": r["user_name"], "value": r[key]}
            for i, r in enumerate(ranked[:20])
        ]

    top_nodes = (
        top_rows("degree", "degree")
        + top_rows("pagerank", "pagerank")
        + top_rows("betweenness", "betweenness")
    )

    # 8. 写出
    write_csv(os.path.join(GRAPH, "nodes.csv"),
              ["node_id", "user_name", "post_count"], nodes)
    write_csv(os.path.join(GRAPH, "edges.csv"),
              ["source", "target", "relation", "weight"], edges)
    write_csv(os.path.join(GRAPH, "edges_post_detail.csv"),
              ["source", "target", "post_id", "mention_count"], post_detail)
    write_csv(os.path.join(GRAPH, "node_centrality.csv"),
              ["node_id", "user_name", "post_count", "in_degree", "out_degree",
               "degree", "pagerank", "betweenness"], centrality_rows)
    write_csv(os.path.join(GRAPH, "top_nodes.csv"),
              ["ranking_type", "rank", "node_id", "user_name", "value"], top_nodes)

    stats = {
        "node_count": n,
        "edge_count": m,
        "directed": G.is_directed(),
        "self_loop_count": len(self_loops),
        "isolated_node_count": len(isolated),
        "active_node_count": active,
        "average_degree": round(sum(deg.values()) / n, 6) if n else 0,
        "average_in_degree": round(sum(in_deg.values()) / n, 6) if n else 0,
        "average_out_degree": round(sum(out_deg.values()) / n, 6) if n else 0,
        "network_density": round(nx.density(G), 10),
        "weakly_connected_components": len(wcc),
        "largest_component_size": len(largest_comp),
        "largest_component_ratio": round(largest_ratio, 6),
        # 附加诊断
        "component_size_distribution": {str(k): v for k, v in
                                        sorted(Counter(len(c) for c in wcc).items())},
        "largest_component_members": sorted(largest_comp),
        "self_loop_edges": [[u, v] for u, v in self_loops],
    }
    with open(os.path.join(GRAPH, "network_statistics.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    # 9. 控制台摘要
    print("=" * 70)
    print(f"node_count               : {n}")
    print(f"edge_count (aggregated)  : {m}")
    print(f"directed                 : {G.is_directed()}")
    print(f"self_loop_count          : {len(self_loops)}")
    print(f"isolated_node_count      : {len(isolated)}")
    print(f"active_node_count        : {active}")
    print(f"average_degree           : {stats['average_degree']}")
    print(f"average_in_degree        : {stats['average_in_degree']}")
    print(f"average_out_degree       : {stats['average_out_degree']}")
    print(f"network_density          : {stats['network_density']}")
    print(f"weakly_connected_components: {len(wcc)}")
    print(f"largest_component_size   : {len(largest_comp)}")
    print(f"largest_component_ratio  : {largest_ratio:.6f}")
    print("--- 边列表 ---")
    for e in edges:
        print(f"  {e['source']} -> {e['target']}  weight={e['weight']}  "
              f"({uid2name.get(e['source'],'?')} -> {uid2name.get(e['target'],'?')})")
    print("=" * 70)


if __name__ == "__main__":
    main()
