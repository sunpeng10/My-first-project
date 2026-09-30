# -*- coding: utf-8 -*-
"""
重建闭合 Mention Network（第三阶段 · 第四步 · v2）
===================================================

相比 v1，v2 额外引入 external_user_mapping.csv（47 个外部账号补采的真实 UID）：
  - 已匹配（verified=True, match_status=matched）的外部账号 → 成为新节点 + 生成边
  - not_found / ambiguous / verified=False → 不生成边、不新增节点

输入（只读）：
  - graph/post_authors.csv
  - graph/mention_relations_raw.csv
  - D:\\WeiboCrawler\\data\\weibo_user_mapping.csv
  - graph/external_user_mapping.csv

输出（全部 v2）：
  - graph/nodes_v2.csv
  - graph/edges_v2.csv
  - graph/edges_post_detail_v2.csv
  - graph/node_centrality_v2.csv
  - graph/top_nodes_v2.csv
  - graph/network_statistics_v2.json
"""

import csv
import json
import os
from collections import Counter, defaultdict

import networkx as nx

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GRAPH = os.path.join(BASE, "graph")
POST_AUTHORS = os.path.join(GRAPH, "post_authors.csv")
MENTION_RAW = os.path.join(GRAPH, "mention_relations_raw.csv")
MAPPING = r"D:\WeiboCrawler\data\weibo_user_mapping.csv"
EXTERNAL = os.path.join(GRAPH, "external_user_mapping.csv")


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
    external = read_csv(EXTERNAL) if os.path.exists(EXTERNAL) else []

    # 内部作者映射
    uid2name = {r["user_id"]: r["user_name"] for r in mapping}
    name2uid = {r["user_name"]: r["user_id"] for r in mapping}

    # 外部账号映射：仅接受 verified=True 且 matched
    ext_name2uid = {}
    ext_uid2name = {}
    ext_matched_count = 0
    ext_not_found = 0
    ext_ambiguous = 0
    for r in external:
        st = (r.get("match_status") or "").strip()
        ver = (r.get("verified") or "").strip().lower() in ("true", "1", "yes")
        if st == "matched" and ver:
            uid = (r.get("target_uid") or "").strip()
            name = (r.get("target_username") or "").strip()
            if uid and name:
                ext_name2uid[name] = uid
                ext_uid2name[uid] = name
                ext_matched_count += 1
        elif st == "ambiguous":
            ext_ambiguous += 1
        else:
            ext_not_found += 1

    # 1. 节点：403 作者 + 已匹配外部账号（post_count=0）
    post_count = Counter(r["author_uid"] for r in posts)
    node_ids = set(post_count.keys()) | set(ext_uid2name.keys())
    nodes = []
    for uid in sorted(node_ids):
        if uid in uid2name:
            nodes.append({"node_id": uid, "user_name": uid2name[uid], "post_count": post_count.get(uid, 0)})
        else:
            nodes.append({"node_id": uid, "user_name": ext_uid2name[uid], "post_count": 0})

    # 2. 边：内部精确匹配 + 外部已确认匹配
    raw_edges = []
    for m in mentions:
        mname = (m.get("mentioned_username") or "").strip()
        target_uid = name2uid.get(mname) or ext_name2uid.get(mname)
        if not target_uid:
            continue
        raw_edges.append({
            "source": m["source_uid"],
            "target": target_uid,
            "post_id": m["image_id"],
            "mention_count": int(m.get("mention_count", 1) or 1),
        })

    agg = defaultdict(int)
    post_detail = []
    for e in raw_edges:
        agg[(e["source"], e["target"])] += e["mention_count"]
        post_detail.append(e)

    edges = [
        {"source": s, "target": t, "relation": "mention", "weight": agg[(s, t)]}
        for (s, t) in sorted(agg.keys())
    ]

    # 3. 有向图
    G = nx.DiGraph()
    G.add_nodes_from(n["node_id"] for n in nodes)
    for e in edges:
        G.add_edge(e["source"], e["target"], weight=e["weight"], relation=e["relation"])

    n = G.number_of_nodes()
    m = G.number_of_edges()
    self_loops = [e for e in G.edges() if e[0] == e[1]]
    isolated = [v for v in G.nodes() if G.degree(v) == 0]
    active = n - len(isolated)

    wcc = list(nx.weakly_connected_components(G))
    largest_comp = max(wcc, key=len) if wcc else set()
    largest_ratio = len(largest_comp) / n if n else 0

    # 4. 中心性
    in_deg = dict(G.in_degree())
    out_deg = dict(G.out_degree())
    deg = dict(G.degree())
    pagerank = nx.pagerank(G)
    betweenness = nx.betweenness_centrality(G)

    id2name = {x["node_id"]: x["user_name"] for x in nodes}
    centrality_rows = [
        {
            "node_id": v,
            "user_name": id2name.get(v, ""),
            "post_count": post_count.get(v, 0),
            "in_degree": in_deg[v],
            "out_degree": out_deg[v],
            "degree": deg[v],
            "pagerank": round(pagerank[v], 8),
            "betweenness": round(betweenness[v], 8),
        }
        for v in sorted(G.nodes())
    ]

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

    # 5. 写出
    write_csv(os.path.join(GRAPH, "nodes_v2.csv"),
              ["node_id", "user_name", "post_count"], nodes)
    write_csv(os.path.join(GRAPH, "edges_v2.csv"),
              ["source", "target", "relation", "weight"], edges)
    write_csv(os.path.join(GRAPH, "edges_post_detail_v2.csv"),
              ["source", "target", "post_id", "mention_count"], post_detail)
    write_csv(os.path.join(GRAPH, "node_centrality_v2.csv"),
              ["node_id", "user_name", "post_count", "in_degree", "out_degree",
               "degree", "pagerank", "betweenness"], centrality_rows)
    write_csv(os.path.join(GRAPH, "top_nodes_v2.csv"),
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
        "external_matched": ext_matched_count,
        "external_not_found": ext_not_found,
        "external_ambiguous": ext_ambiguous,
        "component_size_distribution": {str(k): v for k, v in
                                        sorted(Counter(len(c) for c in wcc).items())},
        "largest_component_members": sorted(largest_comp),
    }
    with open(os.path.join(GRAPH, "network_statistics_v2.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    # 6. 控制台摘要
    print("=" * 70)
    print(f"external matched    : {ext_matched_count}")
    print(f"external not_found  : {ext_not_found}")
    print(f"external ambiguous  : {ext_ambiguous}")
    print(f"node_count          : {n}")
    print(f"edge_count          : {m}")
    print(f"self_loop_count     : {len(self_loops)}")
    print(f"isolated_node_count : {len(isolated)}")
    print(f"active_node_count   : {active}")
    print(f"network_density     : {stats['network_density']}")
    print(f"largest_component   : {len(largest_comp)} ({stats['largest_component_ratio']})")
    print("--- 边列表 ---")
    for e in edges:
        print(f"  {e['source']} -> {e['target']}  w={e['weight']}  "
              f"({id2name.get(e['source'],'?')} -> {id2name.get(e['target'],'?')})")
    print("=" * 70)


if __name__ == "__main__":
    main()
