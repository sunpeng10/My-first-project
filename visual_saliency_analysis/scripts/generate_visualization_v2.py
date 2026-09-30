# -*- coding: utf-8 -*-
"""生成 v2 网络可视化 network_visualization_v2.html（复用 v1 模板）"""
import json
import os

from generate_visualization import HTML_TEMPLATE, read_csv

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GRAPH = os.path.join(BASE, "graph")


def main():
    nodes = read_csv(os.path.join(GRAPH, "nodes_v2.csv"))
    edges = read_csv(os.path.join(GRAPH, "edges_v2.csv"))
    cent = read_csv(os.path.join(GRAPH, "node_centrality_v2.csv"))
    stats = json.load(open(os.path.join(GRAPH, "network_statistics_v2.json"), encoding="utf-8"))

    cent_by_id = {r["node_id"]: r for r in cent}

    node_data = []
    for r in nodes:
        c = cent_by_id.get(r["node_id"], {})
        deg = int(c.get("degree", 0))
        node_data.append({
            "id": r["node_id"],
            "name": r["user_name"],
            "post_count": int(r["post_count"]),
            "degree": deg,
            "pagerank": float(c.get("pagerank", 0)),
            "active": deg > 0,
        })

    edge_data = [{"source": e["source"], "target": e["target"], "weight": int(e["weight"])}
                 for e in edges]

    payload = {
        "nodes": node_data,
        "edges": edge_data,
        "stats": {
            "node_count": stats["node_count"],
            "edge_count": stats["edge_count"],
            "self_loop_count": stats["self_loop_count"],
            "isolated_node_count": stats["isolated_node_count"],
            "active_node_count": stats["active_node_count"],
            "largest_component_size": stats["largest_component_size"],
        },
    }

    html = HTML_TEMPLATE.replace("__DATA__", json.dumps(payload, ensure_ascii=False))
    html = html.replace("微博作者 Mention Network（第一版）", "微博作者 Mention Network（第二版）")
    html = html.replace("边 = 已可靠映射 UID 的 @提及关系", "边 = 已可靠映射 UID 的 @提及关系（含补采的 47 个外部账号 UID）")

    out = os.path.join(GRAPH, "network_visualization_v2.html")
    with open(out, "w", encoding="utf-8") as f:
        f.write(html)
    print("written:", out)


if __name__ == "__main__":
    main()
