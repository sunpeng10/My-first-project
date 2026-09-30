# -*- coding: utf-8 -*-
"""
生成网络可视化 network_visualization.html（自包含，D3 v7 from CDN）

如实表现稀疏网络：8 个活跃节点 + 7 条边用 force 布局，
395 个孤立节点以灰色小点网格静态呈现，不伪造任何连接。
"""

import csv
import json
import os

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GRAPH = os.path.join(BASE, "graph")


def read_csv(p):
    with open(p, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def main():
    nodes = read_csv(os.path.join(GRAPH, "nodes.csv"))
    edges = read_csv(os.path.join(GRAPH, "edges.csv"))
    cent = read_csv(os.path.join(GRAPH, "node_centrality.csv"))
    stats = json.load(open(os.path.join(GRAPH, "network_statistics.json"), encoding="utf-8"))

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
    out = os.path.join(GRAPH, "network_visualization.html")
    with open(out, "w", encoding="utf-8") as f:
        f.write(html)
    print("written:", out)


HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>微博作者 Mention Network · 第一版</title>
<style>
  :root { color-scheme: light dark; }
  * { box-sizing: border-box; }
  body { margin: 0; font-family: -apple-system, "Segoe UI", "Microsoft YaHei", sans-serif; background: #f7f7f8; color: #1c1c1f; }
  @media (prefers-color-scheme: dark) { body { background: #131316; color: #e8e8ea; } }
  .wrap { max-width: 1100px; margin: 0 auto; padding: 20px; }
  h1 { font-size: 20px; margin: 0 0 4px; }
  .sub { color: #6b7280; font-size: 13px; margin-bottom: 14px; }
  .kpis { display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 14px; }
  .kpi { background: #fff; border: 1px solid #e5e7eb; border-radius: 8px; padding: 8px 12px; }
  @media (prefers-color-scheme: dark) { .kpi { background: #1f1f24; border-color: #2e2e34; } }
  .kpi b { font-size: 16px; display: block; }
  .kpi span { font-size: 11px; color: #6b7280; }
  #chart { border: 1px solid #e5e7eb; border-radius: 10px; background: #fff; }
  @media (prefers-color-scheme: dark) { #chart { border-color: #2e2e34; background: #1a1a1f; } }
  .tip { position: absolute; pointer-events: none; background: rgba(17,17,20,.94); color: #fff; padding: 8px 10px; border-radius: 6px; font-size: 12px; line-height: 1.5; opacity: 0; transition: opacity .12s; max-width: 260px; box-shadow: 0 4px 16px rgba(0,0,0,.2); }
  .legend { font-size: 12px; color: #6b7280; margin-top: 8px; display: flex; gap: 16px; flex-wrap: wrap; }
  .legend .dot { display: inline-block; width: 10px; height: 10px; border-radius: 50%; margin-right: 4px; vertical-align: -1px; }
  .note { font-size: 12px; color: #9ca3af; margin-top: 10px; line-height: 1.6; }
</style>
</head>
<body>
<div class="wrap">
  <h1>微博作者 Mention Network（第一版）</h1>
  <div class="sub">有向图：边 = 已可靠映射 UID 的 @提及关系（source → target）；不含转发/评论边，不猜测 UID</div>
  <div class="kpis" id="kpis"></div>
  <div id="chart" style="position: relative;"></div>
  <div class="legend">
    <span><span class="dot" style="background:#2f6fed;"></span>活跃节点（degree&gt;0，大小∝degree）</span>
    <span><span class="dot" style="background:#cbd5e1;"></span>孤立节点（degree=0，共 <b id="isoN"></b> 个，网格静态展示）</span>
    <span>→ 箭头方向 = 提及方向</span>
  </div>
  <div class="note">
    注：孤立节点无任何边，为如实表现稀疏性，以灰色小点网格排布于底部，与活跃节点的 force 布局分离；
    节点之间不存在任何人为连接。自环（作者 @ 自己）以节点上方小圆环表示。
  </div>
</div>
<div class="tip" id="tip"></div>

<script src="https://d3js.org/d3.v7.min.js"></script>
<script>
const DATA = __DATA__;
const W = Math.min(1060, window.innerWidth - 40), H = 640;

const svg = d3.select("#chart").append("svg").attr("width", W).attr("height", H);
const tip = d3.select("#tip");

const active = DATA.nodes.filter(n => n.active);
const isolated = DATA.nodes.filter(n => !n.active);
document.getElementById("isoN").textContent = isolated.length;

// KPI
const s = DATA.stats;
const kpis = [
  ["节点", s.node_count], ["边", s.edge_count],
  ["自环", s.self_loop_count], ["孤立节点", s.isolated_node_count],
  ["活跃节点", s.active_node_count], ["最大连通分量", s.largest_component_size],
];
document.getElementById("kpis").innerHTML = kpis.map(k =>
  `<div class="kpi"><b>${k[1]}</b><span>${k[0]}</span></div>`).join("");

const rScale = d3.scaleSqrt().domain([0, 2]).range([4, 14]);
const color = n => n.active ? "#2f6fed" : "#cbd5e1";

// 箭头标记
svg.append("defs").append("marker")
  .attr("id", "arrow").attr("viewBox", "0 -5 10 10")
  .attr("refX", 22).attr("refY", 0).attr("markerWidth", 7).attr("markerHeight", 7)
  .attr("orient", "auto")
  .append("path").attr("d", "M0,-5L10,0L0,5").attr("fill", "#9ca3af");

// 活跃节点 + 边 force 布局
const aNodes = active.map(n => ({ ...n, x: W/2, y: H*0.32 }));
const links = DATA.edges.filter(e => e.source !== e.target)
  .map(e => ({ source: e.source, target: e.target, weight: e.weight }));
const selfLoops = DATA.edges.filter(e => e.source === e.target);

const sim = d3.forceSimulation(aNodes)
  .force("link", d3.forceLink(links).id(d => d.id).distance(120).strength(0.6))
  .force("charge", d3.forceManyBody().strength(-400))
  .force("center", d3.forceCenter(W/2, H*0.32))
  .force("collide", d3.forceCollide().radius(d => rScale(d.degree)+8));

const linkG = svg.append("g");
const lk = linkG.selectAll("line").data(links).join("line")
  .attr("stroke", "#9ca3af").attr("stroke-width", d => Math.min(2.5, 1 + d.weight*0.2))
  .attr("marker-end", "url(#arrow)");

const nodeG = svg.append("g");
const nG = nodeG.selectAll("circle").data(aNodes).join("circle")
  .attr("r", d => rScale(d.degree)).attr("fill", color)
  .attr("stroke", "#fff").attr("stroke-width", 1.5)
  .call(d3.drag().on("start", dr).on("drag", dg).on("end", de))
  .on("mousemove", hover).on("mouseleave", () => tip.style("opacity", 0));

const lbl = svg.append("g").selectAll("text").data(aNodes).join("text")
  .text(d => d.name).attr("font-size", 10).attr("fill", "#6b7280")
  .attr("text-anchor", "middle").attr("dy", d => -rScale(d.degree) - 4);

// 自环小圆环
const selfG = svg.append("g");
selfG.selectAll("circle").data(selfLoops).join("circle")
  .attr("r", 7).attr("fill", "none").attr("stroke", "#9ca3af").attr("stroke-width", 1.2)
  .attr("opacity", 0.85);

function posSelfLoop(d, n) {
  const n0 = aNodes.find(x => x.id === d.source);
  if (!n0) return { x: 0, y: 0 };
  return { x: n0.x, y: n0.y - 14 };
}

sim.on("tick", () => {
  lk.attr("x1", d => d.source.x).attr("y1", d => d.source.y)
     .attr("x2", d => d.target.x).attr("y2", d => d.target.y);
  nG.attr("cx", d => d.x).attr("cy", d => d.y);
  lbl.attr("x", d => d.x).attr("y", d => d.y);
  selfG.selectAll("circle").attr("cx", d => posSelfLoop(d).x).attr("cy", d => posSelfLoop(d).y);
});

// 孤立节点：静态网格，置于底部
const isoG = svg.append("g");
const cols = Math.max(20, Math.ceil(Math.sqrt(isolated.length * (W/H))));
const gapX = (W - 40) / cols;
const gapY = 12;
isoG.selectAll("circle").data(isolated).join("circle")
  .attr("r", 2).attr("fill", "#cbd5e1").attr("opacity", 0.7)
  .attr("cx", (d, i) => 20 + (i % cols) * gapX + gapX/2)
  .attr("cy", (d, i) => H - 18 - Math.floor(i / cols) * gapY);

function hover(ev, d) {
  tip.style("opacity", 1)
    .html(`<b>${d.name}</b><br>UID：${d.id}<br>发帖：${d.post_count}<br>degree：${d.degree}<br>PageRank：${d.pagerank.toFixed(5)}`)
    .style("left", (ev.clientX + 14) + "px")
    .style("top", (ev.clientY + 14) + "px");
}
function dr(ev, d) { if (!ev.active) sim.alphaTarget(0.3).restart(); d.fx = d.x; d.fy = d.y; }
function dg(ev, d) { d.fx = ev.x; d.fy = ev.y; }
function de(ev, d) { if (!ev.active) sim.alphaTarget(0); d.fx = null; d.fy = null; }
</script>
</body>
</html>
"""


if __name__ == "__main__":
    main()
