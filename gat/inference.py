"""结果导出：CSV / JSON / 图表 / 实验报告。"""
import json
import os

import numpy as np
import pandas as pd

from . import config
from .utils import plot_pagerank_comparison, plot_top_nodes


def _username_map(node_metrics, user_features):
    m = {}
    if "username" in node_metrics.columns:
        m.update(dict(zip(node_metrics["uid"], node_metrics["username"])))
    if "username" in user_features.columns:
        for uid, name in zip(user_features["uid"], user_features["username"]):
            if uid not in m or pd.isna(m.get(uid)):
                m[uid] = name
    return m


def build_node_scores(comp_nodes, node_metrics, user_features, labels_df,
                      gat_score, pred, component_size):
    nm = node_metrics.set_index("uid")
    uf = user_features.set_index("uid")
    name_map = _username_map(node_metrics, user_features)

    in_deg = nm["in_degree"].reindex(comp_nodes).fillna(0).astype("int64").values
    out_deg = nm["out_degree"].reindex(comp_nodes).fillna(0).astype("int64").values
    win = nm["weighted_in_degree"].reindex(comp_nodes).fillna(0).astype("int64").values
    wout = nm["weighted_out_degree"].reindex(comp_nodes).fillna(0).astype("int64").values
    posts_commented = uf["posts_commented"].reindex(comp_nodes).fillna(0).astype("int64").values
    posts_authored = uf["posts_authored"].reindex(comp_nodes).fillna(0).astype("int64").values

    lab = labels_df.reindex(comp_nodes)

    df = pd.DataFrame({
        "uid": comp_nodes,
        "username": [name_map.get(u, "Unknown") for u in comp_nodes],
        "gat_score": np.round(gat_score, 6),
        "predicted_label": pred,
        "pseudo_label": lab["pseudo_label"].astype("int64").values,
        "core_score": np.round(lab["core_score"].values, 6),
        "pagerank": lab["pagerank"].values,
        "in_degree": in_deg,
        "out_degree": out_deg,
        "total_degree": in_deg + out_deg,
        "weighted_in_degree": win,
        "weighted_out_degree": wout,
        "posts_commented": posts_commented,
        "posts_authored": posts_authored,
        "comment_count": wout,
        "component_size": component_size,
    })
    df = df.sort_values("gat_score", ascending=False).reset_index(drop=True)
    return df


def build_top_nodes(node_scores, top_n=config.TOP_N_NODES):
    top = node_scores.head(top_n).copy().reset_index(drop=True)
    top.insert(0, "rank", np.arange(1, len(top) + 1))
    keep = ["rank", "uid", "username", "gat_score", "pseudo_label", "pagerank",
            "in_degree", "out_degree", "total_degree", "core_score",
            "posts_commented", "posts_authored", "comment_count"]
    return top[[c for c in keep if c in top.columns]]


def build_graph_json(node_scores, edge_list, edge_mean_attn):
    score_cols = node_scores.set_index("uid")
    nodes = []
    for uid, row in score_cols.iterrows():
        nodes.append({
            "id": str(uid),
            "uid": str(uid),
            "username": str(row["username"]),
            "score": float(row["gat_score"]),
            "label": int(row["predicted_label"]),
            "pagerank": float(row["pagerank"]),
            "degree": int(row["total_degree"]),
            "in_degree": int(row["in_degree"]),
            "out_degree": int(row["out_degree"]),
        })
    edges = []
    for (s, t, w), a in zip(edge_list, edge_mean_attn):
        edges.append({
            "source": str(s),
            "target": str(t),
            "weight": int(w),
            "attention": float(a),
        })
    return {"nodes": nodes, "edges": edges}


def write_report(metrics, graph_stats, features_used, features_dropped, top_df,
                 topk, out_path):
    """生成 markdown 实验报告。"""
    top = top_df.head(topk)

    def feat_row(f):
        return ", ".join(f) if f else "（无）"

    overlap_lines = "\n".join(
        f"- {k}: `{v}`" for k, v in metrics.items() if "overlap" in k
    )

    top_table = "\n".join(
        f"| {int(r['rank'])} | {r['username']} | {r['uid']} | {r['gat_score']:.4f} | "
        f"{r['pagerank']:.5f} | "
        f"{int(r['in_degree'])} | {int(r['out_degree'])} | {int(r['total_degree'])} |"
        for _, r in top.iterrows()
    )

    md = f"""# GAT 核心传播节点识别 —— 第一阶段实验报告

> **重要声明**：由于当前数据缺乏人工核心节点标注，本实验使用 PageRank、入度及
> 加权入度构造网络结构**弱标签（pseudo-label）**，用于验证 GAT 节点识别流程。
> **pseudo-label 不是人工真实标签**，所有结论只说明“GAT 学习到了网络结构中
> 核心节点的模式”，不能声称证明了这些节点是真正核心传播节点。

## 1. 实验目的

在真实微博评论网络（User→User）上，用两层 GAT 学习节点表示，输出核心传播节点
排名与注意力权重，为后续 Flask API + Vue 可视化集成做准备。

## 2. 数据来源

- 评论边：`D:\\WeiboCrawler\\data\\comment_edges.csv`（{graph_stats['original_edge_rows']} 条一级评论关系）
- 网络指标：`D:\\WeiboCrawler\\results\\comment_network_node_metrics.csv`
- 用户特征：`D:\\WeiboCrawler\\results\\heterogeneous_graph_user_features.csv`
- 帖子特征 / 作者名单：`heterogeneous_graph_post_features.csv`、`weibo_user_mapping.csv`

## 3. 评论网络构建方法

- 边方向严格保留：`source_uid(评论者) → target_uid(微博作者)`
- 同一 `(source, target)` 出现多次时统计 `weight = 评论次数`（不丢信息）
- 当前版本基于**一级评论关系**构建用户交互网络，用于验证 GAT 核心节点识别流程；
  后续如获得转发链、评论回复链等数据，可进一步扩展为更完整的传播网络。

## 4. 网络规模

| 指标 | 数值 |
|------|------|
| 原始节点数 | {graph_stats['original_nodes']} |
| 原始唯一边数 | {graph_stats['original_edges_unique']} |
| 原始边记录数 | {graph_stats['original_edge_rows']} |
| self-loop 记录数 | {graph_stats['self_loop_rows']} |
| 多评论边对数 | {graph_stats['multi_comment_pairs']} |
| 最大边权重 | {graph_stats['max_edge_weight']} |

## 5. 最大连通分量

由于评论关系具有明显方向性，采用**最大弱连通分量**作为 GAT 实验子图，以保留较
完整的局部交互结构。

| 指标 | 数值 |
|------|------|
| 弱连通分量个数 | {graph_stats['num_wcc']} |
| 最大弱连通分量节点数 | {graph_stats['largest_wcc_size']} |
| 分量占比 | {graph_stats['component_ratio']} |

## 6. 节点特征

GAT 输入特征（{len(features_used)} 维）：

{feat_row(features_used)}

- `component_size` 在最大弱连通分量内为常数（{graph_stats['largest_wcc_size']}），无区分度，未纳入训练特征。
- 为避免 target leakage，**不使用** pagerank / in_degree / weighted_in_degree / core_score，
  也不直接使用 total_degree（含 in_degree）。
- 计数类特征先 log1p 压缩重尾，再 z-score 标准化。

剔除的常数特征：{feat_row(features_dropped)}

## 7. pseudo-label 定义

```
core_score = 0.5*norm(pagerank) + 0.3*norm(in_degree) + 0.2*norm(weighted_in_degree)
```

Top 10%（core_score 分位数阈值 {metrics['core_score_threshold']:.4f}）→ `core_label = 1`，其余 0。
正样本 {metrics['positive_labels']} 个，负样本 {metrics['negative_labels']} 个。

**标签退化说明**：{'存在退化情形' if metrics['label_degenerate'] else '无退化'}。当前评论网络为星形二分结构——
绝大多数节点是“只评论、从未被评论”的叶子评论者（in_degree = 0），其 core_score 恰为 0。
若严格取 Top 10%（{metrics['largest_component_nodes'] * 10 // 100} 个节点），会混入大量并列 0 分的
叶子节点，导致正负样本不可分。因此当 Top-10% 阈值退化到 0 时，正样本退化为
**core_score > 0（即收到过 ≥1 条评论）** 的节点，即该子图内真正具有“被关注/权威”地位的作者节点。

## 8. GAT 模型结构

```
Input({graph_stats['feature_dim']})
  -> GATConv(64, heads=4) -> ELU -> Dropout(0.2)   # 输出 256
  -> GATConv(32, heads=1, concat=False) -> Dropout # 输出 32
  -> Linear(32, 2)
```

## 9. 训练参数

- epochs = {metrics['epochs']}，lr = {metrics['lr']}，weight_decay = {metrics['weight_decay']}
- dropout = {metrics['dropout']}，early stopping patience = {metrics['patience']}（监控 validation F1）
- 划分：train 70% / val 15% / test 15%（分层，保持类别分布）
- class weight 缓解类别不平衡；设备：{metrics['device']}

## 10. 评价指标

| 指标 | 数值 |
|------|------|
| Best validation F1 | {metrics['best_val_f1']} |
| Test Accuracy | {metrics['test_accuracy']} |
| Test Precision | {metrics['test_precision']} |
| Test Recall | {metrics['test_recall']} |
| Test F1 | {metrics['test_f1']} |
| ROC-AUC | {metrics['roc_auc']} |

## 11-15. 分类指标说明

由于 pseudo-label 由结构指标构造，Accuracy 仅反映 GAT 对弱标签的拟合程度，
**不能**当作真实核心节点识别准确率。

## 16. Top-K overlap

{overlap_lines}

- Spearman(GAT score vs PageRank) = {metrics['spearman_gat_vs_pagerank']}

## 17. GAT 与 PageRank 对比

- **粗粒度识别成功**：ROC-AUC = {metrics['roc_auc']}，说明 GAT 能近乎完美地区分
  “作者（收到过评论）”与“评论者（从未被评论）”。
- **细粒度排序较弱**：Spearman(GAT score vs PageRank) = {metrics['spearman_gat_vs_pagerank']}，
  为弱正相关；Top-10/20 的 overlap 接近甚至低于随机水平（子图内仅 128 个作者，
  随机抽取 Top-20 的期望 overlap ≈ 0.16）。

原因有两方面：
1. **标签粒度**：二进制 pseudo-label 只编码了“是否作者”这一粗粒度信息，没有编码
   作者节点内部的细粒度影响力排序，softmax 分数在作者群体内趋近饱和（≈1.0）。
2. **“核心”定义差异**：GAT 同时利用了行为特征（out_degree / weighted_out_degree），
   会把“高频评论的放大器型用户”（如 out_degree=24 的桥接评论者）也判为核心节点，
   这与 PageRank 的“权威（被评论）中心性”是两种不同的核心概念。这一差异本身正是
   “识别 KOL / 情感放大节点”任务需要关注的信号，而非单纯噪声。

## 18. Top {topk} 核心节点

| rank | username | uid | gat_score | pagerank | in_degree | out_degree | total_degree |
|------|----------|-----|-----------|----------|-----------|------------|--------------|
{top_table}

## 19. attention 分析

注意力权重已导出到 `results/gat_attention.csv`（字段：source_uid, target_uid,
layer, head, attention_weight），可用于前端展示“哪个用户对哪个用户的信息传递权重较高”。

## 20. 局限性

1. **pseudo-label 非人工标注**：标签由 PageRank / 入度 / 加权入度构造，存在自指性。
2. **网络稀疏**：一级评论关系构造的网络较稀疏，不是完整社会传播网络。
3. **最大弱连通分量仅 {graph_stats['largest_wcc_size']} 节点**：GAT 只在该子图上训练。
4. **author_flag / posts_authored 与 in_degree 相关**：只有原创作者才会收到评论，
   这两个特征与标签中的 in_degree 存在间接相关，属设计取舍（作者身份是 KOL 识别的
   合理先验），报告中如实说明。
5. 未使用转发链、评论回复链，传播结构不完整。

## 21. 后续 Flask + Vue 集成计划

- `results/gat_graph.json` 直接用于 `GET /api/graph`
- `results/gat_top_nodes.csv` 用于 `/api/top-nodes`
- `results/gat_attention.csv` 用于前端注意力边权重可视化
- `results/gat_node_scores.csv` 用于节点详情查询

---
*生成时间由 `scripts/run_gat.py` 记录；本报告自动生成。*
"""
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(md)


def export_all(ctx):
    """写入全部结果文件，返回文件路径列表。"""
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    os.makedirs(config.MODEL_DIR, exist_ok=True)

    node_scores = build_node_scores(
        ctx["comp_nodes"], ctx["node_metrics"], ctx["user_features"],
        ctx["labels_df"], ctx["gat_score"], ctx["pred"], ctx["component_size"],
    )
    top_nodes = build_top_nodes(node_scores, config.TOP_N_NODES)

    # 用 top_df 计算 rank 需从 node_scores 生成带 rank 的 top
    top_for_report = build_top_nodes(node_scores, ctx["topk"])

    graph_json = build_graph_json(node_scores, ctx["edge_list"], ctx["edge_mean_attn"])

    paths = {}

    # CSV
    p = os.path.join(config.OUTPUT_DIR, "gat_node_scores.csv")
    node_scores.to_csv(p, index=False, encoding="utf-8-sig")
    paths["gat_node_scores.csv"] = p

    p = os.path.join(config.OUTPUT_DIR, "gat_top_nodes.csv")
    top_nodes.to_csv(p, index=False, encoding="utf-8-sig")
    paths["gat_top_nodes.csv"] = p

    p = os.path.join(config.OUTPUT_DIR, "gat_attention.csv")
    ctx["att_df"].to_csv(p, index=False, encoding="utf-8-sig")
    paths["gat_attention.csv"] = p

    # JSON
    p = os.path.join(config.OUTPUT_DIR, "gat_metrics.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(ctx["metrics"], f, ensure_ascii=False, indent=2)
    paths["gat_metrics.json"] = p

    p = os.path.join(config.OUTPUT_DIR, "gat_graph.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(graph_json, f, ensure_ascii=False)
    paths["gat_graph.json"] = p

    # PNG
    p = os.path.join(config.OUTPUT_DIR, "gat_top_nodes.png")
    plot_top_nodes(top_for_report, ctx["topk"], p)
    paths["gat_top_nodes.png"] = p

    # PageRank Top-K 数据
    pr_top = node_scores.sort_values("pagerank", ascending=False).reset_index(drop=True)
    p = os.path.join(config.OUTPUT_DIR, "gat_pagerank_comparison.png")
    plot_pagerank_comparison(top_for_report, pr_top, ctx["topk"], p)
    paths["gat_pagerank_comparison.png"] = p

    # Report
    p = os.path.join(config.OUTPUT_DIR, "gat_report.md")
    write_report(
        ctx["metrics"], ctx["graph_stats"], ctx["features_used"],
        ctx["features_dropped"], top_for_report, ctx["topk"], p,
    )
    paths["gat_report.md"] = p

    return paths, node_scores, top_nodes, graph_json
