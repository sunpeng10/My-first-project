# GAT 核心传播节点识别

在真实微博 **User→User 评论网络**上，用两层 Graph Attention Network (GAT) 学习节点表示，识别核心传播节点。是「传播网络 / 核心节点」前端功能的模型部分。

> ⚠️ **局限声明**：由于数据缺乏人工核心节点标注，训练标签是用 PageRank + 入度 + 加权入度构造的 **pseudo-label（弱标签）**，不是人工 Ground Truth。结果应表述为「GAT-based core propagation node identification」，不声称「预测了真实 KOL」。详见 [`results/gat_report.md`](../results/gat_report.md)。

## 数据来源

| 数据 | 路径 | 说明 |
|------|------|------|
| 评论边 | `D:\WeiboCrawler\data\comment_edges.csv` | 8849 条一级评论关系（source=评论者 → target=作者） |

> 该文件在独立采集项目 `WeiboCrawler` 中，未入本仓库；重训 GAT 前需自行准备。

## 运行

在项目根目录 `D:\BASNet` 下：

```bash
python scripts/run_gat.py                   # 完整流程（构建图 → 特征 → 训练 → 推理 → 结果）
python scripts/run_gat.py --mode evaluate   # 跳过训练，加载 models/gat_best.pt 直接推理
python scripts/run_gat.py --topk 20 --epochs 200
```

## 输出（写入 results/）

| 文件 | 说明 |
|------|------|
| `gat_graph.json` | 图结构（nodes / edges），后端 `/api/graph` 读取 |
| `gat_top_nodes.csv` | 核心节点排名，后端 `/api/top-nodes` 读取 |
| `gat_node_scores.csv` | 每节点分数索引，后端 `/api/node/<uid>` 读取 |
| `gat_attention.csv` | 注意力边，后端 `/api/attention` 读取 |
| `gat_metrics.json` | 评估指标 |
| `gat_report.md` | 实验报告 |

## 关键结果（诚实、非调参）

- ROC-AUC 0.9905（作者 / 评论者粗粒度区分极好）
- Test F1 0.52（类别不平衡 + 标签退化所致）
- 网络为**星形二分结构**（多数节点是叶子评论者），最大弱连通分量 2848 节点

## 环境

- torch 2.4.1+cpu，`torch_geometric 2.8.0`（GATConv 在 CPU 上无需 torch_scatter/sparse/cluster）
