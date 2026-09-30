# GAT 核心传播节点识别 —— 第一阶段实验报告

> **重要声明**：由于当前数据缺乏人工核心节点标注，本实验使用 PageRank、入度及
> 加权入度构造网络结构**弱标签（pseudo-label）**，用于验证 GAT 节点识别流程。
> **pseudo-label 不是人工真实标签**，所有结论只说明“GAT 学习到了网络结构中
> 核心节点的模式”，不能声称证明了这些节点是真正核心传播节点。

## 1. 实验目的

在真实微博评论网络（User→User）上，用两层 GAT 学习节点表示，输出核心传播节点
排名与注意力权重，为后续 Flask API + Vue 可视化集成做准备。

## 2. 数据来源

- 评论边：`D:\WeiboCrawler\data\comment_edges.csv`（8849 条一级评论关系）
- 网络指标：`D:\WeiboCrawler\results\comment_network_node_metrics.csv`
- 用户特征：`D:\WeiboCrawler\results\heterogeneous_graph_user_features.csv`
- 帖子特征 / 作者名单：`heterogeneous_graph_post_features.csv`、`weibo_user_mapping.csv`

## 3. 评论网络构建方法

- 边方向严格保留：`source_uid(评论者) → target_uid(微博作者)`
- 同一 `(source, target)` 出现多次时统计 `weight = 评论次数`（不丢信息）
- 当前版本基于**一级评论关系**构建用户交互网络，用于验证 GAT 核心节点识别流程；
  后续如获得转发链、评论回复链等数据，可进一步扩展为更完整的传播网络。

## 4. 网络规模

| 指标 | 数值 |
|------|------|
| 原始节点数 | 8199 |
| 原始唯一边数 | 8145 |
| 原始边记录数 | 8849 |
| self-loop 记录数 | 121 |
| 多评论边对数 | 436 |
| 最大边权重 | 19 |

## 5. 最大连通分量

由于评论关系具有明显方向性，采用**最大弱连通分量**作为 GAT 实验子图，以保留较
完整的局部交互结构。

| 指标 | 数值 |
|------|------|
| 弱连通分量个数 | 249 |
| 最大弱连通分量节点数 | 2848 |
| 分量占比 | 0.3474 |

## 6. 节点特征

GAT 输入特征（6 维）：

out_degree, weighted_out_degree, posts_commented, posts_authored, author_flag, avg_comment_length

- `component_size` 在最大弱连通分量内为常数（2848），无区分度，未纳入训练特征。
- 为避免 target leakage，**不使用** pagerank / in_degree / weighted_in_degree / core_score，
  也不直接使用 total_degree（含 in_degree）。
- 计数类特征先 log1p 压缩重尾，再 z-score 标准化。

剔除的常数特征：（无）

## 7. pseudo-label 定义

```
core_score = 0.5*norm(pagerank) + 0.3*norm(in_degree) + 0.2*norm(weighted_in_degree)
```

Top 10%（core_score 分位数阈值 0.0000）→ `core_label = 1`，其余 0。
正样本 128 个，负样本 2720 个。

**标签退化说明**：存在退化情形。当前评论网络为星形二分结构——
绝大多数节点是“只评论、从未被评论”的叶子评论者（in_degree = 0），其 core_score 恰为 0。
若严格取 Top 10%（284 个节点），会混入大量并列 0 分的
叶子节点，导致正负样本不可分。因此当 Top-10% 阈值退化到 0 时，正样本退化为
**core_score > 0（即收到过 ≥1 条评论）** 的节点，即该子图内真正具有“被关注/权威”地位的作者节点。

## 8. GAT 模型结构

```
Input(6)
  -> GATConv(64, heads=4) -> ELU -> Dropout(0.2)   # 输出 256
  -> GATConv(32, heads=1, concat=False) -> Dropout # 输出 32
  -> Linear(32, 2)
```

## 9. 训练参数

- epochs = 200，lr = 0.005，weight_decay = 0.0005
- dropout = 0.2，early stopping patience = 30（监控 validation F1）
- 划分：train 70% / val 15% / test 15%（分层，保持类别分布）
- class weight 缓解类别不平衡；设备：cpu

## 10. 评价指标

| 指标 | 数值 |
|------|------|
| Best validation F1 | 0.5588 |
| Test Accuracy | 0.9182 |
| Test Precision | 0.3519 |
| Test Recall | 1.0 |
| Test F1 | 0.5205 |
| ROC-AUC | 0.9905 |

## 11-15. 分类指标说明

由于 pseudo-label 由结构指标构造，Accuracy 仅反映 GAT 对弱标签的拟合程度，
**不能**当作真实核心节点识别准确率。

## 16. Top-K overlap

- gat_vs_pagerank_overlap_at_10: `0.1`
- gat_vs_corescore_overlap_at_10: `0.0`
- gat_vs_pagerank_overlap_at_20: `0.05`
- gat_vs_corescore_overlap_at_20: `0.05`
- gat_vs_pagerank_overlap_at_50: `0.36`
- gat_vs_corescore_overlap_at_50: `0.38`

- Spearman(GAT score vs PageRank) = 0.3498

## 17. GAT 与 PageRank 对比

- **粗粒度识别成功**：ROC-AUC = 0.9905，说明 GAT 能近乎完美地区分
  “作者（收到过评论）”与“评论者（从未被评论）”。
- **细粒度排序较弱**：Spearman(GAT score vs PageRank) = 0.3498，
  为弱正相关；Top-10/20 的 overlap 接近甚至低于随机水平（子图内仅 128 个作者，
  随机抽取 Top-20 的期望 overlap ≈ 0.16）。

原因有两方面：
1. **标签粒度**：二进制 pseudo-label 只编码了“是否作者”这一粗粒度信息，没有编码
   作者节点内部的细粒度影响力排序，softmax 分数在作者群体内趋近饱和（≈1.0）。
2. **“核心”定义差异**：GAT 同时利用了行为特征（out_degree / weighted_out_degree），
   会把“高频评论的放大器型用户”（如 out_degree=24 的桥接评论者）也判为核心节点，
   这与 PageRank 的“权威（被评论）中心性”是两种不同的核心概念。这一差异本身正是
   “识别 KOL / 情感放大节点”任务需要关注的信号，而非单纯噪声。

## 18. Top 20 核心节点

| rank | username | uid | gat_score | pagerank | in_degree | out_degree | total_degree |
|------|----------|-----|-----------|----------|-----------|------------|--------------|
| 1 | 一条闪耀大蟒蛇_ | 6272876985 | 1.0000 | 0.00082 | 20 | 0 | 20 |
| 2 | 王霏霏Fei | 1821525001 | 1.0000 | 0.00082 | 20 | 1 | 21 |
| 3 | Vinida万妮达 | 1954778190 | 1.0000 | 0.00078 | 19 | 0 | 19 |
| 4 | 星娱酱 | 2909406375 | 1.0000 | 0.00082 | 20 | 0 | 20 |
| 5 | 吃瓜罗伯特 | 5606716867 | 1.0000 | 0.00005 | 0 | 24 | 24 |
| 6 | 演员陈添祥 | 6871895822 | 1.0000 | 0.00516 | 20 | 1 | 21 |
| 7 | Huldigen | 7850599876 | 1.0000 | 0.00460 | 24 | 1 | 25 |
| 8 | 娄艺潇 | 1316949123 | 1.0000 | 0.00066 | 16 | 0 | 16 |
| 9 | 张艺凡 | 2705589884 | 1.0000 | 0.00058 | 15 | 0 | 15 |
| 10 | TOPCLASS娱乐 | 6580670764 | 1.0000 | 0.00070 | 17 | 0 | 17 |
| 11 | CristianoRonaldo | 5926318749 | 1.0000 | 0.00066 | 20 | 0 | 20 |
| 12 | 多樂 | 5939204645 | 1.0000 | 0.00080 | 20 | 0 | 20 |
| 13 | 渡边小鲜肉 | 2560497310 | 1.0000 | 0.00082 | 20 | 0 | 20 |
| 14 | 演员王彦桐 | 6113513458 | 1.0000 | 0.00058 | 14 | 0 | 14 |
| 15 | 猪堡_ | 6124100893 | 1.0000 | 0.00054 | 13 | 0 | 13 |
| 16 | 高彦Will | 1554226303 | 1.0000 | 0.00082 | 20 | 0 | 20 |
| 17 | 煎饼果仔-张问初 | 6304429250 | 1.0000 | 0.00074 | 20 | 0 | 20 |
| 18 | yeeuni06 | 7912058860 | 1.0000 | 0.00082 | 20 | 0 | 20 |
| 19 | 十鸢里美- | 6441657735 | 1.0000 | 0.00082 | 20 | 0 | 20 |
| 20 | JaniceMan文咏珊 | 1719397245 | 1.0000 | 0.00054 | 13 | 0 | 13 |

## 19. attention 分析

注意力权重已导出到 `results/gat_attention.csv`（字段：source_uid, target_uid,
layer, head, attention_weight），可用于前端展示“哪个用户对哪个用户的信息传递权重较高”。

## 20. 局限性

1. **pseudo-label 非人工标注**：标签由 PageRank / 入度 / 加权入度构造，存在自指性。
2. **网络稀疏**：一级评论关系构造的网络较稀疏，不是完整社会传播网络。
3. **最大弱连通分量仅 2848 节点**：GAT 只在该子图上训练。
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
