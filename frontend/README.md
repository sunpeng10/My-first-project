# 多模态社交媒体舆情分析系统 — 前端

基于 **Vue 3 + Vite** 的科研演示前端，对接现有 Flask 后端（`backend/`），
用于展示三类能力：文本情感显著性、图像视觉显著性、评论传播网络核心节点。

前端只负责 **调用 API 与展示结果**，不包含任何模型推理、不重新训练、不生成 mock 数据。

---

## 1. 环境要求

- Node.js >= 18（已在 Node v24 验证）
- npm >= 9
- 已启动的 Flask 后端（见下文）

## 2. 安装

```bash
cd frontend
npm install
```

## 3. 启动前端

```bash
npm run dev
```

浏览器访问：<http://localhost:5173>

## 4. 启动 Flask 后端

在项目根目录（`D:\BASNet`）：

```bash
python -m backend.app
```

后端监听：<http://127.0.0.1:5000>

> 后端启动时会依次加载 RoBERTa / BASNet / GAT 数据，请耐心等待日志输出
> `[4] Flask API ready`。

## 5. API 地址与代理

- 后端基础地址：`http://127.0.0.1:5000`
- 前端通过 `VITE_API_BASE_URL` 环境变量指定后端地址，默认值即上述地址。

如需自定义，复制 `.env.example` 为 `.env` 并修改：

```
VITE_API_BASE_URL=http://127.0.0.1:5000
```

后端已通过 `flask-cors` 放行 `http://localhost:5173`，无需前端代理。

## 6. 页面功能

| 区域 | 说明 | 对应 API |
| --- | --- | --- |
| 顶部状态 | 后端健康检查（系统在线 / 部分服务不可用） | `GET /api/health` |
| 文本情感分析 | RoBERTa + Captum（IG / Saliency），字/词显著性高亮（词级 jieba 分词聚合，可切换） | `POST /api/text` |
| 图像显著性分析 | BASNet 显著性图 + 7 项视觉指标 | `POST /api/image`、`GET /api/image/saliency/<file>` |
| 传播网络 | vis-network 力导向图，节点大小/颜色按 GAT score | `GET /api/graph` |
| 核心节点 Top 10/20 | 排名表 + ECharts 统计图 | `GET /api/top-nodes` |
| 节点详情 | 点击节点显示字段与 Attention 边 | `GET /api/node/<uid>`、`GET /api/attention` |

## 7. 技术栈与依赖

- [Vue 3](https://vuejs.org/) + [Vite](https://vitejs.dev/) + Vue Router
- [Axios](https://axios-http.com/)（统一封装于 `src/api/index.js`）
- [ECharts](https://echarts.apache.org/)（Top 10 GAT Score 柱状图、GAT/PageRank 对比图）
- [vis-network](https://visjs.github.io/vis-network/)（传播网络图）

## 8. 数据来源

- 传播网络与核心节点数据来自 `results/` 下 GAT 已生成结果：
  - `gat_graph.json`（2848 nodes / 2999 edges）
  - `gat_top_nodes.csv`
  - `gat_node_scores.csv`
  - `gat_attention.csv`
- 文本模型为 RoBERTa（外部项目只读加载），图像模型为 BASNet 已训练权重。
- 前端不读取、不修改任何原始数据文件。

## 9. GAT 网络局限（重要）

- **传播网络基于每条微博最多 20 条一级评论构建。**
- **GAT 核心节点为基于网络结构的 pseudo-label 识别结果，并非人工 Ground Truth。**
- 因此页面使用「核心传播节点 / GAT 核心节点」等表述，
  **不做「真实舆论领袖」「准确识别 KOL」等过强结论**。

---

## 目录结构

```
frontend/
├─ package.json
├─ vite.config.js
├─ index.html
├─ .env / .env.example
├─ src/
│  ├─ main.js
│  ├─ App.vue
│  ├─ assets/main.css
│  ├─ api/index.js
│  ├─ router/index.js
│  ├─ views/Dashboard.vue
│  └─ components/
│     ├─ HeaderBar.vue
│     ├─ TextAnalysisPanel.vue
│     ├─ SaliencyText.vue
│     ├─ ImageAnalysisPanel.vue
│     ├─ SaliencyImage.vue
│     ├─ StatCard.vue
│     ├─ NetworkGraph.vue
│     ├─ TopNodesPanel.vue
│     ├─ ChartsPanel.vue
│     └─ NodeDetailPanel.vue
```
