# Multimodal Public Opinion Analysis API（后端）

把已完成的三个分析能力封装为 Flask REST API，供 Vue 前端调用：

| 能力 | 模型/数据 | API |
|------|-----------|-----|
| 文本情感分析 | RoBERTa + Captum Integrated Gradients | `POST /api/text` |
| 图像视觉显著性 | BASNet | `POST /api/image` |
| 传播网络 / 核心节点 | GAT 结果文件 | `GET /api/graph` 等 |

> 本阶段**只做 API 封装**：不重新训练模型、不修改原始数据、不重构三个项目。

---

## 1. 后端架构

```
backend/
├── app.py            # Flask 启动入口 + 路由 + 错误处理 + 启动日志
├── config.py         # 路径 / 限制 / CORS / API 元信息（唯一需要改路径的地方）
├── schemas.py        # 请求字段校验（文本 / 图片 / limit）
├── text_service.py   # RoBERTa + IG 服务（懒加载单例）
├── image_service.py  # BASNet 服务（懒加载单例）
├── graph_service.py  # GAT 结果读取 + 内存索引（懒加载单例）
├── utils.py          # APIError / 响应构造 / 安全文件名 / 设备选择
├── requirements.txt  # 新增依赖（Flask / flask-cors / requests）
└── README.md
```

服务以**单例**形式在进程内只加载一次模型/数据，不在每个请求里
`from_pretrained` 或重复读取 CSV。

---

## 2. 安装依赖

```bash
pip install flask flask-cors requests
```

其余依赖（`torch`、`transformers`、`captum`、`scikit-image`、`opencv-python`、
`pandas`、`numpy`、`Pillow`）来自项目已有环境，无需重装，也**不要整体导出**
`requirements.txt` 覆盖原环境。

---

## 3. 启动方法

在项目根目录 `D:\BASNet` 下：

```bash
python -m backend.app
# 或
python backend/app.py
```

默认监听 `http://127.0.0.1:5000`。

---

## 4. API 列表

| 方法 | 路径 | 说明 |
|------|------|------|
| GET  | `/api` | API 元信息 |
| GET  | `/api/health` | 各模型加载状态 |
| POST | `/api/text` | 文本情感 + 词级显著性 |
| POST | `/api/image` | 图片显著性（multipart 上传） |
| GET  | `/api/image/saliency/<filename>` | 显著性图文件 |
| GET  | `/api/graph` | 传播网络（nodes/edges） |
| GET  | `/api/top-nodes?limit=20` | 核心节点排名 |
| GET  | `/api/node/<uid>` | 单节点详情 |
| GET  | `/api/attention?uid=xxx&limit=1000` | 注意力边 |

---

## 5. 请求示例

### 文本

```bash
curl -X POST http://127.0.0.1:5000/api/text \
  -H "Content-Type: application/json" \
  -d '{"text": "这个手机太差了"}'
```

```bash
# 指定解释方法（默认 ig，另支持 saliency）
curl -X POST http://127.0.0.1:5000/api/text \
  -H "Content-Type: application/json" \
  -d '{"text": "这个手机太差了", "method": "saliency"}'
```

### 图片

```bash
curl -X POST http://127.0.0.1:5000/api/image \
  -F "file=@test.jpg"
```

### 传播网络

```bash
curl "http://127.0.0.1:5000/api/graph"
curl "http://127.0.0.1:5000/api/top-nodes?limit=5"
curl "http://127.0.0.1:5000/api/node/6272876985"
curl "http://127.0.0.1:5000/api/attention?uid=6272876985&limit=50"
```

---

## 6. 返回示例

### `POST /api/text`

```json
{
  "success": true,
  "data": {
    "text": "这个手机太差了",
    "label": "negative",
    "label_id": 0,
    "confidence": 0.9947,
    "method": "ig",
    "words": [
      {"word": "这", "score": 0.05, "score_raw": -0.0121,
       "token_index": 0, "token_text": "这", "char_range": [0, 1]},
      {"word": "差", "score": 1.0, "score_raw": -0.2638,
       "token_index": 5, "token_text": "差", "char_range": [5, 6]}
    ]
  }
}
```

- `char_range` 为原始文本中的 `[起, 止)` 字符区间，前端据此高亮原文。
- 中文 RoBERTa-wwm 的 token 粒度是**单字**，因此 `words` 为 token 级（不伪造词边界）。
- `score` 为按最大绝对值归一化的显著性大小，`score_raw` 为带符号的原始归因值。

### `POST /api/image`

```json
{
  "success": true,
  "data": {
    "filename": "test.jpg",
    "saliency": {
      "saliency_area_ratio": 0.23,
      "mean_saliency": 0.51,
      "saliency_std": 0.18,
      "saliency_entropy": 0.72,
      "center_bias": 0.63,
      "component_count": 3,
      "largest_component_ratio": 0.61
    },
    "saliency_map_url": "/api/image/saliency/<uuid>.png"
  }
}
```

### `GET /api/graph`

```json
{"success": true, "data": {"nodes": [...], "edges": [...]}}
```

---

## 7. 错误码

所有错误统一返回：

```json
{"success": false, "error": {"code": "CODE", "message": "..."}}
```

| HTTP | code | 触发场景 |
|------|------|----------|
| 400 | `INVALID_REQUEST` | 文本为空/超长、method 不支持、图片非法、limit 非法 |
| 404 | `NOT_FOUND` | UID 不存在、显著性图不存在、路由不存在 |
| 405 | `INVALID_REQUEST` | 方法不允许 |
| 413 | `FILE_TOO_LARGE` | 上传图片超过 10 MB |
| 500 | `INTERNAL_ERROR` | 模型/推理内部错误 |

Python traceback **不会**直接返回给前端，只在终端日志打印。

---

## 8. CORS

开发环境允许 Vue dev server：

```python
CORS_ORIGINS = ["http://localhost:5173", "http://127.0.0.1:5173"]
```

生产环境请**限制 origin**，不要使用 `CORS(app)` 全放开。

---

## 9. 模型加载

- 三个服务（`TextService` / `ImageService` / `GraphService`）在进程内**只加载一次**，
  用线程锁 + `loaded` 标志保证幂等。
- 启动时在 `__main__` 中主动 `load()`，任何失败都会被捕获并记为 `not_loaded`，
  服务仍能启动，`/api/health` 会准确返回 `not_loaded`。
- 设备自动选择：`cuda` 可用则 `cuda`，否则 `cpu`（当前 GAT/BASNet 环境为 CPU）。

---

## 10. 数据来源

| 数据 | 路径 | 说明 |
|------|------|------|
| 文本模型 | `D:\vscode python files\sentiment_baseline\models` | RoBERTa 权重（只读） |
| 图像模型 | `saved_models/basnet_bsi/basnet.pth` | BASNet 权重（只读） |
| 图结构 | `results/gat_graph.json` | 2848 nodes / 2999 edges |
| 核心节点 | `results/gat_top_nodes.csv` | 排名表 |
| 节点详情 | `results/gat_node_scores.csv` | UID 索引 |
| 注意力边 | `results/gat_attention.csv` | 约 14995 条 |

文本模型位于独立项目目录，路径由 `backend/config.py` 的 `SENTIMENT_ROOT` 配置
（可用环境变量 `SENTIMENT_ROOT` 覆盖），**不复制、不重训**。

---

## 11. 当前研究局限

- 核心节点标签为 **network-based pseudo-label**，**不是人工 Ground Truth**。
- 网络基于每条微博**最多 20 条一级评论**构建，**不是完整微博传播网络**。
- GAT 结果应表述为「**GAT-based core propagation node identification**」，
  不声称「GAT 成功预测了真实 KOL」。
- 文本显著性为 token（单字）级，未做中文分词（避免引入未经验证的词边界）。
