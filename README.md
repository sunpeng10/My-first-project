# 多模态微博舆情分析系统

一个面向微博评论数据的**多模态舆情分析**系统：从微博帖子的**文本**、**图片**、**评论传播关系**三个维度出发，分别完成情感分析、视觉显著性检测、核心传播节点识别，并通过 Web 界面统一可视化。适合作为社交媒体舆情分析 / 传播动力学 / 多模态学习的科研演示与复现基线。

---

## 1. 项目背景与目标

社交媒体上一条微博的"传播力"，不仅取决于文案情绪（文本），也取决于配图的视觉冲击力（图像），以及它在评论关系网络中所处的位置（结构）。本项目把这三条线打通：

- **文本**：这条微博表达了什么情绪？哪些词是情绪的关键？
- **图像**：配图的视觉显著性有多强？视觉特征能否预测传播力？
- **结构**：在 User→User 评论网络中，谁是核心传播节点？

最终整合为一个「采集 → 分析 → 可视化」的端到端系统。

## 2. 系统总览（分层架构）

```
┌──────────────────────────────────────────────────────┐
│  前端 frontend/  (Vue 3 + Vite + ECharts + vis-network)│
│  文本分析面板 · 图像显著性面板 · 传播网络图 · 节点详情  │
└─────────────────────────┬────────────────────────────┘
                          │ REST API  http://127.0.0.1:5000
┌─────────────────────────▼────────────────────────────┐
│  后端 backend/  (Flask)                                │
│  /api/text   /api/image   /api/graph  /api/node  ...   │
└──────┬──────────────┬───────────────┬─────────────────┘
       │              │               │
  TextService    ImageService    GraphService   (懒加载单例)
       │              │               │
   RoBERTa        BASNet          results/(GAT 结果)
   (情感+归因)   (显著性+特征)    (图/排名/注意力)
```

数据流自下而上：**采集 → 模型推理 → 结果落盘 → API → 前端展示**，每一层只依赖上一层的结果文件，模型与数据在进程内只加载一次。

## 3. 核心能力与研究方法

### 3.1 文本情感分析（`POST /api/text`）

- **模型**：中文 RoBERTa-wwm 微调的情感分类器
- **可解释性**：Captum Integrated Gradients / Saliency，输出**逐字显著性**，前端按字高亮情绪关键点
- **输出**：情感标签（正向 / 负向）、置信度、每个字（token）的归因分数

### 3.2 图像视觉显著性（`POST /api/image`）

- **模型**：BASNet (Boundary-Aware Salient Object Detection, CVPR 2019)
- **流水线**：500 张微博图片 → 显著性图（Saliency Map）→ 提取 **7 项视觉特征**（显著区域面积占比、平均显著强度、强度离散度、显著性熵、中心偏移、连通域数量、最大连通域占比）
- **下游**：视觉特征与互动指标（点赞 / 评论 / 转发）合并，做显著性与传播力的**相关性分析**

### 3.3 传播网络核心节点（`GET /api/graph` 等）

- **方法**：在真实 User→User 评论网络上，用**两层 GAT (Graph Attention Network)** 学习节点表示
- **标签**：PageRank + 入度 + 加权入度构造的 **pseudo-label**（弱标签，非人工 Ground Truth）
- **输出**：核心节点排名、注意力边、每节点分数，前端用 vis-network 力导向图可视化
- **关键结果**：ROC-AUC 0.9905（作者 / 评论者粗粒度区分极好），Test F1 0.52（受类别不平衡与标签退化影响）

## 4. 数据与处理流水线

```
微博采集 (WeiboCrawler，独立项目)
   │
   ├── 帖子图片 ──> BASNet ──> 显著性特征 ──> 与互动指标合并 ──> 相关性分析
   │                                                          (visual_saliency_analysis/)
   └── 评论边 ────> GAT ────> 核心节点 / 注意力 ──> results/ ──> 后端 GraphService
```

- **采集端**：Python + Selenium，搜索微博并保存为 CSV + 图片（位于独立项目 `WeiboCrawler`，未入本仓库）
- **评论网络**：`comment_edges.csv`（8849 条一级评论关系，source=评论者 → target=作者），网络呈**星形二分结构**（多数节点是叶子评论者）

## 5. 项目结构

```
├── backend/                   # Flask API：封装三大能力（见 backend/README.md）
├── frontend/                  # Vue 3 + Vite 前端（见 frontend/README.md）
├── sentiment/                 # 文本情感训练代码（见 sentiment/README.md）
├── gat/                       # GAT 模型定义 / 训练 / 推理（见 gat/README.md）
├── scripts/run_gat.py         # GAT 完整流程入口
├── results/                   # GAT 已生成结果（图结构 / 排名 / 注意力，后端直接读）
├── models/gat_best.pt         # 已训练 GAT 权重（46 KB）
├── basnet/                    # BASNet 上游（模型 / 推理 / 训练 / 权重）
│   ├── model/                 #   BASNet 模型代码（上游，未修改）
│   ├── data_loader.py         #   数据预处理
│   ├── basnet_test.py         #   推理脚本（已指向微博数据）
│   ├── basnet_train.py        #   训练脚本
│   ├── pytorch_iou/ pytorch_ssim/    # 损失函数
│   └── saved_models/basnet_bsi/     # basnet.pth（未入库，需下载）
└── visual_saliency_analysis/  # 视觉显著性分析流水线（见其 README）
```

## 6. 技术栈

| 层 | 技术 |
|----|------|
| 文本 | PyTorch, Transformers, RoBERTa-wwm, Captum |
| 图像 | PyTorch, BASNet, scikit-image, OpenCV, NumPy |
| 图 | PyTorch Geometric (GATConv) |
| 后端 | Flask, flask-cors |
| 前端 | Vue 3, Vite, Axios, ECharts, vis-network |
| 数据 | Pandas, NumPy, Selenium（采集，独立项目） |

## 7. 快速开始

```bash
# 1. 后端（在项目根目录下）
python -m backend.app          # 监听 http://127.0.0.1:5000

# 2. 前端（另开终端）
cd frontend
npm install
npm run dev                    # http://localhost:5173
```

## 8. 运行依赖与复现状态

| 依赖 | 大小 | 位置 | 状态 |
|------|------|------|------|
| `basnet.pth`（BASNet 权重） | 348 MB | `basnet/saved_models/basnet_bsi/` | 未入库，[GoogleDrive](https://drive.google.com/open?id=1s52ek_4YTDRt_EOkx1FS53u-vJa0c4nu) 下载 |
| RoBERTa 情感模型 | 391 MB | `SENTIMENT_ROOT` 指定 | **待上传** |
| 微博原图 | 267 MB | `D:\WeiboCrawler\images\` | 采集项目产出 |
| `comment_edges.csv` | 1.2 MB | `D:\WeiboCrawler\data\` | 采集项目产出 |

**当前复现状态**：传播网络 / 核心节点 / 节点详情（文字）开箱即用；文本情感、图片显著性、节点原图需补齐上述依赖。

## 9. 已知局限（务必阅读）

- **核心节点标签是 network-based pseudo-label，不是人工 Ground Truth**，GAT 结果应表述为「GAT-based core propagation node identification」，不声称「预测了真实 KOL」。
- 评论网络基于每条微博**最多 20 条一级评论**构建，不是完整微博传播网络。
- 文本显著性为 token（单字）级，未做中文分词（避免引入未经验证的词边界）。

## 10. 上游模型与署名

图像显著性模块基于 **BASNet**（Boundary-Aware Salient Object Detection，Xuebin Qin et al., CVPR 2019），沿用其模型代码与预训练权重，遵守 MIT 协议（见 `LICENSE`）。

- 原仓库：https://github.com/xuebinqin/BASNet
- 权重下载：[GoogleDrive](https://drive.google.com/open?id=1s52ek_4YTDRt_EOkx1FS53u-vJa0c4nu) 或 [百度网盘](https://pan.baidu.com/s/1PrsBdepwrkMWPLSW22FhAg)（提取码 6phq）

```bibtex
@InProceedings{Qin_2019_CVPR,
  author = {Qin, Xuebin and Zhang, Zichen and Huang, Chenyang and Gao, Chao and Dehghan, Masood and Jagersand, Martin},
  title = {BASNet: Boundary-Aware Salient Object Detection},
  booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  month = {June},
  year = {2019}
}
```

## 11. 模块文档

- [`backend/README.md`](backend/README.md) — API 列表、请求示例、错误码、数据来源
- [`frontend/README.md`](frontend/README.md) — 页面功能、环境配置、技术栈
- [`sentiment/README.md`](sentiment/README.md) — 文本情感训练代码（RoBERTa-wwm + Captum）
- [`gat/README.md`](gat/README.md) — GAT 训练 / 推理、pseudo-label 局限
- [`visual_saliency_analysis/README.md`](visual_saliency_analysis/README.md) — 视觉显著性分析流水线
- [`results/gat_report.md`](results/gat_report.md) — GAT 实验报告
