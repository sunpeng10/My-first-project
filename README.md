# 多模态微博舆情分析系统

基于微博评论数据，集成 **文本情感分析、图像视觉显著性、传播网络核心节点识别** 三大能力，并提供 Web 可视化界面（Flask 后端 + Vue 3 前端）。

## 系统能力

| 能力 | 技术 | API | 说明 |
|------|------|-----|------|
| 文本情感分析 | RoBERTa + Captum（IG / Saliency） | `POST /api/text` | 逐字显著性高亮 |
| 图像视觉显著性 | BASNet (CVPR 2019) | `POST /api/image` | 显著性图 + 7 项视觉指标 |
| 传播网络核心节点 | 两层 GAT | `GET /api/graph` 等 | 评论网络核心传播节点识别 |

## 项目结构

```
├── backend/                   # Flask API：封装三大能力（见 backend/README.md）
├── frontend/                  # Vue 3 + Vite 前端（见 frontend/README.md）
├── gat/                       # GAT 模型定义 / 训练 / 推理（见 gat/README.md）
├── scripts/run_gat.py         # GAT 完整流程入口
├── results/                   # GAT 已生成结果（图结构 / 排名 / 注意力，后端直接读）
├── models/gat_best.pt         # 已训练 GAT 权重（46 KB）
├── model/                     # BASNet 官方模型代码（上游，未修改）
├── visual_saliency_analysis/  # 视觉显著性分析流水线（见其 README）
├── basnet_test.py             # BASNet 推理脚本（已指向微博数据）
├── data_loader.py             # 数据预处理
└── saved_models/basnet_bsi/   # basnet.pth 存放处（未入库，需下载）
```

## 数据流

微博采集项目（WeiboCrawler）产出两条数据线：

- **图片 → BASNet → 视觉显著性特征** → 与互动指标合并 → 相关性分析（`visual_saliency_analysis/`）
- **评论边 → GAT → 核心节点 / 注意力** → 写入 `results/` → 后端 `graph_service` 读取

三路结果（RoBERTa 文本、BASNet 图像、GAT 图）在 `backend/` 统一封装为 REST API，由 `frontend/` 可视化。

## 功能可用性（clone 后）

| 功能 | 开箱即用 | 缺少时的影响 |
|------|:---:|------|
| 传播网络 / 核心节点 / 节点详情 | ✅ | 无（`results/` 已入库） |
| 节点详情的原图展示 | ⚠️ | 需微博原图（见「运行依赖」） |
| 文本情感分析 | ❌ | 需 RoBERTa 模型 |
| 图片显著性分析 | ❌ | 需 `basnet.pth` |

> 缺少模型 / 图片时后端仍能启动，`GET /api/health` 会如实返回 `not_loaded`。

## 快速启动

```bash
# 1. 后端（在项目根目录下）
python -m backend.app          # 监听 http://127.0.0.1:5000

# 2. 前端（另开终端）
cd frontend
npm install
npm run dev                    # http://localhost:5173
```

## 运行依赖（clone 后需自行准备）

| 依赖 | 大小 | 存放位置 | 获取方式 |
|------|------|----------|----------|
| BASNet 权重 `basnet.pth` | 348 MB | `saved_models/basnet_bsi/basnet.pth` | [GoogleDrive](https://drive.google.com/open?id=1s52ek_4YTDRt_EOkx1FS53u-vJa0c4nu) 或百度网盘（见文末上游说明） |
| RoBERTa 情感模型 | 391 MB | 任意目录，用 `SENTIMENT_ROOT` 环境变量或 `backend/config.py` 指定 | 待上传（Hugging Face / Release），见 [`backend/README.md`](backend/README.md) |
| 微博原图 | 267 MB | `D:\WeiboCrawler\images\`（可在 `backend/config.py` 改） | 采集项目产出，或运行 `visual_saliency_analysis/scripts/download_weibo_images.py` |

> 这三项均未入库：`basnet.pth` 超过 GitHub 单文件 100 MB 限制；后两者属于独立的采集 / 情感分析项目。

## 模块文档

- [`backend/README.md`](backend/README.md) — API 列表、请求示例、错误码、数据来源
- [`frontend/README.md`](frontend/README.md) — 页面功能、环境配置、技术栈
- [`gat/README.md`](gat/README.md) — GAT 训练 / 推理、pseudo-label 局限
- [`visual_saliency_analysis/README.md`](visual_saliency_analysis/README.md) — 视觉显著性分析流水线
- [`results/gat_report.md`](results/gat_report.md) — GAT 实验报告

## 上游模型与署名

本项目的图像显著性模块基于 **BASNet**（Boundary-Aware Salient Object Detection，Xuebin Qin et al., CVPR 2019），沿用其模型代码与预训练权重，遵守 MIT 协议（见 `LICENSE`）。

- 原仓库：https://github.com/xuebinqin/BASNet
- 权重下载：[GoogleDrive](https://drive.google.com/open?id=1s52ek_4YTDRt_EOkx1FS53u-vJa0c4nu) 或 [百度网盘](https://pan.baidu.com/s/1PrsBdepwrkMWPLSW22FhAg)（提取码 6phq）

### Citation

```
@InProceedings{Qin_2019_CVPR,
  author = {Qin, Xuebin and Zhang, Zichen and Huang, Chenyang and Gao, Chao and Dehghan, Masood and Jagersand, Martin},
  title = {BASNet: Boundary-Aware Salient Object Detection},
  booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  month = {June},
  year = {2019}
}
```
