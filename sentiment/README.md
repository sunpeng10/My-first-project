# 文本情感分析 —— 训练基线

中文情感二分类训练代码（RoBERTa-wwm + Captum 可解释性）。训练出的模型由后端 `backend/text_service.py` 加载，用于 `/api/text` 情感分析与逐字显著性。

## 数据与基座（均为公开资源）

- 基座模型：`hfl/chinese-roberta-wwm-ext`
- 训练数据：`lansinuote/ChnSentiCorp`（公开中文情感数据集，seamew 镜像无网络问题）
- 任务：正 / 负 二分类

## 用法

```bash
python main.py train                          # 训练
python main.py eval                           # 在测试集评估
python main.py predict --text "这家酒店不错"    # 单句预测
python main.py predict                        # 交互式预测
```

训练配置（模型名、数据集、超参、路径）在 `src/config.py`，路径相对项目根目录自动推导，无需修改即可运行。

## 目录结构

```
sentiment/
├── main.py             # CLI 入口（train / eval / predict）
├── src/
│   ├── config.py       # 模型名、数据集、超参、路径
│   ├── data_loader.py  # 数据加载与预处理
│   ├── model.py        # 模型定义
│   ├── trainer.py      # 训练 pipeline
│   ├── evaluate.py     # 评估
│   ├── inference.py    # 预测 + Captum 归因
│   ├── metrics.py      # 评估指标
│   └── utils.py
└── requirements.txt
```

## 与后端的关系

训练完成后，模型保存到 `models/`（或 `outputs/` 下的最佳 checkpoint）。后端 `backend/config.py` 通过 `SENTIMENT_ROOT` 环境变量（默认 `D:\vscode python files\sentiment_baseline`）定位模型目录并只读加载。本仓库未上传的 391MB 权重即训练产物。
