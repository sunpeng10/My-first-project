# 社交媒体图片视觉显著性与传播力相关性分析

Visual Saliency & Virality Correlation Analysis for Social Media Images

## 概述

本项目基于 BASNet (Boundary-Aware Salient Object Detection) 作为基线模型，研究社交媒体图片的视觉显著性（Saliency Map）与其传播力指标（点赞、转发、评论等）之间的相关性。

## 目录结构

```
visual_saliency_analysis/
├── data/
│   ├── images/              # 原始社交媒体图片
│   └── metadata.csv         # 图片元数据（文件名、平台、互动指标等）
├── outputs/
│   └── saliency_maps/       # BASNet 生成的显著性图
├── features/
│   └── visual_features.csv  # 从显著性图提取的数值特征
├── scripts/
│   ├── generate_saliency.py # 调用 BASNet 批量生成显著性图
│   ├── extract_features.py  # 从显著性图提取数值特征
│   └── correlation_analysis.py # 显著性与传播力相关性统计
├── results/                 # 统计分析结果（图表、报告）
└── README.md
```

## 特征维度

从 Saliency Map 中提取的视觉特征包括：
- 显著区域面积占比
- 显著区域数量
- 显著区域空间分布（中心偏移）
- 显著区域对比度
- 显著区域熵

## 传播力指标

metadata.csv 中可包含的指标：
- likes（点赞数）
- shares（转发数）
- comments（评论数）
- impressions（展示量）
- engagement_rate（互动率）

## 基线模型

- BASNet (CVPR 2019) — Boundary-Aware Salient Object Detection
- 模型权重：`saved_models/basnet_bsi/basnet.pth`

## 分析方法

1. 皮尔逊相关系数（Pearson's r）
2. 斯皮尔曼秩相关（Spearman's ρ）
3. 多元线性回归
4. 分组对比（高传播 vs 低传播）

## 使用步骤

1. 将社交媒体图片放入 `data/images/`
2. 编辑 `data/metadata.csv` 填入互动指标
3. 运行 `scripts/generate_saliency.py` 生成显著性图
4. 运行 `scripts/extract_features.py` 提取数值特征
5. 运行 `scripts/correlation_analysis.py` 进行相关性分析

## 作者

BASNet Baseline: Xuebin Qin et al. (2019)
