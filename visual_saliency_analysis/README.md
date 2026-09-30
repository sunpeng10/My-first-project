# 视觉显著性与传播力相关性分析

基于 BASNet 对微博图片做显著性检测，提取视觉特征，与互动指标（点赞 / 评论 / 转发）合并，分析视觉显著性与社交媒体传播力的相关性。数据与代码均在 `visual_saliency_analysis/` 下。

## 目录结构

```
visual_saliency_analysis/
├── data/
│   ├── images/weibo/          # 500 张微博图片 wb_0001.jpg ~ wb_0500.jpg（未入库）
│   ├── weibo_metadata.csv     # 微博互动指标（image_id, text, likes, comments, shares）
│   └── weibo_collection.csv   # 采集模板（9 字段空模板）
├── outputs/
│   ├── saliency_maps/         # 500 张显著性图（未入库）
│   ├── saliency_quality_report.csv  # 异常检测报告（34 条）
│   ├── saliency_visual_check/ # 随机对比图
│   └── dataset_merge_report.txt
├── features/
│   └── visual_features.csv    # 500 行 × 8 列视觉特征
├── graph/                     # 网络构建数据（post_authors.csv 等，后端节点详情读取）
├── results/
│   ├── weibo_visual_dataset.csv   # 合并后完整数据集（500 行 × 12 列）
│   ├── analysis_dataset.csv
│   └── correlation_analysis.csv
└── scripts/                   # 各处理脚本
```

## 视觉特征（8 项）

显著区域面积占比、平均显著强度、强度离散度、显著性熵、中心偏移、连通域数量、最大连通域占比等。

## 处理流水线

1. `scripts/download_weibo_images.py` 下载微博图片
2. `scripts/generate_saliency.py` 批量生成显著性图（调用 BASNet）
3. `scripts/extract_visual_features.py` 提取视觉特征
4. `scripts/merge_visual_propagation_dataset.py` 合并视觉特征与互动指标
5. `scripts/correlation_analysis.py` 相关性分析

## 基线模型

- BASNet (CVPR 2019)，权重 `basnet/saved_models/basnet_bsi/basnet.pth`（未入库，获取方式见根 README）
